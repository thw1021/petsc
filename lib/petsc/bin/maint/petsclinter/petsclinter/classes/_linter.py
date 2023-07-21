#!/usr/bin/env python3
"""
# Created: Mon Jun 20 16:40:24 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from typing import Optional, Any

import re
import weakref
import difflib
import datetime
import itertools
import collections
import clang.cindex as clx
import petsclinter  as pl

from ._diag    import DiagnosticManager, Diagnostic
from ._cursor  import Cursor, CursorLike
from ._path    import Path, PathLike, StrPathLike
from ._src_pos import SourceRange
from ._patch   import Patch

from .._error import ParsingError, KnownUnhandleableCursorError

from .. import util

from ..util._clang import CXTranslationUnit

class WeakList(list):
  """
  Adaptor class to make builtin lists weakly referenceable
  """
  __slots__ = ('__weakref__',)

class Scope:
  """
  Scope encompasses both the logical and lexical reach of a callsite, and is used to
  determine if two function calls may be occur in chronological order. Scopes may be
  approximated by incrementing or decrementing a counter every time a pair of '{}' are
  encountered however it is not that simple. In practice they behave almost identically
  to sets. Every relation between scopes may be formed by the following axioms.

  - Scope A is said to be greater than scope B if one is able to get to scope B from scope A
  e.g.:
  { // scope A
    { // scope B < scope A
      ...
    }
  }
  - Scope A is said to be equivalent to scope B if and only if they are the same object.
  e.g.:
  { // scope A and scope B
    ...
  }

  One notable exception are switch-case statements. Here every 'case' label acts as its
  own scope, regardless of whether a "break" is inserted i.e.:

  switch (cond) { // scope A
  case 1: // scope B begin
    ...
    break; // scope B end
  case 2: // scope C begin
    ...
  case 2:// scope C end, scope D begin
    ...
    break; // scope D end
  }

  Semantics here are weird, as:
  - scope B, C, D < scope A
  - scope B != scope C != scope D
  """
  __slots__ = ('children',)

  children: list[Scope]

  def __init__(self) -> None:
    self.children = []
    return

  def __lt__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return not self >= other

  def __gt__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return self.is_child_of(other)

  def __le__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return not self > other

  def __ge__(self, other: Scope) -> bool:
    assert isinstance(other, Scope)
    return (self > other) or (self == other)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Scope):
      return NotImplemented
    return id(self) == id(other)

  def __ne__(self, other: object) -> bool:
    return not self == other

  def sub(self) -> Scope:
    """
    spawn sub-scope
    """
    child = Scope()
    self.children.append(child)
    return child

  def is_parent_of(self, other: Scope) -> bool:
    """
    self is parent of other
    """
    if self == other:
      return False
    for child in self.children:
      if (other == child) or child.is_parent_of(other):
        return True
    return False

  def is_child_of(self, other: Scope) -> bool:
    """
    self is child of other, or other is parent of self
    """
    return other.is_parent_of(self)

class Addline:
  __slots__    = ('offset',)
  diff_line_re = re.compile(r'^@@ -([0-9,]+) \+([0-9,]+) @@')

  offset: int

  def __init__(self, offset: int) -> None:
    self.offset = offset
    return

  def __call__(self, re_match: re.Match) -> str:
    ll, lr  = re_match.group(1).split(',')
    rl, rr  = re_match.group(2).split(',')
    return f'@@ -{self.offset + int(ll)},{lr} +{self.offset + int(rl)},{rr} @@'

@DiagnosticManager.register(
  ('duplicate-function', 'Check for duplicate function-calls on the same execution path'),
)
class Linter:
  """
  Object to manage the collection and processing of errors during a lint run.
  """
  __slots__ = (
    'flags', 'clang_opts', 'verbose', 'werror', 'err_prefix', 'warn_prefix', 'index', 'errors',
    'warnings', 'patches'
  )

  flags: list[str]
  clang_opts: CXTranslationUnit
  verbose: bool
  werror: bool
  err_prefix: str
  warn_prefix: str
  index: clx.Index
  errors: collections.OrderedDict[Path, collections.OrderedDict[int, WeakList]]
  warnings: list[tuple[StrPathLike, str]]
  patches: collections.defaultdict[Path, list[Patch]]

  def __init__(self, compiler_flags: list[str], clang_options: Optional[CXTranslationUnit] = None, verbose: bool = False, werror: bool = False) -> None:
    r"""Construct a `Linter`

    Parameters
    ----------
    compiler_flags :
      the set of compiler flags to parse with
    clang_options : optional
      the set of clang options to pass to the `clang.cindex.Index.parse()` function, defaults to
      `petsclinter.util.base_clang_options`
    verbose : optional
      whether to print verbose output
    werror : optional
      whether to treat warnings as errors
    """
    if clang_options is None:
      clang_options = util.base_clang_options

    self.flags       = compiler_flags
    self.clang_opts  = clang_options
    self.verbose     = verbose
    self.werror      = werror
    self.err_prefix  = f'{"-" * 92}'
    self.warn_prefix = f'{"%" * 92}'
    self.index       = clx.Index.create()
    self.clear()
    return

  def __str__(self) -> str:
    flag_str   = f'Compiler Flags: {self.flags}'
    clang_str  = f'Clang Options:  {self.clang_opts}'
    show_str   = f'Verbose:        {self.verbose}'
    print_list = [flag_str, clang_str, show_str]
    for v in self.get_all_errors():
      for path, mess in v:
        print_list.append(mess)
    warn_str = self.get_all_warnings(join_to_string=True)
    if warn_str:
      print_list.append(warn_str)
    return '\n'.join(print_list)

  def __enter__(self) -> Linter:
    return self

  def __exit__(self, exception_type: Exception, *args) -> None:
    if not exception_type:
      if self.verbose:
        pl.sync_print(self.get_all_warnings(join_to_string=True))
      pl.sync_print(self.get_all_errors())
    return

  def _check_duplicate_function_calls(self, processed_funcs: dict[str, list[tuple[Cursor, Scope]]]) -> None:
    r"""Check for duplicate instances of functions along the same execution path

    Parameters
    ----------
    processed_funcs :
      a dict mapping parent function names and the list of functions and their scopes

    Notes
    -----
    If two instances of a function have the same `Scope` then they are duplicate and an error is
    logged
    """
    dup_diag = self.diags.duplicate_function
    for function_list in processed_funcs.values():
      seen = {}
      for func, scope in function_list:
        combo = [func.displayname]
        try:
          combo.extend(map(Cursor.get_raw_name_from_cursor, func.get_arguments()))
        except ParsingError:
          continue

        # convert to tuple so it is hashable
        combo_tup = tuple(combo)
        if combo_tup not in seen:
          seen[combo_tup] = (func, scope)
        elif scope >= seen[combo_tup][1]:
          # this combination has already been seen, i.e. this call is duplicate!!
          start      = func.extent.start
          startline  = start.line
          tu         = func.translation_unit
          end        = clx.SourceLocation.from_position(tu, tu.get_file(tu.spelling), startline, -1)
          patch      = Patch(SourceRange.from_locations(start, end), '')
          previous   = seen[combo_tup][0].formatted(
            nbefore=2, nafter=startline - seen[combo_tup][0].extent.start.line
          )
          message    = f'Duplicate function found previous identical usage:\n{previous}'
          self.add_error_from_cursor(func, Diagnostic(dup_diag, message, start, patch=patch))
    return

  @staticmethod
  def find_lintable_expressions(tu: clx.TranslationUnit, symbol_names) -> clx.Cursor:
    r"""Finds all lintable expressions in container symbol_names.

    Parameters
    ----------
    tu :
      the `clang.cindex.TranslationUnit` to search
    symbol_names :
      the names of the symbols to search for and lint

    Notes
    -----
    Note that if a particular expression is not 100% correctly defined (i.e. would the
    file actually compile) then it will not be picked up by clang AST.

    Function-like macros can be picked up, but it will be in the wrong 'order'. The AST is
    built as if you are about to compile it, so macros are handled before any real
    function definitions in the AST, making it impossible to map a macro invocation to
    its 'parent' function.
    """
    UNEXPOSED_DECL = clx.CursorKind.UNEXPOSED_DECL
    SWITCH_STMT    = clx.CursorKind.SWITCH_STMT
    CASE_STMT      = clx.CursorKind.CASE_STMT
    COMPOUND_STMT  = clx.CursorKind.COMPOUND_STMT
    CALL_EXPR      = clx.CursorKind.CALL_EXPR

    def walk_scope_switch(parent: clx.Cursor, scope: Scope):
      """
      Special treatment for switch-case since the AST setup for it is mind-boggingly stupid.
      The first node after a case statement is listed as the cases *child* whereas every other
      node (including the break!!) is the cases *sibling*
      """
      # in case we get here from a scope decrease within a case
      case_scope = scope
      for child in parent.get_children():
        child_kind = child.kind
        if child_kind == CASE_STMT:
          # create a new scope every time we encounter a case, this is now for all intents
          # and purposes the 'scope' going forward. We don't overwrite the original scope
          # since we still need each case scope to be the previous scopes sibling
          case_scope = scope.sub()
          yield from walk_scope(child, scope=case_scope)
        elif child_kind == CALL_EXPR:
          if child.spelling in symbol_names:
            yield (child, possible_parent, case_scope)
            # Cursors that indicate change of logical scope
        elif child_kind == COMPOUND_STMT:
          yield from walk_scope_switch(child, case_scope.sub())

    def walk_scope(parent: clx.Cursor, scope: Optional[Scope] = None):
      """
      Walk the tree determining the scope of a node. here 'scope' refers not only
      to lexical scope but also to logical scope, see Scope object above
      """
      if scope is None:
        scope = Scope()

      for child in parent.get_children():
        child_kind = child.kind
        if child_kind == SWITCH_STMT:
          # switch-case statements require special treatment, we skip to the compound
          # statement
          switch_children = [c for c in child.get_children() if c.kind == COMPOUND_STMT]
          assert len(switch_children) == 1, "Switch statement has multiple '{' operators?"
          yield from walk_scope_switch(switch_children[0], scope.sub())
        elif child_kind == CALL_EXPR:
          if child.spelling in symbol_names:
            yield (child, possible_parent, scope)
        elif child_kind == COMPOUND_STMT:
          # scope has decreased
          yield from walk_scope(child, scope=scope.sub())
        else:
          # same scope
          yield from walk_scope(child, scope=scope)

    # normal lintable cursor kinds, the type of cursors we directly want to deal with
    lintable_kinds          = util.clx_func_call_cursor_kinds | {clx.CursorKind.ENUM_DECL}
    # "extended" lintable kinds.
    extended_lintable_kinds = lintable_kinds | {UNEXPOSED_DECL}

    cursor   = tu.cursor
    filename = tu.spelling
    for possible_parent in cursor.get_children():
      # getting filename is for some reason stupidly expensive, so we do this check first
      parent_kind = possible_parent.kind
      if parent_kind not in extended_lintable_kinds:
        continue
      try:
        if possible_parent.location.file.name != filename:
          continue
      except AttributeError:
        # possible_parent.location.file is None
        continue
      # Sometimes people declare their functions PETSC_EXTERN inline, which would normally
      # trip up the "lintable kinds" detection since the top-level cursor points to a
      # macro (i.e. unexposed decl). In this case we need to check the cursors 1 level
      # down for any lintable kinds.
      if parent_kind == UNEXPOSED_DECL:
        for sub_cursor in possible_parent.get_children():
          if sub_cursor.is_definition() and sub_cursor.kind in lintable_kinds:
            possible_parent = sub_cursor
            break
        else:
          continue
      # if we've gotten this far we have found something worth looking into, so first
      # yield the parent to process any documentation
      yield possible_parent
      if possible_parent.kind in util.clx_func_call_cursor_kinds:
        # then yield any children matching our function calls
        yield from walk_scope(possible_parent)

  @staticmethod
  def get_argument_cursors(func_cursor: CursorLike) -> tuple[Cursor, ...]:
    r"""Given a cursor representing a function, return a tuple of `Cursor`'s of its arguments

    Parameters
    ----------
    func_cursor :
      the function decl cursor

    Returns
    -------
    cursors :
      a tuple of `func_cursors` arguments
    """
    return tuple(Cursor(a, i) for i, a in enumerate(func_cursor.get_arguments(), start=1))

  def clear(self) -> None:
    r"""Resets the linter error, warning, and patch buffers.

    Notes
    -----
    Called automatically before parsing a file
    """
    self.errors   = collections.OrderedDict()
    self.warnings = []
    # This can actually just be a straight list, since each linter object only ever
    # handles a single file, but use dict nonetheless
    self.patches  = collections.defaultdict(list)
    return

  def parse(self, filename: PathLike) -> Linter:
    r"""Parse a file for errors

    Parameters
    ----------
    filename :
      the path of the file to parse

    Returns
    -------
    self :
      the `Linter` instance
    """
    self.clear()
    if self.verbose:
      pl.sync_print('Processing file     ', filename)
    tu = self.index.parse(str(filename), args=self.flags, options=self.clang_opts)
    if self.verbose and tu.diagnostics:
      pl.sync_print('\n'.join(map(str, tu.diagnostics)))
    self.process(tu)
    return self

  def parse_in_memory(self, src: str) -> clx.TranslationUnit:
    r"""Parse a particular source string in memory

    Parameters
    ----------
    src :
      the source string to parse

    Returns
    -------
    tu :
      the translation unit resulting from the parse

    Notes
    -----
    This lets you act as if `src` was some mini file somewhere on disk
    """
    fname = 'tempfile.cpp'
    return clx.TranslationUnit.from_source(
      fname, args=self.flags, unsaved_files=[(fname, src)], options=self.clang_opts
    )

  @DiagnosticManager.register(('parsing-error', 'Generic parsing errors'))
  def process(self, tu: clx.TranslationUnit) -> None:
    r"""Process a translation unit for errors

    Parameters
    ----------
    tu :
      the translation unit to process

    Notes
    -----
    This is the main entry point for the linter
    """
    func_map        = pl.checks._register.check_function_map
    docs_map        = pl.checks._register.check_doc_map
    parsing_diag    = self.process.diags.parsing_error
    processed_funcs = collections.defaultdict(list)

    for results in self.find_lintable_expressions(tu, set(func_map.keys())):
      try:
        if isinstance(results, clx.Cursor):
          docs_map[results.kind](self, Cursor.cast(results))
        else:
          func, parent, scope = results
          func                = Cursor.cast(func)
          parent              = Cursor.cast(parent)
          func_map[func.spelling](self, func, parent)
          processed_funcs[parent.name].append((func, scope))
      except KnownUnhandleableCursorError:
        # ignored
        pass
      except ParsingError as pe:
        tu_cursor = Cursor.cast(tu.cursor)
        self.add_warning_from_cursor(
          tu_cursor, Diagnostic(parsing_diag, str(pe), tu_cursor.extent.start)
        )
    self._check_duplicate_function_calls(processed_funcs)
    return

  def add_error_from_cursor(self, cursor: Cursor, diagnostic: Diagnostic) -> None:
    r"""Given a cursor and a diagnostic, log the error with the linter

    Parameters
    ----------
    cursor :
      the cursor about which the `diagnostic` is concerned
    diagnostic :
      the diagnostic detailing the error
    """
    if diagnostic.disabled():
      return

    assert isinstance(cursor, Cursor)
    filename = cursor.get_file()

    if filename not in self.errors:
      self.errors[filename] = collections.OrderedDict()

    errors    = self.errors[filename]
    cursor_id = cursor.hash
    if cursor_id not in errors:
      errors[cursor_id] = WeakList()

    patch            = diagnostic.patch
    have_patch       = patch is not None
    cursor_id_errors = errors[cursor_id]
    cursor_id_errors.append((
      f'{util.color.bright_red()}{diagnostic.location}: error:{util.color.reset()} {diagnostic.format_message()}',
      have_patch,
      patch.id if have_patch else -1
    ))

    if patch:
      patch.attach(weakref.ref(cursor_id_errors))
      self.patches[filename].append(patch)
    return

  def view_last_error(self) -> None:
    r"""Print the last error added, useful for debugging"""
    for files in reversed(self.errors):
      errors = self.errors[files]
      last   = errors[next(reversed(errors))]
      pl.sync_print(last[0][-1])
      break
    return

  def add_warning(self, filename: StrPathLike, diag: Diagnostic) -> None:
    r"""Add a generic warning given a filename"""
    if self.werror:
      self.add_error_from_cursor(filename, diag)
      return

    if diag.disabled():
      return

    warn_msg = diag.format_message()
    try:
      if warn_msg in self.warnings[-1][1]:
        # we just had the exact same warning, we can ignore it. This happens very often
        # for warnings occurring deep within a macro
        return
    except IndexError:
      pass
    self.warnings.append(
      (filename, f'{util.color.bright_yellow()}{filename}: warning:{util.color.reset()} {warn_msg}')
    )
    return

  def add_warning_from_cursor(self, cursor: Cursor, diag: Diagnostic) -> None:
    r"""Like `Linter.add_error_from_cursor()` but for warnings"""
    if self.werror:
      self.add_error_from_cursor(cursor, diag)
      return

    if diag.disabled():
      return

    assert isinstance(cursor, Cursor)
    warn_str = f'{util.color.bright_yellow()}{diag.location}: warning:{util.color.reset()} {str(cursor)}\n{diag.format_message()}'
    self.warnings.append((cursor.get_file(), warn_str))
    return

  def get_all_errors(self) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    r"""Return all errors collected so far

    Returns
    -------
    all_unresolved :
      a list of tuples of the path and message of unresolved errors (i.e. those without a `Patch`)
    all_resolve :
      a lit of tuples of the path and message of resolved errors (i.e. those with a `Patch`)
    """
    def maybe_add_to_global_list(global_list: list[tuple[Path, str]], local_list: list[str], path: Path) -> None:
      if local_list:
        global_list.append((
          path, '{prefix}\n{}\n{prefix}'.format('\n'.join(local_list), prefix=self.err_prefix)
        ))
      return

    all_unresolved: list[tuple[Path, str]] = []
    all_resolved: list[tuple[Path, str]]   = []
    for path, errors in self.errors.items():
      extracted: tuple[list[str], list[str]] = (
        [], # unresolved
        []  # resolved
      )
      for err_list in errors.values():
        for err, have_patch, _ in err_list:
          extracted[have_patch].append(err)
      maybe_add_to_global_list(all_unresolved, extracted[0], path)
      maybe_add_to_global_list(all_resolved, extracted[1], path)
    return all_unresolved, all_resolved

  def get_all_warnings(self, join_to_string: bool = False):
    r"""Return all warnings collected so far, and optionally join them all as one string

    Parameters
    ----------
    join_to_string : optional
      join the warnings to string

    Returns
    -------
    warnings :
      the list of warnings
    """
    if join_to_string:
      if self.warnings:
        return '\n'.join([
          self.warn_prefix, '\n'.join(s for _, s in self.warnings)[1:], self.warn_prefix
        ])
      return ''
    return self.warnings

  def coalesce_patches(self) -> list[tuple[Path, str]]:
    r"""Given a set of patches, collapse all patches and return the minimal set of diffs required

    Returns
    -------
    patches :
      the list of pairs of coalesced patches and their source files
    """
    def combine(filename: Path, patches: list[Patch]) -> tuple[Path, str]:
      fstr                   = str(filename)
      diffs: list[list[str]] = []
      for patch in patches:
        rn  = datetime.datetime.now().ctime()
        tmp = list(
          difflib.unified_diff(
            patch._make_source().splitlines(True), patch.collapse().splitlines(True),
            fromfile=fstr, tofile=fstr, fromfiledate=rn, tofiledate=rn, n=patch.ctxlines
          )
        )
        tmp[2] = Addline.diff_line_re.sub(Addline(patch.extent.start.line), tmp[2])
        # only the first diff should get the file heading
        diffs.append(tmp[2:] if diffs else tmp)
      return filename, ''.join(itertools.chain.from_iterable(diffs))

    def merge_patches(patch_list: list[Patch], patch: Patch) -> tuple[bool, Patch]:
      patch_extent       = patch.extent
      patch_extent_start = patch_extent.start.line
      for i, previous_patch in enumerate(patch_list):
        prev_patch_extent = previous_patch.extent
        if patch_extent_start == prev_patch_extent.start.line or patch_extent.overlaps(
            prev_patch_extent
        ):
          # this should now be the previous patch on the same line
          merged_patch = previous_patch.merge(patch)
          assert patch_list[i] == previous_patch
          del patch_list[i]
          return True, merged_patch
      return False, patch

    for patch_list in self.patches.values():
      # merge overlapping patches together before we collapse the actual patches
      # themselves
      new_list: list[Patch] = []
      for patch in sorted(patch_list, key=lambda x: x.extent.start.line):
        # we loop until we cannot merge the patch with any additional patches
        while 1:
          merged, patch = merge_patches(new_list, patch)
          if not merged:
            break
        new_list.append(patch)
      patch_list[:] = new_list

    return list(itertools.starmap(combine, self.patches.items()))

  def diagnostics(self) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list, list[tuple[Path, str]]]:
    r"""Return the errors left (unfixed), fixed errors, warnings and avaiable patches. Automatically
    coalesces the patches
    """
    # order is ciritical, coalesce_patches() will prune the patch and warning lists
    patches = self.coalesce_patches()
    errors_left, errors_fixed = self.get_all_errors()
    warnings = self.get_all_warnings()
    return errors_left, errors_fixed, warnings, patches
