#!/usr/bin/env python3
"""
# Created: Thu Jul 27 13:53:56 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from typing import (
  TYPE_CHECKING, Union, TypedDict, TypeVar, NamedTuple, Generic, Protocol, overload, cast as TYPE_CAST
)

from .__version__ import py_version_lt

_ignored = py_version_lt(3, 9)
# Dummy function call conditional to make sure you remember to remove the following line
from typing import List, Dict, Tuple, Type

del _ignored
del py_version_lt

if TYPE_CHECKING:
  import re
  import sys
  import pathlib
  import weakref
  import clang.cindex as clx # type: ignore[import]

  from typing import Optional, Any, NoReturn

  from collections.abc import (
    Iterator, Iterable, Generator, Callable, Collection, Sequence, Container, Mapping
  )

  ##
  # CLASSES
  ##

  from .classes._cursor     import Cursor
  from .classes._linter     import Linter, WeakList
  from .classes._patch      import Patch
  from .classes._path       import Path
  from .classes._pool       import WorkerPoolBase, ParallelPool, SerialPool
  from .classes._src_pos    import SourceLocation, SourceRange
  from .classes._attr_cache import AttributeCache
  from .classes._diag       import (
    DiagnosticMapProxy, DiagnosticMap, DiagnosticsManagerCls, DiagnosticKind, Diagnostic
  )

  PathLike           = Union[pathlib.Path, Path]
  StrPathLike        = Union[PathLike, str]
  CursorLike         = Union[clx.Cursor, Cursor]
  SourceLocationLike = Union[clx.SourceLocation, SourceLocation]
  SourceRangeLike    = Union[clx.SourceRange, SourceRange]
  PoolImpl           = TypeVar('PoolImpl', bound=WorkerPoolBase)
  PathDiffPair       = Tuple[Path, str]
  CondensedDiags     = Dict[Path, List[str]]
  WeakListType       = WeakList[Tuple[str, bool, int]]
  WeakListRef        = weakref.ReferenceType[WeakListType]

  ##
  # DOCS
  ##

  from .classes.docs._doc_str          import Verdict, PetscDocString
  from .classes.docs._doc_section_base import (
    DescribableItem, SectionBase, Synopsis, ParameterList, Prose, VerbatimBlock, InlineList,
  )
  from .classes.docs._doc_section      import (
    FunctionSynopsis, EnumSynopsis, FunctionParameterList, OptionDatabaseKeys, Notes, DeveloperNotes,
    References, FortranNotes, SourceCode, Level, SeeAlso
  )

  SectionImpl        = TypeVar('SectionImpl', bound=SectionBase)
  PetscDocStringImpl = TypeVar('PetscDocStringImpl', bound=PetscDocString)

  ##
  # UTIL
  ##

  from .util._clang   import CXTranslationUnit, ClangFunction
  from .util._color   import Color
  from .util._utility import PrecompiledHeader

  ExceptionKind = TypeVar('ExceptionKind', bound=Exception)

  ##
  # CHECKS
  ##

  FunctionChecker = Callable[[Linter, Cursor, Cursor], None]
  DocChecker      = Callable[[Linter, Cursor], None]
