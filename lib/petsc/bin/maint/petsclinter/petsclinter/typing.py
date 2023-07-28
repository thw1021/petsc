#!/usr/bin/env python3
"""
# Created: Thu Jul 27 13:53:56 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
  import re
  import pathlib
  import clang.cindex as clx # type: ignore[import]

  from typing import Optional, Any, NoReturn, TypeVar

  from collections.abc import (
    Iterator, Iterable, Generator, Callable, Collection, Sequence, Container, Mapping
  )

  ##
  # CLASSES
  ##

  from .classes._cursor  import Cursor
  from .classes._diag    import Diagnostic, DiagnosticMap
  from .classes._linter  import Linter
  from .classes._patch   import Patch
  from .classes._path    import Path
  from .classes._pool    import WorkerPoolBase, ParallelPool, SerialPool
  from .classes._src_pos import SourceLocation, SourceRange

  PathLike           = Union[pathlib.Path, Path]
  StrPathLike        = Union[PathLike, str]
  CursorLike         = Union[clx.Cursor, Cursor]
  SourceLocationLike = Union[clx.SourceLocation, SourceLocation]
  SourceRangeLike    = Union[clx.SourceRange, SourceRange]
  PoolImpl           = TypeVar('PoolImpl', bound=WorkerPoolBase)

  ##
  # DOCS
  ##

  from .classes.docs._doc_str          import Verdict, PetscDocString
  from .classes.docs._doc_section_base import (
    DescribableItem, SectionBase, ParameterList, Prose, VerbatimBlock, InlineList,
  )
  from .classes.docs._doc_section      import (
    Synopsis, FunctionParameterList, OptionDatabaseKeys, Notes, DeveloperNotes, References,
    FortranNotes, SourceCode, Level, SeeAlso
  )

  SectionImpl = TypeVar('SectionImpl', bound=SectionBase)

  ##
  # UTIL
  ##

  from .util._clang   import CXTranslationUnit, PetscCXCursorAndRangeVisitor, ClangFunction
  from .util._color   import Color
  from .util._utility import PrecompiledHeader
