#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:53:35 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..classes._diag import DiagnosticManager

from ..classes.docs._doc_str import PetscDocString

if TYPE_CHECKING:
  from ..classes._linter import Linter
  from ..classes._cursor import Cursor

"""Specific 'driver' function to test a particular docstring archetype"""
def check_petsc_function_docstring(linter: Linter, cursor: Cursor) -> None:
  docstring = PetscDocString(linter, cursor)

  with DiagnosticManager.push_from(docstring.get_pragmas()):
    for section in docstring.parse().sections:
      section.check(linter, cursor, docstring)
  return

def check_petsc_enum_docstring(linter: Linter, cursor: Cursor) -> None:
  docstring = PetscDocString(linter, cursor)

  with DiagnosticManager.push_from(docstring.get_pragmas()):
    for section in docstring.parse().sections:
      section.check(linter, cursor, docstring)
  return
