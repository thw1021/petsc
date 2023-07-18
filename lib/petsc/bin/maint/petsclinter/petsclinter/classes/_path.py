#!/usr/bin/env python3
"""
# Created:
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

import pathlib

from .. import __version__

class Path(type(pathlib.Path())): # type: ignore
  """
  a basic pathlib.Path wrapper with some additional utility backported
  """
  # inheriting pathlib.Path:
  # https://stackoverflow.com/questions/29850801/subclass-pathlib-path-fails
  def append_suffix(self, suffix: str) -> Path:
    r"""Create a path with `suffix` appended, regardless of whether the current path has a suffix
    or not.

    Parameters
    ----------
    suffix:
      the suffix to append

    Returns
    -------
    path:
      the path with the suffix
    """
    suffix     = str(suffix)
    dotstring  = '' if suffix.startswith('.') else '.'
    path: Path = self.with_suffix(f'{self.suffix}{dotstring}{suffix}')
    return path

  def append_name(self, name: str) -> Path:
    r"""Create a path with `name` appended

    Parameters
    ----------
    name:
      the name to append

    Returns
    -------
    path:
      the path with the name
    """
    path: Path = self.with_name(f'{self.stem}{name}')
    return path
