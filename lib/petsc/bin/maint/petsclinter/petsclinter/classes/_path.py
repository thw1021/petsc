#!/usr/bin/env python3
"""
# Created:
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import typing
import pathlib

from .. import __version__

class Path(type(pathlib.Path())):
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
    suffix    = str(suffix)
    dotstring = '' if suffix.startswith('.') else '.'
    return self.with_suffix(f'{self.suffix}{dotstring}{suffix}')

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
    return self.with_name(f'{self.stem}{name}')

  def unlink(self, missing_ok: bool = False) -> None:
    r"""Deletes a path

    Parameters
    ----------
    missing_ok: optional
      don't raise an exception if `self` does not exist

    Raises
    ------
    FileNotFoundError
      if `self` does not exist and `missing_ok` is False
    """
    if __version__.py_version_lt(3, 8):
      try:
        super().unlink()
      except FileNotFoundError as fnfe:
        if missing_ok:
          return
        raise
    else:
      super().unlink(missing_ok=missing_ok)
    return

PathLike    = typing.Union[pathlib.Path, Path]
StrPathLike = typing.Union[PathLike, str]
