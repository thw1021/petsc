#!/usr/bin/env python3
"""
# Created: Thu Dec  8 10:23:34 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
import sys

__MIN_PYTHON_VERSION__ = (3, 6, 0)
__version__            = (1, 0, 0)
__version_str__        = '.'.join(map(str, __version__))

class RedundantMinVersionCheckError(Exception):
  """
  Exception thrown when code checks for minimum python version which is less than the minimum
  requirement for petsclinter
  """
  pass

def py_version_lt(major: int, minor: int, sub_minor: int = 0) -> bool:
  r"""Determines if python version is less than a particular version.

  This should be used whenever back-porting some code as it will automatically raise an
  error if the version check is useless.

  Parameters
  ----------
  major :
    major version number, e.g. 3
  minor :
    minor version number
  sub_minor : optional
    sub-minor or patch version

  Returns
  -------
  ret :
    True if python version is less than `major`.`minor`.`sub_minor`, False otherwise

  Raises
  ------
  RedundantMinVersionCheckError
    If the given version is below petsclinter.__MIN_PYTHON_VERSION__ (and therefore the version check
    is pointless) this raises RedundantMinVersionCheckError. This should not be caught.
  """
  version = (major, minor, sub_minor)
  if version <= __MIN_PYTHON_VERSION__:
    raise RedundantMinVersionCheckError(
      f'Minimum required version {__MIN_PYTHON_VERSION__} already >= checked version {version}. '
      f'There is no need to include this version check!'
    )
  return sys.version_info < version

def version_tuple() -> tuple[int, int, int]:
  r"""Return the package version as a tuple

  Returns
  -------
  version :
    the package version as a tuple
  """
  return __version__

def version_str() -> str:
  r"""Return the package version as a string

  Returns
  -------
  version :
    the package version as a string
  """
  return __version_str__
