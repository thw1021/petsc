#!/usr/bin/env python3
"""
# Created: Tue Aug  1 16:33:38 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import typing
import inspect
import pkgutil
import importlib

EXPORT_SYMBOL_ATTR = '_exported_symbols_'

T = typing.TypeVar('T')

def export_symbol(cls: T) -> T:
  mod = inspect.getmodule(cls)
  assert mod is not None
  if not hasattr(mod, EXPORT_SYMBOL_ATTR):
    setattr(mod, EXPORT_SYMBOL_ATTR, [])
  getattr(mod, EXPORT_SYMBOL_ATTR).append(getattr(cls, '__name__'))
  return cls

def __import_submodules(package, parent, recursive=True):
  """
  Import all submodules of a module, recursively, including subpackages

  :param package: package (name or actual module)
  :type package: str | module
  :rtype: dict[str, types.ModuleType]
  """
  if isinstance(package, str):
    package = importlib.import_module(package)

  results = {}
  for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
    full_name = package.__name__ + '.' + name
    try:
      results[full_name] = importlib.import_module(full_name)
      if recursive and is_pkg:
        results.update(__import_submodules(full_name, parent, recursive=recursive))
    except:
      pass
  return results

def _build__all__(name: str) -> list[str]:
  parent = importlib.import_module(name)
  exported_symbols: list[str] = []
  for mod in __import_submodules(name, parent).values():
    for symbol in inspect.getattr_static(mod, EXPORT_SYMBOL_ATTR, default=[]):
      setattr(parent, symbol, getattr(mod, symbol))
      exported_symbols.append(symbol)
  return exported_symbols
