#!/usr/bin/env python3
"""
# Created: Wed Aug  2 11:30:15 2023 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

if TYPE_CHECKING:
  T = TypeVar('T')

class AttributeCache:
  __slots__ = ('_cache',)

  _cache: dict[str, Any]

  def __init__(self, init_cache: Optional[dict[str, Any]] = None) -> None:
    if init_cache is None:
      init_cache = {}
    else:
      assert isinstance(init_cache, dict)
    self._cache = init_cache
    return

  def _get_cached(self, attr: str, func: Callable[..., T], *args, **kwargs) -> T:
    cache = self._cache
    if attr in cache:
      ret: T = cache[attr]
    else:
      ret         = func(*args, **kwargs)
      cache[attr] = ret
    return ret
