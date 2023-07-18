#!/usr/bin/env python3
"""
# Created: Mon Jun 20 19:33:25 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

from ._src_pos import SourceRange

if TYPE_CHECKING:
  from weakref import ReferenceType

class Delta:
  __slots__ = 'value', 'extent', 'offset'

  value: str
  extent: SourceRange
  offset: int

  def __init__(self, value: str, extent: SourceRange, ctxlines: int) -> None:
    r"""Construct a `Delta`

    Parameters
    ----------
    value :
      the value to replace the `extent` with
    extent :
      a source range to be replaced
    ctxlines :
      number of lines before and after -- context lines -- to include in the source
    """
    self.value  = str(value)
    self.extent = extent
    begin       = extent.start
    self.offset = begin.offset - begin.column - sum(
      map(len, extent.raw(num_context=ctxlines).splitlines(True)[:ctxlines])
    ) + 1
    return

  def deleter(self) -> bool:
    r"""Is this `Delta` a deletion delta

    Returns
    -------
    ret :
      True if this `Delta` deletes its entire range, False otherwise
    """
    return self.value == ''

  def apply(self, src: str, offset: int) -> tuple[str, int]:
    r"""Apply the delta

    Parameters
    ----------
    src :
      the source to modify
    offset :
      offset into `src` at which this delta supposedly applies

    Returns
    -------
    new_src, new_offset :
      the updated `src` and the new offset value
    """
    extent       = self.extent
    src_offset   = offset - self.offset
    begin_offset = extent.start.offset + src_offset
    end_offset   = extent.end.offset + src_offset
    new_src      = f'{src[:begin_offset]}{self.value}{src[end_offset:]}'
    new_offset   = offset + len(new_src) - len(src)
    return new_src, new_offset

  def is_deletion_superset_of(self, other: Delta) -> bool:
    r"""Determine if self's change deletes all of other's extent.

    If this is the case, `other` is a pointless patch and can be discarded

    Parameters
    ----------
    other :
      the other `Delta`

    Returns
    -------
    ret :
      True if `self` deletes all of `other`, False otherwise
    """
    return self.deleter() and other.extent in self.extent

class Patch:
  __global_counter: int = 0
  __slots__             = 'extent', 'ctxlines', 'deltas', 'weak_data', '_cache', 'id'

  extent: SourceRange
  ctxlines: int
  deltas: tuple[Delta, ...]
  weak_data: list[ReferenceType]
  _cache: dict[str, Any]
  id: int

  def __init__(self, src_range: SourceRangeLike, value: str, contextlines: int = 2) -> None:
    r"""Construct a `Patch`

    Parameters
    ----------
    src_range :
      the range in source to patch
    value :
      the replacement string for `src_range`
    contextlines : 2, optional
      the number of lines before and after `src_range` to include in the source
    """
    self.extent    = SourceRange.cast(src_range)
    self.ctxlines  = contextlines
    self.deltas    = (Delta(value, self.extent, self.ctxlines),)
    self.weak_data = []
    self._cache    = {}
    self.id        = Patch.__global_counter
    Patch.__global_counter += 1
    return

  @classmethod
  def from_cursor(cls, cursor: CursorLike, value: str, **kwargs) -> Patch:
    r"""Construct a `Patch` from a cursor

    Parameters
    ----------
    cursor :
      the cursor to take the extent from
    value :
      the value to replace the cursors extent with
    **kwargs : optional
      additional keyword arguments to pass to the `Patch` constructor

    Returns
    -------
    patch :
      the `Patch` object
    """
    return cls(cursor.extent, value, **kwargs)

  def _make_source(self) -> str:
    r"""Instantiate the initial raw source for this `Patch`

    Returns
    -------
    src :
      the text
    """
    return self.extent.raw(num_context=self.ctxlines)

  def _get_cached(self, attr: str, func: Callable, *args, **kwargs) -> Any:
    cache = self._cache
    if attr in cache:
      return cache[attr]
    new_val = cache[attr] = func(*args, **kwargs)
    return new_val

  def _contiguous_extent(self) -> bool:
    r"""Does my extent (which is the union of the extents of my all my deltas) have no holes?

    Returns
    -------
    ret :
      True if this `Patch` is contiguous, False otherwise
    """
    cache_entry = 'contiguous'
    deltas      = self.deltas
    if len(deltas) == 1:
      return self._cache.setdefault(cache_entry, True)
    return self._get_cached(
      cache_entry, all, (p.extent.overlaps(c.extent) for p, c in zip(deltas[:-1], deltas[1:]))
    )


  def discard(self) -> None:
    r"""Drop the error messages corresponding to this patch from the linter"""
    del_indices = []
    self_id     = self.id
    assert self_id >= 0
    for i, weak_ref in enumerate(self.weak_data):
      elist = weak_ref()

      if elist is not None:
        del_indices.append(i)
        # I don't know how this list could ever be longer than a single element, but I
        # guess it does not hurt to handle this case?
        idx = [eid for eid, (_, _, patch_id) in enumerate(elist) if patch_id == self_id]
        if not idx:
          raise RuntimeError('could not locate weakref idx for patch')
        for i in reversed(idx):
          del elist[i] # delete our entry in the error message list

    for i in reversed(del_indices):
      del self.weak_data[i]
    return

  def attach(self, cursor_id_errors: ReferenceType) -> None:
    r"""Aattach a weak reference to this `Patch`s entries in the linters errors

    Parameters
    ----------
    cursor_id_errors :
      the linters cursor-specific list of errors

    Notes
    -----
    This is kind of a cludge and should probably be removed
    """
    self.weak_data.append(cursor_id_errors)
    return

  def is_deletion_superset_of(self, other: Patch) -> bool:
    r"""Determine if any of self's deltas delete all of other's extent.

    Parameters
    ----------
    other :
      the other patch to compare against

    Returns
    -------
    ret :
      True if `self` deletes all of `other`, False otherwise
    """
    assert not self is other
    oextent = other.extent
    if oextent in self.extent:
      deltas = self.deltas
      # first check if any one delta deletes all of other, then check if all deltas are deleters,
      return any(d.deleter() and oextent in d.extent for d in deltas) or \
        (self._contiguous_extent() and all(d.deleter() for d in deltas))
    return False

  @staticmethod
  def cull_deltas(deltas: tuple[Delta, ...]) -> tuple[Delta, ...]:
    r"""Remove any pointless `Delta`s

    Parameters
    ----------
    deltas :
      a set of `Delta`'s to cull

    Returns
    -------
    ret :
      a typle of `Delta`'s which has all deleted subsets removed, i.e. no `Delta` should be a deletion
      superset of any other
    """
    return tuple([
      d_i for d_i in deltas if not any(
        d_j.is_deletion_superset_of(d_i) for d_j in deltas if d_j is not d_i
      )
    ])

  def merge(self, other: Patch) -> Patch:
    r"""Merge a patch with another patch.

    If either of the patches is a 'deletion superset' of the other the redundant patch is discarded
    from the linter.

    Parameters
    ----------
    other :
      the `Patch` to merge with

    Returns
    -------
    ret :
      the merged `Patch`

    Raises
    ------
    ValueError
      if `other` is not a `Patch`
    """
    if not isinstance(other, type(self)):
      raise ValueError(type(other))

    if self is other:
      return self

    if self.is_deletion_superset_of(other):
      other.discard()
      return self

    if other.is_deletion_superset_of(self):
      self.discard()
      return other

    assert self._make_source() == other._make_source(), 'Need to update offset calculation to handle arbitrary src'
    assert self.ctxlines == other.ctxlines, 'Need to update ctxlines to handle arbitrary src'

    self.extent = self.extent.merge_with(other.extent)
    # uncomment when it handles arbitrary source
    # self.src    = self._make_source()
    # fixes and ranges must be applied in order
    combined    = self.cull_deltas(self.deltas + other.deltas)
    argsort     = sorted(range(len(combined)), key=lambda x: combined[x].extent)
    self.deltas = tuple(combined[i] for i in argsort)
    self._cache = {}
    self.weak_data.extend(other.weak_data)
    return self

  def collapse(self) -> str:
    r"""Collapses a the list of fixes and produces into a modified output

    Returns
    -------
    src:
      the patched source
    """
    # Fixes probably should not overwrite each other (for now), so we error out, but this
    # is arguably a completely valid case. I just have not seen an example of it that I
    # can use to debug with yet.
    cache_entry = 'fixed'
    cache       = self._cache
    if cache_entry in cache:
      return cache[cache_entry]

    idx_delta = 0
    src       = old_src = self._make_source()
    for delta in self.deltas:
      src, idx_delta = delta.apply(src, idx_delta)

    assert src != old_src, 'Patch did not seem to do anything!'
    cache[cache_entry] = src
    return src

  def view(self) -> None:
    r"""Visualize the action of the `Patch`"""
    import difflib
    import petsclinter as pl

    idx_delta = 0
    before    = self._make_source()
    for i, delta in enumerate(self.deltas):
      pl.sync_print('Delta:', i, f'({delta})')
      after, idx_delta = delta.apply(before, idx_delta)
      pl.sync_print(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='Original', tofile='Modified'
      )))
      before = after
    return
