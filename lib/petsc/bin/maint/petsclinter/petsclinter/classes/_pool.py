#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:59:46 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import os
import enum
import queue
import multiprocessing as mp
import clang.cindex as clx
import petsclinter  as pl

import abc

from ..util._timeout import timeout, TimeoutError

from ._diag   import DiagnosticManager
from ._linter import Linter

if TYPE_CHECKING:
  from collections.abc import Collection, MutableSequence
  from typing          import Optional, Union, Any, TypeVar, NoReturn

  from ..util._clang import CXTranslationUnit
  from ._path        import Path, PathLike, StrPathLike

  PoolImpl = TypeVar('PoolImpl', bound='WorkerPoolBase')

# directory names to exclude from processing, case sensitive
exclude_dir_names = {
  'tests', 'tutorials', 'output', 'input', 'python', 'benchmarks', 'docs', 'binding', 'contrib',
  'fsrc', 'f90-mod', 'f90-src', 'f90-custom', 'ftn-auto', 'ftn-custom', 'f2003-src', 'ftn-kernels',
}
# directory suffixes to exclude from processing, case sensitive
exclude_dir_suffixes  = ('.dSYM', '.DS_Store')
# file extensions to process, case sensitve
allow_file_extensions = ('.c', '.cpp', '.cxx', '.cu', '.cc', '.h', '.hpp', '.inc')

class WorkerPoolBase(abc.ABC):
  __slots__ = ('verbose', 'warnings', 'errors_left', 'errors_fixed', 'patches')

  verbose: bool
  warnings: list[tuple[Path, str]]
  errors_left: list[tuple[Path, str]]
  errors_fixed: list[tuple[Path, str]]
  patches: list[tuple[Path, str]]

  class QueueSignal(enum.IntEnum):
    """
    Various signals to indicate return type on the data queue from child processes
    """
    WARNING      = enum.auto()
    UNIFIED_DIFF = enum.auto()
    ERRORS_LEFT  = enum.auto()
    ERRORS_FIXED = enum.auto()
    EXIT_QUEUE   = enum.auto()

  def __init__(self, verbose: bool) -> None:
    super().__init__()
    self.verbose      = verbose
    self.warnings     = []
    self.errors_left  = []
    self.errors_fixed = []
    self.patches      = []
    return

  def _print(self, *args, **kwargs) -> None:
    if self.verbose:
      pl.sync_print(*args, **kwargs)
    return

  @abc.abstractmethod
  def _setup(self, compiler_flags: list[str], clang_lib: PathLike, clang_options: CXTranslationUnit, clang_compat_check: bool, werror: bool) -> None:
    return

  @abc.abstractmethod
  def _consume_results(self) -> None:
    return

  @abc.abstractmethod
  def _finalize(self) -> None:
    return

  @abc.abstractmethod
  def put(self, item) -> None:
    return

  def setup(self: PoolImpl, compiler_flags: list[str], clang_lib: Optional[PathLike] = None, clang_options: Optional[CXTranslationUnit] = None, clang_compat_check: bool = True, werror: bool = False) -> PoolImpl:
    r"""Set up a `WorkerPool` instance

    Parameters
    ----------
    compiler_flags :
      the list of compiler flags to pass to the `Linter`
    clang_lib : optional
      the path to libclang
    clang_options: optional
      the options to pass to the `Linter`, defaults to `petsclinter.util.base_clang_options`
    clang_compat_check: optional
      whether to do compatibility checks (if this initializes libclang)
    werror:
      whether to treat warnings as errors

    Returns
    -------
    self:
      the `WorkerPool` instance
    """
    if clang_lib is None:
      assert clx.conf.loaded, 'Must initialize libClang first'
      clang_lib = clx.conf.get_filename()

    if clang_options is None:
      clang_options = pl.util.base_clang_options

    self._setup(compiler_flags, clang_lib, clang_options, clang_compat_check, werror)
    return self

  def walk(self: PoolImpl, src_path_list: list[PathLike], exclude_dirs: Optional[Collection[str]] = None, exclude_dir_suff: Optional[tuple[str, ...]] = None, allow_file_suff: Optional[tuple[str, ...]] = None) -> PoolImpl:
    r"""Walk `src_path_list` and process it

    Parameters
    ----------
    src_path_list :
      a list of paths to process
    exclude_dirs : optional
      a list or set to exclude from processing
    exclude_dir_suff : optional
      a set of suffixes to ignore
    allow_file_suff : optional
      a list of suffixes to explicitly allow

    Returns
    -------
    self :
      the `WorkerPool` instance
    """
    if exclude_dirs is None:
      exclude_dirs = exclude_dir_names
    if exclude_dir_suff is None:
      exclude_dir_suff = exclude_dir_suffixes
    if allow_file_suff is None:
      allow_file_suff = allow_file_extensions

    for src_path in src_path_list:
      if src_path.is_file():
        self.put(src_path)
        continue

      _, dirs, _   = next(os.walk(src_path))
      dir_gen      = (d for d in dirs if d not in exclude_dirs)
      initial_dirs = {str(src_path / d) for d in dir_gen if not d.endswith(exclude_dir_suff)}
      for root, dirs, files in os.walk(src_path):
        self._print('Processing directory', root)
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        dirs[:] = [d for d in dirs if not d.endswith(exclude_dir_suff)]
        for filename in (os.path.join(root, f) for f in files if f.endswith(allow_file_suff)):
          self.put(filename)
        # Every time we reach another top-level node we consume some of the results. This
        # makes the eventual consume-until-empty loop much faster since the queue is not
        # as backed up
        if root in initial_dirs:
          self._consume_results()
    return self

  def finalize(self: PoolImpl) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    r"""Finalize the queue and return the results

    Returns
    -------
    warnings :
      the list of warnings
    errors_left :
      the remaining (unfixed) errors
    errors_fixed :
      the fixed errors
    patches :
      the generated patches

    Notes
    -----
    If running in parallel, and workers fail to finalize in time, calls `self.__crash_and_burn()`
    """
    self._finalize()
    return self.warnings[:], self.errors_left[:], self.errors_fixed[:], self.patches[:]

class ParallelPool(WorkerPoolBase):
  __slots__ = ('input_queue', 'error_queue', 'return_queue', 'lock', 'workers', 'num_workers')

  input_queue: mp.JoinableQueue
  error_queue: mp.Queue
  return_queue: mp.Queue
  lock: mp.synchronize.Lock
  workers: list[mp.Process]
  num_workers: int

  def __init__(self, num_workers: int, verbose: bool) -> None:
    super().__init__(verbose)
    self.input_queue  = mp.JoinableQueue()
    self.error_queue  = mp.Queue()
    self.return_queue = mp.Queue()
    lock              = mp.Lock()
    self.lock         = lock
    self.num_workers  = num_workers

    old_sync_print = pl.sync_print
    def lock_sync_print(*args, **kwargs):
      with lock:
        old_sync_print(*args, **kwargs)
      return
    pl.sync_print = lock_sync_print
    return

  @timeout(seconds=10)
  def __crash_and_burn(self, message: str) -> NoReturn:
    r"""Forcefully annihilate the pool and crash the program

    Parameters
    ----------
    message :
      an informative message to print on crashing

    Raises
    ------
    RuntimeError :
      raises a RuntimeError in all cases
    """
    for worker in self.workers:
      if worker is not None:
        try:
          worker.terminate()
        except:
          pass
    raise RuntimeError(message)

  def _consume_results(self) -> None:
    r"""Consume pending results from the queue

    Raises
    ------
    ValueError :
      if an unknown QueueSignal is returned from the pipe
    """
    self.check()
    return_q = self.return_queue
    try:
      qsize_mess = str(return_q.qsize())
    except NotImplementedError:
      # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.qsize
      #
      # Note that this may raise NotImplementedError on Unix platforms like macOS where
      # sem_getvalue() is not implemented.
      qsize_mess = '0' if return_q.empty() else 'unknown (not implemented on platform)'
    self._print('Estimated number of results:', qsize_mess)

    while not return_q.empty():
      try:
        packet = return_q.get(timeout=1)
      except queue.Empty:
        # this should never really happen since this thread is the only consumer of the
        # queue, but it's here just in case
        break
      for signal, data in packet:
        if signal == self.QueueSignal.ERRORS_LEFT:
          self.errors_left.extend(data)
        elif signal == self.QueueSignal.ERRORS_FIXED:
          self.errors_fixed.extend(data)
        elif signal == self.QueueSignal.UNIFIED_DIFF:
          self.patches.extend(data)
        elif signal == self.QueueSignal.WARNING:
          self.warnings.extend(data)
        else:
          raise ValueError(f'Unknown data returned by return_queue {signal}, {data}')
    return

  def _setup(self, compiler_flags: list[str], clang_lib: PathLike, clang_options: CXTranslationUnit, clang_compat_check: bool, werror: bool) -> None:
    from ..queue_main       import queue_main
    from ..checks._register import check_function_map, classid_map

    self.workers = [
      mp.Process(
        target=queue_main,
        args=(
          clang_lib, clang_compat_check, check_function_map, classid_map, DiagnosticManager,
          compiler_flags, clang_options, self.verbose, werror, self.error_queue, self.return_queue,
          self.input_queue, self.lock
        ),
        name=f'[{i}]'
      )
      for i in range(self.num_workers)
    ]

    for worker in self.workers:
      worker.start()
    return

  def _finalize(self) -> None:
    import time

    # join here to colocate error messages if needs be
    self.input_queue.join()
    self.check()
    # send stop-signal to child processes
    for _ in self.workers:
      self.put(self.QueueSignal.EXIT_QUEUE)

    self._consume_results()
    # If there is a lot of data being sent from the child processes, they may not fully
    # flush in time for us to catch their output in the first consume_results() call.
    #
    # We need to spin (and continue consuming results) until all processes have have
    # exited, since they will only fully exit once they flush their pipes.
    max_timeout = 60 # seconds
    start       = time.time()
    while time.time() - start <= max_timeout:
      live_list = [w.is_alive() for w in self.workers]
      for worker, alive in zip(self.workers, live_list):
        self._print(
          'Checking whether process', worker.name, 'has finished:', 'no' if alive else 'yes'
        )
        if alive:
          worker.join(timeout=1)
          self._consume_results()
      if sum(live_list) == 0:
        break
    else:
      mess = '\n'.join(
        f'{worker.name}: {"alive" if alive else "terminated"}' for worker, alive in zip(self.workers, live_list)
      )
      self.__crash_and_burn(f'Timed out! Workers failed to terminate:\n{mess}')

    self.error_queue.close()
    self.return_queue.close()
    return

  def check(self) -> None:
    r"""Check for errors from the queue

    Notes
    -----
    Calls `self.__crash_and_burn()` if any errors are detected, but does nothing if running in
    serial
    """
    stop_multiproc = False
    timeout_it     = 0
    max_timeouts   = 3
    while not self.error_queue.empty():
      # while this does get recreated for every error, we do not want to needlessly
      # reinitialize it when no errors exist. If we get to this point however we no longer
      # care about performance as we are about to crash everything.
      try:
        exception = self.error_queue.get(timeout=.5)
      except queue.Empty:
        # Queue is not empty (we were in the loop), but we timed out on the get. Should
        # not happen yet here we are. Try a few couple more times, otherwise bail
        timeout_it += 1
        if timeout_it > max_timeouts:
          break
        continue

      err_bars = ''.join(['[ERROR]', 85 * '-', '[ERROR]\n'])
      try:
        err_mess = f'{err_bars}{str(exception)}{err_bars}'
      except:
        err_mess = exception

      print(err_mess, flush=True)
      stop_multiproc = True
      timeout_it     = 0
    if stop_multiproc:
      self.__crash_and_burn('Error in child process detected')
    return

  def put(self, item: Any) -> None:
    # continuously put files onto the queue, if the queue is full we block for
    # queueTimeout seconds and if we still cannot insert to the queue we check
    # children for errors. If no errors are found we try again.
    while 1:
      try:
        self.input_queue.put(item, True, 2)
      except queue.Full:
        # we don't want to join here since a child may have encountered an error!
        self.check()
      else:
        # only get here if put is successful
        break
    return

class SerialPool(WorkerPoolBase):
  __slots__ = ('linter',)

  linter: Linter

  def __init__(self, verbose: bool) -> None:
    super().__init__(verbose)
    return

  def _setup(self, compiler_flags: list[str], clang_lib: PathLike, clang_options: CXTranslationUnit, clang_compat_check: bool, werror: bool) -> None:
    self.linter = Linter(
      compiler_flags, clang_options=clang_options, verbose=self.verbose, werror=werror
    )
    return

  def _consume_results(self) -> None:
    return

  def _finalize(self) -> None:
    return

  def put(self, item: PathLike) -> None:
    err_left, err_fixed, warnings, patches = self.linter.parse(item).diagnostics()
    self.errors_left.extend(err_left)
    self.errors_fixed.extend(err_fixed)
    self.warnings.extend(warnings)
    self.patches.extend(patches)
    return

def WorkerPool(num_workers: int, verbose: bool = False) -> Union[SerialPool, ParallelPool]:
  if num_workers < 0:
      num_workers = max(mp.cpu_count() - 1, 1)

  if num_workers in (0, 1):
    if verbose:
      pl.sync_print(f'Number of worker processes ({num_workers}) too small, disabling multiprocessing')
    return SerialPool(verbose)
  else:
    if verbose:
      pl.sync_print(f'Number of worker processes ({num_workers}) sufficient, enabling multiprocessing')
    return ParallelPool(num_workers, verbose)
