import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from doc import build_man_pages


class BuildManPagesTest(unittest.TestCase):
  def _directories(self, root):
    source = root / 'src' / 'package'
    build = root / 'doc'
    source.mkdir(parents=True)
    (build / 'manualpages').mkdir(parents=True)
    return source,build

  def test_batches_files_with_the_same_manual_section(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      source,build = self._directories(root)
      source.joinpath('makefile').write_text('MANSEC = Sys\n')
      source.joinpath('first.c').write_text('int first;\n')
      source.joinpath('second.c').write_text('int second;\n')

      completed = subprocess.CompletedProcess([],0,'','')
      with mock.patch.object(build_man_pages.os,'listdir',return_value=['makefile','first.c','second.c']), mock.patch.object(build_man_pages.subprocess,'run',return_value=completed) as run:
        errors = build_man_pages.processdir_batched(str(root),str(build),str(source),'doctext')

      self.assertEqual(errors,0)
      run.assert_called_once()
      self.assertEqual(run.call_args.args[0][-2:],['first.c','second.c'])

  def test_preserves_processdir_for_comparison(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      source,build = self._directories(root)
      source.joinpath('makefile').write_text('MANSEC = Sys\n')
      source.joinpath('first.c').write_text('int first;\n')
      source.joinpath('second.c').write_text('int second;\n')

      completed = subprocess.CompletedProcess([],0,'','')
      with mock.patch.object(build_man_pages.subprocess,'run',return_value=completed) as run:
        errors = build_man_pages.processdir(str(root),str(build),str(source),'doctext')

      self.assertEqual(errors,0)
      self.assertEqual(run.call_count,2)


if __name__ == '__main__':
  unittest.main()
