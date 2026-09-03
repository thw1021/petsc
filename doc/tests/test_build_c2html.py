import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from doc import build_c2html


class BuildC2HTMLTest(unittest.TestCase):
  def test_calls_worker_without_generating_makefile(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      build_directory = root / 'doc' / '_build'
      output_directory = build_directory / 'html'
      manualpages_directory = build_directory / 'manualpages'
      source_directory = root / 'src'
      manualpages_directory.mkdir(parents=True)
      source_directory.mkdir()
      manualpages_directory.joinpath('htmlmap').write_text('manual pages\n')
      manualpages_directory.joinpath('mpi.www.index').write_text('MPI pages\n')
      source_directory.joinpath('example.c').write_text('int example;\n')

      original_directory = os.getcwd()
      try:
        with mock.patch.object(build_c2html.subprocess, 'check_output', return_value='revision\n'), mock.patch.object(build_c2html.concurrent.futures, 'ProcessPoolExecutor', build_c2html.concurrent.futures.ThreadPoolExecutor), mock.patch.object(build_c2html.build_c2html_file, 'main_batch', return_value=1) as build_batch:
          build_c2html.main(str(root), str(build_directory), str(output_directory), 'c2html', 'mapnames')
      finally:
        os.chdir(original_directory)

      build_batch.assert_called_once_with(str(root), str(output_directory), 'revision', 'c2html', 'mapnames', ['src/example.c'])
      self.assertFalse(root.joinpath('c2html.mk').exists())


if __name__ == '__main__':
  unittest.main()
