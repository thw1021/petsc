import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from doc import build_c2html_file


class BuildC2HTMLFileTest(unittest.TestCase):
  def test_batches_safe_text_and_falls_back_for_unterminated_text(self):
    texts = ['first\n', '<unfinished tag\n', 'unterminated_identifier']
    with mock.patch.object(build_c2html_file, '_run_mapnames', side_effect=lambda petsc_dir, mapnames, text: text) as run_mapnames:
      mapped = build_c2html_file._map_batch('petsc', 'mapnames', texts)

    self.assertEqual(mapped, texts)
    self.assertEqual(run_mapnames.call_count, 2)
    self.assertIn(texts[0], run_mapnames.call_args_list[1].args[2])
    self.assertIn(texts[1], run_mapnames.call_args_list[1].args[2])
    self.assertEqual(run_mapnames.call_args_list[0], mock.call('petsc', 'mapnames', texts[2]))

  def test_main_replaces_shell_text_filters_with_python(self):
    c2html_output = (
      '<pre width="80">source\n'
      '#if !defined(__SOURCE_H)\n'
      'PetscValidHeaderSpecific(object, 1);\n'
      '#define __SOURCE_H\n'
      '#undef __SOURCE_H\n'
      'EXTERN_C symbol\n'
      'kept\n'
    )
    mapnames_output = '<pre width="80">\nmapped\n'

    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      relative_directory = Path('src/sys')
      source_directory = root / relative_directory
      output_directory = root / 'html'
      source_directory.mkdir(parents=True)
      (output_directory / relative_directory).mkdir(parents=True)
      source_directory.joinpath('example.c').write_text('PETSCFOO_DLLEXPORT int value;\n')

      original_directory = os.getcwd()
      try:
        os.chdir(root)
        with mock.patch.object(build_c2html_file.subprocess, 'check_output', side_effect=[c2html_output, mapnames_output]) as check_output:
          build_c2html_file.main(str(root), str(output_directory), 'revision', 'c2html', 'mapnames', str(relative_directory), 'example.c')
      finally:
        os.chdir(original_directory)

      self.assertEqual(
        check_output.call_args_list,
        [
          mock.call(['c2html', '-n'], text=True, input=' int value;\n'),
          mock.call(
            ['mapnames', '-map', str(root / 'htmlmap.tmp'), '-inhtml'],
            text=True,
            input='<pre width="80">\nsource\nkept\n',
          ),
        ],
      )
      self.assertIn(mapnames_output, output_directory.joinpath(relative_directory, 'example.c.html').read_text())


if __name__ == '__main__':
  unittest.main()
