import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ext'))
import html5_petsc


class HTML5PetscTest(unittest.TestCase):
  def test_manual_page_map_is_cached_on_builder(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      original_directory = os.getcwd()
      try:
        os.chdir(temporary_directory)
        manual_pages = Path('manualpages')
        manual_pages.mkdir()
        manual_pages.joinpath('htmlmap').write_text('man:+Foo++Foo++++man+../Sys/Foo.md#Foo\n')
        builder = SimpleNamespace()
        first_translator = SimpleNamespace(builder=builder)
        second_translator = SimpleNamespace(builder=builder)
        third_translator = SimpleNamespace(builder=SimpleNamespace())
        with mock.patch.object(html5_petsc, 'htmlmap_to_dict', wraps=html5_petsc.htmlmap_to_dict) as parser:
          first_map = html5_petsc.PETScHTMLTranslatorMixin._get_manpage_map(first_translator)
          second_map = html5_petsc.PETScHTMLTranslatorMixin._get_manpage_map(second_translator)
          third_map = html5_petsc.PETScHTMLTranslatorMixin._get_manpage_map(third_translator)
      finally:
        os.chdir(original_directory)

    self.assertIs(first_map, second_map)
    self.assertIsNot(first_map, third_map)
    self.assertEqual(parser.call_count, 2)


if __name__ == '__main__':
  unittest.main()
