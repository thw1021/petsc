import re
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_man_examples_links as example_links


class BuildManExamplesLinksTest(unittest.TestCase):
  def setUp(self):
    self.temporary_directory = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary_directory.name)
    self.build_dir = self.root / 'build'
    self.manual_pages = self.build_dir / 'manualpages'
    self.manual_page = self.manual_pages / 'Sys' / 'Foo.md'
    self.manual_page.parent.mkdir(parents=True)
    self.manual_page.write_text('## Location\n<A HREF="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/interface/foo.c.html#Foo">src/ksp/interface/foo.c</A>\n')
    self.manual_pages.joinpath('manualpages.cit').write_text('man:+Foo++Foo++++man+../Sys/Foo.md#Foo\n')

  def tearDown(self):
    self.temporary_directory.cleanup()

  def write_example(self, relative_path, text='int main(void) { Foo(); }\n'):
    path = self.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

  def example_paths(self):
    text = self.manual_page.read_text().rpartition('\n## Examples\n')[2]
    return re.findall(r'PETSC_DOC_OUT_ROOT_PLACEHOLDER/(src/[^"<]+)\.html', text)

  def test_ranks_package_then_kind_then_help_then_path(self):
    help_text = 'static char help[] = "Foo";\nint main(void) { Foo(); }\n'
    examples = {
      'src/ksp/tutorials/b_plain.c': None,
      'src/ksp/tutorials/a_help.c': help_text,
      'src/ksp/tests/b_plain.c': None,
      'src/ksp/tests/a_help.c': help_text,
      'src/mat/tutorials/b_plain.c': None,
      'src/mat/tutorials/a_help.c': help_text,
      'src/mat/tests/b_plain.c': None,
      'src/mat/tests/a_help.c': help_text,
    }
    for relative_path, text in reversed(examples.items()):
      self.write_example(relative_path, text or 'int main(void) { Foo(); }\n')

    example_links.main(str(self.root), str(self.build_dir))

    self.assertEqual(self.example_paths(), [
      'src/ksp/tutorials/a_help.c',
      'src/ksp/tutorials/b_plain.c',
      'src/ksp/tests/a_help.c',
      'src/ksp/tests/b_plain.c',
      'src/mat/tutorials/a_help.c',
      'src/mat/tutorials/b_plain.c',
      'src/mat/tests/a_help.c',
      'src/mat/tests/b_plain.c',
    ])

  def test_deduplicates_before_deterministic_limit(self):
    with self.manual_pages.joinpath('manualpages.cit').open('a') as fd:
      fd.write('man:+FooAlias++FooAlias++++man+../Sys/Foo.md#FooAlias\n')
    for index in reversed(range(12)):
      self.write_example(f'src/ksp/tutorials/ex{index:02}.c', 'int main(void) { Foo(); FooAlias(); }\n')

    example_links.main(str(self.root), str(self.build_dir))

    self.assertEqual(self.example_paths(), [f'src/ksp/tutorials/ex{index:02}.c' for index in range(10)])

  def test_matches_exact_identifiers_and_help_only_mentions(self):
    self.write_example('src/ksp/tutorials/prefix.c', 'int main(void) { FooBar(); }\n')
    self.write_example('src/ksp/tutorials/help_only.c', 'static const char help[80] = "Foo";\n')

    example_links.main(str(self.root), str(self.build_dir))

    self.assertEqual(self.example_paths(), ['src/ksp/tutorials/help_only.c'])

  def test_infers_package_for_header_defined_symbols(self):
    header_page = self.manual_pages / 'Sys' / 'HeaderSymbol.md'
    header_page.write_text(
      '## Location\n<A HREF="PETSC_DOC_OUT_ROOT_PLACEHOLDER/include/petscksp.h.html#HeaderSymbol">include/petscksp.h</A>\n'
    )
    with self.manual_pages.joinpath('manualpages.cit').open('a') as fd:
      fd.write('man:+HeaderSymbol++HeaderSymbol++++man+../Sys/HeaderSymbol.md#HeaderSymbol\n')

    symbols = example_links.loadmanualpagescit(str(self.root), str(self.build_dir))

    self.assertEqual(symbols['HeaderSymbol'], ('Sys/HeaderSymbol.md', 'ksp'))


if __name__ == '__main__':
  unittest.main()
