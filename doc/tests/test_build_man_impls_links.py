import tempfile
import unittest
from pathlib import Path

from doc import build_man_impls_links as implementation_links


class BuildManImplementationsLinksTest(unittest.TestCase):
  def test_separates_object_header_from_implementations(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      manual_page = Path(temporary_directory) / 'KSP.md'
      manual_page.write_text('typedef struct _p_KSP *KSP;\n')

      implementation_links.processfile(
        '',
        '',
        temporary_directory,
        manual_page.name,
        ['include/petsc/private/kspimpl.h:struct _p_KSP {'],
        ['src/ksp/ksp/impls/cg/cgimpl.h:} KSP_CG;'],
        [],
      )

      text = manual_page.read_text()

    object_header, implementations = text.split('\n## Implementations\n')
    self.assertIn('\n## Object Header\n', object_header)
    self.assertIn('_p_KSP in include/petsc/private/kspimpl.h', object_header)
    self.assertNotIn('_p_KSP', implementations)
    self.assertIn('KSP_CG in src/ksp/ksp/impls/cg/cgimpl.h', implementations)


if __name__ == '__main__':
  unittest.main()
