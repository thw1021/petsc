import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from doc import prepare_docs


class PrepareDocsTest(unittest.TestCase):
  def setUp(self):
    self.temporary_directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary_directory.cleanup)
    self.root = Path(self.temporary_directory.name).resolve()
    self.doc = self.root / 'doc'
    self.doc.mkdir()
    (self.root / 'src').mkdir()
    (self.root / 'include').mkdir()
    self.build = self.root / 'arch-docs' / 'doc'
    self.source = self.build / 'source' / 'doc'

  def prepare(self, files):
    output = ('\0'.join(files) + '\0').encode()
    with mock.patch.object(prepare_docs.subprocess, 'check_output', return_value=output):
      prepare_docs.main(self.root)

  def test_copies_sources_without_modifying_originals(self):
    (self.doc / 'index.md').write_text('# Original\n')
    (self.root / 'include' / 'example.h').write_text('/* include */\n')
    self.prepare(['doc/index.md'])

    self.assertEqual((self.source / 'index.md').read_text(), '# Original\n')
    self.assertFalse((self.source / 'index.md').is_symlink())
    self.assertEqual((self.source / '..' / 'include' / 'example.h').read_text(), '/* include */\n')
    (self.source / 'index.md').write_text('# Generated\n')
    self.assertEqual((self.doc / 'index.md').read_text(), '# Original\n')
    self.assertEqual((self.source / 'images').resolve(), self.build / 'images')
    self.assertEqual((self.source / 'packages-docs').resolve(), self.build / 'packages-docs')
    self.assertEqual(list(self.doc.iterdir()), [self.doc / 'index.md'])

  def test_updates_and_removes_sources_but_retains_manual_pages(self):
    (self.doc / 'index.md').write_text('# Original\n')
    (self.doc / 'removed.md').write_text('# Removed\n')
    self.prepare(['doc/index.md', 'doc/removed.md'])
    manualpage = self.source / 'manualpages' / 'Sys' / 'PetscExample.md'
    manualpage.parent.mkdir(parents=True)
    manualpage.write_text('# PetscExample\n')
    (self.doc / 'index.md').write_text('# Updated\n')
    (self.doc / 'removed.md').unlink()

    self.prepare(['doc/index.md', 'doc/removed.md'])
    self.assertEqual((self.source / 'index.md').read_text(), '# Updated\n')
    self.assertFalse((self.source / 'removed.md').exists())
    self.assertEqual(manualpage.read_text(), '# PetscExample\n')

  def test_clean_preserves_downloads_and_petsc_build(self):
    shutil.copy2(Path(prepare_docs.__file__).with_name('makefile'), self.doc / 'makefile')
    self.prepare(['doc/makefile'])
    output = self.build / '_build' / 'html'
    output.mkdir(parents=True)
    (output / 'index.html').write_text('HTML\n')
    retained = [
      self.build / 'images' / 'image.svg',
      self.build / 'packages-docs' / 'PFLARE' / 'source.c',
      self.build / 'venv' / 'bin' / 'python',
      self.root / 'arch-docs' / 'lib' / 'libpetsc.a',
      self.root / 'arch-other' / 'doc' / 'index.html',
    ]
    for path in retained:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text('Keep\n')
    result = subprocess.run(
      ['make', 'clean', 'PETSC_ARCH=arch-other'], cwd=self.doc,
      env={key: value for key, value in os.environ.items() if key not in ['MAKEFLAGS', 'PETSC_DIR']},
      capture_output=True, text=True,
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertFalse((self.build / 'source').exists())
    self.assertFalse((self.build / '_build').exists())
    self.assertTrue((self.doc / 'makefile').is_file())
    for path in retained:
      self.assertEqual(path.read_text(), 'Keep\n')

  def test_make_targets_always_use_arch_docs(self):
    makefile = Path(prepare_docs.__file__).with_name('makefile')
    shutil.copy2(makefile, self.doc / 'makefile')
    for target in ['html', 'latexpdf', 'website-deploy', 'checklinks_rule']:
      with self.subTest(target=target):
        result = subprocess.run(
          ['make', '-n', 'PETSC_ARCH=arch-other', target],
          cwd=self.doc, env={key: value for key, value in os.environ.items() if key not in ['MAKEFLAGS', 'PETSC_DIR']},
          capture_output=True, text=True, check=True,
        )
        self.assertIn(str(self.source), result.stdout)
        self.assertIn(str(self.build / '_build'), result.stdout)
        self.assertNotIn('arch-other', result.stdout)

    result = subprocess.run(
      ['make', '-n', 'BUILDDIR=../public', 'html'],
      cwd=self.doc, capture_output=True, text=True, check=True,
      env={key: value for key, value in os.environ.items() if key not in ['MAKEFLAGS', 'PETSC_DIR']},
    )
    self.assertIn(str(self.root / 'public'), result.stdout)

  @unittest.skipUnless(importlib.util.find_spec('sphinx') and importlib.util.find_spec('myst_parser'), 'Sphinx and MyST required')
  def test_sphinx_resolves_source_includes_and_generated_pages(self):
    (self.doc / 'conf.py').write_text("extensions = ['myst_parser']\nmaster_doc = 'index'\n")
    (self.doc / 'index.md').write_text(
      '# Documentation\n\n```{literalinclude} /../include/example.h\n```\n\n'
      '```{toctree}\nmanualpages/Sys/PetscExample\n```\n'
    )
    (self.root / 'include' / 'example.h').write_text('/* PETSC_ARCH_DOC_INCLUDE */\n')
    self.prepare(['doc/conf.py', 'doc/index.md'])
    manualpage = self.source / 'manualpages' / 'Sys' / 'PetscExample.md'
    manualpage.parent.mkdir(parents=True)
    manualpage.write_text('# PetscExample\n\nGenerated manual page.\n')
    output = self.build / '_build' / 'html'
    result = subprocess.run(
      [sys.executable, '-B', '-m', 'sphinx', '-W', '-b', 'html', '.', str(output)],
      cwd=self.source, capture_output=True, text=True,
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn('PETSC_ARCH_DOC_INCLUDE', (output / 'index.html').read_text())
    self.assertTrue((output / 'manualpages' / 'Sys' / 'PetscExample.html').is_file())
    self.assertFalse((self.doc / 'manualpages').exists())


if __name__ == '__main__':
  unittest.main()
