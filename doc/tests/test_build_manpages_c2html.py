import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from doc import build_manpages_c2html


class BuildManpagesC2HTMLTest(unittest.TestCase):
  def setUp(self):
    self.temporary_directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary_directory.cleanup)
    self.root = Path(self.temporary_directory.name)
    self.arch = 'arch-test'
    self.variables = self.root / self.arch / 'lib' / 'petsc' / 'conf' / 'petscvariables'
    self.variables.parent.mkdir(parents=True)

  def write_tools(self, missing=()):
    tools = {}
    for variable, executable in [('C2HTML', 'c2html'), ('DOCTEXT', 'doctext'), ('MAPNAMES', 'mapnames')]:
      path = self.root / self.arch / 'bin' / executable
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text('#!/bin/sh\n')
      path.chmod(0o755)
      if variable not in missing: tools[variable] = path
    self.variables.write_text(''.join(f'{name} = {path}\n' for name, path in tools.items()))
    return tools

  def test_gets_tools_from_selected_arch(self):
    expected = self.write_tools()
    with mock.patch.object(build_manpages_c2html.shutil, 'which') as which:
      actual = build_manpages_c2html._get_arch_tools(self.root, self.arch)
    self.assertEqual(actual, {name: os.fspath(path) for name, path in expected.items()})
    which.assert_not_called()

  def test_falls_back_to_path(self):
    self.write_tools(missing=['C2HTML', 'DOCTEXT', 'MAPNAMES'])
    paths = {name: os.fspath(self.root / self.arch / 'bin' / name) for name in ['c2html', 'doctext', 'mapnames']}
    with mock.patch.object(build_manpages_c2html.shutil, 'which', side_effect=paths.get):
      actual = build_manpages_c2html._get_arch_tools(self.root, self.arch)
    self.assertEqual(actual, {'C2HTML': paths['c2html'], 'DOCTEXT': paths['doctext'], 'MAPNAMES': paths['mapnames']})

  def test_rejects_unconfigured_arch(self):
    with self.assertRaisesRegex(RuntimeError, 'is not configured'):
      build_manpages_c2html._get_arch_tools(self.root, self.arch)

  def test_rejects_arch_without_sowing(self):
    self.write_tools(missing=['DOCTEXT', 'MAPNAMES'])
    with mock.patch.object(build_manpages_c2html.shutil, 'which', return_value=None):
      with self.assertRaisesRegex(RuntimeError, 'must provide c2html and sowing'):
        build_manpages_c2html._get_arch_tools(self.root, self.arch)


if __name__ == '__main__':
  unittest.main()
