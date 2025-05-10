import config.package
import os

class Configure(config.package.PythonPackage):
  def __init__(self, framework):
    config.package.PythonPackage.__init__(self, framework)
    self.pkgname         = 'fenics-ufl'
    self.buildLanguages  = ['Cxx']
    self.useddirectly    = 0
