import config.package
import os
import re

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    # NVSHMEM_MAJOR_VERSION and NVSHMEM_MINOR_VERSIO are not the NVSHMEM release version. They are the supported OpenSHMEM spec version.
    #self.versionname       = 'NVSHMEM_VENDOR_STRING' # the string has a format like "NVSHMEM v2.0.2"
    #self.versioninclude    = 'nvshmem_constants.h'
    #self.requiresversion   = 1
    self.buildLanguages    = ['CUDA'] # requires nvcc
    #self.functions         = ['nvshmem_init', 'nvshmem_finalize']
    self.includes          = ['nvshmem.h']
    # NVSHMEM 3.0 split the single libnvshmem into a host shared library and a device static library
    self.liblist           = [['libnvshmem_host.so','libnvshmem_device.a','libcuda.a'], ['libnvshmem.a','libcuda.a'], ['nvshmem.lib','cuda.lib']]
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda       = framework.require('config.packages.CUDA',self)
    self.deps      = [self.cuda]
    return

  def versionToStandardForm(self,ver):
    '''Converts from "NVSHMEMv2.0.2" to standard notation 2.0.2'''
    return re.sub('[a-zA-Z ]*', '', ver)
