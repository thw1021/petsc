import os

import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.minversion        = '7.4.0'
    self.version           = '7.4.0.beta4'
    self.versioninclude    = 'SuiteSparse_config.h'
    self.versionname       = 'SUITESPARSE_MAJOR_VERSION.SUITESPARSE_MINOR_VERSION.SUITESPARSE_PATCH_VERSION'
    self.gitcommit         = 'v'+self.version
    self.download          = ['git://https://github.com/DrTimothyAldenDavis/SuiteSparse','https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/'+self.gitcommit+'.tar.gz']
    self.liblist           = [['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a'],
                              ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a','librt.a'],
                              ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a'],
                              ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a','librt.a']]
    self.functions         = ['umfpack_dl_wsolve','cholmod_l_solve','klu_l_solve','SuiteSparseQR_C_solve']
    self.includes          = ['umfpack.h','cholmod.h','klu.h','SuiteSparseQR_C.h','amd.h']
    self.includedir        = os.path.join('include', 'suitesparse')
    self.hastests          = 1
    self.buildLanguages    = ['Cxx']
    self.hastestsdatafiles = 1
    self.precisions        = ['double']
    self.minCmakeVersion   = (3,20,0)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.deps       = [self.blasLapack,self.mathlib]
    self.cuda       = framework.require('config.packages.cuda',self)
    self.openmp     = framework.require('config.packages.openmp',self)
    self.metis      = framework.require('config.packages.metis',self)
    self.odeps      = [self.openmp,self.cuda,self.metis]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;btf;colamd;ccolamd;cholmod;klu;umfpack;spqr"')
    args.append('-DBLA_VENDOR=Generic')
    args.append('-DBLAS_LIB="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    return args
