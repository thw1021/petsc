#!/usr/bin/env python
# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com

import re
import os
import sys

try:
    import setuptools
except ImportError:
    setuptools = None

topdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, topdir)

pyver = sys.version_info[:2]
if pyver < (2, 6) or (3, 0) <= pyver < (3, 2):
    raise RuntimeError("Python version 2.6, 2.7 or >= 3.2 required")
if pyver == (2, 6) or pyver == (3, 2):
    sys.stderr.write(
        "WARNING: Python %d.%d is not supported.\n" % pyver)

# python-3.11+ requires cython 0.29.32+
if pyver >= (3, 11):
  CYTHON = '0.29.32'
else:
  CYTHON = '0.24'

# --------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------

def F(string):
    Name  = 'PETSc'
    email = 'petsc-maint@mcs.anl.gov'
    return string.format(
        Name=Name,
        email=email,
        name=Name.lower(),
        pyname=Name.lower()+'4py',
    )

def get_name():
    return F('{pyname}')

def get_version():
    try:
        return get_version.result
    except AttributeError:
        pass
    pkg_init_py = os.path.join(F('{pyname}'), '__init__.py')
    with open(os.path.join(topdir, 'src', pkg_init_py)) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
    version = m.groups()[0]
    get_version.result = version
    return version

def description():
    return F('{Name} for Python')

def long_description():
    with open(os.path.join(topdir, 'DESCRIPTION.rst')) as f:
        return f.read()

url      = F('https://gitlab.com/{name}/{name}')
pypiroot = F('https://pypi.io/packages/source')
pypislug = F('{pyname}')[0] + F('/{pyname}')
tarball  = F('{pyname}-%s.tar.gz' % get_version())
download = '/'.join([pypiroot, pypislug, tarball])

classifiers = """
License :: OSI Approved :: BSD License
Operating System :: POSIX
Intended Audience :: Developers
Intended Audience :: Science/Research
Programming Language :: C
Programming Language :: C++
Programming Language :: Cython
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
Development Status :: 5 - Production/Stable
"""

keywords = """
scientific computing
parallel computing
MPI
PETSc
"""

platforms = """
POSIX
Linux
macOS
FreeBSD
"""

metadata = {
    'name'             : get_name(),
    'version'          : get_version(),
    'description'      : description(),
    'long_description' : long_description(),
    'url'              : url,
    'download_url'     : download,
    'classifiers'      : classifiers.strip().split('\n'),
    'keywords'         : keywords.strip().split('\n'),
    'license'          : 'BSD-2-Clause',
    'platforms'        : platforms.strip().split('\n'),
    'author'           : 'Lisandro Dalcin',
    'author_email'     : 'dalcinl@gmail.com',
    'maintainer'       : F('{Name} Team'),
    'maintainer_email' : F('{email}'),
}
metadata.update({
    'requires': ['numpy'],
})

metadata_extra = {
    'long_description_content_type': 'text/rst',
}

# --------------------------------------------------------------------
# Extension modules
# --------------------------------------------------------------------

def sources():
    src = dict(
        source=F('{pyname}/{Name}.pyx'),
        depends=[
            F('{pyname}/*.pyx'),
            F('{pyname}/*.pxd'),
            F('{pyname}/{Name}/*.pyx'),
            F('{pyname}/{Name}/*.pxd'),
            F('{pyname}/{Name}/*.pxi'),
        ],
        workdir='src',
    )
    return [src]

def extensions():
    from os import walk
    from glob import glob
    from os.path import join
    depends = []
    include_dirs = []
    glob_join = lambda *args: glob(join(*args))
    for pth, dirs, files in walk('src'):
        depends += glob_join(pth, '*.h')
        depends += glob_join(pth, '*.c')
    for pkg in ('petsc',):
        if (pkg.upper()+'_DIR') in os.environ:
            pd = os.environ[pkg.upper()+'_DIR']
            pa = os.environ.get('PETSC_ARCH', '')
            depends += glob_join(pd, 'include', '*.h')
            depends += glob_join(pd, 'include', pkg, 'private', '*.h')
            depends += glob_join(pd, pa, 'include', '%sconf.h' % pkg)
    numpy_include = os.environ.get('NUMPY_INCLUDE')
    if numpy_include is not None:
        numpy_includes = [numpy_include]
    else:
        try:
            import numpy
            numpy_includes = [numpy.get_include()]
        except ImportError:
            numpy_includes = []
    include_dirs.extend(numpy_includes)
    ext = dict(
        name=F('{pyname}.lib.{Name}'),
        sources=[F('src/{pyname}/{Name}.c')],
        depends=depends,
        include_dirs=[
            'src',
            F('src/{pyname}/include'),
        ] + include_dirs,
        define_macros=[
            ('MPICH_SKIP_MPICXX', 1),
            ('OMPI_SKIP_MPICXX', 1),
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
    )
    return [ext]

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

from conf.config import setup, Extension
from conf.config import config, build, build_src, build_ext, install
from conf.config import clean, sdist

def get_release():
    suffix = os.path.join('src', 'binding', F('{pyname}'))
    if not topdir.endswith(os.path.join(os.path.sep, suffix)):
        return False
    release = 1
    rootdir = os.path.abspath(os.path.join(topdir, *[os.path.pardir]*3))
    version_h = os.path.join(rootdir, 'include', F('{name}version.h'))
    release_macro = '%s_VERSION_RELEASE' % F('{name}').upper()
    version_re = re.compile(r"#define\s+%s\s+([-]*\d+)" % release_macro)
    if os.path.exists(version_h) and os.path.isfile(version_h):
        with open(version_h, 'r') as f:
            release = int(version_re.search(f.read()).groups()[0])
    return bool(release)

def requires(pkgname, major, minor, release=True):
    minor = minor + int(not release)
    devel = '' if release else '.dev0'
    vmin = "%s.%s%s" % (major, minor, devel)
    vmax = "%s.%s" % (major, minor + 1)
    return "%s>=%s,<%s" % (pkgname, vmin, vmax)

def run_setup():
    setup_args = metadata.copy()
    vstr = setup_args['version'].split('.')[:2]
    x, y = tuple(map(int, vstr))
    release = get_release()
    if not release:
        setup_args['version'] = "%d.%d.0.dev0" %(x, y+1)
    if setuptools:
        setup_args['zip_safe'] = False
        setup_args['install_requires'] = ['numpy']
        for pkg in ('petsc',):
            PKG_DIR = os.environ.get(pkg.upper() + '_DIR')
            if not (PKG_DIR and os.path.isdir(PKG_DIR)):
                package = requires(pkg, x, y, release)
                setup_args['install_requires'] += [package]
        setup_args.update(metadata_extra)
    if setuptools:
        src = os.path.join('src', F('{pyname}'), F('{Name}.c'))
        has_src = os.path.exists(os.path.join(topdir, src))
        has_git = os.path.isdir(os.path.join(topdir, '.git'))
        has_hg  = os.path.isdir(os.path.join(topdir, '.hg'))
        suffix = os.path.join('src', 'binding', F('{pyname}'))
        in_tree = topdir.endswith(os.path.sep + suffix)
        if not has_src or has_git or has_hg or in_tree:
            setup_args['setup_requires'] = ['Cython>='+CYTHON]
    #
    setup(
        packages=[
            F('{pyname}'),
            F('{pyname}.lib'),
        ],
        package_dir={'' : 'src'},
        package_data={
            F('{pyname}'): [
                F('{Name}.pxd'),
                F('{Name}*.h'),
                F('include/{pyname}/*.h'),
                F('include/{pyname}/*.i'),
            ],
            F('{pyname}.lib'): [
                F('{name}.cfg'),
            ],
        },
        ext_modules=[
            Extension(**ext) for ext in extensions()
        ],
        cmdclass={
            'config':     config,
            'build':      build,
            'build_src':  build_src,
            'build_ext':  build_ext,
            'install':    install,
            'clean':      clean,
            'sdist':      sdist,
        },
        **setup_args
    )

# --------------------------------------------------------------------

def cython_req():
    return CYTHON

def cython_chk(VERSION, verbose=True):
    from conf.baseconf import log
    from conf.baseconf import Version
    from conf.baseconf import LegacyVersion
    if verbose:
        warn = lambda msg='': sys.stderr.write(msg+'\n')
    else:
        warn = lambda msg='': None
    #
    try:
        import Cython
    except ImportError:
        warn("*"*80)
        warn()
        warn(" You need Cython to generate C source files.\n")
        warn("   $ python -m pip install cython")
        warn()
        warn("*"*80)
        return False
    #
    REQUIRED = VERSION
    CYTHON_VERSION = Cython.__version__
    if VERSION is not None:
        m = re.match(r"(\d+\.\d+(?:\.\d+)?).*", CYTHON_VERSION)
        if m:
            REQUIRED  = Version(VERSION)
            AVAILABLE = Version(m.groups()[0])
        else:
            REQUIRED  = LegacyVersion(VERSION)
            AVAILABLE = LegacyVersion(CYTHON_VERSION)
        if AVAILABLE < REQUIRED:
            warn("*"*80)
            warn()
            warn(" You need Cython >= {0} (you have version {1}).\n"
                 .format(REQUIRED, CYTHON_VERSION))
            warn("   $ python -m pip install --upgrade cython")
            warn()
            warn("*"*80)
            return False
    #
    if verbose:
        log.info("using Cython version %s" % CYTHON_VERSION)
    return True

def cython_run(
    source, target=None,
    depends=(), includes=(),
    destdir_c=None, destdir_h=None,
    workdir=None, force=False,
    VERSION=None,
):
    from glob import glob
    from conf.baseconf import log
    from conf.baseconf import dep_util
    from conf.baseconf import DistutilsError
    if target is None:
        target = os.path.splitext(source)[0]+'.c'
    cwd = os.getcwd()
    try:
        if workdir:
            os.chdir(workdir)
        alldeps = [source]
        for dep in depends:
            alldeps += glob(dep)
        if not (force or dep_util.newer_group(alldeps, target)):
            log.debug("skipping '%s' -> '%s' (up-to-date)",
                      source, target)
            return
    finally:
        os.chdir(cwd)
    #
    require = 'Cython'
    if VERSION is not None:
        require += '>=%s' % VERSION
    if not cython_chk(VERSION, verbose=False):
        pkgname = re.compile(r'cython(\.|$)', re.IGNORECASE)
        for modname in list(sys.modules.keys()):
            if pkgname.match(modname):
                del sys.modules[modname]
        try:
            import warnings
            import setuptools
            install_setup_requires = setuptools._install_setup_requires
            with warnings.catch_warnings():
                category = setuptools.SetuptoolsDeprecationWarning
                warnings.simplefilter('ignore', category)
                log.info("fetching build requirement %s" % require)
                install_setup_requires(dict(setup_requires=[require]))
        except Exception:
            log.info("failed to fetch build requirement %s" % require)
    if not cython_chk(VERSION):
        raise DistutilsError("requires Cython>=%s" % VERSION)
    #
    log.info("cythonizing '%s' -> '%s'", source, target)
    from conf.cythonize import cythonize
    args = []
    if workdir:
        args += ['--working', workdir]
    args += [source]
    if target:
        args += ['--output-file', target]
    err = cythonize(args)
    if err:
        raise DistutilsError(
            "Cython failure: '%s' -> '%s'" % (source, target))


def build_sources(cmd):
    require = cython_req()
    for source in sources():
        cython_run(
            force=cmd.force,
            VERSION=require,
            **source
        )

build_src.run = build_sources

# --------------------------------------------------------------------

def main():
    run_setup()

if __name__ == '__main__':
    main()

# --------------------------------------------------------------------
