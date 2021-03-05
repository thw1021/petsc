#!/usr/bin/env python
""" Configure PETSc and build and place still-required classic docs"""

import os
import subprocess
import errno
import shutil

def main():
    """ Operations to provide data from the 'classic' PETSc docs system. """
    petsc_dir = os.path.abspath(os.path.join('..', '..', '..'))
    petsc_arch = _configure_minimal_petsc(petsc_dir)
    docs_loc = _build_classic_docs_subset(petsc_dir, petsc_arch)
    html_extra_dir = _populate_html_extra_from_classic_docs(docs_loc)
    return html_extra_dir

def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise


def _configure_minimal_petsc(petsc_dir, petsc_arch='arch-classic-docs') -> None:
    configure = [
        './configure',
        '--with-mpi=0',
        '--with-blaslapack=0',
        '--with-fortran=0',
        '--with-cxx=0',
        '--with-x=0',
        '--with-cmake=0',
        '--with-pthread=0',
        '--with-regexp=0',
        '--download-sowing',
        '--download-c2html',
        '--with-mkl_sparse_optimize=0',
        '--with-mkl_sparse=0',
        'PETSC_ARCH=' + petsc_arch,
    ]
    print('============================================')
    print('Performing a minimal PETSc (re-)configuration')
    print('PETSC_DIR=%s' % petsc_dir)
    print('PETSC_ARCH=%s' % petsc_arch)
    print('(message from', __file__, ')')
    print('============================================')
    subprocess.run(configure, cwd=petsc_dir, check=True)
    # Note: if you want to see configure.log printed out on failure,
    #       catch the subprocess.CalledProcessError exception
    #       and dump the file before re-raising.
    return petsc_arch


def _build_classic_docs_subset(petsc_dir, petsc_arch) -> None:
    """ Build and copy a subset of the classic docs, required for man pages and HTML sources.

    Returns the location of the htmlmap file.

    FIXME: this is biased towards not rebuilding, to save time. This is not ideal yet, because you have to manually wipe things to force a rebuild if, say, you want to see your new man page changes.

    Checks if the expected htmlmap file exists in a standard location.

    If this fails, performs a custom minimalist configuration and uses this to generate
    the file.

    If having to configure and build the docs, this may be quite slow (on the order of 15+ minutes).
    """
    docs_loc = os.path.join(os.getcwd(), '_build_classic')
    # Use htmlmap file as a sentinel
    htmlmap_filename = os.path.join(docs_loc, 'docs', 'manualpages', 'htmlmap')
    if not os.path.isfile(htmlmap_filename):
        command = ['make', 'alldoc1', 'alldoc2',
                   'PETSC_DIR=%s' % petsc_dir,
                   'PETSC_ARCH=%s' % petsc_arch,
                   'LOC=%s' % docs_loc]
        print('============================================')
        print('Building a subset of PETSc classic docs')
        print('PETSC_DIR=%s' % petsc_dir)
        print('PETSC_ARCH=%s' % petsc_arch)
        print(command)
        print('(message from', __file__, ')')
        print('============================================')
        subprocess.run(command, cwd=petsc_dir, check=True)
    return docs_loc


def _populate_html_extra_from_classic_docs(docs_loc) -> str:
    html_extra_dir = 'html_extra_generated'
    for subdir in ['docs', 'include', 'src']:
        if not os.path.isdir(os.path.join(html_extra_dir, subdir)):
            _mkdir_p(html_extra_dir)
            source = os.path.join(docs_loc, subdir)
            target = os.path.join(html_extra_dir, subdir)
            print('============================================')
            print('Copying directory %s from %s to %s' % (subdir, source, target))
            print('(message from', __file__, ')')
            print('============================================')
            shutil.copytree(source, target)
    return html_extra_dir

if __name__ == "__main__":
    main()
