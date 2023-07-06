.. _doc_install:

=======
Install
=======

.. note::
    PETSc is available from several package managers.
    If you are planning to use PETSc **without** :any:`external packages <doc_externalsoftware>` they are possibly the easiest way to install it.

- Conda: https://anaconda.org/conda-forge/petsc  ``conda install -c conda-forge petsc``
- Homebrew: https://formulae.brew.sh/formula/petsc ``brew install petsc``
- Python: https://pypi.org/project/petsc ``python -m pip install mpi4py petsc petsc4py``
- Spack: https://spack.io ``spack install petsc``

The following package managers generally do not stay up-to-date with PETSc releases (if you are a package manager, please contact us and let us know
how we can help ensure the current PETSc release is available with your package manager). We recommend always checking the PETSc version they will
be providing to ensure it is up-to-date. PETSc evolves rapidly with new features so we highly recommend using only the most recent versions.

- Fedora: https://packages.fedoraproject.org/pkgs/petsc/petsc (3.18 version)
- MacPorts: https://ports.macports.org/port/petsc  ``sudo port install petsc`` (3.15 version, do not use)
- Slackware: https://slackbuilds.org/repository/15.0/academic/petsc/?search=petsc (3.17 version, do not use)
- Ubuntu: https://packages.ubuntu.com/jammy/petsc-dev ``sudo apt install petsc-dev`` (3.15 version, do not use)

Information and tutorials on setting up a PETSc installation.

.. toctree::
   :maxdepth: 2

   download
   install_tutorial
   install
   windows
   multibuild
   external_software
   license

