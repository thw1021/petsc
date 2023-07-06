.. _doc_install:

=======
Install
=======

.. note::
    PETSc is available from several package managers.
    If you are not planning to use PETSc with :any:`external packages <doc_externalsoftware>` they are possibly the easiest way to install it.
    But be careful, some package managers have out-of-date versions (if you are a package manager, please contact us and let us know
    how we can help ensure the current PETSc release is available with your manager).

- Conda: https://anaconda.org/conda-forge/petsc  ``conda install -c conda-forge petsc`` (3.19 version)
- Fedora: https://packages.fedoraproject.org/pkgs/petsc/petsc (3.18 version)
- Homebrew: https://formulae.brew.sh/formula/petsc ``brew install petsc``
- MacPorts: https://ports.macports.org/port/petsc  ``sudo port install petsc`` (3.15 version, do not use)
- Python: https://pypi.org/project/petsc ``python -m pip install mpi4py petsc petsc4py``
- Slack: https://spack.io ``spack install petsc`` (3.19 version)
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

