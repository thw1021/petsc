# Build PETSc documentation

`make docs` is heavyweight and state-changing: it configures `arch-docs`, may install or download
dependencies, generates manual-page sources, builds PETSc and petsc4py, and renders the complete
website. Run it only when the user explicitly requests it or approves it with that scope understood.
Apply the same boundary to `make docspdf`, `make -C doc html`, `website-deploy`, and direct Sphinx
invocations because `doc/conf.py` runs the same configure, manual-page, c2html, and petsc4py hooks.

For an audit, review, or edit, inspect the relevant Markdown, toctrees, references, and included
source, and offer the full build only as optional validation. For an authorized build, read the
current `doc/makefile` and `doc/conf.py` before choosing commands. Report the output directory,
warnings, failures, downloads, and generated files.
