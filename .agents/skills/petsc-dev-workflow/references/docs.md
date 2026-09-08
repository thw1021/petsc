# Build PETSc documentation

`make docs` is heavyweight and state-changing: it recreates `petsc-doc-env`, cleans generated
documentation, configures `arch-docs` generates manual-page and C2HTML sources, builds PETSc and
petsc4py, and renders the complete website. The documentation itself is built directly in
`doc/` and is independent of the caller's `PETSC_ARCH`. Run the target only when the user
explicitly requests it or approves this scope.
Apply the same boundary to `make docspdf`, `make -C doc html`, `website-deploy`, and direct Sphinx
invocations because `doc/conf.py` runs the same configure, manual-page, c2html, and petsc4py hooks.

For an audit, review, or edit, inspect the relevant Markdown, toctrees, references, and included
source, and offer the full build only as optional validation. For an authorized build, read the
current `doc/makefile` and `doc/conf.py` before choosing commands.
