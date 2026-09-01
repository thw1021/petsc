# Build PETSc documentation

`make docs` is a heavyweight, state-changing validation command. It configures `arch-docs`, may
install or download dependencies, generates manual-page sources, builds PETSc and petsc4py, and
renders the complete website.

- For an audit or review, inspect the relevant Markdown, toctrees, references, and included source.
  Do not generate documentation or configure `arch-docs`.
- Run it only when the user explicitly requests it or approves it after being told this scope and
  these side effects.
- Do not infer authorization from a request to audit, review, inspect, explain, plan, or edit
  documentation. Inspect the sources and report the full build as an optional validation step.
- Apply the same boundary to `make docspdf`, `make -C doc html`, `website-deploy`, and equivalent
  direct Sphinx invocations. The HTML builders run the configure, manual-page, c2html, and
  petsc4py hooks in `doc/conf.py`; they are not lightweight substitutes for `make docs`.
- For an authorized full build, read the current `doc/makefile` and `doc/conf.py` before selecting
  commands. Report the output directory, warnings, failures, downloads, and generated files.
