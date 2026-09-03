# Initially generated with sphinx-quickstart

# Specify SPHINXOPTS via 'make' invocation to change/reset defaults, SPHINXOPTS_EXTRA to add extra options to defaults
SPHINXOPTS    = --fresh-env --write-all --show-traceback -j auto
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
LINKCHECKDIR  = _build/linkcheck
DOCS_VENV     = ../petsc-doc-env

# First so that "make" without arguments is like "make help".
help:
	-@echo "   make docs - build the HTML website inside a created Python virtual environment (venv)"

# The following does not work because it triggers the loading of docutils that many people may not have in Python
#	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)

.PHONY: help makefile checklinks checklinks_rule clean docs-env docs docspdf website-deploy

checklinks_rule:
	@$(SPHINXBUILD) --builder linkcheck "$(SOURCEDIR)" "$(LINKCHECKDIR)" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)
	@echo
	@echo "Check finished. Report is in $(LINKCHECKDIR)."

docs-env:
	${RM} -r $(DOCS_VENV)
	python3 -m venv $(DOCS_VENV)
	. $(DOCS_VENV)/bin/activate && python -m pip install -r requirements.txt

checklinks: docs-env
	. $(DOCS_VENV)/bin/activate && $(MAKE) checklinks_rule

clean:
	-@${RM} -r $(BUILDDIR)
	-@${RM} -r manualpages/concepts manualpages/singleindex.md manualpages/help.html manualpages/manualpages.cit manualpages/htmlmap* manualpages/*/*.md

# This is a directory target; making it phony would rerun the recipe when the directory already exists.
images:
	git clone https://gitlab.com/petsc/images.git -b main-2 --depth=1 images

website-deploy: images
	$(SPHINXBUILD) --builder dirhtml "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)

# use this rule to build the documentation on your machine when working on PETSc documentation
# TODO: Read sources serially and write HTML in parallel. With the current Sphinx invocation,
# parallel workers spend more time merging the large build environment than they save.
docs: docs-env clean
	. $(DOCS_VENV)/bin/activate && $(MAKE) SPHINXOPTS_EXTRA="--fail-on-warning --keep-going -j 1" html

# use this rule AFTER make docs to build the users manual as a PDF file
docspdf: docs-env
	${RM} -r _build/latex
	. $(DOCS_VENV)/bin/activate && $(MAKE) SPHINXOPTS_EXTRA="--fail-on-warning --keep-going -j 8" latexpdf

# Catch-all target: route all unknown targets to Sphinx using the new "make mode" option.
%: makefile images
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINXOPTS_EXTRA)
