==============================
Developing PETSc Documentation
==============================

.. toctree::
   :maxdepth: 2


General Guidelines
==================

* Good documentation should be like a bonsai tree: alive, on display, frequently tended, and as small as possible (adapted from `these best practices <https://github.com/google/styleguide/blob/gh-pages/docguide/best_practices.md>`__).
* Wrong, irrelevant, or confusing documentation is worse than no documentation.


.. _sphinx_documentation:

Documentation with Sphinx
=========================

`Sphinx <https://www.sphinx-doc.org/en/master/>`__ is a `well-documented <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`__ and widely-used set of Python-based tools
for building documentation. Most content is written using `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__, a simple markup language.

We use Sphinx to coordinate building the documentation for our web page, as well
as a PDF of the Users Manual. To create this PDF, you must have a working
LaTeX installation.

`These slides <https://gitlab.com/psanan/petsc-sphinx-slides>`__ contain an overview of Sphinx and how we use(d) it, as of October, 2020.

The documentation build with Sphinx involves configuring a minimal build
of PETSc and building some of the :any:`classic docs <classic_docs_build>`.

Building the HTML docs locally
------------------------------

We suggest using a `Python 3 virtual environment <https://docs.python.org/3/tutorial/venv.html>`__.

  .. code-block:: console

     > cd $PETSC_DIR
     > python3 -m venv petsc-doc-env
     > . petsc-doc-env/bin/activate
     > pip install -r doc/requirements.txt
     > cd doc
     > make html  # may take several minutes

Then open ``_build/html/index.html`` with your browser.

Notes:

- The above assumes that ``python3`` is Python 3.3 or later. Check with ``python3 --version``.
- You may need to install a package like ``python3-venv``.


.. _sphinx_guidelines:

Sphinx Documentation Guidelines
-------------------------------

* Use the ``.. code-block::`` `directive
  <https://www.sphinx-doc.org/en/1.5/markup/code.html>`__ instead of the ``.. code::``
  `directive <https://docutils.sourceforge.io/docs/ref/rst/directives.html#code>`__ for
  any example code that is not included literally using ``.. literalinclude::``. See
  :ref:`below <doc_devdoc_guide_litinc>` for more details on ``.. literalinclude``.

* Any invocable command line statements longer than a few words should be in
  ``.. code-block::`` sections. Any such statements not in code-block statements must be
  enclosed by double backticks "``". For example ``make all`` is acceptable but

  .. code-block:: console

     > make PETSC_DIR=/my/path/to/petsc PETSC_ARCH=my-petsc-arch all

  should be in a block.

* All code blocks showing invocation of command line must use the "console" block
  directive. E.g.

  .. code-block:: rst

     .. code-block:: console

        > cd $PETSC_DIR/src/snes/interface
        > ./someprog
        output1
        output2

  The only exception of this is when displaying raw output, i.e. with no preceding
  commands. Then one may use just the "::" directive to improve visibility E.g.

  .. code-block:: rst

     ::

        output1
        output2

  which renders as

  ::

     output1
     output2

  Notice that now "output1" and "output2" are not greyed out as previously.

* Any code blocks that show command line invocations must be preceded by the ">"
  character. E.g.

  .. code-block:: rst

     .. code-block:: console

        > ./configure --some-args
        > make libs
        > make ./ex1
        > ./ex1 --some-args


* All environment variables such as ``$PETSC_DIR`` or ``$PATH`` must be preceded by the
  "$" character and be enclosed in double backticks "``". E.g.

  .. code-block:: rst

     Lorem ipsum dolor sit ``$PETSC_DIR``, consectetur adipiscing ``$PETSC_ARCH``...

* When referring to configuration of PETSc, specifically the ``$PETSC_DIR/configure``
  script in plain text (not code blocks), it should always be lower-case, enclosed in
  double backticks "``" and not include "./". E.g.

  .. code-block:: rst

     Lorem ipsum dolor sit ``configure``, consectetur adipiscing elit...

* If using internal section links to to jump to other places within the documentation, use
  explicit labels and namespace them appropriately. Do not use `autosectionlabel
  <https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html>`__
  extension, and do not use implicit links. E.g.

  .. code-block:: rst

     .. _doc_mydoc:

     ======================
     Start Document Heading
     ======================

     .. _doc_mydoc_internalheadline:

     Internal Headline
     =================

  And in some other file

  .. code-block:: rst

     .. _tut_mytutorial:

     ======================
     Start Tutorial Heading
     ======================

     A link- :ref:`my link name <doc_mydoc_internalheadline>`

.. _doc_devdoc_guide_litinc:

* Use the `literalinclude directive <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-literalinclude>`__ to directly include pieces of source code, as in
  the following example. Note that an "absolute" path has been used, which means
  relative to the root for the Sphinx docs (where ``conf.py`` is found).

  .. code-block:: rst

      .. literalinclude:: /../src/sys/error/err.c
         :start-at: PetscErrorCode PetscError(
         :end-at: PetscFunctionReturn(0)
         :append: }

  For robustness to changes in the source files, Use `:start-at:` and related options when possible, noting that you can also use (positive) values of `:lines:` relative to this. For languages other than C, use the `:language:` option to appropriately highlight.

* We use the `sphinxcontrib-bibtex extension <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`__
  to include citations from BibTeX files.
  You must include ``.. bibliography::`` blocks at the bottom of a page including citations (`example <https://gitlab.com/petsc/petsc/-/raw/main/doc/manual/ksp.rst>`__).
  To cite the same reference in more than one page, use `this workaround <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#key-prefixing>`__ on one of them (`example <https://gitlab.com/petsc/petsc/-/raw/main/doc/developers/articles.rst>`__) [#bibtex_footnote]_.

* See special instructions on :any:`docs_images`.

* Prefer formatting styles that are easy to modify and maintain.  In particular, use of `list-table <https://docutils.sourceforge.io/docs/ref/rst/directives.html#list-table>`_ is recommended.

* When using external links with inline URLs, prefer to use `anonymous hyperlink references <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#anonymous-hyperlinks>`__ with two trailing underscores, e.g.

  .. code-block:: rst

      `link text <https://external.org>`__

* Use restraint in adding new Sphinx extensions, in particular those which aren't
  widely-used and well-supported, or those with hidden system dependencies.

.. _docs_images:

Images
======

PETSc's documentation is tightly coupled to the source code and tests, which
motivates including the documentation in the main Git repository. However, this
precludes being able to directly track images with Git, as they are large files
which most clones of the repository will not need, and which which persist indefinitely
in the history of integration branches.

For this reason, we make the non-standard choice to store image files in a
`Git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__. To avoid
potential complication and confusion, we only use the submodule in a limited way:
the main repository only refers to commits on the ``main`` branch of the submodule repository,
and the only operation permitted on the images submodule, outside of a :any:`dedicated cleanup process <docs_images_cleanup>`,
is to add new image files in the ``main`` branch.

Adding new images
-----------------

* Whenever possible,
    - Use SVG files.  SVG is web-friendly and will be automatically converted to PDF using ``rsvg-convert`` [#svg_footnote]_
    - Avoid large files and large numbers of images.
    - Use the structure of the ``doc/`` tree in the main PETSc repository.
* Strongly prefer vector formats like SVG.
* Do not add videos or any other non-image files.
* Either with a new branch or directly (maintainers only), add your new images to the `main` branch in the upstream images repository
* Make sure that your local images submodule state corresponds to this new state, and commit these changes to your branch which uses the new images in the documentation.

Removing, renaming, moving or updating images
---------------------------------------------

Do not directly move, rename, update, or delete images in the images repository.
Simply add a logically-numbered copy of any new version of the image,
and if it is not used in *any* :any:`integration branch <sec_integration_branches>` (``main`` or ``release``),
add to-be-deleted files to the top-level list in the images repository.

.. _docs_images_cleanup:

Cleaning up the images repository
---------------------------------

(Maintainers only)

When the size of the image repository grows too large,

* Create a backup branch named ``main-archive-YYYY-MM-DD`` from the current ``main``
* Create a new commit deleting all files in the to-delete list and clearing the list
* Reset ``main`` to a single commit with this new, cleaned-up state
* Update both ``release`` and ``main`` in the main PETSc repository to point to this new state

.. _classic_docs_build:

Building Classic Documentation
==============================

Some of the documentation is built by a "classic" process as described below.

The documentation tools listed below can be
automatically downloaded and installed by ``configure``.

* `Sowing <http://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz>`__: a text processing tool developed by Bill Gropp.  This produces the PETSc manual pages; see the `Sowing documentation <http://wgropp.cs.illinois.edu/projects/software/sowing/doctext/doctext.htm>`__ and :ref:`manual_page_format`.
* `C2html <http://ftp.mcs.anl.gov/pub/petsc/c2html.tar.gz>`__: A text processing package. This generates the HTML versions of all the source code.

Note: Sowing and c2html have additional dependencies like gcc, g++, and flex and do not
use compilers specified to PETSc configure. [Windows users please install the corresponding
cygwin packages]

.. code-block:: console

    > make alldoc LOC=${PETSC_DIR}

To get a quick preview of manual pages from a single source directory (mainly to debug the manual page syntax):

.. code-block:: console

    > cd $PETSC_DIR/src/snes/interface
    > make LOC=$PETSC_DIR manualpages_buildcite
    > browse $PETSC_DIR/docs/manualpages/SNES/SNESCreate.html  # or suitable command to open the HTML page in a browser


.. rubric:: Footnotes

.. [#bibtex_footnote] The extensions's `development branch <https://github.com/mcmtroffaes/sphinxcontrib-bibtex>`__ `supports our use case better <https://github.com/mcmtroffaes/sphinxcontrib-bibtex/pull/185>`__ (``:footcite:``), which can be investigated if a release is ever made.

.. [#svg_footnote] ``rsvg-convert`` is installable with your package manager, e.g., ``librsvg2-bin`` on Debian/Ubuntu systems).
