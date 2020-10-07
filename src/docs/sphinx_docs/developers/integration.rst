===========================
PETSc Integration Workflows
===========================

Integration Branches
====================

This section explains the workflow used by maintainers to create the
integration branches.

-  ``master`` (soon to be renamed) : stable platform for new development used by the developers and some users.
-  ``release`` : bug fixes against the latest release

Branch ``master``
-----------------

The ``master`` branch contains all features and bug fixes that are believed to be
stable and will be in the next release. Users developing software based
on recently-added features in PETSc should follow ``master``:

New feature branches should start from ``master``.

Branch ``release``
------------------

The ``release`` branch provides bug-fix patches for the latest release.
Bug fixes for the release should be started here:

.. code-block:: none

   $ git checkout -b yourname/fix-component-name release

As with new features, it will be tested and later merged to
``release`` and ``master``. Maintenance releases are tagged on ``release``.


Contributing Workflows
======================

By submitting code, the contributor gives irretrievable consent to the
redistribution and/or modification of the contributed source code as
described in the `PETSc open source license <https://gitlab.com/petsc/petsc/-/blob/master/CONTRIBUTING>`__.

Before filing a merge request
-----------------------------

-  Read :any:`style`
-  If your contribution can be logically decomposed into 2 or more
   separate contributions, submit them in sequence with different
   branches instead of all at once.
-  Include tests which cover any changes to the source code.
-  Before submitting the merge request (MR), run the full test suite -
   i.e ``make alltests TIMEOUT=600`` on your machine
-  Rebase the feature branch over latest ``master`` [so that latest copy of
   ``.gitlab-ci.yaml`` is used], and push
-  Go to https://gitlab.com/petsc/petsc/pipelines/new and submit your
   feature branch
-  If and only if the tests are perfect then submit a merge request (MR)
   otherwise fix your branch, test locally and submit a new pipeline.
-  Do not overdo requesting testing; it is a limited resource, so if you
   realize you don’t need a test you started, cancel it.

Check test results
^^^^^^^^^^^^^^^^^^

-  If you submit a pipeline you should receive an email when it
   completes (with or without error).
-  You can also go to the pipelines page at
   https://gitlab.com/petsc/petsc/pipelines .
-  For an MR, the test pipeline status is displayed near the top of the
   MR page.
-  Please report all “odd” errors in the testing that don’t seem related
   to your branch in issue https://gitlab.com/petsc/petsc/issues/360.

   1. Check the current current threads to see if it is listed and add
      it there. Otherwise, create a new thread.
   2. Also put a message like “This is related to CI (#360)”, in your
      MR/issue. It will then automatically appear in the #360
      discussion, which helps to track what’s going on at the moment.

-  Note that the retry button does NOT use any changes to the branch
   when it retries - it retries exactly what it previously tried. To
   test a fix on one or several specific systems: push the branch and
   start a new pipeline, immediately cancel that pipeline and select
   individual jobs to retry (using the little retry button to the right
   of job name).
-  For errors that occur in the cloud testing you can use
   ``docker run -it --rm -v $(pwd):/build jedbrown/mpich-ccache bash``
   which will drop you into a container with the PETSc repository as the
   working directory in the container.

Submit merge request
--------------------

-  ``git push`` prints a url that can be used to create a merge request.
   Alternatively, use https://gitlab.com/petsc/petsc/merge_requests/new
-  select the correct destination [``master`` or ``release``].
-  select appropriate labels including Workflow:Review
-  If the merge request resolves an outstanding issue (see
   https://gitlab.com/petsc/petsc/issues), you should include a `closing
   pattern <https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#default-closing-pattern>`__
   such as ``Fixes #123`` in the MR’s description so that issue gets
   closed once the MR is merged.

Submit merge requests for suggestions on design, etc.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  You do not need to test the code before submitting
-  Make sure to select DRAFT at the top of the MR page
-  select the additional label Workflow:Request-For-Comment
-  There is also a button ``Add a task list`` (next to numbered list) if
   you edit any Markdown-supporting text area. You can use this to add
   task lists to a WIP MR.

Merge request review process
----------------------------

It is the **submitter’s** responsibility to track the progress of the MR
and ensure it gets merged to master (or release). If the pipeline tests
detect problems it is the **submitter’s** responsibility to fix the
errors.

Gitlab merge requests (MRs) use “threads” to track discussions on MR.
This allows Gitlab and reviewers to track what threads are not yet
resolved.

-  When introducing a new topic (thread) in reviewing a MR make sure you
   submit with ``Start thread`` and not the ``Comment`` green button.
-  When responding to a thread make sure to use ``Reply box`` for that
   thread; do not introduce a new thread or a comment.

The **submitter** must mark threads as resolved as they fix the related
issue.

If the **submitter** feels the MR is not getting reviewed in a timely
manner they may Assign (upper right corner of the screen) to potential
reviewers and request in the discussion these same people to review by @
mentioning them.

When the merge has been approved, all the tests work, and all the
threads have been resolved the **submitter** must set a label to
Workflow:Ready-For-Merge and assign (upper right corner of the screen)
the MR to Satish (@sbalay) so he can merge it.

Docs-only changes
^^^^^^^^^^^^^^^^^

To allow for small, quick changes to documentation, if you have made
**absolutely sure** that your changes only affect documentation, you may
create your merge request, immediately add 
“Workflow:Review-docs” labels, and assign to an integrator
(currently @sbalay) to merge.

If in doubt, use the normal review process.

Remember that documentation changes should be made to the ``release``
branch if they apply to the release version of PETSc.

GitLab Instructions
===================

We use labels to track related groups of activities. To follow labels
(such as GPU or DMNetwork) go to https://gitlab.com/petsc/petsc/-/labels
and click Subscribe on the right side of the table. All merge requests
and issue submissions should supply appropriate labels.

Git Instructions
================

Setup
-----

-  Set your name: ``git config --global user.name  "Your Name"``
-  Set your email: ``git config --global user.email "me@example.com"``
-  Do not push local branches nonexistent on upstream by default:
   ``git config --global push.default simple`` (older versions of git
   require ``git config --global push.default tracking``)

Safety
------

Run this once to avoid accidentally pushing more branches than intended:

.. code-block:: bash

   $ git config --global push.default simple

(older versions of git require
``git config --global push.default tracking``)

Quick Summary of Git Commands for PETSc Developers
--------------------------------------------------

Starting and Working on a New Feature Branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Make sure you start from master: ``git checkout master``

-  Create and switch to a new feature branch:

   ::

        git checkout -b <loginname>/<affected-package>-<short-description>

   For example, Barry’s new feature branch on removing CPP in snes/ will
   use

   ``git checkout -b barry/snes-removecpp``. Use all lowercase and no
   additional underscores.

-  Write code

-  Inspect changes: ``git status``

-  Commit code:

   -  Commit all files changed: ``git commit -a`` or
   -  Commit selected files: ``git commit file1 file2 file1`` or
   -  Add new files to be committed: ``git add file1 file2`` followed by
      ``git commit``. Modified files can be added to a commit in the
      same way.

-  Push feature branch to the remote for review:
   ``git push -u origin barry/snes-removecpp``

   (or equivalently,
   ``git push --set-upstream origin barry/snes-removecpp``)

Switching between and Handling Branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Switch: ``git checkout <branchname>``, for example
   ``git checkout barry/snes-removecpp``

-  Show local and remote-tracking branches: ``git branch -a``

-  Show available remotes: ``git remote -v``

-  Show all branches available on remote: ``git ls-remote``. Use
   ``git remote show origin`` for a complete summary.

-  Delete local branch: ``git branch -d <branchname>`` (only after merge
   to ``master`` is complete)

-  Delete remote branch: ``git push origin :<branchname>`` (mind the
   colon in front of the branch name)

-  Checkout and track a branch available on remote:
   ``git checkout -t knepley/dm-hexfem`` (if you inspect other feature
   branches, e.g. Matt’s hexfem feature branch).

   If you have multiple remotes defined, use
   ``git checkout -t <remotename>/knepley/dm-hexfem``,
   e.g. ``git checkout -t origin/knepley/dm-hexfem``

-  Checkout a branch from remote, but do not track upstream changes on
   remote: ``git checkout --no-track knepley/dm-hexfem``

Reading Commit Logs
^^^^^^^^^^^^^^^^^^^

-  Show logs: ``git log``
-  Show logs for file or folder: ``git log file``
-  Show changes for each log: ``git log -p`` (add file or folder name if
   required)
-  Show diff:

   -  Current working tree: ``git diff path/to/file``
   -  To other commit: ``git diff <SHA1> path/to/file``
   -  Compare version of file in two commits:
      ``git diff <SHA1> <SHA1> path/to/file``

-  Show changes that are in master, but not yet in my current branch:

   -  At any path: ``git log ..master``
   -  Only affecting a path: ``git log ..master src/dm/impls/plex/``
   -  In my branch, but not yet in ``next``: ``git log next.. src/dm/``
   -  Tabulated by author:
      ``git shortlog v3.3..master src/dm/impls/plex``

-  Showing branches:

   -  Not yet stable: ``git branch --all --no-merged master``
   -  Being tested by early users: ``git branch --all --merged next``
   -  Will be in the next release: ``git branch --all --merged master``
   -  Remove ``--all`` to the above to not include remote tracking
      branches (work you have not interacted with yet).

-  Find where to fix a bug:

   -  Find the bad line (e.g., using a debugger)
   -  Find the commit that introduced it: ``git blame path/to/file``
   -  Find the branch containing that commit:
      ``git branch --contains COMMIT`` (usually one topic branch, plus
      ``next``)
   -  Fix bug: ``git checkout topic-branch-name``, fix bug,
      ``git commit``, and merge to ``next``, etc.

Miscellaneous
^^^^^^^^^^^^^

-  Discard changes to a file which are not yet committed:
   ``git checkout path/to/file``
-  Discard all changes to the current working tree: ``git checkout -f``
-  Forward-port local commits to the updated upstream head on master:
   ``git rebase master`` (on feature branch)
-  Delete local branch: ``git branch -D <branchname>``
-  Delete remote branch: ``git push origin :<branchname>`` (only after
   successful integration into ``master``)

Prompt
------

To stay oriented when working with branches, we encourage configuring
`git-prompt <https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh>`__.
In the following, we will include the directory, branch name, and
PETSC_ARCH in our prompt, e.g.

.. code-block:: bash

   ~/Src/petsc (master=) arch-complex
   $ git checkout release
    ~/Src/petsc (release<) arch-complex

The < indicates that our copy of release is behind the repository we are
pulling from. To achieve this we have the following in our .profile (for
bash)

.. code-block:: bash

   source ~/bin/git-prompt.sh  (point this to the location of your git-prompt.sh)
   export GIT_PS1_SHOWDIRTYSTATE=1
   export GIT_PS1_SHOWUPSTREAM="auto"
   export PS1='\w\[\e[1m\]\[\e[35m\]$(__git_ps1 " (%s)")\[\e[0m\] ${PETSC_ARCH}\n\$ '

Tab completion
--------------

To get tab-completion for git commands, first download and then source
`git-completion.bash <https://raw.github.com/git/git/master/contrib/completion/git-completion.bash>`__.

.. _sec_commit_messages:

Writing Commit Messages
-----------------------

.. code-block:: none

   ComponentName: one-line explanation of commit

   After a blank line, write a more detailed explanation of the commit.
   Many tools do not auto-wrap this part, so wrap paragraph text at a
   reasonable length. Commit messages are meant for other people to read,
   possibly months or years later, so describe the rationale for the change
   in a manner that will make sense later.

   If any interfaces have changed, the commit should fix occurrences in
   PETSc itself and the message should state its impact on users.

   If this affects any known issues, include "fix #ISSUENUMBER" or
   "see #ISSUENUM" in the message (without quotes). GitLab will create
   a link to the issue as well as a link from the issue to this commit,
   notifying anyone that was watching the issue. Feel free to link to
   mailing list discussions or [petsc-maint #NUMBER].

Formatted tags in commit messages:

.. code-block:: none

   We have defined several standard tags you should use; this makes it easy
   to search for specific types of contributions. Multiple tags may be used
   in the same commit message.

   * If other people contributed significantly to a commit, perhaps by
   reporting bugs or by writing an initial version of the patch,
   acknowledge them using tags at the end of the commit message.

   Reported-by: Helpful User <helpful@example.com>
   Based-on-patch-by: Original Idea <original@example.com>
   Thanks-to: Incremental Improver <improver@example.com>

   * If work is done for a particular well defined funding
   source or project you should label the commit with one
   or more of the tags

   Funded-by: My funding source
   Project: My project name
   Time: n hours

   Some possible values for Funded-by:
   * P-ECP - preliminary work on the Exascale Computing Project
   * IDEAS - work on interoperability/bug fixes with Hypre, SuperLU, Trilinos
   * PETSc-ODEs - work funded by Emil, Lois, Barry's base ASCRC program
   * PETSc-hierarchical - work funded by the base ASCR program in hierarchical solvers

Commit message template:

.. code-block:: none

   In order to remember tags for commit messages you can create
   a file ~/git/.gitmessage containing the tags. Then on each commit
   git automatically includes these in the editor. Just remember to
   always delete the ones you do not use. For example I have

   Funded-by:
   Project:
   Time:
   Reported-by:
   Thanks-to:

Searching git on commit messages:

.. code-block:: none

   Once you have started using tags it is possible to search the
   commit history for all contributions for a single project etc.

   * Get summary of all commits Funded by a particular source
     git log --all --grep='Funded-by: P-ECP’ --reverse [-stat or -shortstat]

   * Get the number of insertions
    git log --all --grep='Funded-by: P-ECP' --reverse --shortstat | grep changed | cut -f5 -d" " | awk '{total += $NF} END { print total }'

   * Get the number of deletions
    git log --all --grep='Funded-by: P-ECP' --reverse --shortstat | grep changed | cut -f7 -d" " | awk '{total += $NF} END { print total }'

   * Get time
    git log --all --grep='Funded-by: P-ECP' | grep Time: | cut -f2 -d":" | sed s/hours//g | sed s/hour//g |awk '{total += $NF} END { print total }'

Merge commits
^^^^^^^^^^^^^

Do not use ``-m 'useless merge statement'`` when performing a merge.
Instead, let ``git merge`` set up a commit message in your editor. It
will look something like this:

.. code-block:: none

   Merge branch 'master' into yourname/your-feature

   Conflicts:
     path/to/affected/file.c
     other/conflicted/paths.h

(perhaps without a Conflicts section if there are no conflicts). In your
editor, add a short description of *why* you are merging. The final
commit can look something like this:

.. code-block:: none

   Merge branch 'master' into yourname/your-feature

   Obtain symbol visibility (PETSC_INTERN), SNESSetConvergenceHistory()
   bug fix, and SNESConvergedDefault() interface change.

   Conflicts:
     path/to/affected/file.c
     other/conflicted/paths.h

It should either be to obtain a specific feature or because some major
changes affect you. See the Merging section of the `Developer
Instructions <developer-instructions-git>`__ for more on when to use
merges. When merging to an integration branch, a short summary of the
purpose of the topic branch is useful.

Further reading
^^^^^^^^^^^^^^^

http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html

Developing new features
-----------------------

Always start new features on a fresh branch (‘topic branch’) named after
what you intend to develop. **Always branch from ** ``master``:

.. code-block:: bash

   (master) $ git checkout -b yourname/purpose-of-branch
   Switched to a new branch 'yourname/purpose-of-branch'
   (yourname/purpose-of-branch) $

The naming convention for a topic branch is

.. code-block:: none

    <yourname>/<affected-package>-[<affected-package>-...]-<short description>

For example, Matt’s work on finite elements for hexahedra within dmplex
is carried out in the topic branch ``knepley/dmplex-hexfem`` or
``knepley/dmplex-petscsection-hexfem``. Don’t use spaces or underscores,
use lowercase letters only.

Now develop your feature, committing as you go. Write :any:`good commit messages <sec_commit_messages>`.
If you are familiar with
``git rebase``, it can be used at this time to edit your local history,
making its purpose as clear as possible for the reader. When your
feature is ready for review and possible integration, run

.. code-block:: bash

   (yourname/purpose-of-branch) $ git push --set-upstream origin yourname/purpose-of-branch

You can continue to work on this branch, and use ``git push`` to make
your changes visible. Only push on *your* branches.

If you have long-running development of a feature, you will probably
fall behind the master branch. If your branch has not been merged to
another branch (e.g., ``next``) yet, you can replay your changes on top
of the latest ``master`` using

.. code-block:: bash

   (yourname/purpose-of-branch) $ git rebase master

Checking out (Tracking) a Remote Branch
---------------------------------------

If you wish to work on a branch that is available on the remote (shown
via ``git remote show origin``), run

.. code-block:: bash

   git checkout <branchname>

to create a local branch that will merge from the remote branch. If your
local repository is not yet aware of the new branch at the remote
repository, run ``git fetch`` and then repeat the checkout.

Merging
-------

Every branch has a purpose. Merging into branch ``branch-A`` is a
declaration that the purpose of ``branch-A`` is better served by
including those commits that were in ``branch-B``. This is achieved with
the command

.. code-block:: bash

   (branch-A) $ git merge branch-B

Topic branches do not normally contain merge commits, but it is
acceptable to merge from ``master`` or from other topic branches if your
topic depends on a feature or bug fix introduced there. When making such
a merge, use the commit message to state the reason for the merge. Never
merge ``next`` into your branch.

For further philosophy on merges, see

-  `Junio Hamano: Fun with merges and purposes of
   branches <http://gitster.livejournal.com/42247.html>`__
-  `LWN: Rebasing and merging: some git best
   practices <http://lwn.net/Articles/328436/>`__
-  `Linus Torvalds: Merges from
   upstream <http://yarchive.net/comp/linux/git_merges_from_upstream.html>`__
-  `petsc-dev mailing
   list <http://lists.mcs.anl.gov/pipermail/petsc-dev/2013-March/011728.html>`__

Long running development
------------------------

If you support a particular physics code, like
`PyLith <http://www.geodynamics.org/cig/software/pylith>`__, you will
want to certify that a certain branch incorporates new features that it
needs, as well as passes all PETSc tests and follows the latest
development. This tracking branch can be called something like
``knepley/pylith``. You develop new features in separate feature
branches, and integrate into next. Then

.. code-block:: bash

   (next) $ git pull  # get changes from upstream
   (next) $ git merge knepley/new-feature-for-pylith

run tests with application, and

.. code-block:: bash

   (next) $ git push origin next next:knepley/pylith

so that you push your local branch ‘next’ that you just tested with the
application to both ‘next’ and ‘knepley/pylith’.

Application users then follow ‘knepley/pylith’, which is just marking
the state of ‘next’ the last time you tested it. Since it’s always
marking a point on ‘next’, it will fast-forward, so the user just clones
PETSc, then

.. code-block:: bash

   $ git checkout knepley/pylith

and from then on, they will get your the tested state with

.. code-block:: bash

   (knepley/pylith) $ git pull


Racy integration
----------------

Two people occasionally attempt to merge at about the same time, in
which someone will lose the race. It usually goes like this: you
checkout ‘somebranch’, pull to make sure you have everything from
upstream (you didn’t forget this, right?), merge ‘my/topic-branch’,
test, and attempt to push, getting an error like:

.. code-block:: bash

   To git@gitlab.com:petsc/petsc
    ! [rejected]        next -> next (fetch first)
   error: failed to push some refs to 'git@gitlab.com:petsc/petsc'
   hint: Updates were rejected because the remote contains work that you do
   hint: not have locally. This is usually caused by another repository pushing
   hint: to the same ref. You may want to first merge the remote changes (e.g.,
   hint: 'git pull') before pushing again.
   hint: See the 'Note about fast-forwards' in 'git push --help' for details.

**Do NOT perform a non-fast-forward pull on an integration branch.**
(Doing so creates messy history that `does not summarize
nicely <http://git-blame.blogspot.com/2013/09/fun-with-first-parent-history.html>`__
with ``git log --first-parent``.) Instead, you have two choices. The
cleanest is to gracefully lose the race and merge again.
**Recommended:**

.. code-block:: bash

   $ git reset --hard origin/next
   $ git merge my/topic-branch
   ... build and test ...
   $ git push

This produces clean history with no evidence that you encountered a race
and had to try again. `Junio explains in
detail <http://git-blame.blogspot.com/2015/03/fun-with-non-fast-forward.html>`__
why this is preferable. If your merge had significant conflicts or if
the testing you just did was especially onerous, you can switch hats and
merge the result:

.. code-block:: bash

   $ git reset --hard origin/next
   $ git merge ORIG_HEAD
     # edit commit message to state which branch was actually merged
     # due to losing a race to the integration branch.
   ... build and test ...
   $ git push

This keeps both merge commits (which is a record that there was a race,
usually perceived as clutter) but ``git log --first-parent`` still
produces an accurate and concise summary.


Nightly Builds
==============

Logs for the nightly builds are at
http://ftp.mcs.anl.gov/pub/petsc/nightlylogs/index.html
