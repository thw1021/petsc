(mr_fork)=

# Checkout fork merge request branch

`Developers` at times, need to checkout and build changes from a merge request fork branch.

- Using the `COMMIT-SHA` of the branch. This can be obtained from the merge request web page (Commits tab)

  ```console
  % git fetch origin <COMMIT-SHA>
  % git checkout FETCH_HEAD
  ```

- Checkout the branch using the repository `URL`. This `URL+branchname` can be copied from the merge request web page.

  ```console
  % git fetch <URL> <branchname>
  % git checkout FETCH_HEAD
  ```

- Setup a local Git clone to access the merge request branch via the `MR-NUMBER`. Here use the following in `.git/config`:

  ```console
  [remote "origin"]
        url = https://gitlab.com/petsc/petsc.git
        pushurl = git@gitlab.com:petsc/petsc.git
        fetch = +refs/heads/*:refs/remotes/origin/*
        fetch = +refs/merge-requests/*:refs/remotes/origin/merge-requests/*
  ```

  Now, the branch is available to checkout:

  ```console
  % git fetch
  % git checkout origin/merge-requests/<MR-NUMBER>/head
  ```

# Commit and push changes to a merge request fork branch

Only `Owners/Maintainers` can push commits to a merge request fork branch. Here, use the `ssh-URL` for the Git repository.

```console
% git fetch <ssh-URL> <branchname>
% git checkout -b <branchname> FETCH_HEAD
% git push -u <ssh-URL> <branchname>
% (edit/commit)
% git push
```

Notes:

For example, with `merge request` at <https://gitlab.com/petsc/petsc/-/merge_requests/8648> we have:

- `MR-NUMBER` = `8648`
- `COMMIT-SHA` = `a2cf3c576c19da12297b91b93d8d8a7a889de525` ("Copy commit SHA" from the "Commits" tab)
- `URL+branchname` = `https://gitlab.com/petsc/petsc/-/tree/balay/update-packages-download-dir` ("Copying link" of "Source branch")
- `URL` = `https://gitlab.com/petsc/petsc`
- `branchname` = `balay/update-packages-download-dir`
- `ssh-URL` = `git@gitlab.com:petsc/petsc.git`

