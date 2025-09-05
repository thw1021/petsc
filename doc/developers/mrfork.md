(mr_fork)=

# Checkout fork merge request branch

`Developers` at times, need to checkout and build changes from a merge request fork branch.

- Using the COMMIT-SHA of the branch. This can be obtained from the MR web page (commits tab)

  ```console
  % git fetch origin COMMIT-SHA
  % git checkout FETCH_HEAD
  ```

- Checkout the branch using the repository URL. This `URL+branchname` can be copied from the MR web page.

  ```console
  % git fetch URL branchname
  % git checkout FETCH_HEAD
  ```

- Setup local git clone to access the MR branch via the MR-NUMBER. Here use the following in .git/config:

  ```console
  [remote "origin"]
        url = https://gitlab.com/petsc/petsc.git
        fetch = +refs/heads/*:refs/remotes/origin/*
        fetch = +refs/merge-requests/*:refs/remotes/origin/merge-requests/*
  ```

Now the branch is available to checkout:

  ```console
  % git checkout origin/merge-requests/MR-NUMBER/head
  ```

# Commit and push changes to a merge request fork branch

Only `Owners/Maintainers` can push commits to a merge request fork branch. Here use the ssh-URL for the git repository.

   ```console
   % git fetch ssh-URL branchname
   % git checkout -b branchname FETCH_HEAD
   % git push -u ssh-URL branchname
   % <edit/commit>
   % git push
   ```

Notes:

For example, with `URL+branchname` (that is listed in the MR webpage) as: `https://gitlab.com/petsc/petsc/-/tree/main`, we have:

- `URL` = `https://gitlab.com/petsc/petsc`
- `branchname` = `main`
- `ssh-URL` = `git@gitlab.com:petsc/petsc.git`

