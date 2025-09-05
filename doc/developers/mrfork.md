(mr_fork)=

# Checkout branches from a merge request fork branch

`Developers` at times, need to checkout and build changes from a merge request fork branch.

- Using the commit hash of the branch. This can be obtained from the MR web page (commits tab)

  ```console
  % git fetch origin COMMIT-HASH
  % git checkout COMMIT-HASH
  ```

- Checkout the branch using the repo URL. This URL+branchname can be copied from the MR web page.

  ```console
  % git fetch URL branchname
  % git checkout FETCH_HEAD
  ```

- Setup local git clone to access the the MR branch via the MR number. Here use the following in .git/config

  ```console
  [remote "origin"]
        url = https://gitlab.com/petsc/petsc.git
        fetch = +refs/heads/*:refs/remotes/origin/*
        fetch = +refs/merge-requests/*:refs/remotes/origin/merge-requests/*
  ```

Now the branch is available to checkout:

  ```console
  % git checkout origin/MR-NUMBER/head
  ```

# Commit and push changes to a merge request fork branch

Only `Owners/Maintainers` can push commits to a merge request fork branch. Here use the ssh git repo URL.

   ```console
   % git fetch ssh-URL branchname
   % git checkout -b branchname FETCH_HEAD
   % git push -u ssh-URL branchname
   % <edit/commit>
   % git push
   ```

Notes:

For example, with `URL = https://gitlab.com/petsc/petsc`, `branchname = main` we have:

- URL+branchname (that is listed in the MR webpage):  https://gitlab.com/petsc/petsc/-/tree/main
- ssh-URL: git@gitlab.com:petsc

