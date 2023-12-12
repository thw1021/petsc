#!/bin/bash -ex

# for checkclangformat
export PATH=/nfs/gce/projects/petsc/soft/clang+llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH

dest=`lib/petsc/bin/maint/check-merge-branch.sh`
for commit in $(git log --reverse --format=format:%H $dest..HEAD)
do
  git checkout $commit
  git show -q
  ./configure --with-clanguage=cxx
  make vermin
  make checkclangformat
  make checkbadSource
  make checkbadFileChange
  make -f gmakefile check_output
  make check_petsc4py_rst
  make CFLAGS=-Werror CXXFLAGS=-Werror FFLAGS=-Werror all
  make CFLAGS=-Werror CXXFLAGS=-Werror FFLAGS=-Werror check
  make CFLAGS=-Werror CXXFLAGS=-Werror FFLAGS=-Werror allgtests-tap gmakesearch=snes_tutorials-ex48%
done

