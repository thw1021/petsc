#!/usr/bin/env sh

git fetch --unshallow --no-tags origin +release:remotes/origin/release +main:remotes/origin/main

base_release=$(git merge-base --octopus origin/release origin/main HEAD)
base_main=$(git merge-base origin/main HEAD)
if [ ${base_release} = ${base_main} ]; then
    dest=origin/release
else
    dest=origin/main
fi

# Search for and print newly-added files with banned extensions
git diff --diff-filter=A --name-only ${dest} | grep -i \
  -e '\.bmp$' \
  -e '\.eps$' \
  -e '\.exif$' \
  -e '\.gif$' \
  -e '\.gz$' \
  -e '\.jpeg$' \
  -e '\.jpg$' \
  -e '\.png$' \
  -e '\.svg$' \
  -e '\.tar$' \
  -e '\.tif$' \
  -e '\.tiff$' \
  -e '\.zip$' \

# If grep's exit code is zero, there was a match, so declare failure
if [ $? -eq 0 ]; then
  printf -- "ERROR! Files with banned extensions added\n"
  exit 1
else
  printf -- "Success! No files with banned extensions added\n"
  exit 0
fi
