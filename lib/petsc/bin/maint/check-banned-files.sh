#!/bin/bash -ex

dest=$(bash lib/petsc/bin/maint/check-merge-branch.sh)

# Search for and print newly-added files with banned extensions
# If grep's exit code is zero (success), there was a match, so declare overall failure
if git diff --diff-filter=A --name-only ${dest} | grep -i \
  -e '\.bmp$' \
  -e '\.eps$' \
  -e '\.exif$' \
  -e '\.gif$' \
  -e '\.gz$' \
  -e '\.jpeg$' \
  -e '\.jpg$' \
  -e '\.pdf$' \
  -e '\.png$' \
  -e '\.ps$' \
  -e '\.svg$' \
  -e '\.tar$' \
  -e '\.tif$' \
  -e '\.tiff$' \
  -e '\.zip$' \
  ; \
then
  printf -- "ERROR! Files with banned extensions added\n"
  exit 1
else
  printf -- "Success! No files with banned extensions added\n"
fi

file_size_limit=1024 # KiB
if git diff --name-only --diff-filter=A ${dest} \
  | xargs du -k \
  | awk -v limit=$file_size_limit '{if ($1 > limit) {print $0}}' \
  | grep '.*' \
  ; \
then
  printf -- "ERROR! File(s) larger than $file_size_limit KiB added\n"
  exit 1
else
  printf -- "Success! No files larger than $file_size_limit KiB added\n"
fi
