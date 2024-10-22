#!/usr/bin/env python3
import sys
if sys.version_info < (3,5):
  raise RuntimeError('requires python 3.5')
import pathlib
import os
import tempfile
import subprocess
import typing
import json


PathLike = typing.Union[str, os.PathLike]


def get_header_list(petsc_dir: PathLike, petsc_arch: PathLike) -> list[PathLike]:
  """Get all of the header files that define the PETSc C API (all header files included transitively from petsc.h)"""
  with tempfile.TemporaryDirectory() as tmpdirname:
    stem = 'tmp'
    source = pathlib.Path(tmpdirname).joinpath(stem + '.c')
    target = pathlib.Path(tmpdirname).joinpath(stem + '.deps')
    makefile = pathlib.Path(tmpdirname).joinpath('Makefile')

    with open(source, 'w') as fp:
      fp.write('''
#include <petsc.h>

void foo(void){};
''')

    with open(makefile, 'w') as fp:
      fp.write('''
include {d}/lib/petsc/conf/variables
include {d}/{a}/lib/petsc/conf/petscvariables

{t}: {s}
	$(PCC) $(PCC_FLAGS) $(PETSC_CCPPFLAGS) -MM -MF $@ $^

'''.format(d=petsc_dir, a=petsc_arch, t=target, s=source))

    # make the target, which includes one makefile target that
    # has all of the header files as prerequisites
    subprocess.run(["make", "-silent", "-C", tmpdirname, target], check=True)

    headers: list[PathLike] = []
    with open(target, 'r') as fp:
      headers = [pathlib.Path(path).resolve() for path in fp.read().split() if (os.path.isfile(path) and 'include/petsc' in path and pathlib.Path(path).suffix == '.h')]
    return headers


def get_makefile_var(petsc_dir: PathLike, petsc_arch: PathLike, var:str) -> str:
  """get a PETSc makefile variable"""
  result = subprocess.run(['make', '--silent', '-f', pathlib.Path(petsc_dir).joinpath('gmakefile'), 'print', 'PETSC_ARCH=' + str(petsc_arch), 'VAR='+var], check=True, capture_output=True).stdout
  return str(result)


def get_includes(petsc_dir: PathLike, petsc_arch: PathLike) -> list[str]:
  """Get all -I includes used in a PETSc compile command"""
  compile_command = get_makefile_var(petsc_dir, petsc_arch, 'PETSC_COMPILE_SINGLE')
  mpicc_show = get_makefile_var(petsc_dir, petsc_arch, 'MPICC_SHOW')
  includes: list[str] = [t for t in str(compile_command).split() if t.startswith('-I')]
  includes.extend([t for t in str(mpicc_show).split() if t.startswith('-I')])
  return includes


def get_api(clang: PathLike, headers: list[PathLike], petsc_dir: PathLike, petsc_arch: PathLike) -> dict:
  """Get the C API extracted by clang from the header files"""
  includes = get_includes(petsc_dir, petsc_arch)
  cmd = [clang, '-extract-api', '-x', 'c-header']
  cmd.extend([str(p) for p in headers])
  cmd.extend(includes)
  with tempfile.TemporaryDirectory() as tmpdirname:
    output = pathlib.Path(tmpdirname).joinpath('api.json')
    cmd.extend(['-o', str(output)])
    result = subprocess.run(cmd, check=True)
    api: dict = {}
    with open(output, 'r') as fp:
      api = json.load(fp)
    return api


def get_extern_symbols(api: dict) -> list[str]:
  """Get all externally visible symbols"""
  symbols = []
  for symbol in api["symbols"]:
    decl = [a["spelling"] for a in symbol["declarationFragments"]]
    if "extern" in decl:
      symbols.append(symbol["names"]["title"])
  sorted_symbols: list[str] = sorted(symbols)
  return sorted_symbols


def get_structs(api:dict) -> list[str]:
  """Get all structs"""
  struct_token = {"kind": "keyword", "spelling": "struct"}
  structs = set()
  for symbol in api["symbols"]:
    decl = symbol["declarationFragments"]
    for i, t in enumerate(decl):
      if t == struct_token:
        for c in decl[i+1:]:
          if c["kind"] == "typeIdentifier":
            structs.add(c["spelling"])
            break
  sorted_structs: list[str] = sorted(list(structs))
  return sorted_structs

def prefix_api(api: dict, prefix: str) -> None:
  """Get all externally visible symbols"""
  externs = get_extern_symbols(api)
  for symbol in api["symbols"]:
    for a in symbol["declarationFragments"]:
      word = a["spelling"]
      if word in externs:
        a["spelling"] = prefix + word


if __name__ ==  '__main__':
  import sys
  import argparse

  petsc_dir = os.environ['PETSC_DIR']
  petsc_arch = os.environ['PETSC_ARCH']

  clang = None
  try:
    clang = os.environ['CLANG']
  except:
    clang = 'clang'

  headers = get_header_list(petsc_dir, petsc_arch)
  api = get_api(clang, headers, petsc_dir, petsc_arch)
  structs = get_structs(api)
  print(structs)
  #prefix_api(api, "__petsc_double_")
  #for symbol in api["symbols"]:
  #  decl = [a["spelling"] for a in symbol["declarationFragments"]]
  #  if "static " not in decl:
  #    print(''.join(decl))
