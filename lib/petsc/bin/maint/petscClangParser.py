#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os
import clang.cindex

"""
clang.cindex.TranslationUnit does not have all latest flags, but we prefix
with P_ just in case

see: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html#gab1e4965c1ebe8e41d71e90203a723fe9
"""
P_CXTranslationUnit_None = 0x0
P_CXTranslationUnit_DetailedPreprocessingRecord = 0x01
P_CXTranslationUnit_Incomplete = 0x02
P_CXTranslationUnit_PrecompiledPreamble = 0x04
P_CXTranslationUnit_CacheCompletionResults = 0x08
P_CXTranslationUnit_ForSerialization = 0x10
P_CXTranslationUnit_CXXChainedPCH = 0x20
P_CXTranslationUnit_SkipFunctionBodies = 0x40
P_CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = 0x80
P_CXTranslationUnit_CreatePreambleOnFirstParse = 0x100
P_CXTranslationUnit_KeepGoing = 0x200
P_CXTranslationUnit_SingleFileParse = 0x400
P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble = 0x800
P_CXTranslationUnit_IncludeAttributedTypes = 0x1000
P_CXTranslationUnit_VisitImplicitAttributes = 0x2000
P_CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles = 0x4000
P_CXTranslationUnit_RetainExcludedConditionalBlocks = 0x8000

mathCursors = (
  clang.cindex.CursorKind.INTEGER_LITERAL,
  clang.cindex.CursorKind.UNARY_OPERATOR
  )

def findFunctionCallExpr(tu,macroNames):
  """
  Finds all function call expressions in list macroNames.

  Note that if a particular function call is not 100% correctly defined (i.e. would the
  file actually compile) then it will not be picked up by clang AST.

  Function-like macros can be picked up, but it will be in the wrong 'order'. The AST is
  built as if you are about to compile it, so macros are handled before any real
  function definitions in the AST, making it impossible to map a macro invocation to
  its 'parent' function.
  """
  def findFunctionCallsRecurse(cursor,funcCalls=[],funcStack=[]):
    for c in cursor.get_children():
      if c.location.file is None: continue
      elif c.location.file.name != filename: continue
      elif c.kind == clang.cindex.CursorKind.FUNCTION_DECL:
        funcStack.append(c)
      elif c.kind == clang.cindex.CursorKind.CALL_EXPR:
        if c.spelling in macroNames:
          funcCalls.append((c,funcStack[-1]))
      funcCalls,funcStack = findFunctionCallsRecurse(c,funcCalls=funcCalls,funcStack=funcStack)
    if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
      try: funcStack.pop()
      except IndexError: pass
    return funcCalls,funcStack

  filename = tu.cursor.spelling
  funcCalls,_ = findFunctionCallsRecurse(tu.cursor)
  return funcCalls

class ArgCursor(object):
  def __init__(self,cursor,idx):
    def stringOnlyAlpha(string):
      for c in string:
        if not c.isalpha():
          return False
      return True

    if cursor.spelling:
      self.name = cursor.spelling
    else:
      # try to convert **&obj+73 to obj
      if cursor.kind in mathCursors:
        tokens = [t.spelling for t in cursor.get_tokens()]
      else:
        tokens = [t.spelling for t in cursor.get_tokens() if stringOnlyAlpha(t.spelling)]
      self.name = ''.join(tokens)
    if cursor.type.get_pointee().spelling:
      ctemp = cursor.type.get_pointee()
      if ctemp.spelling:
        self.typename = ctemp.get_canonical().spelling
    elif cursor.type.get_canonical().spelling:
      self.typename = cursor.type.get_canonical().spelling
    else:
      self.typename = cursor.type.spelling
    self.argidx = idx
    self.cursor = cursor
    return

  def __repr__(self):
    loc = self.cursor.location
    locStr = ':'.join([loc.file.name,str(loc.column),str(loc.line)])
    return "'"+self.name+"' of type '"+self.typename+"' at "+locStr

def checkMatchingClassid(badSource,obj,objClassid):
  """
  Does the classid match the particular PETSc type
  """
  classidMap = {
    "struct _p_Mat *" : "MAT_CLASSID",
    "_p_Mat *" : "MAT_CLASSID",
    "struct _p_Vec *" : "VEC_CLASSID",
    "_p_Vec *" : "VEC_CLASSID",
    "struct _p_PetscRandom *" : "PETSC_RANDOM_CLASSID",
    "_p_PetscRandom *" : "PETSC_RANDOM_CLASSID",
    "struct _p_PetscViewer *" : "PETSC_VIEWER_CLASSID",
    "_p_PetscViewer *" : "PETSC_VIEWER_CLASSID",
    "struct _p_PetscDraw *" : "PETSC_DRAW_CLASSID",
    "_p_PetscDraw *" :"PETSC_DRAW_CLASSID",
    }
  try:
    expectedClassid = classidMap[obj.typename]
  except KeyError:
    badSource.append("Unkown class "+str(obj))
    return badSource
  if expectedClassid != objClassid.name:
    badSource.append("Classid doesn't match. Expected '"+expectedClassid+"' found "+str(objClassid)+"'.\nUse classid for "+str(obj))
  return badSource

def checkMatchingArgNum(badSource,obj,idx,parentArgs):
  """
  Is the Arg # correct w.r.t. the function arguments
  """
  if idx.cursor.canonical.kind not in mathCursors:
    badSource.append(" ".join(["Index value is of unexpected type","'"+str(idx.cursor.canonical.kind)+"'","not"," or ".join(["'"+str(s)+"'" for s in mathCursors]),"for",str(idx)]))
    return badSource
  try:
    idxNum = int(idx.name)
  except ValueError:
    badSource.append(" ".join(["Potential argument mismatch, could not determine integer value for",str(idx)]))
    return badSource
  parentArgNames = tuple(s.name for s in parentArgs)
  try:
    matchLoc = parentArgNames.index(obj.name)
  except ValueError:
    badSource.append("Parent function '"+parentArgs[0].cursor.semantic_parent.spelling+"' arguments:\n  "+"\n  ".join(str(i+1)+": "+str(s) for i,s in enumerate(parentArgs))+"\nDon't contain the object "+str(obj))
    return badSource
  if idxNum != parentArgs[matchLoc].argidx:
    badSource.append("Argument number doesn't match. Expected '"+str(parentArgs[matchLoc].argidx)+"' found '"+str(idxNum)+"' instead for "+str(obj))
  return badSource

def checkPetscValidHeaderSpecific(badSource,func,parent):
  """
  Specific check for PetscValidHeaderSpecific, can be made more general
  """
  funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
  parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  obj,classid,idx = funcArgs
  badSource = checkMatchingClassid(badSource,obj,classid)
  badSource = checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return badSource

def checkPetscValidPointer(badSource,func,parent):
  """
  Specific check for PetscValidPointer
  """
  funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
  parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  obj,idx = funcArgs
  badSource = checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return badSource

checkDict = {
  "PetscValidPointer"        : checkPetscValidPointer,
  "PetscValidHeaderSpecific" : checkPetscValidHeaderSpecific
}
def main(petscDir,petscArch,clangDir=None,clangLib=None,verbose=False):
  def getClangSysIncludes():
    import subprocess
    """
    Get system clangs set of default include search directories.

    Because for some reason these are hardcoded by the compilers and so libclang does not have them.
    """
    output = subprocess.run(["clang","-E","-x","c++","/dev/null","-v"],capture_output=True,check=True,universal_newlines=True)
    output.check_returncode()
    # goes to stderr because of /dev/null
    includes = output.stderr.split("#include <...> search starts here:\n")[1]
    includes = includes.split("End of search list.")[0].replace("(framework directory)","")
    includes = includes.split("\n")
    includes = ["-I"+os.path.abspath(i.strip()) for i in includes if i]
    return includes


  if not clang.cindex.Config.loaded:
    clang.cindex.Config.set_compatibility_check(True)
    if clangDir:
      clangDir = os.path.expanduser(os.path.expandvars(clangDir))
      clang.cindex.Config.set_library_path(os.path.abspath(clangDir))
    elif clangLib:
      clangLib = os.path.expanduser(os.path.expandvars(clangLib))
      clang.cindex.Config.set_library_file(os.path.abspath(clangLib))
    else:
      raise RuntimeError("Must supply either clangdir or clangloc")
  with open(os.path.join(petscDir,petscArch,"lib","petsc","conf","petscvariables"),"r") as pv:
    line = pv.readline()
    while line:
      if line.startswith("PETSC_CC_INCLUDES"):
        extraIncludes = line.split("=")[1]
        break
      line = pv.readline()
  extraIncludes = extraIncludes.strip().split(" ")
  sysincludes = getClangSysIncludes()
  flags = sysincludes+["-x","c++","-include",os.path.join(petscDir,"include","petscastfix.hpp")]+extraIncludes
  if verbose: print("Compile flags","\n".join(flags))

  pchClangOptions = (
    P_CXTranslationUnit_CreatePreambleOnFirstParse |
    P_CXTranslationUnit_ForSerialization
  )

  baseClangOptions = (
    P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble
  )

  # Create the global index object
  index = clang.cindex.Index.create()

  # Create a precompiled header from petsc.h, this saves a lot of time since this
  # includes almost every sub-header in petsc
  petscHeader = os.path.join(petscDir,"include","petsc.h")
  petscPrecompiledHeader = os.path.join(petscDir,"include","petsc_ast_precompile.pch")
  if verbose: print("Creating precompiled header",petscPrecompiledHeader)
  tu = index.parse(petscHeader,args=flags,options=pchClangOptions)
  if tu.diagnostics: print([d for d in tu.diagnostics])
  tu.save(petscPrecompiledHeader)
  pchIncl = ["-include-pch",petscPrecompiledHeader]

  # exclude these directories
  excludeDirs = set(["f90-mod","f90-src","f90-custom","output","python","fsrc","ftn-auto","ftn-custom"])
  suffixes = (".c",)

  # change dirs to $PETSC_DIR/src since we are pretending to be the makefile
  srcDir = os.path.join(petscDir,"src")
  oldloc = os.getcwd()
  os.chdir(srcDir)
  for mansec in ["sys","vec","mat","dm","ksp","snes","ts","tao"]:
    msdir = os.path.join(srcDir,mansec)
    for root,dirs,files in os.walk(msdir):
      if verbose: print("Processing directory",root)
      dirs[:] = [d for d in dirs if d not in excludeDirs]
      files[:] = [f for f in files if f.endswith(suffixes)]
      for f in files:
        if verbose: print("-- Processing file",f)
        tu = index.parse(os.path.join(root,f),args=flags+pchIncl,options=baseClangOptions)
        if tu.diagnostics:
          diags = {d.spelling : 0 for d in tu.diagnostics}
          for k in diags: print(k)

        badSource = []
        for func,parent in findFunctionCallExpr(tu,checkDict.keys()):
          badSource = checkDict[func.spelling](badSource,func,parent)
        for s in badSource: print(s)
  os.chdir(oldloc)
  if os.path.exists(petscPrecompiledHeader):
    if verbose: print("Deleteing precompiled header",petscPrecompiledHeader)
    os.remove(petscPrecompiledHeader)
  return

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Set options for clang static analysis tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--PETSC_DIR",required=False,help="If this option is unused defaults to environment variable $PETSC_DIR",dest="petscdir")
  parser.add_argument("--PETSC_ARCH",required=False,help="If this option is unused defaults to environment variable $PETSC_ARCH",dest="petscarch")
  parser.add_argument("--verbose",required=False,action="store_true")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--clang_dir",help="Directory containing libclang.[so|dylib|dll]",dest="clangdir")
  group.add_argument("--clang_lib",help="Location of libclang.[so|dylib|dll]",dest="clanglib")
  args = parser.parse_args()

  try:
    petscdir = args.petscdir if args.petscdir is not None else os.environ["PETSC_DIR"]
  except KeyError as ke:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options") from ke
  try:
    petscarch = args.petscarch if args.petscarch is not None else os.environ["PETSC_ARCH"]
  except KeyError as ke:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options") from ke
  main(petscdir,petscarch,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose)
