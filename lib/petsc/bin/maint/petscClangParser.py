#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os
import clang.cindex
import multiprocessing as mp

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

mathCursors = set([clang.cindex.CursorKind.INTEGER_LITERAL,clang.cindex.CursorKind.UNARY_OPERATOR])

strTokens = set([clang.cindex.TokenKind.IDENTIFIER])

pchClangOptions = (
  P_CXTranslationUnit_CreatePreambleOnFirstParse |
  P_CXTranslationUnit_ForSerialization
)

baseClangOptions = (P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble)

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
        tokenlist = [t.spelling for t in cursor.get_tokens() if t.kind in strTokens]
        tokens = tokenlist[0]
        try:
          tokens[0]
        except IndexError as ie:
          raise RuntimeError(" ".join(["Empty token array",str(tokens),"from",str(tokenlist)])) from ie
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
    "_p_Tao *" : "TAO_CLASSID",
    "_p_TaoLineSearch *" : "TAOLINESEARCH_CLASSID",
    "_p_TS *" : "TS_CLASSID",
    "_p_TSAdapt *" : "TSADAPT_CLASSID",
    "_p_TSTrajector *" : "TSTRAJECTORY_CLASSID",
    "_p_PetscConvEst *" : "PETSC_OBJECT_CLASSID",
    "_p_LineSearch *" : "SNESLINESEARCH_CLASSID",
    "_p_SNES *" : "SNES_CLASSID",
    "_p_DMSNES *" : "DMSNES_CLASSID",
    "_p_DMKSP *" : "DMKSP_CLASSID",
    "_p_KSP *" : "KSP_CLASSID",
    "_p_KSPGuess *" : "KSPGUESS_CLASSID",
    "_p_PC *" : "PC_CLASSID",
    "_p_DM *" : "DM_CLASSID",
    "_p_DMLabel *" : "DMLABEL_CLASSID",
    "_p_PetscPartitioner *" : "PETSCPARTITIONER_CLASSID",
    "_p_DMField *" : "DMFIELD_CLASSID",
    "_p_DMAdaptor *" : "DM_CLASSID",
    "_p_DMPlexCellRefiner *" : "DM_CLASSID",
    "_p_PetscDS *" : "PETSCDS_CLASSID",
    "_p_PetscWeakForm *" : "PETSCWEAKFORM_CLASSID",
    "_p_PetscQuadrature *" : "PETSCQUADRATURE_CLASSID",
    "_p_PetscLimiter *" : "PETSCLIMITER_CLASSID",
    "_p_PetscFV *" : "PETSCFV_CLASSID",
    "_p_PetscDualSpace *" : "PETSCDUALSPACE_CLASSID",
    "_p_PetscFE *" : "PETSCFE_CLASSID",
    "_p_PetscSpace *" : "PETSCSPACE_CLASSID",
    "_p_Mat *" : "MAT_CLASSID",
    "_p_MatNullSpace *" : "MAT_NULLSPACE_CLASSID",
    "_p_MatTransposeColoring *" : "MAT_TRANSPOSECOLORING_CLASSID",
    "_p_MatColoring *" : "MAT_COLORING_CLASSID",
    "_p_MatFDColoring *" : "MAT_FDCOLORING_CLASSID",
    "_p_MatMFFD *" : "MATMFFD_CLASSID",
    "_p_MatCoarsen *" : "MAT_COARSEN_CLASSID",
    "_p_MatPartitioning *" : "MAT_PARTITIONING_CLASSID",
    "_p_Vec *" : "VEC_CLASSID",
    "_p_VecTagger *" : "VEC_TAGGER_CLASSID",
    "_p_PF *" : "PF_CLASSID",
    "_p_PetscRandom *" : "PETSC_RANDOM_CLASSID",
    "_p_PetscViewer *" : "PETSC_VIEWER_CLASSID",
    "_p_PetscDraw *" :"PETSC_DRAW_CLASSID",
    "_p_PetscDrawSP *" : "PETSC_DRAWSP_CLASSID",
    "_p_PetscDrawBar *" : "PETSC_DRAWBAR_CLASSID",
    "_p_PetscDrawAxis *" : "PETSC_DRAWAXIS_CLASSID",
    "_p_PetscDrawLG *" : "PETSC_DRAWLG_CLASSID",
    "_p_PetscDrawHG *" : "PETSC_DRAWHG_CLASSID",
    "_p_PetscContainer *" : "PETSC_CONTAINER_CLASSID",
    "_p_PetscSection *" : "PETSC_SECTION_CLASSID",
    "_p_PetscSectionSym *" : "PETSC_SECTION_SYM_CLASSID",
    "_p_IS *" : "IS_CLASSID",
    "_p_ISLocalToGlobalMapping *" : "IS_LTOGM_CLASSID",
    "_p_PetscSF *" : "PETSCSF_CLASSID",
    "_p_AO *" : "AO_CLASSID",
  }
  try:
    expectedClassid = classidMap[obj.typename]
  except KeyError:
    badSource.append("Unkown class "+str(obj))
    raise RuntimeError
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
    # If the parent arguments don't contain the symbol then we cannot check for correct
    # numbering
    badSource.append("Parent function '"+parentArgs[0].cursor.semantic_parent.spelling+"' arguments:\n  "+"\n  ".join(str(i+1)+": "+str(s) for i,s in enumerate(parentArgs))+"\nDon't contain the object "+str(obj))
    return badSource
  if idxNum != parentArgs[matchLoc].argidx:
    badSource.append("Argument number doesn't match. Expected '"+str(parentArgs[matchLoc].argidx)+"' found '"+str(idxNum)+"' for "+str(obj))
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

def queueWorker(clangLib,args,options,verbose,exceptions,queue,lock):
  clang.cindex.Config.set_library_file(clangLib)
  index = clang.cindex.Index.create()
  printPrefix = mp.current_process().name+" --"
  while True:
    filename = queue.get()
    try:
      if verbose:
        with lock:
          print(printPrefix,"Processing file",filename)
      tu = index.parse(filename,args=args,options=options)
      if verbose and tu.diagnostics:
        diags = {" ".join([printPrefix,filename,":",d.spelling]) : 0 for d in tu.diagnostics}
        diags = "\n".join(diags.keys())
        with lock:
          print(diags)
      badSource = []
      for func,parent in findFunctionCallExpr(tu,checkDict.keys()):
        badSource = checkDict[func.spelling](badSource,func,parent)
      if badSource:
        badSourceStr = printPrefix+" "+"\n".join(badSource)
        with lock:
          print(badSourceStr)
    except Exception as exp:
      import traceback
      err = traceback.format_exc()
      preamble = " ".join([printPrefix,"Error detected while processing",filename])
      exceptions.put(preamble+"\n"+err)
    queue.task_done()
  return

def main(petscDir,petscArch,clangDir=None,clangLib=None,verbose=False):
  if not clang.cindex.Config.loaded:
    clang.cindex.Config.set_compatibility_check(True)
    if clangDir:
      clangDir = os.path.abspath(os.path.expanduser(os.path.expandvars(clangDir)))
      clang.cindex.Config.set_library_path(clangDir)
    elif clangLib:
      clangLib = os.path.abspath(os.path.expanduser(os.path.expandvars(clangLib)))
      clang.cindex.Config.set_library_file(clangLib)
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

  # Create a precompiled header from petsc.h, this saves a lot of time since this
  # includes almost every sub-header in petsc
  petscHeader = os.path.join(petscDir,"include","petsc.h")
  petscPrecompiledHeader = os.path.join(petscDir,"include","petsc_ast_precompile.pch")
  if verbose: print("Creating precompiled header",petscPrecompiledHeader)
  index = clang.cindex.Index.create()
  tu = index.parse(petscHeader,args=flags,options=pchClangOptions)
  if tu.diagnostics: print([d for d in tu.diagnostics])
  tu.save(petscPrecompiledHeader)
  pchIncl = ["-include-pch",petscPrecompiledHeader]

  # exclude these directories
  excludeDirs = set(["f90-mod","f90-src","f90-custom","output","python","fsrc","ftn-auto","ftn-custom","f2003-src","ftn-kernels"])
  excludeDirSuffixes = (".dSYM",)
  suffixes = (".c",)

  # Spin up a queue and threads
  maxWorkers = mp.cpu_count()-1
  fpq = mp.JoinableQueue(maxWorkers)
  exq = mp.Queue()
  fpl = mp.Lock()
  # Get the library file to pass to subprocesses
  clangLib = clang.cindex.Config().get_filename()
  for i in range(maxWorkers):
    workerName = "[{i}]".format(i=i)
    worker = mp.Process(target=queueWorker,args=(clangLib,flags+pchIncl,baseClangOptions,verbose,exq,fpq,fpl,),name=workerName,daemon=True)
    worker.start()

  # change dirs to $PETSC_DIR/src since we are pretending to be the makefile
  srcDir = os.path.join(petscDir,"src")
  oldloc = os.getcwd()
  os.chdir(srcDir)
  stop = False
  for mansec in ["sys","vec","mat","dm","ksp","snes","ts","tao"]:
    for root,dirs,files in os.walk(os.path.join(srcDir,mansec)):
      if verbose: print("[ROOT] Processing directory",root)
      dirs[:] = [d for d in dirs if d not in excludeDirs]
      dirs[:] = [d for d in dirs if not d.endswith(excludeDirSuffixes)]
      files[:] = [os.path.join(root,f) for f in files if f.endswith(suffixes)]
      if files:
        for f in files:
          fpq.put(f)
        # Join here to collocate error messages to a directory
        fpq.join()
        while not exq.empty():
          stop = True
          exception = exq.get()
          print("[ERROR] ------------------------------------------------------------------------------------ [ERROR]")
          print(exception)
          print("[ERROR] ------------------------------------------------------------------------------------ [ERROR]")
        if stop:
          raise RuntimeError("Error in child process detected")
  exq.close()
  fpq.close()
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
