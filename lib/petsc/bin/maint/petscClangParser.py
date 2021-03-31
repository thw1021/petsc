#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os
import clang.cindex
import petscClangParserUtil
import multiprocessing as mp

"""
clang.cindex.TranslationUnit does not have all latest flags, but we prefix
with P_ just in case

see: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html#gab1e4965c1ebe8e41d71e90203a723fe9
"""
P_CXTranslationUnit_None                                 = 0x0
P_CXTranslationUnit_DetailedPreprocessingRecord          = 0x01
P_CXTranslationUnit_Incomplete                           = 0x02
P_CXTranslationUnit_PrecompiledPreamble                  = 0x04
P_CXTranslationUnit_CacheCompletionResults               = 0x08
P_CXTranslationUnit_ForSerialization                     = 0x10
P_CXTranslationUnit_CXXChainedPCH                        = 0x20
P_CXTranslationUnit_SkipFunctionBodies                   = 0x40
P_CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = 0x80
P_CXTranslationUnit_CreatePreambleOnFirstParse           = 0x100
P_CXTranslationUnit_KeepGoing                            = 0x200
P_CXTranslationUnit_SingleFileParse                      = 0x400
P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble    = 0x800
P_CXTranslationUnit_IncludeAttributedTypes               = 0x1000
P_CXTranslationUnit_VisitImplicitAttributes              = 0x2000
P_CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles     = 0x4000
P_CXTranslationUnit_RetainExcludedConditionalBlocks      = 0x8000

funcCallCursors = set([clang.cindex.CursorKind.FUNCTION_DECL,clang.cindex.CursorKind.CALL_EXPR])

mathCursors = set([clang.cindex.CursorKind.INTEGER_LITERAL,clang.cindex.CursorKind.UNARY_OPERATOR])

castCursors = set([clang.cindex.CursorKind.CSTYLE_CAST_EXPR])

refDeclCursors = set([clang.cindex.CursorKind.UNEXPOSED_EXPR,clang.cindex.CursorKind.MEMBER_REF_EXPR])

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
    """
    Unused because slow, only for debugging
    """
    for c in cursor.get_children():
      try:
        if c.location.file.name != filename: continue
      except AttributeError:
        # c.location.file is None
        continue
      if c.kind == clang.cindex.CursorKind.FUNCTION_DECL:
        funcStack.append(c)
      elif c.kind == clang.cindex.CursorKind.CALL_EXPR:
        if c.spelling in macroNames:
          funcCalls.append((c,funcStack[-1]))
      funcCalls,funcStack = findFunctionCallsRecurse(c,funcCalls=funcCalls,funcStack=funcStack)
    if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
      try: funcStack.pop()
      except IndexError: pass
    return funcCalls,funcStack

  funcCalls = []
  cursor,filename = tu.cursor,tu.cursor.spelling
  for possibleParent in cursor.get_children():
    try:
      if possibleParent.location.file.name != filename: continue
    except AttributeError:
      # possibleParent.location.file is None
      continue
    if possibleParent.kind not in funcCallCursors: continue
    # If we've gotten this far we have found a function definition
    for funcChild in possibleParent.walk_preorder():
      if funcChild.kind == clang.cindex.CursorKind.CALL_EXPR:
        if funcChild.spelling in macroNames:
          funcCalls.append((funcChild,possibleParent))
  return funcCalls

class ArgCursor(object):
  @staticmethod
  def getNameFromCursor(cursor):
    if cursor.spelling:
      name = cursor.spelling
    else:
      # Try to convert **&(PetscObject)obj+73 to obj
      if cursor.kind in mathCursors:
        tokens = [t.spelling for t in cursor.get_tokens()]
        name = ''.join(tokens)
      elif cursor.kind in castCursors:
        # Need to extract the castee from the caster
        castee = [c for c in cursor.get_children() if c.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR]
        if len(castee) != 1:
          # If we don't have 1 symbol left then we're in trouble
          raise RuntimeError("Cannot determine castee from the caster for cursor at {loc}".format(loc=str(cursor.location)))
        # Easer to make a whole new temp cursor and have it figure out
        # the naming for us than duplicate the code
        name = ArgCursor.getNameFromCursor(castee[0])
      else:
        tokenlist = [t.spelling for t in cursor.get_tokens() if t.kind in strTokens]
        tokens = tokenlist[0]
        try:
          tokens[0]
        except IndexError as ie:
          raise RuntimeError(" ".join(["Empty token array",str(tokens),"from",str(tokenlist)])) from ie
        name = ''.join(tokens)
      if not name:
        raise RuntimeError("Cannot determine name of symbol")
    return name

  @staticmethod
  def getTypenameFromCursor(cursor):
    if cursor.type.get_pointee().spelling:
      ctemp = cursor.type.get_pointee()
      if ctemp.get_canonical().spelling:
        typename = ctemp.get_canonical().spelling
      else:
        typename = ctemp.spelling
    elif cursor.type.get_canonical().spelling:
      typename = cursor.type.get_canonical().spelling
    else:
      typename = cursor.type.spelling
    return typename

  def __init__(self,cursor,idx=-1):
    self.name = ArgCursor.getNameFromCursor(cursor)
    self.typename = ArgCursor.getTypenameFromCursor(cursor)
    self.argidx = idx
    self.cursor = cursor
    return

  def __repr__(self):
    loc = self.cursor.location
    locStr = ':'.join([loc.file.name,str(loc.column),str(loc.line)])
    return "'"+self.name+"' of type '"+self.typename+"' at "+locStr

class BadSource(object):
  def __init__(self,prefix,printErrorMessages=True,printWarningMessages=True,lock=None):
    self.errors = []
    self.warnings = []
    self.prefix = prefix
    self.printErrorMessages = printErrorMessages
    self.printWarningMessages = printWarningMessages
    self.lock = lock
    return

  def __repr__(self):
    if self.printErrorMessages: self.printErrors(self.prefix)
    if self.printWarningMessages: self.printWarnings(self.prefix)
    return

  def __enter__(self):
    return self

  def __exit__(self,*args):
    self.__repr__()
    return

  def __print(self,msg):
    if self.lock:
      with self.lock:
        print(msg)
    else:
      print(msg)
    return

  def addError(self,errMsg):
    self.errors.append(errMsg)
    return

  def printErrors(self,prefix):
    if self.errors:
      errmsg = prefix+" "+"\n".join(self.errors)
      self.__print(errmsg)
    return

  def addWarning(self,warnMsg):
    self.warnings.append(warnMsg)
    return

  def printWarnings(self,prefix):
    if self.warnings:
      warnmsg = prefix+" "+"\n".join(self.warnings)
      self.__print(warnmsg)
    return

petscClassIdMap = {
  "_p_AO *"                     : "AO_CLASSID",
  "_p_Characteristic *"         : "CHARACTERISTIC_CLASSID",
  "_p_DM *"                     : "DM_CLASSID",
  "_p_DMAdaptor *"              : "DM_CLASSID",
  "_p_DMField *"                : "DMFIELD_CLASSID",
  "_p_DMKSP *"                  : "DMKSP_CLASSID",
  "_p_DMLabel *"                : "DMLABEL_CLASSID",
  "_p_DMPlexCellRefiner *"      : "DM_CLASSID",
  "_p_DMSNES *"                 : "DMSNES_CLASSID",
  "_p_DMTS *"                   : "DMTS_CLASSID",
  "_p_IS *"                     : "IS_CLASSID",
  "_p_ISLocalToGlobalMapping *" : "IS_LTOGM_CLASSID",
  "_p_KSP *"                    : "KSP_CLASSID",
  "_p_KSPGuess *"               : "KSPGUESS_CLASSID",
  "_p_LineSearch *"             : "SNESLINESEARCH_CLASSID",
  "_p_Mat *"                    : "MAT_CLASSID",
  "_p_MatCoarsen *"             : "MAT_COARSEN_CLASSID",
  "_p_MatColoring *"            : "MAT_COLORING_CLASSID",
  "_p_MatFDColoring *"          : "MAT_FDCOLORING_CLASSID",
  "_p_MatMFFD *"                : "MATMFFD_CLASSID",
  "_p_MatNullSpace *"           : "MAT_NULLSPACE_CLASSID",
  "_p_MatPartitioning *"        : "MAT_PARTITIONING_CLASSID",
  "_p_MatTransposeColoring *"   : "MAT_TRANSPOSECOLORING_CLASSID",
  "_p_PC *"                     : "PC_CLASSID",
  "_p_PF *"                     : "PF_CLASSID",
  "_p_PetscContainer *"         : "PETSC_CONTAINER_CLASSID",
  "_p_PetscConvEst *"           : "PETSC_OBJECT_CLASSID",
  "_p_PetscDS *"                : "PETSCDS_CLASSID",
  "_p_PetscDraw *"              : "PETSC_DRAW_CLASSID",
  "_p_PetscDrawAxis *"          : "PETSC_DRAWAXIS_CLASSID",
  "_p_PetscDrawBar *"           : "PETSC_DRAWBAR_CLASSID",
  "_p_PetscDrawHG *"            : "PETSC_DRAWHG_CLASSID",
  "_p_PetscDrawLG *"            : "PETSC_DRAWLG_CLASSID",
  "_p_PetscDrawSP *"            : "PETSC_DRAWSP_CLASSID",
  "_p_PetscDualSpace *"         : "PETSCDUALSPACE_CLASSID",
  "_p_PetscFE *"                : "PETSCFE_CLASSID",
  "_p_PetscFV *"                : "PETSCFV_CLASSID",
  "_p_PetscLimiter *"           : "PETSCLIMITER_CLASSID",
  "_p_PetscPartitioner *"       : "PETSCPARTITIONER_CLASSID",
  "_p_PetscQuadrature *"        : "PETSCQUADRATURE_CLASSID",
  "_p_PetscRandom *"            : "PETSC_RANDOM_CLASSID",
  "_p_PetscSF *"                : "PETSCSF_CLASSID",
  "_p_PetscSection *"           : "PETSC_SECTION_CLASSID",
  "_p_PetscSectionSym *"        : "PETSC_SECTION_SYM_CLASSID",
  "_p_PetscSpace *"             : "PETSCSPACE_CLASSID",
  "_p_PetscViewer *"            : "PETSC_VIEWER_CLASSID",
  "_p_PetscWeakForm *"          : "PETSCWEAKFORM_CLASSID",
  "_p_SNES *"                   : "SNES_CLASSID",
  "_p_TS *"                     : "TS_CLASSID",
  "_p_TSAdapt *"                : "TSADAPT_CLASSID",
  "_p_TSGLLEAdapt *"            : "TSGLLEADAPT_CLASSID",
  "_p_TSTrajectory *"           : "TSTRAJECTORY_CLASSID",
  "_p_Tao *"                    : "TAO_CLASSID",
  "_p_TaoLineSearch *"          : "TAOLINESEARCH_CLASSID",
  "_p_Vec *"                    : "VEC_CLASSID",
  "_p_VecTagger *"              : "VEC_TAGGER_CLASSID",
}

def checkMatchingClassid(badSource,obj,objClassid):
  """
  Does the classid match the particular PETSc type
  """
  try:
    expectedClassid = petscClassIdMap[obj.typename]
  except KeyError:
    # Raise exception here since this isn't a bad source, moreso a failure of
    # this script since we should know about all classids
    raise RuntimeError("Unkown or invalid class "+str(obj))
  if expectedClassid != objClassid.name:
    badSource.addError("Classid doesn't match for {obj}. Expected '{expected}' found '{found}'".format(obj=str(obj),expected=expectedClassid,found=objClassid.name))
  return

def checkMatchingArgNum(badSource,obj,idx,parentArgs):
  """
  Is the Arg # correct w.r.t. the function arguments
  """
  def matchFromObjectDefinition(obj,parentArgNames):
    """
    Try and see if the cursor corresponds to:
    myFunction(barType bar)
    ...
    foo = bar->baz;
    macro(foo,barIdx)
    """
    defCursor = obj.cursor.get_definition()
    if defCursor.location == obj.cursor.location:
      # definition didn't move, so probably no definition
      raise ValueError
    elif defCursor.kind == clang.cindex.CursorKind.VAR_DECL:
      # found definition, so were in business
      # Parents here is an odd choice of words since on the very same line I loop
      # over children, but then again clangs AST has an odd semantic for parents/children
      potentialParents = [child for child in defCursor.walk_preorder() if child.kind in refDeclCursors]
      # Weed out all the self-references
      potentialParents = [parent for parent in potentialParents if parent.spelling != defCursor.spelling]
      # At this point we should be left with a single token, otherwise somethings
      # gone wrong, either way, probably a bug
      if len(potentialParents) != 1:
        raise RuntimeError("Could not determine parent of definition {defn} for object {obj}".format(defn=str(ArgCursor(defCursor)),obj=str(obj)))
      parent = potentialParents[0]
      name = ArgCursor.getNameFromCursor(parent)
      return parentArgNames.index(name)
    raise ValueError


  if idx.cursor.canonical.kind not in mathCursors:
    badSource.addError(" ".join(["Index value is of unexpected type","'"+str(idx.cursor.canonical.kind)+"'","not"," or ".join(["'"+str(s)+"'" for s in mathCursors]),"for",str(idx)]))
    return
  try:
    idxNum = int(idx.name)
  except ValueError:
    badSource.addWarning(" ".join(["Potential argument mismatch, could not determine integer value for",str(idx)]))
    return
  parentArgNames = tuple(s.name for s in parentArgs)
  try:
    matchLoc = parentArgNames.index(obj.name)
  except ValueError:
    try:
      matchLoc = matchFromObjectDefinition(obj,parentArgNames)
    except ValueError:
      # If the parent arguments don't contain the symbol and we couldn't determine a
      # definition then we cannot check for correct numbering, so we cannot do
      # anything here but emit a warning
      badSource.addWarning("Parent function '{parfn}()' does not contain the object {obj}, cannot deterine index correctness.\nParent arguments:\n  ".format(parfn=parentArgs[0].cursor.semantic_parent.spelling,obj=str(obj))+"\n  ".join(str(i+1)+": "+str(s) for i,s in enumerate(parentArgs)))
      return
  if idxNum != parentArgs[matchLoc].argidx:
    badSource.addError("Argument number doesn't match. Expected '{expected}' found '{found}' for {obj}".format(expected=str(parentArgs[matchLoc].argidx),found=str(idxNum),obj=str(obj)))
  return

def checkPetscValidHeaderSpecific(badSource,func,parent):
  """
  Specific check for PetscValidHeaderSpecific, can be made more general
  """
  funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
  parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  obj,classid,idx = funcArgs
  checkMatchingClassid(badSource,obj,classid)
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidPointer(badSource,func,parent):
  """
  Specific check for PetscValidPointer
  """
  funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
  parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  obj,idx = funcArgs
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

checkDict = {
  "PetscValidPointer"        : checkPetscValidPointer,
  "PetscValidHeaderSpecific" : checkPetscValidHeaderSpecific
}

def queueWorker(clangLib,args,options,verbose,errorMismatch,exceptions,queue,lock):
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
      with BadSource(printPrefix,lock=lock) as badSource:
        for func,parent in findFunctionCallExpr(tu,checkDict.keys()):
          checkDict[func.spelling](badSource,func,parent)
    except Exception:
      import traceback
      preamble = " ".join([printPrefix,"Error detected while processing",filename])
      exceptions.put("\n".join([preamble,traceback.format_exc()]))
    queue.task_done()
  return

def main(petscDir,petscArch,clangDir=None,clangLib=None,verbose=False,errorMismatch=None,threads=True):
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
  if tu.diagnostics: print("\n".join([d.spelling for d in tu.diagnostics]))
  tu.save(petscPrecompiledHeader)
  pchIncl = ["-include-pch",petscPrecompiledHeader]

  # exclude these directories
  excludeDirs = set(["f90-mod","f90-src","f90-custom","output","input","python","fsrc","ftn-auto","ftn-custom","f2003-src","ftn-kernels"])
  excludeDirSuffixes = (".dSYM",)
  suffixes = (".c",)

  # Get the library file to pass to subprocesses
  clangLib = clang.cindex.Config().get_filename()
  # For some reason clang doesn't like this when precompiling the header
  flags.append("-Wno-nullability-completeness")
  compilerFlags = flags+pchIncl
  clangOptions = baseClangOptions
  if threads:
    # Spin up a queue and threads
    maxWorkers = mp.cpu_count()-1
    fileProcessorQueue = mp.JoinableQueue(maxWorkers)
    exceptionSignalQueue = mp.Queue()
    fileProcessorLock = mp.Lock()
    for i in range(maxWorkers):
      workerName = "[{i}]".format(i=i)
      worker = mp.Process(target=queueWorker,args=(clangLib,compilerFlags,clangOptions,verbose,errorMismatch,exceptionSignalQueue,fileProcessorQueue,fileProcessorLock,),name=workerName,daemon=True)
      worker.start()

  # change dirs to $PETSC_DIR/src since we are pretending to be the makefile
  srcDir = os.path.join(petscDir,"src")
  oldloc = os.getcwd()
  os.chdir(srcDir)
  #for mansec in ["sys","vec","mat","dm","ksp","snes","ts","tao"]:
  for mansec in ["sys"]:
    for root,dirs,files in os.walk(os.path.join(srcDir,mansec)):
      if verbose: print("[ROOT] Processing directory",root)
      dirs[:] = [d for d in dirs if d not in excludeDirs]
      dirs[:] = [d for d in dirs if not d.endswith(excludeDirSuffixes)]
      files[:] = [os.path.join(root,f) for f in files if f.endswith(suffixes)]
      if threads:
        stopThreads = False
        for filename in files:
          fileProcessorQueue.put(filename)
        # Join here to collocate error messages to a directory
        fileProcessorQueue.join()
        while not exceptionSignalQueue.empty():
          stopThreads = True
          exception = exceptionSignalQueue.get()
          print("[ERROR] ------------------------------------------------------------------------------------ [ERROR]")
          print(exception)
          print("[ERROR] ------------------------------------------------------------------------------------ [ERROR]")
        if stopThreads: raise RuntimeError("Error in child process detected")
      else: # threads
        for filename in files:
          printPrefix = "[ROOT]"
          if verbose: print(printPrefix,"Processing file",filename)
          tu = index.parse(filename,args=compilerFlags,options=clangOptions)
          if verbose and tu.diagnostics:
            diags = {" ".join([printPrefix,filename,":",d.spelling]) : 0 for d in tu.diagnostics}
            diags = "\n".join(diags.keys())
            print(diags)
          with BadSource(printPrefix) as badSource:
            for func,parent in findFunctionCallExpr(tu,checkDict.keys()):
              checkDict[func.spelling](badSource,func,parent)
  if threads:
    fileProcessorQueue.close()
    exceptionSignalQueue.close()
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
  parser.add_argument("--error",required=False,nargs="*",default=["None"],help="Error on linter diagnostic instead of continuing, optionally provide list of macros to error for")
  parser.add_argument("--no-threads",required=False,action="store_false",help="Don't use threads",dest="threads")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--clang_dir",help="Directory containing libclang.[so|dylib|dll]",dest="clangdir")
  group.add_argument("--clang_lib",help="Location of libclang.[so|dylib|dll]",dest="clanglib")
  args = parser.parse_args()
  if "None" in args.error  or not args.error:
    if not args.error:
      raise RuntimeError("not implemented yet")
    args.error = None
  else:
    raise RuntimeError("not implemented yet")
    for errarg in args.error:
      if errarg not in checkDict.keys():
        raise RuntimeError(" ".join(["Unknown check",errarg]))

  try:
    petscdir = args.petscdir if args.petscdir is not None else os.environ["PETSC_DIR"]
  except KeyError as ke:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options") from ke
  try:
    petscarch = args.petscarch if args.petscarch is not None else os.environ["PETSC_ARCH"]
  except KeyError as ke:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options") from ke
  main(petscdir,petscarch,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,errorMismatch=args.error,threads=args.threads)
