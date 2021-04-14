#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os,enum
import clang.cindex as clx
import petscClangParserUtil

def fastUnifiedDiff(listA,listB,fromfile="",tofile="",fromfiledate="",tofiledate="",n=0,lineterm="\n"):
  """
  Optimized version of difflib.unified_diff. difflib.SequenceMatcher is unbelievably slow but we can aggresively cut corners since we know the general location of all of the differences. This function only really serves to format the changes into the unified diff format.
  """
  import difflib,itertools
  def formatRangeUnified(pre,start,stop):
    "Convert range to the 'ed' format"
    start += pre
    stop += pre
    # Per the diff spec at http://www.unix.org/single_unix_specification/
    beginning = start+1     # lines start numbering with one
    length = stop-start
    if length == 1:
      return "{}".format(beginning)
    if not length:
      beginning -= 1        # empty ranges begin at line just before the range
    return "{},{}".format(beginning,length)

  if not fromfiledate or not tofiledate:
    import datetime
    rn = datetime.datetime.now().ctime()
    if not fromfiledate:
      fromfiledate = rn
    if not tofiledate:
      tofiledate   = rn
  fromdate = "\t{}".format(fromfiledate)
  todate   = "\t{}".format(tofiledate)
  yield "--- {}{}{}".format(fromfile,fromdate,lineterm)
  yield "+++ {}{}{}".format(tofile,todate,lineterm)
  deletes = set(["replace","delete"])
  inserts = set(["replace","insert"])

  # find consecutive streaks of values, do this by taking the difference between a value
  # and its index. If the values are consecutive val-idx(val) will be equal.
  for _,g in itertools.groupby(enumerate(val for _,val in listA),lambda x: x[0]-x[1]):
    groupIdxs = list(g)
    lineStart = min(l for _,l in groupIdxs)
    groupA,groupB = [listA[i][0] for i,_ in groupIdxs],[listB[i][0] for i,_ in groupIdxs]
    for group in difflib.SequenceMatcher(a=groupA,b=groupB).get_grouped_opcodes(n):
      first,last  = group[0],group[-1]
      file1_range = formatRangeUnified(lineStart,first[1],last[2])
      file2_range = formatRangeUnified(lineStart,first[3],last[4])
      yield "@@ -{} +{} @@{}".format(file1_range,file2_range,lineterm)

      for tag,i1,i2,j1,j2 in group:
        if tag == "equal":
          for line in groupA[i1:i2]:
            yield " "+line
          continue
        if tag in deletes:
          for line in groupA[i1:i2]:
            yield "-"+line
        if tag in inserts:
          for line in groupB[j1:j2]:
            yield "+"+line


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

# Cursors that may be attached to function-like usage
funcCallCursors = set([clx.CursorKind.FUNCTION_DECL,clx.CursorKind.CALL_EXPR])

# Cursors that may be attached to mathemateical operations or types
mathCursors = set([clx.CursorKind.INTEGER_LITERAL,clx.CursorKind.UNARY_OPERATOR,clx.CursorKind.BINARY_OPERATOR])

# Cursors that contain base literal types
literalCursors = set([clx.CursorKind.INTEGER_LITERAL,clx.CursorKind.STRING_LITERAL])

# Cursors that may be attached to casting
castCursors = set([clx.CursorKind.CSTYLE_CAST_EXPR])

# Cursors that may be attached when types are converted
convertCursors = castCursors|set([clx.CursorKind.UNEXPOSED_EXPR])

strTokens = set([clx.TokenKind.IDENTIFIER])

# General Array types
arrayTypes = set([clx.TypeKind.INCOMPLETEARRAY,clx.TypeKind.CONSTANTARRAY,clx.TypeKind.VARIABLEARRAY])

# Specific types
enumTypes   = set([clx.TypeKind.ENUM])
boolTypes   = enumTypes|set([clx.TypeKind.BOOL])
charTypes   = set([clx.TypeKind.CHAR_S,clx.TypeKind.UCHAR])
mpiIntTypes = set([clx.TypeKind.INT])
intTypes    = enumTypes|mpiIntTypes|set([clx.TypeKind.USHORT,clx.TypeKind.SHORT,clx.TypeKind.UINT,clx.TypeKind.LONGLONG,clx.TypeKind.ULONGLONG])
realTypes   = set([clx.TypeKind.FLOAT,clx.TypeKind.DOUBLE,clx.TypeKind.LONGDOUBLE,clx.TypeKind.FLOAT128])
scalarTypes = realTypes|set([clx.TypeKind.COMPLEX])

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

class QueueSignal(enum.IntEnum):
  UNIFIED_DIFF = enum.auto()
  ERRORS_LEFT  = enum.auto()
  EXIT_QUEUE   = enum.auto()

class ArgCursorWarning(Exception):
  """
  Mostly to just have a custom "something went wrong when building an ArgCursor" to except
  for rather than using a built-in type. These are errors that are meant to be caught and logged
  rather than stopping execution alltogether.

  This should make it so that actual errors aren't hidden.
  """
  pass

class ArgCursor(object):
  @staticmethod
  def getNameFromCursor(cursor):
    """
    Try to convert **&(PetscObject)obj[i]+73 to obj
    """
    def errorViewFromCursor(cursor):
      """
      Something has gone wrong, and we try to extract as much information from the cursor as
      possible for the exception. Nothing is guaranteed to be useful here.
      """
      name = cursor.displayname
      kind = cursor.kind
      # Does not yet raise exception so we can call it here
      typename = ArgCursor.getTypenameFromCursor(cursor)
      loc = cursor.location
      locStr = ':'.join([loc.file.name,str(loc.column),str(loc.line)])
      return "'{name}' of kind '{kind}' of type '{typename}' at {locStr}".format(name=name,kind=kind,typename=typename,locStr=locStr)

    name = None
    if cursor.spelling:
      name = cursor.spelling
    elif cursor.kind in mathCursors:
      tokens = [t.spelling for t in cursor.get_tokens()]
      name = ''.join(tokens)
    elif cursor.kind in castCursors:
      # Need to extract the castee from the caster
      castee = [c for c in cursor.get_children() if c.kind == clx.CursorKind.UNEXPOSED_EXPR]
      if len(castee) != 1:
        # If we don't have 1 symbol left then we're in trouble, as we probably didn't
        # pick the right cursors above
        raise RuntimeError("Cannot determine castee from the caster for cursor {obj}".format(obj=errorViewFromCursor(cursor)))
      # Easer to do some mild recursion to figure out the naming for us than duplicate
      # the code. Perhaps this should have some sort of recursion check
      name = ArgCursor.getNameFromCursor(castee[0])
    elif (cursor.type.get_canonical().kind == clx.TypeKind.POINTER) or (cursor.kind == clx.CursorKind.UNEXPOSED_EXPR):
      pointees = []
      if cursor.type.get_pointee().kind  == clx.TypeKind.CHAR_S:
        # For some reason preprocessor macros that contain strings don't propogate
        # their spelling up to the primary cursor, so we need to plumb through
        # the various sub-cursors to find it.
        pointees = [c for c in cursor.walk_preorder() if c.kind in literalCursors]
      elif clx.CursorKind.ARRAY_SUBSCRIPT_EXPR in {c.kind for c in cursor.get_children()}:
        # in the form of obj[i], so we try and weed out the iterator variable
        pointees = [c for c in cursor.walk_preorder() if c.type.get_canonical().kind in arrayTypes]
        if not pointees:
          # wasn't a pure array, so we try pointer
          pointees = [c for c in cursor.walk_preorder() if c.type.kind == clx.TypeKind.POINTER]
      pointees = list({p.spelling: p for p in pointees}.values())
      if len(pointees) == 1:
          name = ArgCursor.getNameFromCursor(pointees[0])
    if not name:
      # Catchall last attempt, we become the compiler and parse the tokens ourselves
      tokenList = [t for t in cursor.get_tokens() if t.kind in strTokens]
      # Remove iterator variables
      tokenList = [t for t in tokenList if t.cursor.kind not in mathCursors]
      # removes all cursors that have duplicate spelling
      tokenList = list({t.spelling: t for t in tokenList}.values())
      if len(tokenList) != 1:
        srcstr = petscClangParserUtil.getSourceFromCursor(cursor)
        # For whatever reason (perhaps because its macro stringization hell) PETSC_HASH_MAP
        # and PetscKernel_XXX absolutely __brick__ the AST. The resultant cursors have no
        # children, no name, no tokens, and a completely incorrect SourceLocation.
        # They are for all intents and purposes uncheckable :)
        if "PETSC_HASH" in srcstr:
          if "_MAP" in srcstr:
            raise ArgCursorWarning("Encountered unparsable PETSC_HASH_MAP for cursor {cursor}".format(cursor=errorViewFromCursor(cursor)))
          elif "_SET" in srcstr:
            raise ArgCursorWarning("Encountered unparsable PETSC_HASH_SET for cursor {cursor}".format(cursor=errorViewFromCursor(cursor)))
        elif "PetscKernel_" in srcstr:
          raise ArgCursorWarning("Encountered unparsable PetscKernel_XXX for cursor {cursor}".format(cursor=errorViewFromCursor(cursor)))
        else:
          raise RuntimeError("Unexpected number of tokens for cursor {obj}".format(obj=errorViewFromCursor(cursor)))
      name = tokenList[0].spelling
      if not name:
        raise RuntimeError("Cannot determine name of symbol from cursor {obj}".format(obj=errorViewFromCursor(cursor)))
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

  @staticmethod
  def getDerivedTypeNameFromCursor(cursor):
    return cursor.type.spelling

  @staticmethod
  def getSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,trim=False):
    return petscClangParserUtil.getSourceFromCursor(cursor,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,trim=trim)

  def getSource(self,nbefore=0,nafter=0,nboth=0,trim=False):
    return ArgCursor.getSourceFromCursor(self,nbefore=nbefore,nafter=nafter,nboth=nboth,trim=trim)

  @staticmethod
  def viewSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,ret=True):
    return petscClangParserUtil.viewSourceFromCursor(cursor,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,ret=ret)

  def viewSource(self,nbefore=0,nafter=0,nboth=0,ret=True):
    return ArgCursor.viewSourceFromCursor(self,nbefore=nbefore,nafter=nafter,nboth=nboth,ret=ret)

  @staticmethod
  def viewAstFromCursor(cursor):
    return print("\n".join(petscClangParserUtil.viewAstFromCursor(cursor)))

  def viewAst(self):
    return ArgCursor.viewAstFromCursor(self)

  def __init__(self,cursor,idx=-12345):
    assert isinstance(cursor,clx.Cursor)
    self.__cursor        = cursor
    self.name            = ArgCursor.getNameFromCursor(cursor)
    self.typename        = ArgCursor.getTypenameFromCursor(cursor)
    self.derivedtypename = ArgCursor.getDerivedTypeNameFromCursor(cursor)
    self.argidx          = idx
    return

  def __getattr__(self,attr):
    """
    Allows us to essentialy fake being a clang cursor, if __getattribute__ fails
    (i.e. the value wasn't found in self), then we try the cursor. So we can do things
    like self.translation_unit, but keep all of our variables out of the cursors
    namespace
    """
    return getattr(self.__cursor,attr)

  def __repr__(self):
    loc    = self.location
    locStr = ':'.join([loc.file.name,str(loc.column),str(loc.line)])
    srcStr = self.viewSource(ret=True)
    return "{loc}\n'{name}' of derived type '{derivedtype}', canonical type '{type}'\n{src}\n".format(loc=locStr,name=self.name,derivedtype=self.derivedtypename,type=self.typename,src=srcStr)

class SourceFix(object):
  def __init__(self,cursor,value):
    self.filename  = cursor.location.file.name
    self.startLine = cursor.extent.start.line
    assert self.startLine >= 1
    self.src       = ArgCursor.getSourceFromCursor(cursor)
    self.fixed     = None
    self.begins    = [cursor.extent.start.column-1]
    self.ends      = [cursor.extent.end.column-1]
    self.replace   = [self.src[self.begins[0]:self.ends[0]]]
    self.deltas    = [str(value)]
    self.fixDepth  = 0
    return

  def appendFix(self,fix):
    assert isinstance(fix,SourceFix)
    if self.src != fix.src:
      raise RuntimeError("Cannot combine fixes that do not share identical source!")
    self.begins.extend(fix.begins)
    self.ends.extend(fix.ends)
    self.replace.extend(fix.replace)
    self.deltas.extend(fix.deltas)
    return

  def mergeCollapse(self):
    """
    Collapses a list of fixes and produces a fixed src line.
    Fixes probably should not overwrite each other (for now), so we error out, but this
    is arguably a completely valid case. I just have not seen an example of it that I
    can use to debug with yet.
    """
    if self.fixDepth == len(self.deltas): # already collapsed, no need to do it again
      assert self.fixed
      return
    idxDelta = 0
    newSrc   = self.src
    for begin,end,replace,delta in zip(self.begins,self.ends,self.replace,self.deltas):
      if replace not in newSrc:
        raise RuntimeError("Target replacement not in src anymore, fix no longer relevant")
      if (begin+idxDelta < 0) or (end+idxDelta > len(newSrc)):
        raise RuntimeError("Idx out of bounds of src, fix not viable")
      newSrcTemp = newSrc[:begin+idxDelta]+delta+newSrc[end+idxDelta:]
      idxDelta   = len(newSrcTemp)-len(newSrc)
      newSrc     = newSrcTemp
    self.fixDepth = len(self.deltas)
    self.fixed    = newSrc
    return

class BadSource(object):
  def __init__(self,prefix,printWarningMessages=True,lock=None):
    self.errors               = []
    self.warnings             = []
    # This can actually just be a straight list, since each badSource object only ever
    # handles a single file, but use dict nonetheless
    self.diffs                = dict()
    self.prefix               = prefix
    self.printWarningMessages = printWarningMessages
    self.lock                 = lock
    self.errPrefix            = " ".join([prefix,85*"-"])
    self.warnPrefix           = " ".join([prefix,85*"%"])
    return

  def __repr__(self):
    prefixStr = "Prefix:       '{prefix}'".format(prefix=self.prefix)
    lockStr   = "Lock:          {lock}".format(lock="True" if self.lock else "False")
    showStr   = "Show warnings: {warn}".format(warn=self.printWarningMessages)
    printList = [prefixStr,lockStr,showStr]
    errorStr  = self.getAllErrors()
    if errorStr: printList.append(errorStr)
    warnStr   = self.getWarnings()
    if warnStr: printList.append(warnStr)
    return "\n".join(printList)

  def __enter__(self):
    return self

  def __exit__(self,excType,*args):
    if not excType:
      if self.printWarningMessages:
        self.__print(self.getWarnings())
      self.__print(self.getAllErrors())
    return

  def __print(self,msg):
    if msg:
      if self.lock:
        with self.lock:
          print(msg)
      else:
        print(msg)
    return

  def addErrorFromCursor(self,locCursor,errMsg,fix=None):
    errPrefix = str(locCursor)
    errMess = "".join(["\nERROR {errno}: ".format(errno=len(self.errors)),errPrefix,"\n",errMsg])
    self.errors.append((errMess,fix != None))
    try:
      self.diffs[fix.filename].append(fix)
    except KeyError:
      self.diffs[fix.filename] = [fix]
      return
    except AttributeError:
      # fix = None, return
      return
    # check if this is a compound error, i.e. an additional error on the same line
    # in which case we need to combine with previous fix
    if fix.startLine == self.diffs[fix.filename][-2].startLine:
      # remove ourselves from the list
      fix = self.diffs[fix.filename].pop()
      # this should now be the previous fix on the same line, so we combine with it
      self.diffs[fix.filename][-1].appendFix(fix)
    return

  def getAllErrors(self):
    if self.errors:
      return "\n".join([self.errPrefix,"\n".join(err for err,_ in self.errors)[1:],self.errPrefix])
    return

  def getErrorsLeft(self):
    errLeft = [err for err,fixed in self.errors if not fixed]
    if errLeft:
      return "\n".join([self.errPrefix,"\n".join(errLeft)[1:],self.errPrefix])
    return

  def addWarning(self,warnMsg):
    self.warnings.append("".join(["\nWARNING {warnno}: ".format(warnno=len(self.warnings)),warnMsg]))
    return

  def addWarningFromCursor(self,locCursor,warnMsg):
    warnPrefix = str(locCursor)
    self.warnings.append("".join(["\nWARNING {warnno}: ".format(warnno=len(self.warnings)),warnPrefix,"\n",warnMsg]))
    return

  def getWarnings(self):
    if self.warnings:
      return "\n".join([self.warnPrefix,"\n".join(self.warnings)[1:],self.warnPrefix])
    return

  def coalesceDiffs(self):
    combinedDiffs = []
    for filename,diffs in self.diffs.items():
      minLine,maxLine = diffs[0].startLine-1,diffs[-1].startLine
      for d in diffs:
        d.mergeCollapse()
      srcList   = [(d.src,d.startLine) for d in diffs]
      fixedList = [(d.fixed,d.startLine) for d in diffs]
      unified   = fastUnifiedDiff(srcList,fixedList,fromfile=filename,tofile=filename)
      combinedDiffs.append((filename,"".join(unified)))
    return combinedDiffs

class FilterFunctor(object):
  def __init__(self,expected,funcCursor,pointer=False,notPointerHook=None,pointerHook=None,successHook=None,failureHook=None,**kwargs):
    self.expectedTypeKinds            = expected
    self.funcCursor                   = funcCursor
    self.pointer                      = pointer
    self.unexpectedNotPointerFunction = notPointerHook
    self.unexpectedPointerFunction    = pointerHook
    self.successFunction              = successHook
    self.failureFunction              = failureHook
    self.extraArgs                    = kwargs
    return

  def unexpectedNotPointerHook(self,badSource,obj,objType):
    try:
      self.unexpectedNotPointerFunction(badSource,obj,objType,**vars(self))
    except TypeError:
      badSource.addErrorFromCursor(obj,"Object of clang type {otype} is not a pointer. Expected pointer of one of the following types: {types}".format(otype=objType.kind,types=self.expectedTypeKinds))
    return

  def unexpectedPointerHook(self,badSource,obj,objType):
    try:
      self.unexpectedPointerFunction(badSource,obj,objType,**vars(self))
    except TypeError:
      badSource.addErrorFromCursor(obj,"Object of clang type {otype} is a pointer when it should not be".format(otype=objType.kind))
    return

  def successHook(self,badSource,obj,objType):
    try:
      self.successFunction(badSource,obj,objType,**vars(self))
    except TypeError:
      pass
    return

  def failureHook(self,badSource,obj,objType):
    try:
      # currently no function does this, but in case a "try one more thing" function,
      # one can implement it in the if not handled clause
      handled = self.failureFunction(badSource,obj,objType,**vars(self))
    except TypeError:
      handled = False
    if not handled:
      badSource.addErrorFromCursor(obj,"Object of clang type {otype} is not in expected types: {types}".format(otype=objType.kind,types=self.expectedTypeKinds))
    return


"""Generic test functions"""
def checkIsPetscScalarAndNotPetscReal(badSource,obj,objType,**kwargs):
  if "PetscScalar" not in obj.derivedtypename:
    funcCursor = kwargs["funcCursor"]
    if "PetscReal" in obj.derivedtypename:
      validFunc = kwargs["extraArgs"]["validFunc"]
      call = [c for c in funcCursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO]
      assert len(call) == 1
      fix = SourceFix(call[0],validFunc)
      badSource.addErrorFromCursor(obj,"Incorrect use of {funcName}(), use {validFunc}() instead".format(funcName=funcCursor.displayname,validFunc=validFunc),fix=fix)
    else:
      badSource.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscScalars".format(funcName=funcCursor.displayname))
  return

def checkIsPetscRealAndNotPetscScalar(badSource,obj,objType,**kwargs):
  if "PetscReal" not in obj.derivedtypename:
    funcCursor = kwargs["funcCursor"]
    if "PetscScalar" in obj.derivedtypename:
      validFunc = kwargs["extraArgs"]["validFunc"]
      call = [c for c in funcCursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO]
      assert len(call) == 1
      fix = SourceFix(call[0],validFunc)
      badSource.addErrorFromCursor(obj,"Incorrect use of {funcName}(), use {validFunc}() instead".format(funcName=funcCursor.displayname,validFunc=validFunc),fix=fix)
    else:
      badSource.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscReals".format(funcName=funcCursor.displayname))
  return

def checkIntIsNotPetscBool(badSource,obj,objType,**kwargs):
  if "PetscBool" in obj.derivedtypename:
    funcCursor,validFunc = kwargs["funcCursor"],kwargs["extraArgs"]["validFunc"]
    call = [c for c in funcCursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO]
    assert len(call) == 1
    fix = SourceFix(call[0],validFunc)
    badSource.addErrorFromCursor(obj,"Incorrect use of {funcName}(), use {validFunc}() instead".format(funcName=funcCursor.displayname,validFunc=validFunc),fix=fix)
  return

def checkMPIIntIsNotPetscInt(badSource,obj,objType,**kwargs):
  if "PetscInt" in obj.derivedtypename:
    funcCursor,validFunc = kwargs["funcCursor"],kwargs["extraArgs"]["validFunc"]
    call = [c for c in funcCursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO]
    assert len(call) == 1
    fix = SourceFix(call[0],preferredFuncName)
    badSource.addErrorFromCursor(obj,"Incorrect use of {funcName}(), use {validFunc}() instead".format(funcName=funcCursor.displayname,validFunc=validFunc),fix=fix)
  return

def checkIsPetscBool(badSource,obj,objType,**kwargs):
  if ("PetscBool" not in obj.derivedtypename) and ("bool" not in obj.typename):
    funcCursor = kwargs["funcCursor"]
    badSource.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscBool or bool".format(funcName=funcCursor.displayname))
  return

def checkMatchingClassid(badSource,obj,objClassid):
  """
  Does the classid match the particular PETSc type
  """
  try:
    expectedClassid = petscClassIdMap[obj.typename]
  except KeyError:
    # The class doesn't exist, perhaps they passed in a wonky type, at the very least it
    # isn't a petsc type
    if not (obj.typename.startswith("_p_") or obj.typename.startswith("_n_")):
      classFromClassId = list(petscClassIdMap.keys())[list(petscClassIdMap.values()).index(objClassid.name)]
      badSource.addErrorFromCursor(obj,"Classid doesn't match. Expected type '{typen}' for '{classid}'".format(typen=classFromClassId,classid=objClassid.name))
      return
    # Raise exception here since this isn't a bad source, moreso a failure of
    # this script since it should know about all petsc classes
    raise RuntimeError("Unkown or invalid class "+str(obj))
  if objClassid.name != expectedClassid:
    fix = SourceFix(objClassid,expectedClassid)
    badSource.addErrorFromCursor(obj,"Classid doesn't match. Expected '{expected}' found '{found}'".format(expected=expectedClassid,found=objClassid.name),fix=fix)
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
    or
    macro(bar->baz,barIdx)
    """
    defCursor = obj.get_definition()
    if not defCursor: raise ValueError
    potentialParents = []
    if defCursor.location == obj.location:
      # definition didn't move, so probably no definition, check for this before though
      # to explicitly catch it, since it should be handleable
      raise RuntimeError
    elif defCursor.kind == clx.CursorKind.VAR_DECL:
      # found definition, so were in business
      # Parents here is an odd choice of words since on the very same line I loop
      # over children, but then again clangs AST has an odd semantic for parents/children
      for defChild in defCursor.get_children():
        if defChild.kind in convertCursors:
          potentialParentsTemp = [child for child in defChild.walk_preorder() if child.kind == clx.CursorKind.DECL_REF_EXPR]
          # Weed out any self-references
          potentialParentsTemp = [parent for parent in potentialParentsTemp if parent.spelling != defCursor.spelling]
          potentialParents.extend(potentialParentsTemp)
    elif defCursor.kind == clx.CursorKind.FIELD_DECL:
      # we have deduced that the original cursor may refer to a struct member
      # reference, so we go back and see if indeed this is the case
      for memberChild in obj.get_children():
        if memberChild.kind == clx.CursorKind.MEMBER_REF_EXPR:
          potentialParentsTemp = [c for c in memberChild.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR]
          potentialParentsTemp = [parent for parent in potentialParentsTemp if parent.spelling != memberChild.spelling]
          potentialParents.extend(potentialParentsTemp)
    if potentialParents:
      if len(potentialParents) > 1:
        # If >1 cursor, probably a bug since we should have weeded something out
        raise RuntimeError("Cannot determine a unique definition cursor for object")
      name = ArgCursor.getNameFromCursor(potentialParents[0])
      return parentArgNames.index(name)
    raise ValueError

  if idx.canonical.kind not in mathCursors:
    badSource.addWarningFromCursor(idx,"Index value is of unexpected type '{kind}'".format(kind=idx.canonical.kind))
    return
  try:
    idxNum = int(idx.name)
  except ValueError:
    badSource.addWarningFromCursor(idx,"Potential argument mismatch, could not determine integer value")
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
      parentFunc = ArgCursor(parentArgs[0].semantic_parent)
      badSource.addWarningFromCursor(obj,"Cannot determine index correctness, parent function '{parfn}()' seemingly does not contain the object:\n\n{proofParent}".format(parfn=parentFunc.name,proofParent=parentFunc.viewSource(ret=True)))
      return
  if idxNum != parentArgs[matchLoc].argidx:
    errMess = "Argument number doesn't match for '{badObj}'. Found '{found}' expected '{expected}' from\n\n{proofParent}".format(badObj=obj.name,expected=str(parentArgs[matchLoc].argidx),found=str(idxNum),proofParent=parentArgs[matchLoc].viewSource(ret=True))
    fix = SourceFix(idx,parentArgs[matchLoc].argidx)
    badSource.addErrorFromCursor(idx,errMess,fix=fix)
  return

def checkMatchingSpecificType(badSource,obj,filterFunctor):
  """
  Checks that obj is of a particular kind, for example char. Can optionally handle pointers too.
  """
  objType = obj.canonical.type.get_canonical()
  if filterFunctor.pointer:
    if objType.kind in filterFunctor.expectedTypeKinds:
      filterFunctor.unexpectedNotPointerHook(badSource,obj,objType)
      return
    if objType.kind == clx.TypeKind.INCOMPLETEARRAY:
      objType = objType.element_type
      # get rid of any nested array types
      while objType.kind in arrayTypes:
        objType = objType.element_type
    if objType.kind == clx.TypeKind.POINTER:
      objType = objType.get_pointee()
      # get rid of any nested pointer types
      while objType.kind == clx.TypeKind.POINTER:
        objType = objType.get_pointee()
  else:
    if objType.kind in arrayTypes or objType.kind == clx.TypeKind.POINTER:
      filterFunctor.unexpectedPointerHook(badSource,obj,objType)
      return
  if objType.kind in filterFunctor.expectedTypeKinds:
    filterFunctor.successHook(badSource,obj,objType)
  else:
    filterFunctor.failureHook(badSource,obj,objType)
  return


"""Specific 'driver' function to test a particular macro archetype"""
def checkPetscValidHeaderSpecificType(badSource,func,parent):
  """
  Specific check for PetscValidHeaderSpecificType(obj,classid,idx,type)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    # add warning since it isn't a source error but rather a parsing failure
    badSource.addWarning(str(acw))
    return
  # Don't need the type
  obj,classid,idx,_ = funcArgs
  checkMatchingClassid(badSource,obj,classid)
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidHeaderSpecific(badSource,func,parent):
  """
  Specific check for PetscValidHeaderSpecific(obj,classid,idx)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    # add warning since it isn't a source error but rather a parsing failure
    badSource.addWarning(str(acw))
    return
  obj,classid,idx = funcArgs
  checkMatchingClassid(badSource,obj,classid)
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidHeader(badSource,func,parent):
  """
  Specific check for PetscValidHeader(obj,idx)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    badSource.addWarning(str(acw))
    return
  obj,idx = funcArgs
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidPointer(badSource,func,parent,filterFunctor=None):
  """
  Specific check for PetscValidPointer(obj,idx)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    badSource.addWarning(str(acw))
    return
  obj,idx = funcArgs
  if filterFunctor:
    assert filterFunctor.pointer == True
    checkMatchingSpecificType(badSource,obj,filterFunctor)
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidCharPointer(badSource,func,parent):
  """
  Specific check for PetscValidCharPointer(obj,idx)
  """
  charFilter = FilterFunctor(charTypes,func,pointer=True)
  checkPetscValidPointer(badSource,func,parent,filterFunctor=charFilter)
  return

def checkPetscValidIntPointer(badSource,func,parent):
  """
  Specific check for PetscValidIntPointer(obj,idx)
  """
  intFilter = FilterFunctor(intTypes,func,pointer=True,successHook=checkIntIsNotPetscBool,validFunc="PetscValidBoolPointer")
  checkPetscValidPointer(badSource,func,parent,filterFunctor=intFilter)
  return

def checkPetscValidBoolPointer(badSource,func,parent):
  """
  Specific check for PetscValidBoolPointer(obj,idx)
  """
  boolFilter = FilterFunctor(boolTypes,func,pointer=True,successHook=checkIsPetscBool)
  checkPetscValidPointer(badSource,func,parent,filterFunctor=boolFilter)
  return

def checkPetscValidScalarPointer(badSource,func,parent):
  """
  Specific check for PetscValidScalarPointer(obj,idx)
  """
  scalarFilter = FilterFunctor(scalarTypes,func,pointer=True,successHook=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidRealPointer")
  checkPetscValidPointer(badSource,func,parent,filterFunctor=scalarFilter)
  return

def checkPetscValidRealPointer(badSource,func,parent):
  """
  Specific check for PetscValidRealPointer(obj,idx)
  """
  realFilter = FilterFunctor(realTypes,func,pointer=True,successHook=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidScalarPointer")
  checkPetscValidPointer(badSource,func,parent,filterFunctor=realFilter)
  return

def checkPetscCheckSameType(badSource,func,parent):
  """
  Specific check for PetscCheckSameType(objA,idxA,objB,idxB)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    badSource.addWarning(str(acw))
    return
  objA,idxA,objB,idxB = funcArgs
  checkMatchingArgNum(badSource,objA,idxA,parentArgs)
  checkMatchingArgNum(badSource,objB,idxB,parentArgs)
  return

def checkPetscValidType(badSource,func,parent):
  """
  Specific check for PetscValidType(obj,idx)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    badSource.addWarning(str(acw))
    return
  obj,idx = funcArgs
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscCheckSameComm(badSource,func,parent):
  """
  Specific check for PetscCheckSameComm(objA,idxA,objB,idxB)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    badSource.addWarning(str(acw))
    return
  objA,idxA,objB,idxB = funcArgs
  checkMatchingArgNum(badSource,objA,idxA,parentArgs)
  checkMatchingArgNum(badSource,objB,idxB,parentArgs)
  return

def checkPetscValidLogicalCollective(badSource,func,parent,filterFunctor):
  """
  Generic check for PetscValidLogicalCollectiveXXX(pobj,obj,idx)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    badSource.addWarning(str(acw))
    return
  # dont need the petsc object, nothing to check there
  _,obj,idx = funcArgs
  assert filterFunctor.pointer == False
  checkMatchingSpecificType(badSource,obj,filterFunctor)
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidLogicalCollectiveScalar(badSource,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveScalar(pobj,obj,idx)
  """
  scalarFilter = FilterFunctor(scalarTypes,func,successHook=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidLogicalCollectiveReal")
  checkPetscValidLogicalCollective(badSource,func,parent,scalarFilter)
  return

def checkPetscValidLogicalCollectiveReal(badSource,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveReal(pobj,obj,idx)
  """
  realFilter = FilterFunctor(realTypes,func,successHook=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidLogicalCollectiveScalar")
  checkPetscValidLogicalCollective(badSource,func,parent,realFilter)
  return

def checkPetscValidLogicalCollectiveInt(badSource,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveInt(pobj,obj,idx)
  """
  intFilter = FilterFunctor(intTypes,func,successHook=checkIntIsNotPetscBool,validFunc="PetscValidLogicalCollectiveBool")
  checkPetscValidLogicalCollective(badSource,func,parent,intFilter)
  return

def checkPetscValidLogicalCollectiveMPIInt(badSource,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveMPIInt(pobj,obj,idx)
  """
  mpiIntFilter = FilterFunctor(mpiIntTypes,func,successHook=checkMPIIntIsNotPetscInt,validFunc="PetscValidLogicalCollectiveInt")
  checkPetscValidLogicalCollective(badSource,func,parent,mpiIntFilter)
  return

def checkPetscValidLogicalCollectiveBool(badSource,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveBool(pobj,obj,idx)
  """
  boolFilter = FilterFunctor(boolTypes,func,successHook=checkIsPetscBool)
  checkPetscValidLogicalCollective(badSource,func,parent,boolFilter)
  return

def checkPetscValidLogicalCollectiveEnum(badSource,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveEnum(pobj,obj,idx)
  """
  enumFilter = FilterFunctor(enumTypes,func)
  checkPetscValidLogicalCollective(badSource,func,parent,enumFilter)
  return


mansecs          = ["sys","vec","mat","dm","ksp","snes","ts","tao"]
checkFunctionMap = {
  "PetscValidHeaderSpecificType"      : checkPetscValidHeaderSpecificType,
  "PetscValidHeaderSpecific"          : checkPetscValidHeaderSpecific,
  "PetscValidHeader"                  : checkPetscValidHeader,
  "PetscValidPointer"                 : checkPetscValidPointer,
  "PetscValidCharPointer"             : checkPetscValidCharPointer,
  "PetscValidIntPointer"              : checkPetscValidIntPointer,
  "PetscValidBoolPointer"             : checkPetscValidBoolPointer,
  "PetscValidScalarPointer"           : checkPetscValidScalarPointer,
  "PetscValidRealPointer"             : checkPetscValidRealPointer,
  "PetscCheckSameType"                : checkPetscCheckSameType,
  "PetscValidType"                    : checkPetscValidType,
  "PetscCheckSameComm"                : checkPetscCheckSameComm,
  "PetscValidLogicalCollectiveScalar" : checkPetscValidLogicalCollectiveScalar,
  "PetscValidLogicalCollectiveReal"   : checkPetscValidLogicalCollectiveReal,
  "PetscValidLogicalCollectiveInt"    : checkPetscValidLogicalCollectiveInt,
  "PetscValidLogicalCollectiveMPIInt" : checkPetscValidLogicalCollectiveMPIInt,
  "PetscValidLogicalCollectiveBool"   : checkPetscValidLogicalCollectiveBool,
  "PetscValidLogicalCollectiveEnum"   : checkPetscValidLogicalCollectiveEnum,
}

"""Utility and pre-check setup"""
def updateMansecs(mansecFilter):
  """
  Remove directories from mansecs if they are not in mansecFilter
  """
  if mansecFilter:
    global mansecs

    mansecFilter = set(mansecFilter)
    mansecs      = [m for m in mansecs if m in mansecFilter]
  return

def updateCheckFunctionMap(filterChecks):
  """
  Remove checks from checkFunctionMap if they are not in filterChecks
  """
  if filterChecks:
    global checkFunctionMap

    # note the list, this makes a copy of the keys allowing us to delete entries "in place"
    for key in list(checkFunctionMap.keys()):
      if key not in filterChecks:
        del checkFunctionMap[key]
  return

def updatePetscClassIdMap(petscDir):
  """
  add additional petsc classes by traipsing through the headers and looking for typedefs
  """
  import re
  global petscClassIdMap

  includeBaseDir = os.path.join(petscDir,"include")
  regclass       = re.compile("(\s*typedef\s+struct\s+)(_[pn]_[A-Za-z_]*\s+\*)")
  for root,_,filenames in os.walk(includeBaseDir):
    for fname in filenames:
      if fname.endswith(".h"):
        with open(os.path.join(root,fname),"r") as rfile:
          line = rfile.readline()
          while line:
            fl = regclass.search(line)
            if fl:
              struct = fl.group(2)
              if struct not in petscClassIdMap:
                petscClassIdMap[struct] = "ERROR_UNKNOWN_PETSC_CLASSID"
            line = rfile.readline()
  return

def updatePetscScalarType(petscDir,petscArch):
  import re
  global scalarTypes

  confFile   = os.path.join(petscDir,petscArch,"include","petscconf.h")
  regcomplex = re.compile("^#define\s*PETSC_USE_COMPLEX\s")
  with open(confFile,"r") as rfile:
    line = rfile.readline()
    while line:
      if regcomplex.search(line): return
      line = rfile.readline()
  # petsc is configured to NOT use complex numbers, so we remove them from scalartypes
  scalarTypes.remove(clx.TypeKind.COMPLEX)
  return

def getPetscExtraIncludes(petscDir,petscArch):
  import re

  # keep these separate, since ORDER MATTERS HERE. Imagine that for example the
  # mpiInclude dir has copies of old petsc headers, you don't want these to come first
  # in the include search path and hence override those found in petsc/include.

  # You might be thinking that seems suspiciously specific, but I was this close to filing
  # a bug report for python believing that cdll.load() was not deterministic...
  petscIncludes = []
  mpiIncludes   = []
  with open(os.path.join(petscDir,petscArch,"lib","petsc","conf","petscvariables"),"r") as pv:
    ccinc  = re.compile("^PETSC_CC_INCLUDES\s*=")
    mpiinc = re.compile("^MPI_INCLUDE\s*=")
    shoinc = re.compile("^MPICC_SHOW\s*=")
    line   = pv.readline()
    while line:
      if ccinc.search(line):
        petscIncludes.append(line.split("=")[1])
      elif mpiinc.search(line) or shoinc.search(line):
        mpiIncludes.append(line.split("=")[1])
      line = pv.readline()
  extraIncludes = [l.strip().split(" ") for l in petscIncludes+mpiIncludes if l]
  extraIncludes = [item for sublist in extraIncludes for item in sublist if item.startswith("-I")]
  seen          = set()
  extraIncludes = [item for item in extraIncludes if not item in seen and not seen.add(item)]
  return extraIncludes

def getClangSysIncludes():
  import subprocess,sys
  """
  Get system clangs set of default include search directories.

  Because for some reason these are hardcoded by the compilers and so libclang does not have them.
  """
  if sys.version_info >= (3,7):
    output = subprocess.run(["clang","-E","-x","c++","/dev/null","-v"],capture_output=True,check=True,universal_newlines=True)
  else:
    output = subprocess.run(["clang","-E","-x","c++","/dev/null","-v"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,universal_newlines=True)
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
  cursor,filename = tu.cursor,tu.cursor.spelling
  for possibleParent in cursor.get_children():
    # getting filename is for some reason stupidly expensive, so we do this check first
    if possibleParent.kind not in funcCallCursors: continue
    try:
      if possibleParent.location.file.name != filename: continue
    except AttributeError:
      # possibleParent.location.file is None
      continue
    # if we've gotten this far we have found a function definition
    for funcChild in possibleParent.walk_preorder():
      if funcChild.kind == clx.CursorKind.CALL_EXPR:
        if funcChild.spelling in macroNames:
          yield (funcChild,possibleParent)


"""Main functions for root and queue processes"""
def queueMain(clangLib,checkFunctionFilter,petscDir,petscArch,args,options,verbose,printWarnings,exceptions,dataQueue,queue,lock):
  import multiprocessing as mp

  proc = mp.current_process().name
  postfix = " --"
  printPrefix = proc+postfix[:len("[ROOT]")-len(proc)]
  if verbose:
    with lock:
      print(printPrefix,15*"=","Performing setup",15*"=")
  if not clx.conf.loaded:
    clx.conf.set_library_file(clangLib)
  index       = clx.Index.create()
  errorPrefix = printPrefix+" Exception detected while processing"
  updateCheckFunctionMap(checkFunctionFilter)
  updatePetscClassIdMap(petscDir)
  updatePetscScalarType(petscDir,petscArch)
  if verbose:
    with lock:
      print(printPrefix,15*"=","Entering queue",15*"=")
  while True:
    filename = queue.get()
    if filename == QueueSignal.EXIT_QUEUE:
      queue.task_done()
      break
    try:
      if verbose:
        with lock:
          print(printPrefix,"Processing file     ",filename)
      tu = index.parse(filename,args=args,options=options)
      if tu.diagnostics and (verbose or printWarnings):
        diags = {" ".join([printPrefix,filename+":",d.spelling]) for d in tu.diagnostics}
        diags = "\n".join(diags)
        with lock:
          print(diags)
      with BadSource(printPrefix,printWarningMessages=printWarnings,lock=lock) as badSource:
        for func,parent in findFunctionCallExpr(tu,checkFunctionMap.keys()):
          checkFunctionMap[func.spelling](badSource,func,parent)
        dataQueue.put((QueueSignal.UNIFIED_DIFF,badSource.coalesceDiffs()))
        dataQueue.put((QueueSignal.ERRORS_LEFT,badSource.getErrorsLeft()))
    except:
      import traceback
      preamble = " ".join([errorPrefix,filename])
      exceptions.put("\n".join([preamble,traceback.format_exc()]))
    queue.task_done()
  exceptions.close()
  dataQueue.close()
  if verbose:
    with lock:
      print(printPrefix,15*"=","Exiting queue",15*"=")
  return

def main(petscDir,petscArch,clangDir=None,clangLib=None,verbose=False,multiproc=True,mansecFilter=None,checkFunctionFilter=None,printWarnings=False,maxWorkers=0,applyPatches=False):
  if not clx.conf.loaded:
    clx.conf.set_compatibility_check(True)
    if clangDir:
      clangDir = os.path.abspath(os.path.expanduser(os.path.expandvars(clangDir)))
      clx.conf.set_library_path(clangDir)
    elif clangLib:
      clangLib = os.path.abspath(os.path.expanduser(os.path.expandvars(clangLib)))
      clx.conf.set_library_file(clangLib)
    else:
      raise clx.LibclangError("Must supply either clangDir or clangLib")

  rootPrintPrefix = "[ROOT]"
  pchClangOptions = (P_CXTranslationUnit_CreatePreambleOnFirstParse |
                     P_CXTranslationUnit_Incomplete |
                     P_CXTranslationUnit_ForSerialization)
  baseClangOptions = (P_CXTranslationUnit_PrecompiledPreamble |
                      P_CXTranslationUnit_SkipFunctionBodies |
                      P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble)
  forceCxxFlag    = ["-x","c++"]
  miscFlags       = ["-Wno-nullability-completeness"]
  sysincludes     = getClangSysIncludes()
  extraIncludes   = getPetscExtraIncludes(petscDir,petscArch)
  compilerFlags   = sysincludes+forceCxxFlag+extraIncludes+miscFlags
  if verbose: print("\n".join([rootPrintPrefix+" Compile flags:",*compilerFlags]))

  # create a precompiled header from petsc.h, and all of the major "impl" headers,
  # this saves a lot of time since this includes almost every sub-header in petsc.
  # Including petsc.h first should define almost everything we need so no side effects
  # from including headers in the wrong order below
  updateMansecs(mansecFilter)
  mansecimpls     = [m+"impl.h" for m in mansecs]+["isimpl.h","dtimpl.h","dmpleximpl.h","petscfeimpl.h","dmlabelimpl.h","sfimpl.h","viewerimpl.h","characteristicimpl.h"]
  megaHeaderLines = ["#include <petscastfix.hpp>","#include <petsc.h>"]
  for headerFile in os.listdir(os.path.join(petscDir,"include","petsc","private")):
    if headerFile in mansecimpls or headerFile.startswith(("hash","pc")):
      megaHeaderLines.append("#include <petsc/private/{headerFile}>".format(headerFile=headerFile))

  megaHeader = "\n".join(megaHeaderLines)+"\n" # extra newline for last line
  petscPrecompiledHeader = os.path.join(petscDir,"include","petsc_ast_precompile.h.pch")
  if verbose: print("\n".join([rootPrintPrefix+" Mega header:",megaHeader]))
  if verbose: print(rootPrintPrefix,"Creating precompiled header",petscPrecompiledHeader)
  index = clx.Index.create()
  tu = index.parse("megaHeader.hpp",args=compilerFlags,unsaved_files=[("megaHeader.hpp",megaHeader)],options=pchClangOptions)
  if tu.diagnostics:
    print("\n".join(map(str,tu.diagnostics)))
    raise clx.LibclangError("Warnings generated when creating the precompiled header. This usually means that the libclang set is faulty!")
  tu.save(petscPrecompiledHeader)
  compilerFlags.extend(["-include-pch",petscPrecompiledHeader])

  if multiproc:
    import multiprocessing as mp
    # -1 since num workers+root = numCpu
    if not maxWorkers: maxWorkers = max(mp.cpu_count()-1,1)
    if maxWorkers == 1:
      multiproc = False
      print(rootPrintPrefix,"Number of processes ({numproc}) too small. Not using multiprocessing".format(numproc=maxWorkers))
  # need a second "if multiproc" since the above might turn multiproc off. it might have
  # been a great place to use "goto" to jump to the else but since thats apparently far
  # too complex a construct according to python devs we do this stupid song and dance
  if multiproc:
    # get the library file to pass to subprocesses
    clangLib = clx.conf.get_filename()
    # spin up a queue and multiproc
    fileProcessorQueue = mp.JoinableQueue(3*maxWorkers)
    exceptionSignalQueue = mp.Queue()
    dataQueue = mp.Queue()
    fileProcessorLock = mp.Lock()
    workerArgs = (clangLib,checkFunctionFilter,petscDir,petscArch,compilerFlags,baseClangOptions,verbose,printWarnings,exceptionSignalQueue,dataQueue,fileProcessorQueue,fileProcessorLock,)
    for i in range(maxWorkers):
      workerName = "[{i}]".format(i=i)
      worker = mp.Process(target=queueMain,args=workerArgs,name=workerName,daemon=True)
      worker.start()
    # need these later for error printing
    errBars = "[ERROR]"+(85*"-")+"[ERROR]\n"
    errBars = [errBars,errBars]
  else:
    # apply the filters if we aren't using workers
    updateCheckFunctionMap(checkFunctionFilter)
    updatePetscClassIdMap(petscDir)
    updatePetscScalarType(petscDir,petscArch)

  # exclude these directories
  excludeDirs = set(["f90-mod","f90-src","f90-custom","output","input","python","fsrc","ftn-auto","ftn-custom","f2003-src","ftn-kernels","tests","tutorials"])
  excludeDirSuffixes = (".dSYM",)
  # allow these file suffixes
  allowFileSuffixes = (".c",".cpp",".cxx",".cu",)
  errorsLeft,diffs = [],[]
  for mansec in mansecs:
    for root,dirs,files in os.walk(os.path.join(petscDir,"src",mansec)):
      if verbose: print(rootPrintPrefix,"Processing directory",root)
      dirs[:] = [d for d in dirs if d not in excludeDirs]
      dirs[:] = [d for d in dirs if not d.endswith(excludeDirSuffixes)]
      files[:] = [os.path.join(root,f) for f in files if f.endswith(allowFileSuffixes)]
      if multiproc:
        for filename in files:
          fileProcessorQueue.put(filename)
      else:
        for filename in files:
          if verbose: print(rootPrintPrefix,"Processing file     ",filename)
          tu = index.parse(filename,args=compilerFlags,options=baseClangOptions)
          if tu.diagnostics and (verbose or printWarnings):
            diags = {" ".join([rootPrintPrefix,str(d)]) for d in tu.diagnostics}
            diags = "\n".join(diags)
            print(diags)
          with BadSource(rootPrintPrefix,printWarningMessages=printWarnings) as badSource:
            for func,parent in findFunctionCallExpr(tu,checkFunctionMap.keys()):
              checkFunctionMap[func.spelling](badSource,func,parent)
            diffs.extend(badSource.coalesceDiffs())
            errorsLeft.append(badSource.getErrorsLeft())
    if multiproc:
      stopMultiproc = False
      # join here to colocate error messages to a mansec
      fileProcessorQueue.join()
      while not exceptionSignalQueue.empty():
        exception = exceptionSignalQueue.get()
        errMess = str(exception).join(errBars)
        print(errMess)
        stopMultiproc = True
      if stopMultiproc: raise RuntimeError("Error in child process detected")
  if multiproc:
    # send stop-signal to child processes
    for _ in range(maxWorkers):
      fileProcessorQueue.put(QueueSignal.EXIT_QUEUE)
    fileProcessorQueue.close()
    # wait for queue to close
    fileProcessorQueue.join()
    exceptionSignalQueue.close()
    while not dataQueue.empty():
      signal,returnData = dataQueue.get()
      if signal == QueueSignal.ERRORS_LEFT:
        errorsLeft.append(returnData)
      elif signal == QueueSignal.UNIFIED_DIFF:
        diffs.extend(returnData)
    dataQueue.close()
  if os.path.exists(petscPrecompiledHeader):
    if verbose: print(rootPrintPrefix,"Deleting precompiled header",petscPrecompiledHeader)
    os.remove(petscPrecompiledHeader)
  errorsLeft = [e for e in errorsLeft if e] # remove any None's
  diffs      = [d for d in diffs if d]
  if diffs:
    import time

    srcDir,patchDir = os.path.join(petscDir,"src"),os.path.join(petscDir,"petscLintPatches")
    try:
      os.mkdir(patchDir)
    except FileExistsError:
      pass
    manglePostfix = "_"+str(int(time.time()))+".patch"
    for filename,diff in diffs:
      filename    = filename.replace(srcDir,"").replace(os.path.sep,"_")[1:]
      mangledFile = filename.split(".")[0]+manglePostfix
      mangledFile = os.path.join(patchDir,mangledFile)
      with open(mangledFile,"w") as fd:
        if verbose: print(rootPrintPrefix,"Writing patch to file",mangledFile)
        fd.write(diff)
    if applyPatches:
      import subprocess,glob,sys

      if verbose: print(rootPrintPrefix,"Applying patches from patch directory",patchDir)
      rootDir = "-d"+os.path.abspath(os.path.sep)
      patchGlob = patchDir+os.path.sep+"*"+manglePostfix
      for patch in glob.iglob(patchGlob):
        if verbose: print(rootPrintPrefix,"Applying patch",patch)
        if sys.version_info >= (3,7):
          output = subprocess.run(["patch",rootDir,"-p0","--unified","--dry-run","-i",patch],check=True,universal_newlines=True,capture_output=True)
        else:
          output = subprocess.run(["patch",rootDir,"-p0","--unified","--dry-run","-i",patch],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,universal_newlines=True)
        print(output.stdout)
  if errorsLeft:
    print(rootPrintPrefix,27*"=","UNCORRECTABLE ERRORS BEGIN",30*"=")
    print("\n".join(errorsLeft))
    print(rootPrintPrefix,27*"=","UNCORRECTABLE ERRORS END",32*"=")
    print(rootPrintPrefix,"Some errors could not be automatically corrected via the diff files, see above")
  elif diffs:
    print(rootPrintPrefix,27*"=","NO UNCORRECTABLE ERRORS REMAIN",26*"=")
    if applyPatches:
      print(rootPrintPrefix,"All errors fixed via patch files written to",patchDir)
    else:
      print(rootPrintPrefix,"All errors fixable via patch files written to",patchDir)
  return

if __name__ == "__main__":
  import argparse

  try:
    petscDir = os.environ["PETSC_DIR"]
  except KeyError:
    petscDir = None
  try:
    petscArch = os.environ["PETSC_ARCH"]
  except KeyError:
    petscArch = None

  parser = argparse.ArgumentParser(description="set options for clang static analysis tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  grouplibclang = parser.add_argument_group(title="libclang location settings")
  group = grouplibclang.add_mutually_exclusive_group(required=True)
  group.add_argument("--clang_dir",help="directory containing libclang.[so|dylib|dll]",dest="clangdir")
  group.add_argument("--clang_lib",help="direct location of libclang.[so|dylib|dll]",dest="clanglib")
  grouppetsc = parser.add_argument_group(title="petsc location settings")
  grouppetsc.add_argument("--PETSC_DIR",required=False,default=petscDir,help="if this option is unused defaults to environment variable $PETSC_DIR",dest="petscdir")
  grouppetsc.add_argument("--PETSC_ARCH",required=False,default=petscArch,help="if this option is unused defaults to environment variable $PETSC_ARCH",dest="petscarch")
  parser.add_argument("--verbose",required=False,action="store_true",help="verbose progress printed to screen")
  parser.add_argument("--show-warnings",required=False,action="store_true",help="show ast matching warnings",dest="warn")
  parser.add_argument("--filter-functions",required=False,nargs="+",choices=list(checkFunctionMap.keys()),help="filter for errors from available function names",dest="filterfunc")
  parser.add_argument("--filter-mansec",required=False,nargs="+",choices=mansecs,help="run only over specified mansecs, defaults to all",dest="filtermansec")
  parser.add_argument("--no-multiprocessing",required=False,action="store_false",help="use multiprocessing",dest="multiproc")
  parser.add_argument("--jobs",required=False,type=int,default=0,nargs="?",help="number of multiprocessing jobs, 0 defaults to number of processors on machine")
  parser.add_argument("--apply-patches",required=False,action="store_true",help="apply patches automatically instead of saving to file",dest="apply")
  args = parser.parse_args()

  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.verbose:
    args.warn = True
  main(args.petscdir,args.petscarch,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,multiproc=args.multiproc,mansecFilter=args.filtermansec,checkFunctionFilter=args.filterfunc,printWarnings=args.warn,maxWorkers=args.jobs,applyPatches=args.apply)
