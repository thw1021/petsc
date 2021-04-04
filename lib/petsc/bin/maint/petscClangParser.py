#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os
import clang.cindex as clx
import petscClangParserUtil

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

# Cursors that may be attached to function-like usage
funcCallCursors = set([clx.CursorKind.FUNCTION_DECL,clx.CursorKind.CALL_EXPR])

# Cursors that may be attached to mathemateical operations or types
mathCursors = set([clx.CursorKind.INTEGER_LITERAL,clx.CursorKind.UNARY_OPERATOR])

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
charTypes   = set([clx.TypeKind.CHAR_S,clx.TypeKind.UCHAR])
intTypes    = enumTypes|set([clx.TypeKind.USHORT,clx.TypeKind.SHORT,clx.TypeKind.INT,clx.TypeKind.UINT,clx.TypeKind.LONGLONG,clx.TypeKind.ULONGLONG])
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

class PetscCursor(clx.Cursor):
  """
  This class exists purely for purpose of making viewing a cursor easier
  """
  @classmethod
  def cast(cls, cursor: clx.Cursor):
    """
    Cast an clang cursor into a petsc cursor
    """
    assert isinstance(cursor,clx.Cursor)
    cursor.__class__ = cls  # can now use our __repr__
    assert isinstance(cursor,PetscCursor)
    return cursor

  def __repr__(self):
    return "\n".join(petscClangParserUtil.viewAstFromCursor(self))

  def viewSource(self,nbefore=10,nafter=10,nboth=10,ret=False):
    return petscClangParserUtil.viewSourceFromCursor(self,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,ret=ret)

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
    Try to convert **&(PetscObject)obj+73 to obj
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
    elif cursor.type.get_canonical().kind == clx.TypeKind.POINTER:
      if cursor.type.get_pointee().kind  == clx.TypeKind.CHAR_S:
        # For some reason preprocessor macros that contain strings don't propogate
        # their spelling up to the primary cursor, so we need to plumb through
        # the various sub-cursors to find it.
        pointees = [c for c in cursor.walk_preorder() if c.kind in literalCursors]
        if len(pointees) == 1:
          name = ArgCursor.getNameFromCursor(pointees[0])
          srcstr = petscClangParserUtil.viewSourceFromCursor(cursor,ret=True)
          if "PetscValidHeaderSpecificType" not in srcstr:
            raise RuntimeError
    # Catchall last attempt, literally parse the tokens
    if not name:
      tokenlist = [t for t in cursor.get_tokens() if t.kind in strTokens]
      # Remove iterator variables
      tokenlist = [t for t in tokenlist if t.cursor.type.get_canonical().kind != clx.TypeKind.INT]
      if len(tokenlist) != 1:
        srcstr = petscClangParserUtil.viewSourceFromCursor(cursor,ret=True)
        # For whatever reason (perhaps because its macro stringization hell) PETSC_HASH_MAP
        # and PetscKernel_XXX absolutely __brick__ the AST. The resultant cursors have no
        # children, no name, no tokens, and a completely incorrect SourceLocation.
        # They are for all intents and purposes uncheckable :)
        if "PETSC_HASH_MAP" in srcstr:
          raise ArgCursorWarning("Encountered unparsable PETSC_HASH_MAP for cursor {cursor}".format(cursor=errorViewFromCursor(cursor)))
        elif "PetscKernel_" in srcstr:
          raise ArgCursorWarning("Encountered unparsable PetscKernel_XXX for cursor {cursor}".format(cursor=errorViewFromCursor(cursor)))
        else:
          raise RuntimeError("Unexpected number of tokens: "+str(tokenlist))
      name = tokenlist[0].spelling
      if not name:
        raise ArgCursorWarning("Cannot determine name of symbol from cursor {obj}".format(obj=errorViewFromCursor(cursor)))
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

  def __init__(self,cursor,idx=-12345):
    cursor = PetscCursor.cast(cursor)
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
  def __init__(self,prefix,printWarningMessages=True,lock=None):
    self.errors = []
    self.warnings = []
    self.prefix = prefix
    self.printWarningMessages = printWarningMessages
    self.lock = lock
    return

  def __repr__(self):
    prefixStr = "Prefix:       '{prefix}'".format(prefix=self.prefix)
    lockStr   = "Lock:          {lock}".format(lock="True" if self.lock else "False")
    showStr   = "Show warnings: {warn}".format(warn=self.printWarningMessages)
    printList = [prefixStr,lockStr,showStr]
    errorStr  = self.getErrors(self.prefix)
    if errorStr:
      printList.append(errorStr)
    warnStr   = self.getWarnings(self.prefix)
    if warnStr:
      printList.append(warnStr)
    return "\n".join(printList)

  def __enter__(self):
    return self

  def __exit__(self,*args):
    if self.printWarningMessages:
      self.__print(self.getWarnings(self.prefix))
    self.__print(self.getErrors(self.prefix))
    return

  def __print(self,msg):
    if msg:
      if self.lock:
        with self.lock:
          print(msg)
      else:
        print(msg)
    return

  def addError(self,errMsg):
    self.errors.append(" ".join(["ERROR:",errMsg]))
    return

  def getErrors(self,prefix):
    if self.errors:
      return prefix+" "+"\n".join(self.errors)
    return

  def addWarning(self,warnMsg):
    self.warnings.append(" ".join(["WARNING:",warnMsg]))
    return

  def getWarnings(self,prefix):
    if self.warnings:
      warnPrefix = " ".join([prefix,85*"%","\n"])
      return warnPrefix+"\n".join(self.warnings)+"\n"+warnPrefix
    return


"""Generic test functions"""
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
      badSource.addError("{obj} doesn't match expected type '{typen}' for '{classid}'".format(obj=str(obj),typen=classFromClassId,classid=objClassid.name))
      return
    # Raise exception here since this isn't a bad source, moreso a failure of
    # this script since it should know about all petsc classes
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
    or
    macro(bar->baz,barIdx)
    """
    potentialParents = []
    defCursor = obj.cursor.get_definition()
    if not defCursor: raise ValueError
    if defCursor.location == obj.cursor.location:
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
      for memberChild in obj.cursor.get_children():
        if memberChild.kind == clx.CursorKind.MEMBER_REF_EXPR:
          potentialParentsTemp = [c for c in memberChild.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR]
          potentialParentsTemp = [parent for parent in potentialParentsTemp if parent.spelling != memberChild.spelling]
          potentialParents.extend(potentialParentsTemp)
    if potentialParents:
      if len(potentialParents) > 1:
        # If >1 cursor, probably a bug since we should have weeded something out
        raise RuntimeError("Cannot determine a unique definition cursor for object {obj}".format(obj=obj))
      elif len(potentialParents) < 1:
        # if <1 cursor then probably some funky stuff is happening, we log it
        raise RuntimeError("Could not map definition of object {obj} to parent function arguments".format(obj=obj))
      name = ArgCursor.getNameFromCursor(potentialParents[0])
      return parentArgNames.index(name)
    raise ValueError

  if idx.cursor.canonical.kind not in mathCursors:
    badSource.addWarning(" ".join(["Index value is of unexpected type","'"+str(idx.cursor.canonical.kind)+"'","not"," or ".join(["'{s}'".format(s=s) for s in mathCursors]),"for",str(idx)]))
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
    except ValueError as ve:
      # If the parent arguments don't contain the symbol and we couldn't determine a
      # definition then we cannot check for correct numbering, so we cannot do
      # anything here but emit a warning
      errMessPrefix = "Parent function '{parfn}()' does not contain the object {obj}, cannot determine index correctness.".format(parfn=parentArgs[0].cursor.semantic_parent.spelling,obj=str(obj))
      errMessSuffix = "Parent arguments:\n "+"\n ".join(str(s.argidx)+": "+str(s) for s in parentArgs)
      if str(ve): errMessPrefix = "\nSpecifically: ".join([errMessPrefix,str(ve)])
      badSource.addWarning("\n".join([errMessPrefix,errMessSuffix]))
      return
  if idxNum != parentArgs[matchLoc].argidx:
    badSource.addError("Argument number doesn't match. Expected '{expected}' found '{found}' for {obj}".format(expected=str(parentArgs[matchLoc].argidx),found=str(idxNum),obj=str(obj)))
  return

def checkMatchingSpecificPointerType(badSource,obj,expectedTypeKinds,filterTypeKinds):
  """
  Checks that obj is of a particular pointer kind, for example char*
  """
  objType = obj.cursor.canonical.type.get_canonical()
  if objType.kind in expectedTypeKinds:
    raise RuntimeError
    badSource.addError("\n".join(["Object {obj} of clang type {otype} is not a pointer to expected types:".format(obj=str(obj),otype=objType.kind),"\n".join(map(str,expectedTypeKinds))]))
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
  if objType.kind in expectedTypeKinds:
    try:
      filterTypeKinds(badSource,obj)
    except TypeError:
      # 'NoneType' object is not callable
      pass
  else:
    badSource.addError("\n".join(["Object {obj} of clang type {otype} is not in expected types:".format(obj=str(obj),otype=objType.kind),"\n".join(map(str,expectedTypeKinds))]))
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
    # add warning since it isn't a source error but rather a parsing failure
    badSource.addWarning(str(acw))
    return
  obj,idx = funcArgs
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidPointer(badSource,func,parent,pointerFilter=None):
  """
  Specific check for PetscValidPointer(obj,idx)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    # add warning since it isn't a source error but rather a parsing failure
    badSource.addWarning(str(acw))
    return
  obj,idx = funcArgs
  if pointerFilter is not None:
    expected,filterFunc = pointerFilter
    checkMatchingSpecificPointerType(badSource,obj,expected,filterFunc)
  checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return

def checkPetscValidCharPointer(badSource,func,parent):
  """
  Specific check for PetscValidCharPointer(obj,idx)
  """
  checkPetscValidPointer(badSource,func,parent,pointerFilter=(charTypes,None))
  return

def checkPetscValidIntPointer(badSource,func,parent):
  """
  Specific check for PetscValidIntPointer(obj,idx)
  """
  def checkIntIsNotPetscBool(badSource,obj):
    if "PetscBool" in obj.typename:
      badSource.addError("Incorrect use of PetscValidIntPointer() for object {obj}. Use PetscValidBoolPointer() instead".format(obj=str(obj)))
    return

  checkPetscValidPointer(badSource,func,parent,pointerFilter=(intTypes,checkIntIsNotPetscBool))
  return

def checkPetscValidBoolPointer(badSource,func,parent):
  """
  Specific check for PetscValidBoolPointer(obj,idx)
  """
  def checkIsPetscBool(badSource,obj):
    if "PetscBool" not in obj.typename:
      badSource.addError("Incorrect use of PetscValidBoolPointer() for object {obj}. PetsccValidBoolPointer() should only be used for PetscBool".format(obj=str(obj)))
    return

  checkPetscValidPointer(badSource,func,parent,pointerFilter=(enumTypes,checkIsPetscBool))
  return

def checkPetscValidScalarPointer(badSource,func,parent):
  """
  Specific check for PetscValidScalarPointer(obj,idx)
  """
  def checkIsPetscScalar(badSource,obj):
    if "PetscScalar" not in obj.cursor.type.spelling:
      badSource.addError("Incorrect use of PetscValidScalarPointer() for object {obj}. PetscValidScalarPointer() should only be used for PetscScalars".format(obj=str(obj)))
    return

  checkPetscValidPointer(badSource,func,parent,pointerFilter=(scalarTypes,checkIsPetscScalar))
  return

def checkPetscValidRealPointer(badSource,func,parent):
  """
  Specific check for PetscValidRealPointer(obj,idx)
  """
  def checkIsPetscReal(badSource,obj):
    if "PetscReal" not in obj.cursor.type.spelling:
      badSource.addError("Incorrect use of PetscValidRealPointer() for object {obj}. PetscValidRealPointer() should only be used for PetscReals".format(obj=str(obj)))
    return

  checkPetscValidPointer(badSource,func,parent,pointerFilter=(realTypes,checkIsPetscReal))
  return

def checkPetscCheckSameType(badSource,func,parent):
  """
  Specific check for PetscCheckSameType(objA,idxA,objB,idxB)
  """
  try:
    funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
    parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  except ArgCursorWarning as acw:
    # add warning since it isn't a source error but rather a parsing failure
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
    # add warning since it isn't a source error but rather a parsing failure
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
    # add warning since it isn't a source error but rather a parsing failure
    badSource.addWarning(str(acw))
    return
  objA,idxA,objB,idxB = funcArgs
  checkMatchingArgNum(badSource,objA,idxA,parentArgs)
  checkMatchingArgNum(badSource,objB,idxB,parentArgs)
  return

checkFunctionMap = {
  "PetscValidHeaderSpecificType" : checkPetscValidHeaderSpecificType,
  "PetscValidHeaderSpecific"     : checkPetscValidHeaderSpecific,
  "PetscValidHeader"             : checkPetscValidHeader,
  "PetscValidPointer"            : checkPetscValidPointer,
  "PetscValidCharPointer"        : checkPetscValidCharPointer,
  "PetscValidIntPointer"         : checkPetscValidIntPointer,
  "PetscValidBoolPointer"        : checkPetscValidBoolPointer,
  "PetscValidScalarPointer"      : checkPetscValidScalarPointer,
  "PetscValidRealPointer"        : checkPetscValidRealPointer,
  "PetscCheckSameType"           : checkPetscCheckSameType,
  "PetscValidType"               : checkPetscValidType,
  "PetscCheckSameComm"           : checkPetscCheckSameComm,
}

"""Utility and pre-check setup"""
def updateCheckFunctionMap(filterChecks):
  """
  Remove checks from checkFunctionMap if they are not in filterChecks
  """
  global checkFunctionMap

  if filterChecks:
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
  regclass = re.compile("(\s*typedef\s+struct\s+)(_[pn]_[A-Za-z_]*\s+\*)")
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

  confFile = os.path.join(petscDir,petscArch,"include","petscconf.h")
  regcomplex = re.compile("^#define\s*PETSC_USE_DEBUG\s")
  with open(confFile,"r") as rfile:
    line = rfile.readline()
    while line:
      fl = regcomplex.search(line)
      if fl: return
      line = rfile.readline()
  # petsc is configured to NOT use complex numbers, so we remove them from scalartypes
  scalarTypes.remove(clx.TypeKind.COMPLEX)
  return

def getPetscExtraIncludes(petscDir,petscArch):
  import re

  with open(os.path.join(petscDir,petscArch,"lib","petsc","conf","petscvariables"),"r") as pv:
    refccinc = re.compile("^PETSC_CC_INCLUDES\s*=")
    line     = pv.readline()
    while line:
      incl = refccinc.search(line)
      if incl:
        extraIncludes = line.split("=")[1]
        break
      line = pv.readline()
  extraIncludes = extraIncludes.strip().split(" ")
  return extraIncludes

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
      if funcChild.kind == clx.CursorKind.CALL_EXPR:
        if funcChild.spelling in macroNames:
          yield (funcChild,possibleParent)


"""Main functions for root and queue processes"""
def queueMain(clangLib,checkFunctionFilter,petscDir,petscArch,args,options,verbose,printWarnings,exceptions,queue,lock):
  import multiprocessing as mp

  printPrefix = mp.current_process().name+" --"
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
    if filename == "__EXIT_QUEUE__":
      queue.task_done()
      exceptions.close()
      break
    try:
      if verbose:
        with lock:
          print(printPrefix,"Processing file",filename)
      tu = index.parse(filename,args=args,options=options)
      if (verbose or printWarnings) and tu.diagnostics:
        diags = {" ".join([printPrefix,filename+":",d.spelling]) for d in tu.diagnostics}
        diags = "\n".join(diags)
        with lock:
          print(diags)
      with BadSource(printPrefix,printWarningMessages=printWarnings,lock=lock) as badSource:
        for func,parent in findFunctionCallExpr(tu,checkFunctionMap.keys()):
          checkFunctionMap[func.spelling](badSource,func,parent)
    except Exception:
      import traceback
      preamble = " ".join([errorPrefix,filename])
      exceptions.put("\n".join([preamble,traceback.format_exc()]))
    queue.task_done()
  if verbose:
    with lock:
      print(printPrefix,15*"=","Exiting queue",15*"=")
  return

def miniTest(compilerFlags,clangOptions):
  import sys
  testCode = """
  #include <petscastfix.hpp>
  void func1(const unsigned char arrayBob[][3],PetscInt *arrayJames)
  {
   PetscValidIntPointer(arrayBob,4);
   return;
  }
  void func2(const char arrayFred[])
  {
   PetscValidCharPointer(arrayFred,23);
   return;
  }
  void func3(char *arrayAlice)
  {
   PetscValidCharPointer(arrayAlice,2);
   return;
  }
  int main(int argc, char *argv[])
  {
   int *arrayJames;
   unsigned char arrayBob[][3];
   char arrayFred[];
   char *arrayAlice;
   func1(arrayBob,arrayJames);
   func2(arrayFred);
   func3(arrayAlice);
   return 0;
  }
"""
  index = clx.Index.create()
  tu = index.parse("mytest.cpp",args=compilerFlags,unsaved_files=[("mytest.cpp",testCode)],options=clangOptions)
  with BadSource("[ROOT]") as badSource:
    for func,parent in findFunctionCallExpr(tu,checkFunctionMap.keys()):
      checkFunctionMap[func.spelling](badSource,func,parent)
  sys.exit(0)
  return

def main(petscDir,petscArch,clangDir=None,clangLib=None,verbose=False,multiproc=True,checkFunctionFilter=None,printWarnings=False):
  import multiprocessing as mp

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

  pchClangOptions = (P_CXTranslationUnit_CreatePreambleOnFirstParse |
                     P_CXTranslationUnit_ForSerialization)
  baseClangOptions = (P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble)
  rootPrintPrefix = "[ROOT]"
  extraIncludes   = getPetscExtraIncludes(petscDir,petscArch)
  sysincludes     = getClangSysIncludes()
  petscastinclude = ["-include",os.path.join(petscDir,"include","petscastfix.hpp")]
  forceCxxFlag    = ["-x","c++"]
  compilerFlags   = sysincludes+forceCxxFlag+petscastinclude+extraIncludes+["-Wno-nullability-completeness"]
  if verbose: print("\n".join([rootPrintPrefix+" Compile flags:",*compilerFlags]))

  # create a precompiled header from petsc.h, and all of the major "impl" headers,
  # this saves a lot of time since this includes almost every sub-header in petsc.
  # Including petsc.h first should define almost everything we need so no side effects
  # from including headers in the wrong order below
  mansecs         = ["sys","vec","mat","dm","ksp","snes","ts","tao"]
  mansecimpls     = [m+"impl.h" for m in mansecs]
  megaHeaderLines = ["#include <petscastfix.hpp>\n","#include <petsc.h>\n"]
  for headerFile in os.listdir(os.path.join(petscDir,"include","petsc","private")):
    if headerFile in mansecimpls:
      megaHeaderLines.append("#include <petsc/private/{headerFile}>\n".format(headerFile=headerFile))
  megaHeader = "".join(megaHeaderLines)
  if verbose: print("\n".join([rootPrintPrefix+" Mega header:",megaHeader]))
  petscPrecompiledHeader = os.path.join(petscDir,"include","petsc_ast_precompile.pch")
  if verbose: print(rootPrintPrefix,"Creating precompiled header",petscPrecompiledHeader)
  index = clx.Index.create()
  tu = index.parse("megaHeader.hpp",args=compilerFlags,unsaved_files=[("megaHeader.hpp",megaHeader)],options=pchClangOptions)
  if tu.diagnostics:
    print("\n".join([d.spelling for d in tu.diagnostics]))
    raise clx.LibclangError("Warnings generated when creating the precompiled header. This usually means that the libclang set is faulty!")
  tu.save(petscPrecompiledHeader)
  pchIncludes = ["-include-pch",petscPrecompiledHeader]

  # get the library file to pass to subprocesses
  clangLib = clx.conf.get_filename()
  compilerFlags = forceCxxFlag+sysincludes+pchIncludes+extraIncludes+["-Wno-nullability-completeness"]
  #miniTest(compilerFlags,baseClangOptions)
  if multiproc:
    # spin up a queue and multiproc
    maxWorkers = mp.cpu_count()-1
    fileProcessorQueue = mp.JoinableQueue(3*maxWorkers)
    exceptionSignalQueue = mp.Queue()
    fileProcessorLock = mp.Lock()
    workerArgs = (clangLib,checkFunctionFilter,petscDir,petscArch,compilerFlags,baseClangOptions,verbose,printWarnings,exceptionSignalQueue,fileProcessorQueue,fileProcessorLock,)
    for i in range(maxWorkers):
      workerName = "[{i}]".format(i=i)
      worker = mp.Process(target=queueMain,args=workerArgs,name=workerName,daemon=True)
      worker.start()
    # need these later for error printing
    errBars = "[ERROR]"+(85*"-")+"[ERROR]\n"
    errBars = [errBars,errBars]
  else:
    updateCheckFunctionMap(checkFunctionFilter)
    updatePetscClassIdMap(petscDir)
    updatePetscScalarType(petscDir,petscArch)

  # change dirs to $PETSC_DIR/src since we are pretending to be the makefile
  srcDir = os.path.join(petscDir,"src")
  oldloc = os.getcwd()
  os.chdir(srcDir)
  # exclude these directories
  excludeDirs = set(["f90-mod","f90-src","f90-custom","output","input","python","fsrc","ftn-auto","ftn-custom","f2003-src","ftn-kernels"])
  excludeDirSuffixes = (".dSYM",)
  # allow these file suffixes
  allowFileSuffixes = (".c",)
  for mansec in mansecs:
  #for mansec in ["dm"]:
    for root,dirs,files in os.walk(os.path.join(srcDir,mansec)):
      if verbose: print(rootPrintPrefix,"Processing directory",root)
      dirs[:] = [d for d in dirs if d not in excludeDirs]
      dirs[:] = [d for d in dirs if not d.endswith(excludeDirSuffixes)]
      files[:] = [os.path.join(root,f) for f in files if f.endswith(allowFileSuffixes)]
      if multiproc:
        for filename in files:
          fileProcessorQueue.put(filename)
      else:
        for filename in files:
          if verbose: print(rootPrintPrefix,"Processing file",filename)
          tu = index.parse(filename,args=compilerFlags,options=baseClangOptions)
          if (verbose or printWarnings) and tu.diagnostics:
            diags = {" ".join([rootPrintPrefix,filename+":",d.spelling]) for d in tu.diagnostics}
            diags = "\n".join(diags)
            print(diags)
          with BadSource(rootPrintPrefix,printWarningMessages=printWarnings) as badSource:
            for func,parent in findFunctionCallExpr(tu,checkFunctionMap.keys()):
              checkFunctionMap[func.spelling](badSource,func,parent)
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
      fileProcessorQueue.put("__EXIT_QUEUE__")
    fileProcessorQueue.close()
    # wait for queue to close
    fileProcessorQueue.join()
    exceptionSignalQueue.close()
  os.chdir(oldloc)
  if os.path.exists(petscPrecompiledHeader):
    if verbose: print(rootPrintPrefix,"Deleting precompiled header",petscPrecompiledHeader)
    os.remove(petscPrecompiledHeader)
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
  parser.add_argument("--filter",required=False,nargs="+",choices=list(checkFunctionMap.keys()),help="filter for errors from available function names")
  parser.add_argument("--no-multiprocessing",required=False,action="store_false",help="use multiprocessing",dest="multiproc")
  args = parser.parse_args()

  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.verbose:
    args.warn = True
  main(args.petscdir,args.petscarch,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,multiproc=args.multiproc,checkFunctionFilter=args.filter,printWarnings=args.warn)
