#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os
import sys
import enum
import itertools
import collections
import ctypes
import difflib
import re
import multiprocessing as mp
import multiprocessing.queues
import petscClangLinterUtil as pclu
from petscClangLinterUtil import (
  Scope,PetscSourceLocation,PetscSourceRange,PetscCXCursorAndRangeVisitor,
  CXCursorAndRangeVisitorCallBackProto,PetscPath
)
try:
  import clang.cindex as clx
except ModuleNotFoundError as mnfe:
  if mnfe.name == "clang":
    raise RuntimeError("Must run e.g. 'python -m pip install clang' to use linter") from mnfe
  raise # whatever it is they should know about it

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

# clang options used for parsing files
baseClangOptions = (
  P_CXTranslationUnit_PrecompiledPreamble |
  P_CXTranslationUnit_SkipFunctionBodies  |
  P_CXTranslationUnit_LimitSkipFunctionBodiesToPreamble
)

# clang options for creating the precompiled megaheader
basePCHClangOptions = (
  P_CXTranslationUnit_CreatePreambleOnFirstParse |
  P_CXTranslationUnit_Incomplete                 |
  P_CXTranslationUnit_ForSerialization           |
  P_CXTranslationUnit_KeepGoing
)

# Cursors that may be attached to function-like usage
funcCallCursors = {clx.CursorKind.FUNCTION_DECL,clx.CursorKind.CALL_EXPR}

# Cursors that may be attached to mathemateical operations or types
mathCursors     = {
  clx.CursorKind.INTEGER_LITERAL,
  clx.CursorKind.UNARY_OPERATOR,
  clx.CursorKind.BINARY_OPERATOR
}

# Cursors that contain base literal types
literalCursors  = {
  clx.CursorKind.INTEGER_LITERAL,
  clx.CursorKind.STRING_LITERAL
}

# Cursors that may be attached to casting
castCursors     = {
  clx.CursorKind.CSTYLE_CAST_EXPR,
  clx.CursorKind.CXX_STATIC_CAST_EXPR,
  clx.CursorKind.CXX_DYNAMIC_CAST_EXPR,
  clx.CursorKind.CXX_REINTERPRET_CAST_EXPR,
  clx.CursorKind.CXX_CONST_CAST_EXPR,
  clx.CursorKind.CXX_FUNCTIONAL_CAST_EXPR
}

# Cursors that may be attached when types are converted
convertCursors  = castCursors|{clx.CursorKind.UNEXPOSED_EXPR}

varTokens       = {clx.TokenKind.IDENTIFIER}

functionTypes   = {clx.TypeKind.FUNCTIONPROTO,clx.TypeKind.FUNCTIONNOPROTO}

# General Array types, note this doesn't contain the pointer type since that is usually handled
# differently
arrayTypes      = {
  clx.TypeKind.INCOMPLETEARRAY,
  clx.TypeKind.CONSTANTARRAY,
  clx.TypeKind.VARIABLEARRAY
}

# Specific types
enumTypes   = {clx.TypeKind.ENUM}
# because PetscBool is an enum...
boolTypes   = enumTypes|{clx.TypeKind.BOOL}
charTypes   = {clx.TypeKind.CHAR_S,clx.TypeKind.UCHAR}
mpiIntTypes = {clx.TypeKind.INT}
intTypes    = enumTypes|mpiIntTypes|{
  clx.TypeKind.USHORT,
  clx.TypeKind.SHORT,
  clx.TypeKind.UINT,
  clx.TypeKind.LONG,
  clx.TypeKind.LONGLONG,
  clx.TypeKind.ULONGLONG
}
realTypes   = {
  clx.TypeKind.FLOAT,
  clx.TypeKind.DOUBLE,
  clx.TypeKind.LONGDOUBLE,
  clx.TypeKind.FLOAT128
}
scalarTypes = realTypes|{clx.TypeKind.COMPLEX}

"""
Adding new classes
------------------

You must register new instances of PETSc classes in the classIdMap which expects its
contents to be in the form:

"CaseSensitiveNameOfPrivateStruct *" : "CaseSensitiveNameOfCorrespondingClassId",

See below for examples.

* please add your new class in alphabetical order and preserve the alignment! *

The automated way to do it (in emacs) is to slap it in the first entry then highlight
the the contents (i.e. excluding "classIdMap = {" and the closing "}") and do:

1. M-x sort-fields RET
2. M-x align-regexp RET : RET
"""
classIdMap = {
  "_p_AO *"                     : "AO_CLASSID",
  "_p_Characteristic *"         : "CHARACTERISTIC_CLASSID",
  "_p_DM *"                     : "DM_CLASSID",
  "_p_DMAdaptor *"              : "DM_CLASSID",
  "_p_DMField *"                : "DMFIELD_CLASSID",
  "_p_DMKSP *"                  : "DMKSP_CLASSID",
  "_p_DMLabel *"                : "DMLABEL_CLASSID",
  "_p_DMPlexTransform *"        : "DMPLEXTRANSFORM_CLASSID",
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

# directory names to exclude from processing, case sensitive
excludeDirNames     = {"tests","tutorials","output","input","python","fsrc","f90-mod","f90-src","f90-custom","ftn-auto","ftn-custom","f2003-src","ftn-kernels","benchmarks","docs","binding","contrib"}
# directory suffixes to exclude from processing, case sensitive
excludeDirSuffixes  = (".dSYM",".DS_Store")
# file extensions to process, case sensitve
allowFileExtensions = (".c",".cpp",".cxx",".cu",".cc",".h",".hpp")

class ParsingError(Exception):
  __doc__="""
  Mostly to just have a custom "something went wrong when trying to perform a check" to except
  for rather than using a built-in type. These are errors that are meant to be caught and logged
  rather than stopping execution alltogether.

  This should make it so that actual errors aren't hidden.
  """
  pass


class PetscDocString(object):
  __doc__="""
  Container to encapsulate a sowing docstring and retrieve various objects for it.
  Essentially a PetscCursor for comments.
  """

  class DefaultSection(object):
    __doc__ = """Container for a single section of the docstring, has members:
    'name'     - the name of this section
    'required' - is this section required in the docstring
    'keywords' - keyword header-titles, i.e. "Input Parameter", or "Level", must be correctly cased
    'raw'      - the raw text in the section
    'extent'   - the SourceRange for the whole section
    'lines'    - a tuple of each line of text and its SourceRange in the section
    'items'    - a tuple of extracted tokens of interest, e.g. the level value, options parameters,
                 function parameters, etc.
    """
    __slots__ = "name","required","keywords","raw","extent","lines","items"

    def __init__(self,name,required=False,keywords=None):
      assert isinstance(name,str)
      if keywords is None:
        keywords = (name.title(),)
      assert isinstance(keywords,(list,tuple))
      self.name     = name
      self.required = required
      self.keywords = tuple(keywords)
      self.clear()
      return

    def __str__(self):
      return "\n".join([
        "Type:   {}".format(type(self)),
        "Name:   {}".format(self.name),
        "Extent: {}".format(self.extent),
      ])

    def __bool__(self):
      return bool(self.raw) or bool(self.lines) or bool(self.extent)

    def clear(self):
      self.raw    = None
      self.extent = None
      self.lines  = None
      return

    def fill(self,data):
      assert len(data)
      try:
        assert not self, "Refilling section!"
      except AssertionError:
        import ipdb; ipdb.set_trace()
      self.clear()
      self.lines  = tuple(data)
      self.raw    = "\n".join(s for _,s in self.lines)
      self.extent = PetscSourceRange.fromLocations(
        self.lines[0][0].start,self.lines[-1][0].end
      )
      return

    def setup(self,docstring,inspectLine=lambda *args:None):
      if not self.lines:
        if self.required:
          name = self.transform(self.name)[1]
          mess = "Required subheading(s) '{}' not found in docstring".format(name)
          docstring.addErrorFromSourceRange(mess,docstring.extent,highlight=False)
        return

      seen = collections.defaultdict(list)
      for srcloc,line in self.lines:
        if ":" in line:
          possibleHeader = line.split(":")[0].strip()
          seen[possibleHeader.casefold()].append(
            docstring.makeSourceRange(possibleHeader,line,srcloc.start.line)
          )
        # let each section type determine if this line is useful
        inspectLine(srcloc,line)
        # if valid:
      #     items.append((docstring.makeSourceRange(item,line,srcloc.start.line),item.strip()))
      # self.items = tuple(items)
      for heading,where in seen.items():
        if len(where) <= 1:
          continue
        srclist        = []
        nbefore,nafter = 2,0
        for i,loc in enumerate(where):
          startline = loc.start.line
          if i:
            nbefore = startline-prevLineBegin-1
            if i == len(where)-1:
              nafter = 2
          srclist.append(loc.formatted(numBeforeContext=nbefore,numAfterContext=nafter,trim=False))
          prevLineBegin = startline
        error = "Multiple '{}' subheadings. Much like Highlanders, there can only be one:\n{}".format(self.transform(self.name)[1],"\n".join(srclist))
        docstring._linter.addErrorFromCursor(docstring.cursor,error)
      return

    @staticmethod
    def isHeading(item):
      if isinstance(item,tuple):
        assert len(item) == 2
        assert isinstance(item[0],PetscSourceRange) and isinstance(item[1],str)
        text = item[1]
      elif isinstance(item,str):
        text = item
      else:
        raise NotImplementedError
      return ":" in text and not text.lstrip().startswith(("+",".","-"))

    @staticmethod
    def transform(text):
      return text.istitle(),text.title()


  class Synopsis(DefaultSection):
    def setup(self,ds,*args,**kwargs):
      found = False
      items = []

      def inspector(loc,line):
        nonlocal found
        if found: return
        pre,dash,rest = line.partition(" - ")
        if dash and rest:
          item = pre.strip()
          items.append((ds.makeSourceRange(item,line,loc.start.line),item))
          found = True
        return

      super().setup(ds,*args,inspectLine=inspector,**kwargs)
      self.items = tuple(items)
      return

  class ParameterList(DefaultSection):
    def setup(self,ds,*args,**kwargs):
      subheading = 0
      groups     = collections.defaultdict(list)

      def checkOptPrefixes(groups):
        return

      def checkOptStartsWith(item,descr,char):
        loc,line = item
        if not line.startswith(char):
          eloc = ds.makeSourceRange(line[0],line,loc.start.line)
          mess = "{} parameter list entry must start with '{}'".format(descr,char)
          ds.addErrorFromSourceRange(mess,eloc,patch=SourceFix.fromSourceRange(eloc,char))
        return

      def inspector(loc,line):
        if not line or line.isspace():
          return
        if self.isHeading(line) and len(groups.keys()):
          nonlocal subheading
          subheading += 1
        lstrip = line.lstrip()
        if lstrip.startswith(("+",".","-")):
          #item = lstrip[1:].split("-",maxsplit=1)[0].strip()
          groups[subheading].append((loc,line))
        return

      super().setup(ds,*args,inspectLine=inspector,**kwargs)
      self.items = dict(groups)
      for _,opts in sorted(self.items.items()):
        numOpts = len(opts)
        if numOpts == 0:
          import ipdb; ipdb.set_trace()
        elif numOpts == 1:
          # only 1 option, should start with '.'
          checkOptStartsWith(opts[0],"Solitary",".")
        else:
          # more than 1, should be "+", then however many ".", then last is "-"
          checkOptStartsWith(opts[0],"First multi","+")
          for opt in opts[1:-1]:
            checkOptStartsWith(opt,"Multi",".")
          checkOptStartsWith(opts[-1],"Last multi","-")
      return

  class InlineList(DefaultSection):
    def setup(self,ds,*args,**kwargs):
      items = []

      def inspector(loc,line):
        item   = None
        lstrip = line.lstrip()
        if ":" in lstrip:
          item = lstrip.split(":")[1].strip()
        else:
          item = lstrip.rstrip()
        if item:
          items.append((ds.makeSourceRange(item,line,loc.start.line),item))
        return

      super().setup(ds,*args,inspectLine=inspector,**kwargs)
      self.items = tuple(items)
      return

  class Sections(object):
    __slots__ = "_sections"

    def __init__(self):
      self._sections = collections.OrderedDict()
      return

    def __getattr__(self,attr):
      try:
        return self._sections[attr]
      except KeyError as ke:
        raise AttributeError from ke

    def __iter__(self):
      yield from self._sections.values()

    def __contains__(self,section):
      return self.registered(section)

    def find(self,heading):
      for head,section in self._sections.items():
        if head in heading:
          return section
      sectionNames   = self._sections.keys()
      closestMatches = difflib.get_close_matches(heading,sectionNames,n=1)
      if not len(closestMatches):
        # try if we can find a sub-word
        for head in heading.split():
          closestMatches = difflib.get_close_matches(head,sectionNames,n=1)
          if len(closestMatches):
            break
      if len(closestMatches):
        print(80*"*","CLOSEST MATCHES FOUND {} FOR {}".format(closestMatches,heading),80*"*",sep="\n")
        return self._sections[closestMatches[0]]
      import ipdb; ipdb.set_trace()
      raise ValueError(heading)

    def registered(self,section):
      if isinstance(section,PetscDocString.DefaultSection):
        return section.name in self._sections
      elif isinstance(section,str):
        return section in self._sections
      raise NotImplementedError(section)

    def addSection(self,section):
      assert not self.registered(section)
      self._sections[section.name] = section
      return

    def keywords(self):
      return (keyword for section in self for keyword in section.keywords)


  sections        = Sections()
  registered      = False
  sowingTypes     = {"@","S","E"}
  clxToSowingType = {
    clx.TypeKind.FUNCTIONPROTO : ("@","functions")
  }
  __slots__ = "_linter","cursor","raw","extent","indent"

  @classmethod
  def registerSection(cls,section):
    return cls.sections.addSection(section)

  @classmethod
  def registerDefaultSections(cls):
    if cls.registered:
      return
    defaultHeadings = (
      cls.Synopsis("synopsis",required=True),
      cls.ParameterList("parameters",keywords=("Input Parameter","Output Parameter")),
      cls.DefaultSection("notes"),
      cls.ParameterList("options"),
      cls.InlineList("seealso",keywords=(".seealso",),required=True),
      cls.InlineList("level",required=True),
      cls.DefaultSection("references")
    )
    for section in defaultHeadings:
      cls.registerSection(section)
    cls.registered = True
    return

  @classmethod
  def fromCursor(cls,linter,cursor):
    return cls(linter,cursor,*PetscCursor.getCommentAndRangeFromCursor(cursor))

  @classmethod
  def isValidDocstring(cls,cursor,raw):
    if not raw or not isinstance(raw,str):
      return False

    # if we find sowing chars, its probably a docstring
    if raw.lstrip().startswith("/*@") or raw.rstrip().endswith("@*/"):
      return True

    # if we find these keywords, likely this is a docstring
    if any(keyword.casefold() in raw.casefold() for keyword in cls.sections.keywords()):
      # if it doesn't end with _private or _internal then its very likely a docstring
      if not PetscCursor.getNameFromCursor(cursor).casefold().endswith(("_private","_internal")):
        return True
    return False


  def __init__(self,linter,cursor,raw,extent,indent=2):
    assert isinstance(linter,PetscLinter)
    self.registerDefaultSections()
    if not self.isValidDocstring(cursor,raw):
      raise ParsingError("Not a docstring!")
    self._linter = linter
    self.cursor  = cursor
    self.raw     = raw
    self.extent  = PetscSourceRange.cast(extent)
    self.indent  = indent
    return


  def makeSourceLocation(self,lineno,col):
    return PetscSourceLocation.fromPosition(self.cursor.translation_unit,lineno,col)

  def makeSourceRange(self,token,string,lineno):
    colBegin = string.index(token)+1
    colEnd   = colBegin+len(token)
    begin    = self.makeSourceLocation(lineno,colBegin)
    end      = self.makeSourceLocation(lineno,colEnd)
    return PetscSourceRange.fromLocations(begin,end)

  def addErrorFromSourceRange(self,error,crange,patch=None,**kwargs):
    assert isinstance(error,str)
    error = ":\n".join([error,crange.formatted(numContext=kwargs.pop("numContext",2),**kwargs)])
    self._linter.addErrorFromCursor(self.cursor,error,patch=patch)
    return


  def _checkValidSowingChars(self):
    sowingType,layType = self.clxToSowingType[self.cursor.type.kind]
    # check the beginning
    splitlines  = self.raw.splitlines()
    line        = splitlines[0]
    beginSowing = line.split("/*")[1].split()
    try:
      beginSowing = beginSowing[0]
    except IndexError:
      self.addErrorFromSourceRange(
        "Invalid comment begin line, does not contain sowing identifier. Expected '/*{}' for {}".format(sowingType,layType),
        self.makeSourceRange(line,line,self.extent.start.line)
      )
      beginSowing = [sowingType]
    if beginSowing[0] not in self.sowingTypes:
      import ipdb; ipdb.set_trace()
      raise ParsingError
    beginSowing = "".join(beginSowing)
    # check that nothing else is on the comment begin line
    lsplit = line.strip().split(maxsplit=1)
    if len(lsplit) != 1:
      rest    = lsplit[1]
      restloc = self.makeSourceRange(rest,line,self.extent.start.line)
      self.addErrorFromSourceRange(
        "Invalid comment begin line, must only contain '/*' and sowing identifier",
        restloc,patch=SourceFix.fromSourceRange(restloc,"\n"+(" "*self.indent)+rest)
      )
    # now check the end
    line      = splitlines[-1]
    endSowing = line.split("*/")[0].split()
    try:
      endSowing = endSowing[-1]
    except IndexError:
      pass
    else:
      if sorted(endSowing) != sorted(beginSowing) and 0:
        # TODO: REVIEW: should this check exist?
        correct = beginSowing[::-1]
        endline = self.extent.end.line
        self.addErrorFromSourceRange(
          "Invalid comment end line, sowing identifier(s) do not match begin identifier(s). Expected '{}*/' found '{}*/'".format(correct,endSowing),
          self.makeSourceRange(endSowing,line,endline),patch=SourceFix.fromSourceRange(
            self.makeSourceRange(line,line,endline),line.replace(endSowing,correct)
          )
        )
    return

  def _checkValidDocstringSpacing(self):
    endLine     = self.extent.end.line+1
    cursorStart = self.cursor.extent.start
    if endLine != cursorStart.line:
      # there is at least 1 (probably empty) line between the comment end and whatever it
      # is describing
      self.addErrorFromSourceRange(
        "Invalid line-spacing between docstring and the symbol it describes. The docstring must appear immediately above its target",
        self.makeSourceRange("","",endLine),highlight=False,
        patch=SourceFix(self.makeSourceLocation(endLine,1),cursorStart,"")
      )
    return

  def _reset(self):
    for s in self.sections:
      s.clear()
    return

  def parse(self):
    def smartIndent(line):
      lstrip = line.lstrip()
      if lstrip.startswith(("+",".","-")):
        return lstrip
      return self.indent*" "+lstrip

    self._reset()
    self._checkValidDocstringSpacing()
    self._checkValidSowingChars()
    rawData        = []
    section        = self.sections.synopsis
    invalidIndents = []
    for lineno,line in enumerate(self.raw.splitlines(),start=self.extent.start.line):
      lstrip = line.lstrip()
      if lstrip.startswith("/*") or lstrip.endswith("*/"):
        continue

      lrange = self.makeSourceRange(line,line,lineno)
      # if the line is regular (not empty, or a parameter list), check that line is
      # indented correctly
      indent = len(line)-len(lstrip)
      if lstrip and (indent != self.indent) and not lstrip.startswith((".","+","-")):
        self.addErrorFromSourceRange(
          "Invalid indentation ({}), all regular (non-empty, non-parameter) text must be indented to {} columns".format(indent,self.indent),
          self.makeSourceRange(" "*indent,line,lineno),
          patch=SourceFix.fromSourceRange(lrange,smartIndent(line))
        )
      if ":" in lstrip:
        prevline = rawData[-1][1] if len(rawData) else None
        if prevline and not prevline.isspace():
          self.addErrorFromSourceRange(
            "Missing empty line between sections, must have one before this section",
            self.makeSourceRange("","",lineno),highlight=False,
            patch=SourceFix.fromSourceRange(lrange,"\n"+smartIndent(lstrip))
          )
        newSection = self.sections.find(lstrip.split(":",maxsplit=1)[0].strip().casefold())
        if newSection != section:
          section.fill(rawData)
          rawData = []
          section = newSection
      rawData.append((lrange,line))

    section.fill(rawData)
    for section in self.sections:
      section.setup(self)
    return self


  def checkValidSolitarySectionHeadings(self,section,headings,delim=":"):
    for loc,text in headings:
      _,sep,after = text.partition(delim)
      assert sep
      if after.rstrip():
        self.addErrorFromSourceRange(
          "Heading must appear alone on a line, any content must be on the next line",
          self.makeSourceRange(after,text,loc.start.line)
        )
    return

  def checkValidSectionHeaderSpelling(self,section,headings,transform=None,delim=":"):
    if transform is None:
      transform = section.transform

    for loc,text in headings:
      before,sep,_ = text.partition(delim)
      assert sep
      heading = before.strip()
      if any(k in heading for k in section.keywords):
        continue

      headingLoc       = self.makeSourceRange(heading,text,loc.start.line)
      wasValid,correct = transform(heading)
      if not wasValid and any(k in correct for k in section.keywords):
        self.addErrorFromSourceRange(
          "Invalid header formatting. Expected '{}' found '{}'".format(correct,heading),
          headingLoc,patch=SourceFix.fromSourceRange(headingLoc,correct)
        )
        continue

      closest = difflib.get_close_matches(correct,section.keywords,n=1)
      if len(closest):
        match = closest[0]
        self.addErrorFromSourceRange(
          "Unknown section header '{}', assuming you meant '{}'".format(heading,match),
          headingLoc,patch=SourceFix.fromSourceRange(headingLoc,match)
        )
      else:
        import ipdb; ipdb.set_trace()
    return


class PetscCursor(object):
  __doc__="""
  A utility wrapper around clang.cindex.Cursor that makes retrieving certain useful properties (such as demangled names) from a cursor easier.
  Also provides a host of utility functions that get and (optionally format) the source code around a particular cursor. As it is a wrapper any
  operation done on a clang Cursor may be performed directly on a PetscCursor (although this object does not pass the isinstance() check).

  See __getattr__ below for more info.
  """
  __slots__ = "__cursor","name","typename","derivedtypename","argidx"

  def __init__(self,cursor,idx=-12345):
    assert isinstance(cursor,(clx.Cursor,PetscCursor))
    if isinstance(cursor,PetscCursor):
      self.__cursor        = cursor.clangCursor()
      self.name            = cursor.name
      self.typename        = cursor.typename
      self.derivedtypename = cursor.derivedtypename
      self.argidx          = cursor.argidx if idx == -12345 else idx
    else:
      self.__cursor        = cursor
      self.name            = self.getNameFromCursor(cursor)
      self.typename        = self.getTypenameFromCursor(cursor)
      self.derivedtypename = self.getDerivedTypenameFromCursor(cursor)
      self.argidx          = idx
    return

  @classmethod
  def asPetscCursor(cls,cursor):
    __doc__="""like numpy.asanyarray but for PetscCursors"""
    assert isinstance(cursor,(clx.Cursor,cls))
    return cls(cursor) if isinstance(cursor,clx.Cursor) else cursor

  def clangCursor(self):
    __doc__="""return the internal clang cursor"""
    return self.__cursor

  def __getattr__(self,attr):
    __doc__="""
    Allows us to essentialy fake being a clang cursor, if __getattribute__ fails
    (i.e. the value wasn't found in self), then we try the cursor. So we can do things
    like self.translation_unit, but keep all of our variables out of the cursors
    namespace
    """
    return getattr(self.__cursor,attr)

  def __str__(self):
    return "\n".join([self.getFormattedLocationString(),self.getFormattedBlurb()])

  @classmethod
  def errorViewFromCursor(cls,cursor):
    __doc__="""
    Something has gone wrong, and we try to extract as much information from the cursor as
    possible for the exception. Nothing is guaranteed to be useful here.
    """
    name = cursor.displayname
    kind = cursor.kind
    loc  = cursor.location
    try:
      fname  = loc.file.name
    except AttributeError:
      fname  = "UNKNOWN_FILE"
    locStr   = ":".join([fname,str(loc.column),str(loc.line)])
    # Does not yet raise exception so we can call it here
    typename = cls.getTypenameFromCursor(cursor)
    srcStr   = cls.getFormattedSourceFromCursor(cursor,nboth=2)
    return "'{}' of kind '{}' of type '{}' at {}:\n{}".format(name,kind,typename,locStr,srcStr)

  @classmethod
  def getNameFromCursor(cls,cursor):
    __doc__="""
    Try to convert **&(PetscObject)obj[i]+73 to obj
    """
    if isinstance(cursor,cls):
      return cursor.name
    name = None
    if cursor.spelling:
      name = cursor.spelling
    elif cursor.kind in mathCursors:
      if cursor.kind == clx.CursorKind.BINARY_OPERATOR:
        # we arbitrarily use the first token here since we assume that it is the important
        # one.
        operands = list(cursor.get_children())
        # its certainly funky when a binary operation doesn't have a binary system of
        # operands
        assert len(operands) == 2, "Found {} operands for binary operator when only expecting 2 for cursor {}".format(len(operands),cls.errorViewFromCursor(cursor))
        name = operands[0].spelling
      else:
        # just a plain old number or unary operator
        name = "".join(t.spelling for t in cursor.get_tokens())
    elif cursor.kind in castCursors:
      # Need to extract the castee from the caster
      castee = [c for c in cursor.get_children() if c.kind == clx.CursorKind.UNEXPOSED_EXPR]
      # If we don't have 1 symbol left then we're in trouble, as we probably didn't
      # pick the right cursors above
      assert len(castee) == 1, "Cannot determine castee from the caster for cursor {}".format(cls.errorViewFromCursor(cursor))
      # Easer to do some mild recursion to figure out the naming for us than duplicate
      # the code. Perhaps this should have some sort of recursion check
      name = cls.getNameFromCursor(castee[0])
    elif (cursor.type.get_canonical().kind == clx.TypeKind.POINTER) or (cursor.kind == clx.CursorKind.UNEXPOSED_EXPR):
      pointees = []
      if cursor.type.get_pointee().kind  == clx.TypeKind.CHAR_S:
        # For some reason preprocessor macros that contain strings don't propagate
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
      if len(pointees) > 1:
        # sometimes array subscripts can creep in
        pointees = [c for c in pointees if c.kind not in mathCursors]
      if len(pointees) == 1:
        name = cls.getNameFromCursor(pointees[0])
    if not name:
      # Catchall last attempt, we become the very thing we swore to destroy and parse the
      # tokens ourselves
      tokenList = [t for t in cursor.get_tokens() if t.kind in varTokens]
      # Remove iterator variables
      tokenList = [t for t in tokenList if t.cursor.kind not in mathCursors]
      # removes all cursors that have duplicate spelling
      tokenList = list({t.spelling: t for t in tokenList}.values())
      if len(tokenList) != 1:
        # For whatever reason (perhaps because its macro stringization hell) PETSC_HASH_MAP
        # and PetscKernel_XXX absolutely __brick__ the AST. The resultant cursors have no
        # children, no name, no tokens, and a completely incorrect SourceLocation.
        # They are for all intents and purposes uncheckable :)
        srcstr = cls.getRawSourceFromCursor(cursor)
        errstr = cls.errorViewFromCursor(cursor)
        if "PETSC_HASH" in srcstr:
          if "_MAP" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_MAP for cursor {}".format(errstr))
          elif "_SET" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_SET for cursor {}".format(errstr))
          raise RuntimeError("Unhandled unparsable PETSC_HASH_XXX for cursor {}".format(errstr))
        elif "PetscKernel_" in srcstr:
          raise ParsingError("Encountered unparsable PetscKernel_XXX for cursor {}".format(errstr))
        elif ("PetscOptions" in srcstr) or ("PetscObjectOptions" in srcstr):
          raise ParsingError("Encountered unparsable Petsc[Object]OptionsBegin for cursor {}".format(errstr))
        raise RuntimeError("Unexpected number of tokens ({}) for cursor {}".format(len(tokenList),errstr))
      name = tokenList[0].spelling
      assert name, "Cannot determine name of symbol from cursor {}".format(cls.errorViewFromCursor(cursor))
    return name

  @classmethod
  def getRawNameFromCursor(cls,cursor):
    __doc__="""
    if getNameFromCursor tries to convert **&(PetscObject)obj[i]+73 to obj then this function
    tries to extract **&(PetscObject)obj[i]+73 in the cleanest way possible
    """
    name = "".join(t.spelling for t in cursor.get_tokens())
    if not name:
      try:
        # now we try for the formatted name
        name = cls.getNameFromCursor(cursor)
      except ParsingError:
        srcstr = cls.getRawSourceFromCursor(cursor)
        errstr = cls.errorViewFromCursor(cursor)
        if "PETSC_HASH" in srcstr:
          if "_MAP" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_MAP for cursor {}".format(errstr))
          elif "_SET" in srcstr:
            raise ParsingError("Encountered unparsable PETSC_HASH_SET for cursor {}".format(errstr))
          raise RuntimeError("Unhandled unparsable PETSC_HASH_XXX for cursor {}".format(errstr))
        elif "PetscKernel_" in srcstr:
          raise ParsingError("Encountered unparsable PetscKernel_XXX for cursor {}".format(errstr))
        elif ("PetscOptions" in srcstr) or ("PetscObjectOptions" in srcstr):
          raise ParsingError("Encountered unparsable Petsc[Object]OptionsBegin for cursor {}".format(errstr))
        raise RuntimeError("Could not determine useful name for cursor {}".format(errstr))
    return name

  @classmethod
  def getTypenameFromCursor(cls,cursor):
    __doc__="""
    Try to get the most canonical type from a cursor so DM -> _p_DM *
    """
    if isinstance(cursor,cls):
      return cursor.typename
    type    = cursor.type
    pointee = type.get_pointee()
    if pointee.spelling:
      pointeeCanonSpelling = pointee.get_canonical().spelling
      return pointeeCanonSpelling if pointeeCanonSpelling else pointee.spelling
    canonSpelling = type.get_canonical().spelling
    return canonSpelling if canonSpelling else type.spelling

  @staticmethod
  def getDerivedTypenameFromCursor(cursor):
    __doc__="""
    Get the least canonical type form a cursor so DM -> DM
    """
    return cursor.type.spelling

  @staticmethod
  def getRawSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,trim=False):
    return pclu.getRawSourceFromCursor(
      cursor,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,trim=trim
    )

  def getRawSource(self,**kwargs):
    return self.getRawSourceFromCursor(self,**kwargs)

  @staticmethod
  def getFormattedSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,view=False):
    if cursor.kind == clx.CursorKind.FUNCTION_DECL:
      begin  = cursor.extent.start
      # -1 gives you EOL
      fnline = PetscSourceLocation.fromPosition(cursor.translation_unit,begin.line,-1)
      extent = PetscSourceRange.fromLocations(cursor.extent.start,fnline)
    else:
      extent = cursor.extent
    return pclu.getFormattedSourceFromSourceRange(
      extent,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,view=view
    )

  def getFormattedSource(self,**kwargs):
    return self.getFormattedSourceFromCursor(self,**kwargs)

  def view(self):
    print(self.getFormattedSource(nboth=5))
    return

  @staticmethod
  def getFormattedLocationStringFromCursor(cursor):
    loc = cursor.location
    return ":".join([loc.file.name,str(loc.column),str(loc.line)])

  def getFormattedLocationString(self):
    return self.getFormattedLocationStringFromCursor(self)

  @classmethod
  def getFormattedBlurbFromCursor(cls,cursor):
    cursor = cls.asPetscCursor(cursor)
    return "'{}' of derived type '{}', canonical type '{}'\n{}\n".format(
      cursor.name,cursor.derivedtypename,cursor.typename,cursor.getFormattedSource(nboth=2)
    )

  def getFormattedBlurb(self):
    return self.getFormattedBlurbFromCursor(self)

  @staticmethod
  def viewAstFromCursor(cursor):
    return print("\n".join(pclu.viewAstFromCursor(cursor)))

  def viewAst(self):
    return self.viewAstFromCursor(self)

  @staticmethod
  def getOrRegisterClangFunction(funcName,argtypes,rettype):
    try:
      func = getattr(clx.conf.lib,funcName)
      if (func.argtypes is None) and (func.errcheck is None):
        # if this hasn't been registered before these will be none
        raise AttributeError
    except AttributeError:
      # have to do the book-keeping ourselves since it may not be properly hooked up
      clx.register_function(clx.conf.lib,(funcName,argtypes,rettype),False)
      func = getattr(clx.conf.lib,funcName)
    return func

  @classmethod
  def findCursorReferencesFromCursor(cls,cursor):
    __doc__="""
    Brute force find and collect all references in a file that pertain to a particular
    cursor. Essentially refers to finding every reference to the symbol that the cursor
    represents, so this function is only useful for first-class symbols (i.e. variables,
    functions)
    """
    foundCursors = []
    def callBackFunc(ctx,cursor,srcRange):
      # The "cursor" returned here is actually just a CXCursor, not the real
      # clx.Cursor that we lead python to believe in our function prototype. Luckily we
      # have all we need to remake the python object from scratch
      cursor = clx.Cursor.from_location(ctx.translation_unit,srcRange.start)
      try:
        foundCursors.append(cls(cursor))
      except ParsingError:
        pass
      except Exception as exc:
        string = "Full error full error message below:"
        print('='*30,"CXCursorAndRangeVisitor Error",'='*30)
        print("It is possible that this is a false positive! E.g. some 'unexpected number of tokens' errors are due to macro instantiation locations being misattributed.\n",string,"\n","-"*len(string),"\n",exc,sep="")
        print('='*30,"CXCursorAndRangeVisitor End Error",'='*26)
      return 1 # continue

    callBack   = CXCursorAndRangeVisitorCallBackProto(callBackFunc)
    pyCtx      = ctypes.py_object(cursor) # pyCtx = (PyObject *)cursor;
    cxCallback = PetscCXCursorAndRangeVisitor(pyCtx,callBack)
    func       = cls.getOrRegisterClangFunction(
      "clang_findReferencesInFile",[clx.Cursor,clx.File,PetscCXCursorAndRangeVisitor],ctypes.c_uint
    )
    func(cursor.clangCursor(),cursor.location.file,cxCallback)
    return foundCursors

  def findCursorReferences(self):
    return self.findCursorReferencesFromCursor(self)

  @classmethod
  def getCommentAndRangeFromCursor(cls,cursor):
    cursor = cls.asPetscCursor(cursor)
    func   = cls.getOrRegisterClangFunction(
      "clang_Cursor_getCommentRange",[clx.Cursor],clx.SourceRange
    )
    return cursor.raw_comment,func(cursor.clangCursor())

  def getCommentAndRange(self):
    return self.getCommentAndRangeFromCursor(self)


class SourceFix(object):
  DELETE    = -1
  __slots__ = "extent","filename","ctxlines","src","ranges","deltas","offset","fixed","fixDepth"

  def __init__(self,begin,end,value,contextlines=2):
    def validrange(begin,end):
      begin = PetscSourceLocation.cast(begin)
      end   = PetscSourceLocation.cast(end)
      if begin.line == end.line:
        assert begin.column < end.column, "end col {} <= begin col {}, ill-formed source fix".format(end.column,begin.column)
      elif begin.line > end.line:
        mess = "end line {} < begin line {}, ill-formed source fix".format(end.line,begin.line)
        raise AssertionError(mess)
      return PetscSourceRange.fromLocations(begin,end)

    self.extent   = validrange(begin,end)
    self.filename = PetscPath(self.extent.start.file.name)
    self.ctxlines = contextlines
    self.src      = self.extent.raw(numContext=self.ctxlines)
    self.ranges   = [self.extent]
    self.deltas   = [value if value is self.DELETE else str(value)]
    # this is an error, since previous detection should not have created a fix
    assert self.deltas[0] != self.extent.raw(), "trying to replace {} with itself".format(self.src)
    # if we have context, this is the character offset into self.src such that
    # self.src[self.begins[i].offset-self.offset]
    # gives the start of the src snippet to replace
    begin         = self.ranges[0].start
    offset        = begin.column-1+sum(map(len,self.src.splitlines(True)[:self.ctxlines]))
    self.offset   = begin.offset-offset
    self.fixed    = None
    self.fixDepth = 0
    return

  @classmethod
  def fromSourceRange(cls,srcRange,value,**kwargs):
    return cls(srcRange.start,srcRange.end,value,**kwargs)

  @classmethod
  def fromCursor(cls,cursor,value,**kwargs):
    return cls.fromSourceRange(cursor.extent,value,**kwargs)

  def dofix(self,n=1):
    newSrc   = self.src
    idxDelta = 0
    for rng,i in zip(sorted(self.ranges),range(n)):
      begin       = rng.start
      end         = rng.end
      beginoffset = begin.offset-self.offset
      endoffset   = end.offset-self.offset
      delta       = self.deltas[i]
      newSrcTemp  = "".join([newSrc[:beginoffset+idxDelta],delta,newSrc[endoffset+idxDelta:]])
      idxDelta    = len(newSrcTemp)-len(newSrc)
      newSrc      = newSrcTemp
    print(newSrcTemp)
    return newSrcTemp

  def appendFix(self,other):
    assert isinstance(other,type(self))
    assert self.src == other.src, "Cannot combine fixes that do not share identical source!"
    self.extent = self.extent.mergeWith(other.extent)
    for mine,theirs in itertools.zip_longest(self.ranges,other.ranges):
      if mine is None or theirs is None:
        break
      if mine.overlaps(theirs):
        import ipdb; ipdb.set_trace()

    # fixes and ranges must be applied in order
    newRanges    = self.ranges+other.ranges
    rangeArgsort = sorted(range(len(newRanges)),key=newRanges.__getitem__)
    self.ranges  = [newRanges[i] for i in rangeArgsort]
    newDeltas    = self.deltas+other.deltas
    self.deltas  = [newDeltas[i] for i in rangeArgsort]
    return

  def collapse(self):
    __doc__="""
    Collapses a list of fixes and produces a fixed src line.
    Fixes probably should not overwrite each other (for now), so we error out, but this
    is arguably a completely valid case. I just have not seen an example of it that I
    can use to debug with yet.
    """
    if self.fixDepth == len(self.deltas): # already collapsed, no need to do it again
      assert self.fixed, "Fix depth {} = number of deltas {} but no fixed string exists".format(
        self.fixDepth,len(self.deltas)
      )
      return

    idxDelta = 0
    newSrc   = self.src
    for rng,delta in zip(self.ranges,self.deltas):
      begin,end   = rng.start,rng.end
      assert begin in self.extent and end in self.extent, "Idx out of bounds of src, fix not viable"
      beginoffset = begin.offset-self.offset
      endoffset   = end.offset-self.offset
      newSrcTemp  = "".join([newSrc[:beginoffset+idxDelta],delta,newSrc[endoffset+idxDelta:]])
      idxDelta    = len(newSrcTemp)-len(newSrc)
      newSrc      = newSrcTemp
    self.fixDepth = len(self.deltas)
    self.fixed    = newSrc
    return


class PetscLinter(object):
  __doc__="""
  Object to manage the collection and processing of errors during a lint run.
  """
  __slots__ = "flags","clangOpts","prefix","verbose","werror","lock","errPrefix","warnPrefix","errors","warnings","patches","index"

  def __init__(self,compilerFlags,clangOptions=baseClangOptions,prefix="[ROOT]",verbose=False,werror=False,lock=None):
    self.flags      = compilerFlags
    self.clangOpts  = clangOptions
    self.prefix     = prefix
    self.verbose    = verbose
    self.werror     = werror
    self.lock       = lock
    self.errPrefix  = " ".join([prefix,85*"-"])
    self.warnPrefix = " ".join([prefix,85*"%"])
    self.errors     = collections.OrderedDict()
    self.warnings   = []
    # This can actually just be a straight list, since each linter object only ever
    # handles a single file, but use dict nonetheless
    self.patches    = {}
    self.index      = clx.Index.create()
    return

  def __str__(self):
    prefixStr = "Prefix:        '{}'".format(self.prefix)
    flagStr   = "Compiler Flags: {}".format(self.flags)
    clangStr  = "Clang Options:  {}".format(self.clangOpts)
    lockStr   = "Lock:           {}".format(self.lock is not None)
    showStr   = "Verbose:        {}".format(self.verbose)
    printList = [prefixStr,flagStr,clangStr,lockStr,showStr]
    errorStr  = self.getAllErrors()
    if errorStr: printList.append(errorStr)
    warnStr   = self.getAllWarnings(joinToString=True)
    if warnStr: printList.append(warnStr)
    return "\n".join(printList)

  def __enter__(self):
    return self

  def __exit__(self,excType,*args):
    if not excType:
      if self.verbose:
        self.__print(self.getAllWarnings(joinToString=True))
      self.__print(self.getAllErrors())
    return

  def __print(self,*args,**kwargs):
    args = tuple([a for a in args if a])
    if not args and not kwargs:
      return
    if self.lock:
      with self.lock:
        print(*args,**kwargs)
    else:
      print(*args,**kwargs)
    return

  @staticmethod
  def findFunctionCallExpr(tu,functionNames):
    __doc__="""
    Finds all function call expressions in container functionNames.

    Note that if a particular function call is not 100% correctly defined (i.e. would the
    file actually compile) then it will not be picked up by clang AST.

    Function-like macros can be picked up, but it will be in the wrong 'order'. The AST is
    built as if you are about to compile it, so macros are handled before any real
    function definitions in the AST, making it impossible to map a macro invocation to
    its 'parent' function.
    """
    def walkScopeSwitch(parent,scope):
      __doc__="""
      special treatment for switch-case since the AST setup for it is mind-boggingly stupid.
      The first node after a case statement is listed as the cases *child* whereas every other
      node (including the break!!) is the cases *sibling*
      """
      CASE_KIND     = clx.CursorKind.CASE_STMT
      COMPOUND_KIND = clx.CursorKind.COMPOUND_STMT
      CALL_KIND     = clx.CursorKind.CALL_EXPR
      # in case we get here from a scope decrease within a case
      caseScope = scope
      for child in parent.get_children():
        childKind = child.kind
        if childKind == CASE_KIND:
          # create a new scope every time we encounter a case, this is now for all intents
          # and purposes the 'scope' going forward. We don't overwrite the original scope
          # since we still need each case scope to be the previous scopes sibling
          caseScope = scope.sub()
          yield from walkScope(child,scope=caseScope)
        elif childKind == CALL_KIND:
          if child.spelling in functionNames:
            yield (child,possibleParent,caseScope)
            # Cursors that indicate change of logical scope
        elif childKind == COMPOUND_KIND:
          yield from walkScopeSwitch(child,caseScope.sub())

    def walkScope(parent,scope=Scope()):
      __doc__="""
      walk the tree determining the scope of a node. here 'scope' refers not only
      to lexical scope but also to logical scope, see Scope object above
      """
      SWITCH_KIND   = clx.CursorKind.SWITCH_STMT
      COMPOUND_KIND = clx.CursorKind.COMPOUND_STMT
      CALL_KIND     = clx.CursorKind.CALL_EXPR
      for child in parent.get_children():
        childKind = child.kind
        if childKind == SWITCH_KIND:
          # switch-case statements require special treatment, we skip to the compound
          # statement
          switchChildren = [c for c in child.get_children() if c.kind == COMPOUND_KIND]
          assert len(switchChildren) == 1, "Switch statement has multiple '{' operators?"
          yield from walkScopeSwitch(switchChildren[0],scope.sub())
        elif childKind == CALL_KIND:
          if child.spelling in functionNames:
            yield (child,possibleParent,scope)
        elif childKind == COMPOUND_KIND:
          # scope has decreased
          yield from walkScope(child,scope=scope.sub())
        else:
          # same scope
          yield from walkScope(child,scope=scope)

    cursor   = tu.cursor
    filename = cursor.spelling
    for possibleParent in cursor.get_children():
      # getting filename is for some reason stupidly expensive, so we do this check first
      if possibleParent.kind not in funcCallCursors: continue
      try:
        if possibleParent.location.file.name != filename: continue
      except AttributeError:
        # possibleParent.location.file is None
        continue
      # if we've gotten this far we have found a function definition, so first yield the
      # parent
      yield possibleParent
      # then yield any children matching our function calls
      yield from walkScope(possibleParent)

  def clear(self):
    __doc__="""
    Resets the linter error, warning, and patch buffers.
    Called automatically before parsing a file
    """
    self.errors   = collections.OrderedDict()
    self.warnings = []
    self.patches  = {}
    return

  def parse(self,filename):
    __doc__="""
    parse a file for errors
    """
    self.clear()
    if self.verbose: self.__print(self.prefix,"Processing file     ",filename)
    tu = self.index.parse(str(filename),args=self.flags,options=self.clangOpts)
    if self.verbose and tu.diagnostics:
      self.__print("\n".join({" ".join([self.prefix,d]) for d in map(str,tu.diagnostics)}))
    self.process(tu)
    return self

  def process(self,tu):
    __doc__="""
    process a translation unit for errors
    """
    processedFuncs = collections.defaultdict(list)
    for results in self.findFunctionCallExpr(tu,set(checkFunctionMap.keys())):
      if isinstance(results,clx.Cursor):
        checkDocMap[results.kind](self,results)
        continue
      func,parent,scope = results
      try:
        checkFunctionMap[func.spelling](self,func,parent)
      except ParsingError as pe:
        self.addWarning(tu.cursor.spelling,str(pe))
      processedFuncs[PetscCursor.getNameFromCursor(parent)].append(
        (PetscCursor.asPetscCursor(func),scope)
      )
    for pname,functionList in processedFuncs.items():
      seen = {}
      for func,scope in functionList:
        combo = [func.displayname]
        try:
          combo.extend(PetscCursor.getRawNameFromCursor(a) for a in func.get_arguments())
        except ParsingError:
          continue
        combo = tuple(combo)
        if combo not in seen:
          seen[combo] = (func,scope)
        elif scope >= seen[combo][1]:
          seenStart = seen[combo][0].extent.start.line
          start     = func.extent.start
          startline = start.line
          end       = PetscSourceLocation.fromPosition(func.translation_unit,startline,-1)
          patch     = SourceFix(start,end,"")
          message   = "Duplicate function found previous identical usage:\n\n{}".format(
            seen[combo][0].getFormattedSource(nbefore=2,nafter=startline-seenStart)
          )
          self.addErrorFromCursor(func,message,patch=patch)
    return

  @staticmethod
  def getArgumentCursors(funcCursor):
    __doc__="""
    given a cursor representing a function, return a tuple of PetscCursor's of its arguments
    """
    return tuple([PetscCursor(a,i+1) for i,a in enumerate(funcCursor.get_arguments())])

  def addErrorFromCursor(self,cursor,errorMessage,patch=None):
    __doc__="""
    given a cursor attach a diagnostic error message to it, and optionally a fix
    """
    cursor   = PetscCursor.asPetscCursor(cursor)
    filename = PetscPath(cursor.location.file.name)
    if filename not in self.errors:
      self.errors[filename] = collections.OrderedDict()
    cursorId = cursor.hash
    if cursorId not in self.errors[filename]:
      header = "\nERROR {}: {}\n".format(len(self.errors[filename]),str(cursor))
      self.errors[filename][cursorId] = (header,[],[])
    self.errors[filename][cursorId][1].append(errorMessage)
    self.errors[filename][cursorId][2].append(patch is not None)
    try:
      assert filename == patch.filename
      self.patches[patch.filename].append(patch)
    except KeyError:
      self.patches[patch.filename] = [patch]
      return
    except AttributeError:
      assert patch is None
      # patch = None, return
      return
    # check if this is a compound error, i.e. an additional error on the same line
    # in which case we need to combine with previous patch
    pex = patch.extent
    for prevPatch in self.patches[patch.filename][:-1]:
      prepex = prevPatch.extent
      if pex.overlaps(prepex) or pex.start.line == prepex.start.line:
        # remove ourselves from the list
        patch = self.patches[patch.filename].pop()
        # this should now be the previous patch on the same line, so we combine with it
        prevPatch.appendFix(patch)
        break
    return

  def addWarning(self,filename,warnMsg):
    __doc__="""
    add a generic warning given a filename
    """
    if self.werror:
      return self.addErrorFromCursor(filename,warnMsg)
    try:
      if warnMsg in self.warnings[-1][1]:
        # we just had the exact same warning, we can ignore it. This happens very often
        # for warnings occurring deep within a macro
        return
    except IndexError:
      pass
    warnStr = "".join(["\nWARNING {}: ".format(len(self.warnings)),warnMsg])
    self.warnings.append((filename,warnStr))
    return

  def addWarningFromCursor(self,cursor,warnMsg):
    __doc__="""
    given a cursor attach a diagnostic warning message to it
    """
    if self.werror:
      return self.addErrorFromCursor(cursor,warnMsg)
    warnFile = cursor.location.file.name
    warnStr  = "".join(["\nWARNING {}: ".format(len(self.warnings)),str(cursor),"\n",warnMsg])
    self.warnings.append((warnFile,warnStr))
    return

  def getAllErrors(self):
    __doc__="""
    return all errors collected so far in a tuple
    """
    def maybeAddToGlobalList(globalList,localList,path):
      if len(localList):
        globalList.append((
          path,"{prefix}\n{}\n{prefix}".format("\n".join(localList)[1:],prefix=self.errPrefix)
        ))
      return

    def maybeAddToLocalList(localList,thing,mask,header):
      string = "\n\n".join(itertools.compress(thing,mask))
      if string:
        localList.append("".join((header,string)))
      return

    allUnresolved,allResolved = [],[]
    for path,errors in self.errors.items():
      unresolved,resolved = [],[]
      for header,errs,mask in errors.values():
        maybeAddToLocalList(resolved,errs,mask,header)
        maybeAddToLocalList(unresolved,errs,(not m for m in mask),header)
      maybeAddToGlobalList(allUnresolved,unresolved,path)
      maybeAddToGlobalList(allResolved,resolved,path)
    return allUnresolved,allResolved

  def getAllWarnings(self,joinToString=False):
    __doc__="""
    return all warnings collected so far, and optionally join them all as one string
    """
    if joinToString:
      if len(self.warnings):
        return "\n".join([
          self.warnPrefix,"\n".join(s for _,s in self.warnings)[1:],self.warnPrefix
        ])
      return ""
    return self.warnings

  def coalescePatches(self):
    __doc__="""
    given a set of patches, collapse all patches and return the minimal set of diffs required
    """
    import datetime

    class Addline(object):
      __slots__ = "offset"
      def __init__(self,offset):
        self.offset = offset
        return

      def __call__(self,match):
        ll,lr = match.group(1).split(",")
        rl,rr = match.group(2).split(",")
        return "@@ -{},{} +{},{} @@".format(self.offset+int(ll),lr,self.offset+int(rl),rr)

    combinedPatches = []
    for filename,patches in self.patches.items():
      fstr  = str(filename)
      diffs = []
      for p in sorted(patches,key=lambda x: x.extent.start.line):
        p.collapse()
        rn  = datetime.datetime.now().ctime()
        tmp = list(difflib.unified_diff(
          p.src.splitlines(True),p.fixed.splitlines(True),
          fromfile=fstr,tofile=fstr,fromfiledate=rn,tofiledate=rn,n=p.ctxlines
        ))
        tmp[2] = re.sub(r"^@@ -([0-9,]+) \+([0-9,]+) @@",Addline(p.extent.start.line),tmp[2])
        # only the first diff should get the file heading
        diffs.append("".join(tmp[2:] if len(diffs) else tmp))
      combinedPatches.append((filename,"".join(diffs)))
    return combinedPatches

  def diagnostics(self):
    errorsLeft,errorsFixed = self.getAllErrors()
    warnings = self.getAllWarnings()
    patches  = self.coalescePatches()
    return errorsLeft,errorsFixed,warnings,patches


class WorkerPool(mp.queues.JoinableQueue):
  __slots__ = (
    "parallel","errorQueue","returnQueue","lock","workers","numWorkers","timeout",
    "verbose","prefix","warnings","errorsLeft","errorsFixed","patches","linter"
  )

  class QueueSignal(enum.IntEnum):
    __doc__="""
    Various signals to indicate return type on the data queue from child processes
    """
    WARNING      = enum.auto()
    UNIFIED_DIFF = enum.auto()
    ERRORS_LEFT  = enum.auto()
    ERRORS_FIXED = enum.auto()
    EXIT_QUEUE   = enum.auto()

  def __init__(self,numWorkers=-1,timeout=2,verbose=False,prefix="[ROOT]",**kwargs):
    if numWorkers < 0:
      numWorkers = max(mp.cpu_count()-1,1)
    super().__init__(numWorkers,**kwargs,ctx=mp.get_context())
    if numWorkers in {0,1}:
      if verbose:
        print(prefix,"Number of worker processes ({}) too small, disabling multiprocessing".format(numWorkers))
      self.parallel    = False
      self.errorQueue  = None
      self.returnQueue = None
      self.lock        = None
    else:
      if verbose:
        print(prefix,"Number of worker processes ({}) sufficient, enabling multiprocessing".format(numWorkers))
      self.parallel    = True
      self.errorQueue  = mp.Queue()
      self.returnQueue = mp.Queue()
      self.lock        = mp.Lock()
    self.numWorkers  = numWorkers
    self.timeout     = timeout
    self.verbose     = verbose
    self.prefix      = prefix
    self.warnings    = []
    self.errorsLeft  = []
    self.errorsFixed = []
    self.patches     = []
    return

  def setup(self,compilerFlags,clangLib=None,clangOptions=baseClangOptions,werror=False):
    if clangLib is None:
      assert clx.conf.loaded, "Must initialize libClang first"
      clangLib = clx.conf.get_filename()
    if self.parallel:
      commonKwArgs = {
        "target" : queueMain,
        "args"   : (
          clangLib,checkFunctionMap,classIdMap,compilerFlags,clangOptions,self.verbose,werror,
          self.errorQueue,self.returnQueue,self,self.lock
        ),
        "daemon" : True
      }
      self.workers = [
        mp.Process(**commonKwArgs,name=n) for n in map("[{}]".format,range(self.numWorkers))
      ]
      for worker in self.workers:
        worker.start()
    else:
      self.linter = PetscLinter(compilerFlags,clangOptions=clangOptions,prefix=self.prefix,verbose=self.verbose,werror=werror)
    return self

  def walk(self,srcPath,excludeDirs=excludeDirNames,excludeDirSuff=excludeDirSuffixes,allowFileSuff=allowFileExtensions):
    if srcPath.is_file():
      self.put(srcPath)
    else:
      for root,dirs,files in os.walk(srcPath):
        if self.verbose: print(self.prefix,"Processing directory",root)
        dirs[:] = [d for d in dirs if d not in excludeDirs]
        dirs[:] = [d for d in dirs if not d.endswith(excludeDirSuff)]
        for filename in (os.path.join(root,f) for f in files if f.endswith(allowFileSuff)):
          self.put(filename)
    return

  def put(self,filename,*args):
    if self.parallel:
      import queue
      # continuously put files onto the queue, if the queue is full we block for
      # queueTimeout seconds and if we still cannot insert to the queue we check
      # children for errors. If no errors are found we try again.
      while True:
        try:
          super().put(filename,True,self.timeout)
          break # only get here if put is successful
        except queue.Full:
          # we don't want to join here since a child may have encountered an error!
          self.check()
    else:
      errLeft,errFixed,warnings,patches = self.linter.parse(filename).diagnostics()
      self.errorsLeft.extend(errLeft)
      self.errorsFixed.extend(errFixed)
      self.warnings.append(warnings)
      self.patches.extend(patches)
    return

  def check(self,join=False):
    if self.parallel:
      stopMultiproc = False
      # join here to colocate error messages if needs be
      if join: self.join()
      while not self.errorQueue.empty():
        # while this does get recreated for every error, we do not want to needlessly
        # reinitialize it when no errors exist. If we get to this point however we no longer
        # care about performance as we are about to crash everything.
        errBars   = "".join(["[ERROR]",85*"-","[ERROR]\n"])
        errBars   = [errBars,errBars]
        exception = self.errorQueue.get()
        try:
          errMess = str(exception).join(errBars)
        except:
          errMess = exception
        print(errMess)
        stopMultiproc = True
      if stopMultiproc:
        raise RuntimeError("Error in child process detected")
    return

  def finalize(self):
    if self.parallel:
      self.check(join=True)
      self.errorQueue.close()
      # send stop-signal to child processes
      for _ in self.workers:
        self.put(self.QueueSignal.EXIT_QUEUE)
      while not self.returnQueue.empty():
        signal,returnData = self.returnQueue.get()
        if signal == self.QueueSignal.ERRORS_LEFT:
          self.errorsLeft.extend(returnData)
        elif signal == self.QueueSignal.ERRORS_FIXED:
          self.errorsFixed.extend(returnData)
        elif signal == self.QueueSignal.UNIFIED_DIFF:
          self.patches.extend(returnData)
        elif signal == self.QueueSignal.WARNING:
          self.warnings.append(returnData)
        else:
          raise ValueError("Unknown data returned by returnQueue {}, {}".format(signal,returnData))
      self.returnQueue.close()
      self.close()
      for worker in self.workers:
        # spin until every child exits, we need to do this otherwise the final summary
        # output is totally garbled
        worker.join()
        if sys.version_info >= (3,7):
          worker.close()
    self.errorsLeft  = [e for e in self.errorsLeft  if e] # remove any None's
    self.errorsFixed = [e for e in self.errorsFixed if e]
    self.warnings    = [w for w in self.warnings    if w]
    self.patches     = [p for p in self.patches     if p]
    return self.warnings,self.errorsLeft,self.errorsFixed,self.patches


"""
TODO:
- make an object to hold all the PetscLinter diagnostics per file
- make PetscLinter.diagnostics() be the one-stop shop
- make warnings into a list with filename
- integrate warnings into the test output as well
- implement fixits for stuff we can fix
- figure out how to handle overwriting fixits...
- figure out how to handle in-out parameters
"""
"""utilities for checking docstrings"""
def checkDocValidSynopsis(linter,cursor,docstring):
  synopsis = docstring.sections.synopsis
  if not synopsis:
    import ipdb; ipdb.set_trace()
    # wtf
    return

  foundSynopsis = False
  cursorName    = PetscCursor.getNameFromCursor(cursor)
  for lineLoc,line in synopsis.lines:
    symbol,dash,rest = line.partition("-")
    if len(dash) and len(rest):
      foundSynopsis = True
      symbol        = symbol.strip()
      if symbol != cursorName:
        loc = docstring.makeSourceRange(symbol,line,lineLoc.start.line)
        if len(difflib.get_close_matches(symbol,[cursorName],n=1)):
          mess  = "Docstring name '{}' does not match symbol. Assuming you meant '{}'".format(
            symbol,cursorName
          )
          patch = SourceFix.fromSourceRange(loc,cursorName)
        else:
          mess  = "Docstring name '{}' does not match symbol name '{}'".format(symbol,cursorName)
          patch = None
        docstring.addErrorFromSourceRange(mess,loc,patch=patch)
      break
  if foundSynopsis is False:
    docstring.addErrorFromSourceRange("Docstring missing synopsis. Expected '{} - a very useful description here'".format(cursorName),synopsis.extent,highlight=False)
  return

def checkDocValidFunctionParameters(linter,cursor,docstring):
  fnargs = linter.getArgumentCursors(cursor)
  if len(fnargs) == 0:
    # if the function has no arguments and we have nothing to do here
    return
  try:
    params = docstring.sections.parameters
  except AttributeError:
    import ipdb; ipdb.set_trace()
    linter.addErrorFromCursor(cursor,"Function parameters are all undocumented")
    return
    raise RuntimeError("no params, but function has args; this should be handled")

  headings = [l for l in params.lines if params.isHeading(l)]
  docstring.checkValidSectionHeaderSpelling(params,headings)
  docstring.checkValidSolitarySectionHeadings(params,headings)

  requiresC    = []
  POINTER_KIND = clx.TypeKind.POINTER
  for arg in fnargs:
    canon = arg.type.get_canonical()
    if canon.kind == POINTER_KIND:
      it = 0
      while canon.kind == POINTER_KIND:
        if it > 100:
          # there is no chance that someone has a variable over 100 pointers deep, so
          # clearly something is wrong
          emess = "Ran for {} iterations (> 100) trying to get pointer type for\n{}\n".format(
            it,arg.errorViewFromCursor(arg),"\n".join(pclu.viewAstFromCursor(arg))
          )
          raise RuntimError(emess)
        canon = canon.get_pointee()
        it   += 1
      dataKind = canon.kind
      if dataKind in charTypes:
        requiresC.append((arg,"char"))
      elif dataKind in functionTypes:
        requiresC.append((arg,"function"))

  if len(requiresC) and not docstring.raw.startswith("/*@C"):
    line   = docstring.raw.split(maxsplit=1)[0]
    crange = docstring.makeSourceRange(line,line,docstring.extent.start.line)
    blame  = "\n".join("  {}. '{}' of derived type '{}' (is a {} pointer)".format(i+1,a.name,a.derivedtypename,why) for i,(a,why) in enumerate(requiresC))
    mess   = "Function requires custom fortran interface but missing 'C' from docstring header. Due to\n{}".format(blame)
    docstring.addErrorFromSourceRange(mess,crange,patch=SourceFix.fromSourceRange(crange,line+"C"))

  def argLen(text):
    return len(text[1:].split("-",maxsplit=1)[0].rstrip())+1

  fnargnames = [a.name for a in fnargs]
  fnargseen  = [False]*len(fnargs)
  notfound   = []
  for id,group in sorted(params.items.items()):
    maxArgLen = 0
    indices   = []
    remove    = set()
    for i,(loc,text) in enumerate(group.copy()):
      arg = text.split()[1].strip()
      try:
        idx = fnargnames.index(arg)
      except ValueError:
        notfound.append((arg,*group[i]))
        remove.add(i)
      else:
        fnargseen[idx] = True
        indices.append(idx)
        maxArgLen = max(maxArgLen,argLen(text))
    group = [g for i,g in enumerate(group) if i not in remove]
    for loc,text in group:
      pos = text[1:].find("-")
      if pos == -1:
        if not text or text.isspace():
          continue
        import ipdb; ipdb.set_trace()
      if pos != maxArgLen:
        alen  = argLen(text)
        fixed = "{:{}}{}".format(text[:alen],maxArgLen,text[pos:])[alen:]
        floc  = docstring.makeSourceRange(text[alen:],text,loc.end.line)
        docstring.addErrorFromSourceRange(
          "Dashes must be aligned to largest column ({}) within argument list".format(maxArgLen+1),
          docstring.makeSourceRange(text[pos:pos+3],text,loc.end.line),
          patch=SourceFix.fromSourceRange(floc,fixed)
        )

  fnargsleft = [name for seen,name in zip(fnargseen,fnargnames) if not seen]
  if notfound:
    fnargsleft2 = fnargsleft.copy()
    for arg,loc,text in notfound:
      message = "Extra docstring parameter '{}' not found in function arguments:\n{}".format(
        arg,loc.formatted(numContext=2)
      )
      closest = difflib.get_close_matches(arg,fnargsleft2,n=1)
      if closest:
        match     = closest[0]
        argcursor = [c for c in fnargs if c.name == match][0]
        message  += "\n\ndid you mean {}".format(argcursor.getFormattedBlurb())
        fnargsleft2.remove(match)
      linter.addErrorFromCursor(cursor,message)
  for arg in fnargsleft:
    docstring.addErrorFromSourceRange(
      "Undocumented parameter '{}' not found in parameter section".format(arg),params.extent,
      highlight=False
    )
  return

def checkDocValidLevel(linter,cursor,docstring):
  level = docstring.sections.level
  if not level:
    # if no level, nothing to check here, error will already have been logged
    return

  docstring.checkValidSectionHeaderSpelling(level,[l for l in level.lines if level.isHeading(l)])

  validLevels = ("beginner","intermediate","advanced","developer","deprecated")
  expected    = ", or ".join([", ".join(validLevels[:-1]),validLevels[-1]])
  newLineWithProperIndent = ":\n"+docstring.indent*" "
  for loc,levelName in level.items:
    if levelName not in validLevels:
      locase = levelName.casefold()
      if locase in validLevels:
        docstring.addErrorFromSourceRange(
          "Level subheading must be lowercase, expected '{}' found '{}'".format(locase,levelName),
          loc,patch=SourceFix.fromSourceRange(loc,locase)
        )
      else:
        closeMatches = difflib.get_close_matches(locase,validLevels,n=1)
        if closeMatches:
          match = closeMatches[0]
          mess  = "Unknown Level subheading '{}', assuming you meant '{}'".format(levelName,match)
          patch = SourceFix.fromSourceRange(loc,match)
        else:
          mess  = "Unknown Level subheading '{}', expected one of {}".format(levelName,expected)
          patch = None
        docstring.addErrorFromSourceRange(mess,loc,patch=patch)
  for loc,line in level.lines:
    continue # TODO FIX ME, need to be able to handle the below
    if line and ":" not in line:
      # if you get a "prevloc" and "prevline" not defined error here this means that we
      # are erroring out on the first trip round this loop and somehow have a
      # lone-standing 'beginner' or whatever without an explicit "Level:" line...
      errorMessage = "Level values must be on the same line as the 'Level' heading, not on separate line:\n{}".format(prevloc.mergeWith(loc).formatted(numContext=2,highlight=False))
      # This is a stupid hack to solve a multifaceted issue. Suppose you have
      # Level:
      # BLABLABLA
      # The first fix above does a tolower() transformation
      # Level:
      # blabla
      # while this fix would apply a join transformation
      # Level: BLABLA
      # See the issue already? Since we sort the transformations by line the second
      # transformation would actually end up going *first*, meaning that the lowercase
      # transformation is no longer valid for patch...

      # create a range starting at newline of previous line going until the first
      # non-space character on the next line
      delrange = PetscSourceRange.fromPositions(
        cursor.translation_unit,prevloc.end.line,-1,loc.start.line,len(line)-len(line.lstrip())
      )
      # given '  Level:\n  blabla'
      #                ^^^
      #                 |
      #              delrange
      # delete delrange from it to get '  Level: blabla'
      linter.addErrorFromCursor(cursor,errorMessage,patch=SourceFix.fromSourceRange(delrange,""))
    prevloc  = loc
    prevline = line
  return

def checkDocValidSeealso(linter,cursor,docstring):
  seealso = docstring.sections.seealso
  if not seealso:
    import ipdb; ipdb.set_trace()
    return

  def transform(text):
    return text.islower(),text.lower()

  def isHeading(line):
    return ":" in line[1] and line[1].startswith(".")

  docstring.checkValidSectionHeaderSpelling(
    seealso,list(filter(isHeading,seealso.lines)),transform=transform
  )
  return


"""Specific 'driver' function to test a particular docstring archetype"""
def checkPetscFunctionDocString(linter,function):
  try:
    docstring = PetscDocString.fromCursor(linter,function).parse()
  except ParsingError:
    return # error already logged with linter

  checkDocValidSynopsis(linter,function,docstring)
  checkDocValidFunctionParameters(linter,function,docstring)
  checkDocValidLevel(linter,function,docstring)
  checkDocValidSeealso(linter,function,docstring)
  return


checkDocMap = {
  clx.CursorKind.FUNCTION_DECL : checkPetscFunctionDocString,
}

"""utilities for checking functions"""
def alwaysTrue(*args,**kwargs):
  return True

def alwaysFalse(*args,**kwargs):
  return False

def addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName):
  __doc__="""
  shorthand for extracting a fix from a function cursor
  """
  call = [c for c in funcCursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO]
  assert len(call) == 1
  fix = SourceFix.fromCursor(call[0],validFuncName)
  linter.addErrorFromCursor(obj,"Incorrect use of {}(), use {}() instead".format(funcCursor.displayname,validFuncName),patch=fix)
  return

def convertToCorrectPetscValidLogicalCollectiveXXX(linter,obj,objType,**kwargs):
  __doc__="""
  Try to glean the correct PetscValidLogicalCollectiveXXX from the type, used as a failure hook in the validlogicalcollective checks.
  """
  validFuncName = None
  objTypeKind   = objType.kind
  if objTypeKind in scalarTypes:
    if "PetscReal" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveReal"
    elif "PetscScalar" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveScalar"
  elif objTypeKind in enumTypes:
    if "PetscBool" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveBool"
    else:
      validFuncName = "PetscValidLogicalCollectiveEnum"
  elif objTypeKind in intTypes:
    if "PetscInt" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveInt"
    elif "PetscMPIInt" in obj.derivedtypename:
      validFuncName = "PetscValidLogicalCollectiveMPIInt"
  if validFuncName:
    funcCursor = kwargs["funcCursor"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName)
    return True
  return False

def convertToCorrectPetscValidXXXPointer(linter,obj,objType,**kwargs):
  __doc__="""
  Try to glean the correct PetscValidLogicalXXXPointer from the type, used as a failure hook in the validpointer checks.
  """
  validFuncName = None
  objTypeKind   = objType.kind
  if (objTypeKind == clx.TypeKind.RECORD) or (objTypeKind == clx.TypeKind.VOID):
    # pointer to struct or void pointer, use PetscValidPointer() instead
    validFuncName = "PetscValidPointer"
  elif objTypeKind in charTypes:
    validFuncName = "PetscValidCharPointer"
  elif objTypeKind in scalarTypes:
    if "PetscReal" in obj.derivedtypename:
      validFuncName = "PetscValidRealPointer"
    elif "PetscScalar" in obj.derivedtypename:
      validFuncName = "PetscValidScalarPointer"
  elif objTypeKind in enumTypes:
    if "PetscBool" in obj.derivedtypename:
      validFuncName = "PetscValidBoolPointer"
  elif objTypeKind in intTypes:
    if ("PetscInt" in obj.derivedtypename) or ("PetscMPIInt" in obj.derivedtypename):
      validFuncName = "PetscValidIntPointer"
  if validFuncName:
    funcCursor = kwargs["funcCursor"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName)
    return True
  return False

def checkIsPetscScalarAndNotPetscReal(linter,obj,objType,**kwargs):
  __doc__="""
  Used as a success hook, since a scalar may (depending on how petsc was configured) pass the type check for reals, so we must double check the name
  """
  if "PetscScalar" not in obj.derivedtypename:
    funcCursor = kwargs["funcCursor"]
    if "PetscReal" in obj.derivedtypename:
      validFunc = kwargs["validFunc"]
      addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
    else:
      linter.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscScalars".format(funcName=funcCursor.displayname))
  return True

def checkIsPetscRealAndNotPetscScalar(linter,obj,objType,**kwargs):
  if "PetscReal" not in obj.derivedtypename:
    funcCursor = kwargs["funcCursor"]
    if "PetscScalar" in obj.derivedtypename:
      validFunc = kwargs["validFunc"]
      addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
    else:
      linter.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscReals".format(funcName=funcCursor.displayname))
  return True

def checkIntIsNotPetscBool(linter,obj,objType,**kwargs):
  if "PetscBool" in obj.derivedtypename:
    funcCursor,validFunc = kwargs["funcCursor"],kwargs["validFunc"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
  return True

def checkMPIIntIsNotPetscInt(linter,obj,objType,**kwargs):
  if "PetscInt" in obj.derivedtypename:
    funcCursor,validFunc = kwargs["funcCursor"],kwargs["validFunc"]
    addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
  return True

def checkIsPetscBool(linter,obj,objType,**kwargs):
  if ("PetscBool" not in obj.derivedtypename) and ("bool" not in obj.typename):
    funcCursor = kwargs["funcCursor"]
    linter.addErrorFromCursor(obj,"Incorrect use of {funcName}(), {funcName}() should only be used for PetscBool or bool".format(funcName=funcCursor.displayname))
  return True

def checkIsPetscObject(linter,obj):
  __doc__="""
  Returns True if obj is a valid PetscObject, otherwise False. Automatically adds the error to the linter. Raises RuntimeError if obj is a PetscObject that isn't registered in the classIdMap.
  """
  if not obj.typename.startswith("_p_"):
    linter.addErrorFromCursor(obj,"Non-PETSc type when PETSc object expected.")
    return False
  elif obj.typename not in classIdMap:
    # Raise exception here since this isn't a bad source, moreso a failure of
    # this script since it should know about all petsc classes
    errorMessage = "{}\nUnknown or invalid PETSc class '{}'. If you are introducing a new class, you must register it with this linter! See {} and search for 'Adding new classes' for more information\n".format(obj,obj.derivedtypename,PetscPath(__file__).resolve())
    raise RuntimeError(errorMessage)
  validObject = True
  pObjType    = obj.type.get_canonical().get_pointee()
  # Must have a struct here, e.g. _p_Vec
  assert pObjType.kind == clx.TypeKind.RECORD,"Symbol does not appear to be a struct!"
  objFields = [f for f in pObjType.get_fields()]
  if len(objFields) >= 2:
    if (PetscCursor.getTypenameFromCursor(objFields[0]) != "_p_PetscObject"):
      validObject = False
  else:
    validObject = False
  if not validObject:
    objDecl = PetscCursor.asPetscCursor(pObjType.get_declaration())
    if len(objFields) == 0:
      linter.addWarningFromCursor(obj,"Object '{}' is prefixed with '_p_' to indicate it is a PetscObject but cannot determine fields. Likely the header containing definition of the object is in a nonstandard place:\n\n{}\n{}".format(objDecl.typename,objDecl.getFormattedLocationString(),objDecl.getFormattedSource(nafter=2)))
    else:
      linter.addErrorFromCursor(obj,"Object '{}' is prefixed with '_p_' to indicate it is a PetscObject but its definition is missing a PETSCHEADER as the first struct member:\n\n{}\n{}".format(objDecl.typename,objDecl.getFormattedLocationString(),objDecl.getFormattedSource(nafter=2)))
  return validObject

def checkMatchingClassid(linter,obj,objClassid):
  __doc__="""
  Does the classid match the particular PETSc type
  """
  checkIsPetscObject(linter,obj)
  expectedClassid = classIdMap[obj.typename]
  if objClassid.name != expectedClassid:
    fix = SourceFix.fromCursor(objClassid,expectedClassid)
    linter.addErrorFromCursor(obj,"Classid doesn't match. Expected '{}' found '{}'".format(expectedClassid,objClassid.name),patch=fix)
  return

def checkTraceableToParentArgs(obj,parentArgNames):
  __doc__="""
  Try and see if the cursor can be linked to parent function arguments. If it can be successfully linked return the index of the matched object otherwise raises ParsingError.

  myFunction(barType bar)
  ...
  fooType foo = bar->baz;
  macro(foo,barIdx);
  /* or */
  macro(bar->baz,barIdx);
  /* or */
  initFooFromBar(bar,&foo);
  macro(foo,barIdx);
  """
  potentialParents = []
  defCursor        = obj.get_definition()
  if defCursor:
    assert defCursor.location != obj.location, "Object has definition cursor, yet the cursor did not move. This should be handled!"
    if defCursor.kind == clx.CursorKind.VAR_DECL:
      # found definition, so were in business
      # Parents here is an odd choice of words since on the very same line I loop
      # over children, but then again clangs AST has an odd semantic for parents/children
      convertOrDereferenceCursors = convertCursors|{clx.CursorKind.UNARY_OPERATOR}
      for defChild in defCursor.get_children():
        if defChild.kind in convertOrDereferenceCursors:
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
  elif obj.kind in convertCursors:
    curs = [PetscCursor(c,obj.argidx) for c in obj.walk_preorder() if c.kind == clx.CursorKind.DECL_REF_EXPR]
    if len(curs) > 1:
      curs = [c for c in curs if c.displayname == obj.name]
    assert len(curs) == 1, "Could not uniquely determine base cursor from conversion cursor {}".format(obj)
    obj = curs[0]
    # for cases with casting + struct member reference:
    #
    # macro((type *)bar->baz,barIdx);
    #
    # the object "name" will (rightly) refer to 'baz', but since this is an inline
    # "definition" it doesn't show up in get_definition(), thus we check here
    potentialParents.append(obj)
  if not potentialParents:
    # this is the if-all-else-fails approach, first we search the __entire__ file for
    # references to the cursor. Once we have some matches we take the earliest one
    # as this one is in theory where the current cursor is instantiated. Then we select
    # the best match for the possible instantiating cursor and recursively call this
    # function. This section stops when the cursor definition is of type PARM_DECL (i.e.
    # defined as the function parameter to the parent function).
    refsAll = obj.findCursorReferences()
    # don't care about uses of object __after__ the macro, and don't want to pick up
    # the actual macro location either
    refsAll = [r for r in refsAll if r.location.line < obj.location.line]
    # we just tried those and they didn't work, also more importantly weeds out the
    # instantiation line if this is an intermediate cursor in a recursive call to this
    # function
    argRefs = [r for r in refsAll if r.kind not in {clx.CursorKind.VAR_DECL,clx.CursorKind.FIELD_DECL}]
    if not len(argRefs):
      # it's not traceable to a function argument, so maybe its a global static variable
      if len([r for r in refsAll if r.storage_class in {clx.StorageClass.STATIC}]):
        # a global variable is not a function argumment, so this is unhandleable
        raise ParsingError("PETSC_CLANG_STATIC_ANALYZER_IGNORE")

    assert len(argRefs), "Could not determine the origin of cursor {}".format(obj)
    # take the first, as this is the earliest
    firstRef = argRefs[0]
    tu,loc   = firstRef.translation_unit,firstRef.location
    srcLen   = len(firstRef.getRawSource())
    # why the following song and dance? Because you cannot walk the AST backwards, and
    # in the case that the current cursor is in a function call we need to access
    # our co-arguments to the function, i.e. "adjacent" branches since they should link
    # to (or be) in the parent functions argument list. So we have to
    # essentially reparse this line to be able to start from the top.
    lineStart = PetscSourceLocation.fromPosition(tu,loc.line,1)
    lineEnd   = PetscSourceLocation.fromPosition(tu,loc.line,srcLen+1)
    lineRange = PetscSourceRange.fromLocations(lineStart,lineEnd).sourceRange
    tGroup    = list(clx.TokenGroup.get_tokens(tu,lineRange))
    funcProto = [i for i,t in enumerate(tGroup) if t.cursor.type.get_canonical().kind in functionTypes]
    if funcProto:
      assert len(funcProto) == 1, "Could not determine unique function prototype from {} for provenance of {}".format("".join([t.spelling for t in tGroup]),obj)
      idx        = funcProto[0]
      lambdaExpr = lambda t: (t.spelling != ")") and t.kind in varTokens
      iterator   = map(lambda x: x.cursor,itertools.takewhile(lambdaExpr,tGroup[idx+2:]))
    # we now have completely different cursor selected, so we recursively call this
    # function
    else:
      # not a function call, must be an assignment statement, meaning we should now
      # assert that the current obj is being assigned to
      assert PetscCursor.getNameFromCursor(tGroup[0].cursor) == obj.name
      # find the binary operator, it will contain the most comprehensive AST
      iterator = tGroup[[x.spelling for x in tGroup].index("=")].cursor.walk_preorder()
      iterator = [c for c in iterator if c.kind == clx.CursorKind.DECL_REF_EXPR]
    altCursor = [c for c in iterator if PetscCursor.getNameFromCursor(c) != obj.name]
    potentialParents.extend(altCursor)
  if not potentialParents:
    raise ParsingError
  # arguably at this point anything other than len(potentialParents) should be 1,
  # and anything else can be considered a failure of this routine (therefore a RTE)
  # as it should be able to detect the definition.
  assert len(potentialParents) == 1, "Cannot determine a unique definition cursor for object"
  # If >1 cursor, probably a bug since we should have weeded something out
  parent = potentialParents[0]
  if parent.get_definition().kind == clx.CursorKind.PARM_DECL:
    name = PetscCursor.getNameFromCursor(parent)
    try:
      loc  = parentArgNames.index(name)
    except ValueError as ve:
      # name isn't in the parent arguments, so we raise parsing error from it
      raise ParsingError from ve
  else:
    parent = PetscCursor(parent,obj.argidx)
    # deeper into the rabbit hole
    loc = checkTraceableToParentArgs(parent,parentArgNames)
  return loc

def checkMatchingArgNum(linter,obj,idx,parentArgs):
  __doc__="""
  Is the Arg # correct w.r.t. the function arguments
  """
  if idx.canonical.kind not in mathCursors:
    # sometimes it is impossible to tell if the index is correct so this is a warning not
    # an error. For example in the case of a loop:
    # for (i = 0; i < n; ++i) PetscValidIntPointer(arr+i,i);
    linter.addWarningFromCursor(idx,"Index value is of unexpected type '{}'".format(idx.canonical.kind))
    return
  try:
    idxNum = int(idx.name)
  except ValueError:
    linter.addWarningFromCursor(idx,"Potential argument mismatch, could not determine integer value")
    return
  parentArgNames = tuple(s.name for s in parentArgs)
  try:
    matchLoc = parentArgNames.index(obj.name)
  except ValueError:
    try:
      matchLoc = checkTraceableToParentArgs(obj,parentArgNames)
    except ParsingError as pe:
      # If the parent arguments don't contain the symbol and we couldn't determine a
      # definition then we cannot check for correct numbering, so we cannot do
      # anything here but emit a warning
      if "PETSC_CLANG_STATIC_ANALYZER_IGNORE" in pe.args:
        return
      if len(parentArgs):
        parentFunc = PetscCursor(parentArgs[0].semantic_parent)
        parentFuncName = parentFunc.name+"()"
        parentFuncSrc  = parentFunc.getFormattedSource()
      else:
        # parent function has no arguments (very likely that "obj" is a global variable)
        parentFuncName = "UNKNOWN FUNCTION"
        parentFuncSrc  = "  <could not determine parent function signature from arguments>"
      linter.addWarningFromCursor(obj,"Cannot determine index correctness, parent function '{}' seemingly does not contain the object:\n\n{}".format(parentFuncName,parentFuncSrc))
      return
  if idxNum != parentArgs[matchLoc].argidx:
    errMess = "Argument number doesn't match for '{}'. Found '{}' expected '{}' from\n\n{}".format(obj.name,str(idxNum),str(parentArgs[matchLoc].argidx),parentArgs[matchLoc].getFormattedSource())
    fix = SourceFix.fromCursor(idx,parentArgs[matchLoc].argidx)
    linter.addErrorFromCursor(idx,errMess,patch=fix)
  return

def checkMatchingSpecificType(linter,obj,expectedTypeKinds,pointer,unexpectedNotPointerFunction=alwaysFalse,unexpectedPointerFunction=alwaysFalse,successFunction=alwaysTrue,failureFunction=alwaysFalse,**kwargs):
  __doc__="""
  Checks that obj is of a particular kind, for example char. Can optionally handle pointers too.

  Nonstandard arguments:

  expectedTypeKinds            - the base type that you want obj to be, e.g. clx.TypeKind.ENUM
                                 for PetscBool
  pointer                      - should obj be a pointer to your type?
  unexpectedNotPointerFunction - pointer is TRUE, the object matches the base type but IS NOT
                                 a pointer
  unexpectedPointerFunction    - pointer is FALSE, the object matches the base type but IS a
                                 pointer
  successFunction              - the object matches the type and pointer specification
  failureFunction              - the object does NOT match the base type

  The hooks must return whether they handled the failure, this can mean either determining
  that the object was correct all along, or that a more helpful error message was logged
  and/or that a fix was created.
  """
  objType = obj.canonical.type.get_canonical()
  if pointer:
    if objType.kind in expectedTypeKinds:
      if not unexpectedNotPointerFunction(linter,obj,objType,**kwargs):
        linter.addErrorFromCursor(obj,"Object of clang type {} is not a pointer. Expected pointer of one of the following types: {}".format(objType.kind,expectedTypeKinds))
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
      if not unexpectedPointerFunction(linter,obj,objType,**kwargs):
        linter.addErrorFromCursor(obj,"Object of clang type {} is a pointer when it should not be".format(objType.kind))
      return
  if objType.kind in expectedTypeKinds:
    handled = successFunction(linter,obj,objType,**kwargs)
    if not handled:
      errorMessage = "{}\nType checker successfully matched object of type {} to (one of) expected types:\n- {}\n\nBut user supplied on-successful-match hook '{}' returned non-truthy value '{}' indicating unhandled error!".format(obj,objType.kind,'\n- '.join(map(str,expectedTypeKinds)),successFunction,handled,expectedTypeKinds,objType.kind)
      raise RuntimeError(errorMessage)
  else:
    if not failureFunction(linter,obj,objType,**kwargs):
      linter.addErrorFromCursor(obj,"Object of clang type {} is not in expected types: {}".format(objType.kind,expectedTypeKinds))
  return


"""Specific 'driver' function to test a particular function archetype"""
def checkObjIdxGenericN(linter,func,parent):
  __doc__="""
  For generic checks where the form is func(obj1,idx1,...,objN,idxN)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  for obj,idx in zip(funcArgs[::2],funcArgs[1::2]):
    checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidHeaderSpecificType(linter,func,parent):
  __doc__="""
  Specific check for PetscValidHeaderSpecificType(obj,classid,idx,type)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  # Don't need the type
  obj,classid,idx,_ = funcArgs
  checkMatchingClassid(linter,obj,classid)
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidHeaderSpecific(linter,func,parent):
  __doc__="""
  Specific check for PetscValidHeaderSpecific(obj,classid,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  obj,classid,idx = funcArgs
  checkMatchingClassid(linter,obj,classid)
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidPointerAndType(linter,func,parent,expectedTypes,unexpectedNotPointerFunction=alwaysFalse,unexpectedPointerFunction=alwaysFalse,successFunction=alwaysTrue,failureFunction=convertToCorrectPetscValidXXXPointer,**kwargs):
  __doc__="""
  Generic check for PetscValidXXXPointer(obj,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  obj,idx = funcArgs
  checkMatchingSpecificType(
    linter,obj,expectedTypes,True,
    unexpectedNotPointerFunction=unexpectedNotPointerFunction,
    unexpectedPointerFunction=unexpectedPointerFunction,
    successFunction=successFunction,
    failureFunction=failureFunction,
    funcCursor=func,
    **kwargs
  )
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidCharPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidCharPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,charTypes)
  return

def checkPetscValidIntPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidIntPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,intTypes,successFunction=checkIntIsNotPetscBool,validFunc="PetscValidBoolPointer")
  return

def checkPetscValidBoolPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidBoolPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,boolTypes,successFunction=checkIsPetscBool)
  return

def checkPetscValidScalarPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidScalarPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,scalarTypes,successFunction=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidRealPointer")
  return

def checkPetscValidRealPointer(linter,func,parent):
  __doc__="""
  Specific check for PetscValidRealPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,realTypes,successFunction=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidScalarPointer")
  return

def checkPetscValidLogicalCollective(linter,func,parent,expectedTypes,unexpectedNotPointerFunction=alwaysFalse,unexpectedPointerFunction=alwaysFalse,successFunction=alwaysTrue,failureFunction=convertToCorrectPetscValidLogicalCollectiveXXX,**kwargs):
  __doc__="""
  Generic check for PetscValidLogicalCollectiveXXX(pobj,obj,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  # dont need the petsc object, nothing to check there
  _,obj,idx = funcArgs
  checkMatchingSpecificType(
    linter,obj,expectedTypes,False,
    unexpectedNotPointerFunction=unexpectedNotPointerFunction,
    unexpectedPointerFunction=unexpectedPointerFunction,
    successFunction=successFunction,
    failureFunction=failureFunction,
    funcCursor=func,
    **kwargs
  )
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidLogicalCollectiveScalar(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveScalar(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,scalarTypes,successFunction=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidLogicalCollectiveReal")
  return

def checkPetscValidLogicalCollectiveReal(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveReal(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,realTypes,successFunction=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidLogicalCollectiveScalar")
  return

def checkPetscValidLogicalCollectiveInt(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveInt(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,intTypes,successFunction=checkIntIsNotPetscBool,validFunc="PetscValidLogicalCollectiveBool")
  return

def checkPetscValidLogicalCollectiveMPIInt(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveMPIInt(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,mpiIntTypes,successFunction=checkMPIIntIsNotPetscInt,validFunc="PetscValidLogicalCollectiveInt")
  return

def checkPetscValidLogicalCollectiveBool(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveBool(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,boolTypes,successFunction=checkIsPetscBool)
  return

def checkPetscValidLogicalCollectiveEnum(linter,func,parent):
  __doc__="""
  Specific check for PetscValidLogicalCollectiveEnum(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,enumTypes)
  return


checkFunctionMap = {
  "PetscValidHeaderSpecificType"       : checkPetscValidHeaderSpecificType,
  "PetscValidHeaderSpecific"           : checkPetscValidHeaderSpecific,
  "PetscValidHeader"                   : checkObjIdxGenericN,
  "PetscValidPointer"                  : checkObjIdxGenericN,
  "PetscValidCharPointer"              : checkPetscValidCharPointer,
  "PetscValidIntPointer"               : checkPetscValidIntPointer,
  "PetscValidBoolPointer"              : checkPetscValidBoolPointer,
  "PetscValidScalarPointer"            : checkPetscValidScalarPointer,
  "PetscValidRealPointer"              : checkPetscValidRealPointer,
  "PetscCheckSameType"                 : checkObjIdxGenericN,
  "PetscValidType"                     : checkObjIdxGenericN,
  "PetscCheckSameComm"                 : checkObjIdxGenericN,
  "PetscCheckSameTypeAndComm"          : checkObjIdxGenericN,
  "PetscValidLogicalCollectiveScalar"  : checkPetscValidLogicalCollectiveScalar,
  "PetscValidLogicalCollectiveReal"    : checkPetscValidLogicalCollectiveReal,
  "PetscValidLogicalCollectiveInt"     : checkPetscValidLogicalCollectiveInt,
  "PetscValidLogicalCollectiveMPIInt"  : checkPetscValidLogicalCollectiveMPIInt,
  "PetscValidLogicalCollectiveBool"    : checkPetscValidLogicalCollectiveBool,
  "PetscValidLogicalCollectiveEnum"    : checkPetscValidLogicalCollectiveEnum,
  "VecNestCheckCompatible2"            : checkObjIdxGenericN,
  "VecNestCheckCompatible3"            : checkObjIdxGenericN,
  "MatCheckPreallocated"               : checkObjIdxGenericN,
  "MatCheckProduect"                   : checkObjIdxGenericN,
  "MatCheckSameLocalSize"              : checkObjIdxGenericN,
  "MatCheckSameSize"                   : checkObjIdxGenericN,
  "PetscValidDevice"                   : checkObjIdxGenericN,
  "PetscCheckCompatibleDevices"        : checkObjIdxGenericN,
  "PetscValidDeviceContext"            : checkObjIdxGenericN,
  "PetscCheckCompatibleDeviceContexts" : checkObjIdxGenericN,
  "PetscSFCheckGraphSet"               : checkObjIdxGenericN,
}

"""Utility and pre-check setup"""
def subprocessRun(*args,**kwargs):
  __doc__="""
  lightweight wrapper to hoist the ugly version check out of the regular code, turns a subprocess.CalledProcessError into a RuntimeError with more diagnostics
  """
  import subprocess

  if sys.version_info < (3,7):
    if kwargs.pop("capture_output",None):
      kwargs.setdefault("stdout",subprocess.PIPE)
      kwargs.setdefault("stderr",subprocess.PIPE)
  try:
    output = subprocess.run(*args,**kwargs)
  except subprocess.CalledProcessError as cpe:
    emess = "Subprocess error:\nstderr:\n{}\nstdout:\n{}\n{}".format(cpe.stderr,cpe.stdout,cpe)
    raise RuntimeError(emess) from cpe
  return output

def tryToFindLibclangDir():
  __doc__="""
  Crudely tries to find libclang directory first using ctypes.util.find_library(), then llvm-config, and then finally checks a few places on macos
  """
  import ctypes.util

  llvmLibDir = ctypes.util.find_library("clang")
  if not llvmLibDir:
    try:
      output = subprocessRun(["llvm-config","--libdir"],capture_output=True,universal_newlines=True,check=True)
      llvmLibDir = output.stdout.strip()
    except FileNotFoundError:
      # FileNotFoundError: [Errno 2] No such file or directory: 'llvm-config'
      # try to find llvmLibDir by hand
      import platform

      if platform.system().casefold() == "darwin":
        try:
          output = subprocessRun(["xcode-select","-p"],capture_output=True,universal_newlines=True,check=True)
          xcodeDir = output.stdout.strip()
          if xcodeDir == "/Applications/Xcode.app/Contents/Developer": # default Xcode path
            llvmLibDir = PetscPath(xcodeDir,"Toolchains","XcodeDefault.xctoolchain","usr","lib")
          elif xcodeDir == "/Library/Developer/CommandLineTools":      # CLT path
            llvmLibDir = PetscPath(xcodeDir,"usr","lib")
        except FileNotFoundError:
          # FileNotFoundError: [Errno 2] No such file or directory: 'xcode-select'
          pass
  return llvmLibDir

def initializeLibclang(clangDir=None,clangLib=None):
  __doc__="""
  Set the required library file or directory path to initialize libclang
  """
  if not clx.conf.loaded:
    clx.conf.set_compatibility_check(True)
    if clangLib:
      clangLib = PetscPath(clangLib).resolve()
      clx.conf.set_library_file(clangLib)
    elif clangDir:
      clangDir = PetscPath(clangDir).resolve()
      clx.conf.set_library_path(clangDir)
    else:
      raise RuntimeError("Must supply either clang directory path or clang library path")
  return clangDir,clangLib

def filterCheckFunctionMap(filterChecks):
  __doc__="""
  Remove checks from checkFunctionMap if they are not in filterChecks
  """
  if filterChecks:
    global checkFunctionMap

    # note the list, this makes a copy of the keys allowing us to delete entries "in place"
    for key in list(checkFunctionMap.keys()):
      if key not in filterChecks:
        del checkFunctionMap[key]
  return

def getPetscExtraIncludes(petscDir,petscArch):
  # keep these separate, since ORDER MATTERS HERE. Imagine that for example the
  # mpiInclude dir has copies of old petsc headers, you don't want these to come first
  # in the include search path and hence override those found in petsc/include.

  # You might be thinking that seems suspiciously specific, but I was this close to filing
  # a bug report for python believing that cdll.load() was not deterministic...
  petscIncludes = []
  mpiIncludes   = []
  cxxflags      = []
  with open(PetscPath(petscDir,petscArch,"lib","petsc","conf","petscvariables"),"r") as pv:
    ccinc  = re.compile("^PETSC_CC_INCLUDES\s*=")
    mpiinc = re.compile("^MPI_INCLUDE\s*=")
    shoinc = re.compile("^MPICC_SHOW\s*=")
    cxxflg = re.compile("^CXX_FLAGS\s*=")
    for line in pv:
      if ccinc.search(line):
        petscIncludes.append(line.split("=",1)[1])
      elif mpiinc.search(line) or shoinc.search(line):
        mpiIncludes.append(line.split("=",1)[1])
      elif cxxflg.search(line):
        cxxflags.append(line.split("=",1)[1])
  cxxflags      = [l.strip().split(" ") for l in cxxflags if l]
  cxxflags      = [flag for flags in cxxflags for flag in flags if flag.startswith("-std=")]
  cxxflags      = [cxxflags[-1]] if cxxflags else [] # take only the last one
  extraIncludes = [l.strip().split(" ") for l in petscIncludes+mpiIncludes if l]
  extraIncludes = [item for sublist in extraIncludes for item in sublist if item.startswith("-I")]
  seen          = set()
  extraIncludes = [item for item in extraIncludes if not item in seen and not seen.add(item)]
  return cxxflags+extraIncludes

def getClangSysIncludes():
  __doc__="""
  Get system clangs set of default include search directories.

  Because for some reason these are hardcoded by the compilers and so libclang does not have them.
  """
  output = subprocessRun(
    ["clang","-E","-x","c++","/dev/null","-v"],
    capture_output=True,check=True,universal_newlines=True
  )
  # goes to stderr because of /dev/null
  includes = output.stderr.split("#include <...> search starts here:\n")[1]
  includes = includes.split("End of search list.")[0].replace("(framework directory)","")
  includes = includes.splitlines()
  includes = ["".join(["-I",str(PetscPath(i.strip()).resolve())]) for i in includes if i]
  return includes

def buildCompilerFlags(petscDir,petscArch,extraCompilerFlags=[],verbose=False,printPrefix="[ROOT]"):
  __doc__="""
  build the baseline set of compiler flags, these are passed to all translation unit parse attempts
  """
  miscFlags        = [
    "-DPETSC_CLANG_STATIC_ANALYZER",
    "-xc++",
    "-Wno-empty-body",
    "-Wno-writable-strings",
    "-Wno-array-bounds",
    "-Wno-nullability-completeness",
    "-fparse-all-comments",
  ]
  sysincludes      = getClangSysIncludes()
  petscIncludes    = getPetscExtraIncludes(petscDir,petscArch)
  compilerFlags    = sysincludes+miscFlags+petscIncludes+extraCompilerFlags
  if verbose: print("\n".join([printPrefix+" Compile flags:",*compilerFlags]))
  return compilerFlags

def buildPrecompiledHeader(petscDir,compilerFlags,extraHeaderIncludes=[],verbose=False,printPrefix="[ROOT]",pchClangOptions=basePCHClangOptions):
  __doc__="""
  create a precompiled header from petsc.h, and all of the private headers, this not only saves a lot of time, but is critical to finding struct definitions. Header contents are not parsed during the actual linting, since this balloons the parsing time as libclang provides no builtin auto header-precompilation like the normal compiler does.

  Including petsc.h first should define almost everything we need so no side effects from including headers in the wrong order below.
  """
  if not isinstance(petscDir,PetscPath):
    petscDir = PetscPath(petscDir).resolve()

  index             = clx.Index.create()
  precompiledHeader = petscDir/"include"/"petsc_ast_precompile.pch"
  megaHeaderLines   = [
    # Kokkos needs to go first since it mucks with complex
    ("petscvec_kokkos.hpp","#include <petscvec_kokkos.hpp>"),
    ("petsc.h","#include <petsc.h>")
  ]
  privateDirName    = petscDir/"include"/"petsc"/"private"
  megaHeaderName    = "megaHeader.hpp"

  # build a megaheader from every header in private first
  for header in privateDirName.iterdir():
    if header.suffix in (".h",".hpp"):
      megaHeaderLines.append((header.name,"#include <petsc/private/{}>".format(header.name)))

  # loop until we get a completely clean compilation, any problematic headers are discarded
  while True:
    megaHeader = "\n".join(hfi for _,hfi in megaHeaderLines)+"\n"  # extra newline for last line
    tu = index.parse(
      megaHeaderName,
      args=compilerFlags,unsaved_files=[(megaHeaderName,megaHeader)],options=pchClangOptions
    )
    diags = {}
    for diag in tu.diagnostics:
      try:
        filename = diag.location.file.name
      except AttributeError:
        # file is None
        continue
      basename,filename = os.path.split(filename)
      if filename not in diags:
        # save the problematic header name as well as its path (a surprise tool that will
        # help us later)
        diags[filename] = (basename,diag)
    for dirname,diag in tuple(diags.values()):
      # the reason this is done twice is because as usual libclang hides
      # everything in children. Suppose you have a primary header A (which might be
      # include/petsc/private/headerA.h), header B and header C. Header B and C are in
      # unknown locations and all we know is that Header A includes B which includes C.
      #
      # Now suppose header C is missing, meaning that Header A needs to be removed.
      # libclang isn't gonna tell you that without some elbow grease since that would be
      # far too easy. Instead it raises the error about header B, so we need to link it
      # back to header A.
      if dirname != privateDirName:
        # problematic header is NOT in include/petsc/private, so we have a header B on our
        # hands
        for child in diag.children:
          # child of header B here is header A not header C
          try:
            filename = child.location.file.name
          except AttributeError:
            # file is None
            continue
          # filter out our fake header
          if filename != megaHeaderName:
            # this will be include/petsc/private, headerA.h
            basename,filename = os.path.split(filename)
            if filename not in diags:
              diags[filename] = (basename,diag)
    if diags:
      diagerrs = "\n"+"\n".join(str(d) for _,d in diags.values())
      if verbose: print(printPrefix,"Included header has errors, removing",diagerrs)
      megaHeaderLines = [(hdr,hfi) for hdr,hfi in megaHeaderLines if hdr not in diags]
    else:
      break
  if extraHeaderIncludes:
    # now include the other headers but this time immediately crash on errors, let the
    # user figure out their own busted header files
    megaHeader = megaHeader+"\n".join(extraHeaderIncludes)
    if verbose: print("\n".join([printPrefix+" Mega header:",megaHeader]))
    tu = index.parse(
      megaHeaderName,
      args=compilerFlags,unsaved_files=[(megaHeaderName,megaHeader)],options=pchClangOptions
    )
    if tu.diagnostics:
      print("\n".join(map(str,tu.diagnostics)))
      raise clx.LibclangError("\n\nWarnings or errors generated when creating the precompiled header. This usually means that the provided libclang setup is faulty. If you used the auto-detection mechanism to find libclang then perhaps try specifying the location directly.")
  elif verbose:
    print("\n".join([printPrefix+" Mega header:",megaHeader]))
  precompiledHeader.unlink(missing_ok=True)
  tu.save(precompiledHeader)
  compilerFlags.extend(["-include-pch",str(precompiledHeader)])
  if verbose: print(printPrefix,"Saving precompiled header",precompiledHeader)
  return precompiledHeader


"""Main functions for root and queue processes"""
def testMain(petscDir,testPath,outputDir,patches,errorsFixed,errorsLeft,replace=False,verbose=False):
  import tempfile
  import shutil

  class TemporaryCopy(object):
    def __init__(self,fname):
      self.fname = PetscPath(fname).resolve()
      assert self.fname.exists()
      return

    def __enter__(self):
      self.tmp     = tempfile.NamedTemporaryFile(delete=True,suffix=self.fname.suffix)
      self.tmpPath = PetscPath(self.tmp.name).resolve()
      assert self.tmpPath.exists()
      shutil.copy2(str(self.fname),str(self.tmpPath))
      return self

    def __exit__(self,*args):
      PetscPath.unlink(self.origFile(),missing_ok=True)
      PetscPath.unlink(self.rejFile(),missing_ok=True)
      del self.tmp
      return

    def origFile(self):
      return self.tmpPath.append_suffix(".orig")

    def rejFile(self):
      return self.tmpPath.append_suffix(".rej")


  def test(generatedOutput,referenceFile):
    shortRefName = referenceFile.relative_to(petscDir)
    if replace:
      print("\tREPLACE",shortRefName)
      referenceFile.write_text("".join(generatedOutput))
      return
    if not referenceFile.exists():
      return "Missing reference file '{}'\n".format(referenceFile)
    return "".join(difflib.unified_diff(
      referenceFile.read_text().splitlines(True),generatedOutput,
      fromfile=str(shortRefName),tofile="Generated Output",n=0
    ))


  # sanitize the output so that it will be equal across systems
  def sanitizeOutputFile(text):
    if text is None:
      return []
    return [l.replace(str(petscDir),".") for l in text.splitlines(True)]

  def sanitizePatchFile(text):
    if text is None:
      return []
    # skip the diff header with file names
    return text.splitlines(True)[2:]

  def renamePatchFileTarget(text,newPath):
    lines    = text.splitlines(True)
    outFile  = lines[0].split()[1]
    lines[0] = lines[0].replace(outFile,str(newPath))
    lines[1] = lines[1].replace(outFile,str(newPath))
    return "".join(lines)


  patchError = {}
  rootDir    = "--directory="+petscDir.anchor
  patches    = dict(patches)
  output     = {p:["<--- FIXED --->",s,"<--- LEFT --->"] for p,s in errorsFixed}
  for path,string in errorsLeft:
    if path not in output:
      output[path] = ["<--- FIXED --->\n<--- LEFT --->"]
    output[path].append(string)
  output = {key: "\n".join(val if len(val) == 4 else val+[""]) for key,val in output.items()}
  if testPath.is_dir():
    cSuffixes = ("*.c","*.cxx","*.cpp","*.cc","*.CC")
    fileList  = [item for sublist in map(testPath.glob,cSuffixes) for item in sublist]
  else:
    fileList  = [testPath]
  for testFile in fileList:
    outputBase = outputDir/testFile.stem
    outputFile = outputBase.with_suffix(".out")
    patchFile  = outputBase.with_suffix(".patch")
    shortName  = testFile.relative_to(petscDir)

    print("\tTEST   ",shortName)

    outputErrors = [
      test(sanitizeOutputFile(output.get(testFile)),outputFile),
      test(sanitizePatchFile(patches.get(testFile)),patchFile)
    ]

    # no point in checking the patch, we have already replaced
    if not replace:
      # make sure the patch can be applied
      with TemporaryCopy(testFile) as tmpSrc:
        with tempfile.NamedTemporaryFile(delete=True,suffix=".patch") as tmpPatch:
          tmpPatchPath = PetscPath(tmpPatch.name).resolve()
          tmpPatchPath.write_text(renamePatchFileTarget(patches.get(testFile),tmpSrc.tmpPath))
          try:
            patchOutput = subprocessRun(
              ["patch",rootDir,"--strip=0","--unified","--input={}".format(tmpPatchPath)],
              check=True,universal_newlines=True,capture_output=True
            )
          except RuntimeError as re:
            emess = "Application of patch based on {} failed:\n{}\n".format(testFile,str(re))
            rej   = tmpSrc.rejFile()
            if rej.exists():
              emess += "\n{}:\n{}".format(rej,rej.read_text())
            outputErrors.append(emess)

    outputErrors = [e for e in outputErrors if e]
    if outputErrors:
      print("\tNOT OK ",shortName)
      patchError[testFile] = "\n".join(outputErrors)
    else:
      print("\tOK     ",shortName)
  if patchError:
    errBars = "".join(["[ERROR]",85*"-","[ERROR]"])
    errBars = [errBars+"\n",errBars]
    for errFile in patchError:
      print(patchError[errFile].join(errBars))
    return 21
  return 0

def queueMain(clangLib,checkFunctionMapU,classIdMapU,compilerFlags,clangOptions,verbose,werror,errorQueue,returnQueue,fileQueue,lock):
  __doc__="""
  main function for worker processes in the queue, does pretty much the same thing the main process would do in their place
  """
  def updateGlobals(updatedCheckFunctionMap,updatedClassIdMap):
    global checkFunctionMap,classIdMap # in a function so the "globalness" doesn't leak
    checkFunctionMap = updatedCheckFunctionMap
    classIdMap       = updatedClassIdMap
    return

  def lockPrint(*args,**kwargs):
    if verbose:
      with lock:
        print(*args,**kwargs)
    return

  # in case errors are thrown before setup is complete
  errorPrefix = "[UNKNOWN_CHILD]"
  filename    = "QUEUE SETUP"
  try:
    updateGlobals(checkFunctionMapU,classIdMapU)
    proc        = mp.current_process().name
    printPrefix = proc+" --"[:len("[ROOT]")-len(proc)]
    errorPrefix = " ".join([printPrefix,"Exception detected while processing"])
    lockPrint(printPrefix,15*"=","Performing setup",15*"=")
    initializeLibclang(clangLib=clangLib)
    linter = PetscLinter(compilerFlags,clangOptions=clangOptions,prefix=printPrefix,verbose=verbose,werror=werror,lock=lock)
    lockPrint(printPrefix,15*"=","Entering queue  ",15*"=")
    while True:
      filename = fileQueue.get()
      if filename == WorkerPool.QueueSignal.EXIT_QUEUE:
        fileQueue.task_done()
        break
      errLeft,errFixed,warnings,patches = linter.parse(filename).diagnostics()
      returnQueue.put((WorkerPool.QueueSignal.UNIFIED_DIFF,patches))
      returnQueue.put((WorkerPool.QueueSignal.ERRORS_LEFT ,errLeft))
      returnQueue.put((WorkerPool.QueueSignal.ERRORS_FIXED,errFixed))
      returnQueue.put((WorkerPool.QueueSignal.WARNING     ,warnings))
      fileQueue.task_done()
    lockPrint(printPrefix,15*"=","Exiting queue   ",15*"=")
  except:
    try:
      # attempt to send the traceback back to parent
      import traceback
      preamble = " ".join([errorPrefix,str(filename)])
      errorQueue.put("\n".join([preamble,traceback.format_exc()]))
    except Exception as e:
      # if this fails then I guess we really are screwed
      errorQueue.put("[UNKNOWN CHILD] UNKNOWN ERROR\n{}\n".format(e))
    finally:
      try:
        # in case we had any work from the queue we need to release it but only after
        # putting our exception on the queue
        fileQueue.task_done()
      except ValueError:
        # task_done() called more times than get(), means we threw before getting the
        # filename
        pass
  errorQueue.close()
  returnQueue.close()
  return

def main(petscDir,petscArch,srcPath=None,clangDir=None,clangLib=None,verbose=False,workers=-1,checkFunctionFilter=None,patchDir=None,applyPatches=False,extraCompilerFlags=[],extraHeaderIncludes=[],testOutputDir=None,replaceTests=False,werror=False):
  __doc__="""
  entry point for linter

  Positional arguments:
  petscDir  -- $PETSC_DIR
  petscArch -- $PETSC_ARCH

  Keyword arguments:
  srcPath             -- alternative directory (or single file) to use as src root (default: $PETSC_DIR/src)
  clangDir            -- directory containing libclang.[so|dylib|dll] (default: None)
  clangLib            -- direct path to libclang.[so|dylib|dll], overrrides clangDir if set (default: None)
  verbose             -- display debugging statements (default: False)
  workers             -- number of processes for multiprocessing, -1 is number of system CPU's-1, 0 or 1 for serial computation (default: -1)
  checkFunctionFilter -- list of function names as strings to only check for, none == all of them. For example ["PetscValidPointer","PetscValidHeaderSpecific"] (default: None)
  patchDir            -- directory to store patches if they are generated (default: $PETSC_DIR/petscLintPatches)
  applyPatches        -- automatically apply patch files to source if they are generated (default: False)
  extraCompilerFlags  -- list of extra compiler flags to append to petsc and system flags. For example ["-I/my/non/standard/include","-Wsome_warning"] (default: None)
  extraHeaderIncludes -- list of #include statements to append to the precompiled mega-header, these must be in the include search path. Use extraCompilerFlags to make any other search path additions. For example ["#include <slepc/private/epsimpl.h>"] (default: None)
  testOutputDir       -- directory containing test output to compare patches against, use special keyword '__at_src__' to use srcPath/output (default: None)
  replaceTests        -- replace output files in testOutputDir with patches generated (default: False)
  werror              -- treat all linter-generated warnings as errors (default: False)
  """

  # pre-processing setup
  if bool(applyPatches) and bool(testOutputDir):
    raise RuntimeError("Test directory and apply patches are both non-zero. It is probably not a good idea to apply patches over the test directory!")
  clangDir,clangLib = initializeLibclang(clangDir=clangDir,clangLib=clangLib)
  petscDir = PetscPath(petscDir).resolve()
  srcPath  = petscDir/"src" if srcPath is None else PetscPath(srcPath).resolve()
  patchDir = petscDir/"petscLintPatches" if patchDir is None else PetscPath(patchDir).resolve()
  if testOutputDir == "__at_src__":
    if srcPath.is_dir():
      testOutputDir = srcPath/"output"
    elif srcPath.is_file():
      testOutputDir = srcPath.parent/"output"
    else:
      raise RuntimeError("Got neither a directory or file as srcPath",srcPath)

  if testOutputDir is not None:
    assert testOutputDir.exists(), "Test Output Directory {} does not appear to exist".format(testOutputDir)

  filterCheckFunctionMap(checkFunctionFilter)
  rootPrintPrefix   = "[ROOT]"
  compilerFlags     = buildCompilerFlags(
    petscDir,petscArch,
    extraCompilerFlags=extraCompilerFlags,verbose=verbose,printPrefix=rootPrintPrefix
  )
  precompiledHeader = buildPrecompiledHeader(
    petscDir,compilerFlags,extraHeaderIncludes=extraHeaderIncludes,verbose=verbose
  )

  pool = WorkerPool(numWorkers=workers,verbose=verbose)
  pool.setup(compilerFlags,werror=werror).walk(srcPath)
  warnings,errorsLeft,errorsFixed,patches = pool.finalize()
  if verbose: print(rootPrintPrefix,"Deleting precompiled header",precompiledHeader)
  precompiledHeader.unlink()
  if testOutputDir is not None:
    return testMain(
      petscDir,srcPath,testOutputDir,patches,errorsFixed,errorsLeft,
      replace=replaceTests,verbose=verbose
    )
  elif patches:
    import time

    patchDir.mkdir(exist_ok=True)
    manglePostfix = "".join(["_",str(int(time.time())),".patch"])
    rootDir       = "--directory="+patchDir.anchor
    for fname,patch in patches:
      mangledRel = fname.append_name(manglePostfix)
      if mangledRel.parent != srcPath.parent: # not in same directory
        mangledRel = mangledRel.relative_to(srcPath)
      mangledFile = patchDir/str(mangledRel).replace(os.path.sep,"_")
      if verbose: print(rootPrintPrefix,"Writing patch to file",mangledFile)
      mangledFile.write_text(patch)
    if applyPatches:
      if verbose: print(rootPrintPrefix,"Applying patches from patch directory",patchDir)
      for patchFile in patchDir.glob("*"+manglePostfix):
        if verbose: print(rootPrintPrefix,"Applying patch",patchFile)
        output = subprocessRun(
          ["patch",rootDir,"--strip=0","--unified","--input=".format(patchFile)],
          check=True,universal_newlines=True,capture_output=True
        )
        if verbose: print(output.stdout)
  ret       = 0
  formatStr = " ".join([rootPrintPrefix,"{:=^85}"])
  if warnings and verbose:
    print(formatStr.format(" Found Warnings "))
    print("\n".join(s for tup in warnings for _,s in tup))
    print(formatStr.format(" End warnings "))
  if errorsFixed and verbose:
    print(formatStr.format(" Fixed Errors " if applyPatches else " Fixable Errors "))
    print("\n".join(e for _,e in errorsFixed))
    print(formatStr.format(" End Fixed Errors "))
  if errorsLeft:
    print(formatStr.format(" Unfixable Errors "))
    print("\n".join(e for _,e in errorsLeft))
    print(formatStr.format(" End Unfixable Errors "))
    print("Some errors or warnings could not be automatically corrected via the patch files")
    ret = 11
  if patches:
    if applyPatches:
      print("All fixable errors or warnings successfully patched")
    else:
      print("Patch files written to",patchDir)
      print("Apply manually using:")
      print("  patch {} --strip=0 --unified --input={}".format(rootDir,patchDir/("*"+manglePostfix)))
      if ret != 0:
        ret = 12
  return ret


if __name__ == "__main__":
  import argparse
  def str2bool(v):
    if isinstance(v,bool):
      return v
    v = v.casefold()
    if v in {"yes","true","t","y","1"}:
      return True
    elif v in {"no","false","f","n","0",""}:
      return False
    else:
      raise argparse.ArgumentTypeError("Boolean value expected, got '{}'".format(v))

  clangDir = tryToFindLibclangDir()
  try:
    petscDir      = os.environ["PETSC_DIR"]
    defaultSrcDir = PetscPath(petscDir).resolve()/"src"
  except KeyError:
    petscDir      = None
    defaultSrcDir = "$PETSC_DIR/src"
  try:
    petscArch = os.environ["PETSC_ARCH"]
  except KeyError:
    petscArch = None

  parser = argparse.ArgumentParser(description="set options for clang static analysis tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  grouplibclang = parser.add_argument_group(title="libclang location settings")
  group = grouplibclang.add_mutually_exclusive_group(required=False)
  group.add_argument("--clang_dir",nargs="?",help="directory containing libclang.[so|dylib|dll], if not given attempts to automatically detect it via llvm-config",default=clangDir,dest="clangdir")
  group.add_argument("--clang_lib",nargs="?",help="direct location of libclang.[so|dylib|dll], overrides clang directory if set",dest="clanglib")
  grouppetsc = parser.add_argument_group(title="petsc location settings")
  grouppetsc.add_argument("--PETSC_DIR",required=False,default=petscDir,help="if this option is unused defaults to environment variable $PETSC_DIR",dest="petscdir")
  grouppetsc.add_argument("--PETSC_ARCH",required=False,default=petscArch,help="if this option is unused defaults to environment variable $PETSC_ARCH",dest="petscarch")
  parser.add_argument("-s","--src-dir",required=False,default=defaultSrcDir,help="Alternate base directory of source tree (e.g. $SLEPC_DIR/src)",dest="src")
  parser.add_argument("-v","--verbose",required=False,type=str2bool,nargs="?",const=True,default=False,help="verbose progress printed to screen")
  filterFuncChoices = ", ".join(list(checkFunctionMap.keys()))
  parser.add_argument("-f","--functions",required=False,nargs="+",choices=list(checkFunctionMap.keys()),metavar="FUNCTIONNAME",help="filter to display errors only related to list of provided function names, default is all functions. Choose from available function names: "+filterFuncChoices,dest="funcs")
  parser.add_argument("-j","--jobs",required=False,type=int,const=-1,default=-1,nargs="?",help="number of multiprocessing jobs, -1 means number of processors on machine")
  parser.add_argument("-p","--patch-dir",required=False,help="directory to store patches in if they are generated, defaults to SRC_DIR/../petscLintPatches",dest="patchdir")
  parser.add_argument("-a","--apply-patches",required=False,type=str2bool,nargs="?",const=True,default=False,help="automatically apply patches that are saved to file",dest="apply")
  parser.add_argument("--CXXFLAGS",required=False,nargs="+",default=[],help="extra flags to pass to CXX compiler",dest="cxxflags")
  parser.add_argument("--test",required=False,nargs="?",const="__at_src__",help="test the linter for correctness. Optionally provide a directory containing the files against which to compare patches, defaults to SRC_DIR/output if no argument is given. The files of correct patches must be in the format [path_from_src_dir_to_testFileName].out")
  parser.add_argument("--replace",required=False,type=str2bool,nargs="?",const=True,default=False,help="replace output files in test directory with patches generated")
  parser.add_argument("--werror",required=False,type=str2bool,nargs="?",const=True,default=False,help="treat all warnings as errors")
  args = parser.parse_args()

  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.clanglib:
    args.clangdir = None

  ret = main(args.petscdir,args.petscarch,srcPath=args.src,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,workers=args.jobs,checkFunctionFilter=args.funcs,patchDir=args.patchdir,applyPatches=args.apply,extraCompilerFlags=args.cxxflags,testOutputDir=args.test,replaceTests=args.replace,werror=args.werror)
  sys.exit(ret)
