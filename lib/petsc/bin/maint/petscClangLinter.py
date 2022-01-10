#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
try:
  import clang.cindex as clx
except ModuleNotFoundError as mnfe:
  if mnfe.name == "clang":
    raise RuntimeError("Must run e.g. 'python -m pip install clang' to use linter") from mnfe
  raise mnfe # whatever it is they should know about it

import os
import sys
import enum
import itertools
import collections
import ctypes
import difflib
import re
import weakref
import operator
import multiprocessing as mp
import multiprocessing.queues
import petscClangLinterUtil as pclu
from petscClangLinterUtil import (
  CXTranslationUnit,Scope,PetscSourceLocation,PetscSourceRange,
  PetscCXCursorAndRangeVisitor,CXCursorAndRangeVisitorCallBackProto,PetscPath,ParsingError,
  Diagnostic,DiagnosticManager
)

# clang options used for parsing files
baseClangOptions = (
  CXTranslationUnit.PrecompiledPreamble |
  CXTranslationUnit.SkipFunctionBodies  |
  CXTranslationUnit.LimitSkipFunctionBodiesToPreamble
)

# clang options for creating the precompiled megaheader
basePCHClangOptions = (
  CXTranslationUnit.CreatePreambleOnFirstParse |
  CXTranslationUnit.Incomplete                 |
  CXTranslationUnit.ForSerialization           |
  CXTranslationUnit.KeepGoing
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

class PetscDocStringBase(object):
  @staticmethod
  def diagnosticFlag(text,prefix="doc"):
    if isinstance(text,str):
      return collections.deque((prefix,text) if prefix == "doc" else ("doc",prefix,text))
    if not isinstance(text,collections.deque):
      text = collections.deque(text)
    if not text[0].startswith(prefix):
      text.appendleft(prefix)
    return text


@DiagnosticManager.register(
  ("internal-linkage","Verify that symbols with internal linkage don't have docstrings"),
  ("sowing-chars","Verify that sowing begin and end indicators match the symbol type"),
  ("spacing","Verify that dosctrings occur immediately above that which they describe"),
  ("indentation","Verify that docstring text is correctly indented"),
  ("section-spacing","Verify that there section headers are separated by at least 1 empty line"),
  ("section-header-solitary","Verify that qualifying section headers are alone on their line"),
  ("section-header-spelling","Verify section headers are correctly spelled"),
  ("section-header-maybe-header","Check for lines that seem like they are supposed to be headers"),
  ("section-header-fishy-header","Check for headers that seem like they should not be headers")
)
class PetscDocString(PetscDocStringBase):
  """
  Container to encapsulate a sowing docstring and retrieve various objects for it.
  Essentially a PetscCursor for comments.
  """

  class SectionBase(object):
    """Container for a single section of the docstring, has members:
    'name'     - the name of this section
    'required' - is this section required in the docstring
    'titles'   - header-titles, i.e. "Input Parameter", or "Level", must be spelled correctly
    'keywords' - keywords to help match an unknown title to a section
    'raw'      - the raw text in the section
    'extent'   - the SourceRange for the whole section
    'lines'    - a tuple of each line of text and its SourceRange in the section
    'items'    - a tuple of extracted tokens of interest, e.g. the level value, options parameters,
                 function parameters, etc.
    """
    __slots__ = "name","required","titles","keywords","raw","extent","lines","items"

    def __init__(self,name,required=False,keywords=None,titles=None):
      assert isinstance(name,str)
      if titles is None:
        titles = (name.title(),)
      if keywords is None:
        keywords = (name.title(),)
      self.name     = name
      self.required = required
      self.titles   = tuple(titles)
      self.keywords = tuple(set(tuple(keywords)+self.titles))
      self.clear()
      return

    def __str__(self):
      return "\n".join([
        "Type:   {}".format(type(self)),
        "Name:   {}".format(self.name),
        "Extent: {}".format(self.extent),
      ])

    def __bool__(self):
      return bool(self.lines)


    def clear(self):
      self.raw    = ""
      self.extent = None
      self.lines  = []
      return

    def fill(self,data):
      assert len(data)
      self.lines.extend(list(data))
      self.raw    = "\n".join(s for _,s in self.lines)
      self.extent = PetscSourceRange.fromLocations(self.lines[0][0].start,self.lines[-1][0].end)
      return

    def setup(self,docstring,inspectLine=None):
      if not self:
        if self.required:
          diag = self.diags.section_header_missing
          mess = "Required section '{}' not found".format(self.titles[0])
          docstring.addErrorFromSourceRange(diag,mess,docstring.extent,highlight=False)
        return

      inspect   = inspectLine is not None
      isHeading = docstring._getIsHeading(self)
      seen      = collections.defaultdict(list)

      for loc,line in self.lines:
        if isHeading(line):
          possibleHeader = line.split(":" if ":" in line else None,maxsplit=1)[0].strip()
          seen[possibleHeader.casefold()].append(
            docstring.makeSourceRange(possibleHeader,line,loc.start.line)
          )
        if inspect:
          # let each section type determine if this line is useful
          inspectLine(loc,line)

      uniqueHeadDiag = self.diags.section_header_unique
      for heading,where in seen.items():
        if len(where) <= 1:
          continue
        lasti   = len(where)-1
        srclist = []
        nbefore = 2
        nafter  = 0
        for i,loc in enumerate(where):
          startline = loc.start.line
          if i:
            nbefore = startline-prevLineBegin-1
            if i == lasti:
              nafter = 2
          srclist.append(loc.formatted(numBeforeContext=nbefore,numAfterContext=nafter,trim=False))
          prevLineBegin = startline
        mess = "Multiple '{}' subheadings. Much like Highlanders, there can only be one:\n{}".format(self.transform(self.name),"\n".join(srclist))
        docstring._linter.addErrorFromCursor(docstring.cursor,Diagnostic(uniqueHeadDiag,mess))
      return


    @staticmethod
    def transform(text):
      return text.title()

    @staticmethod
    def checkIndentAllowed():
      return True

    @staticmethod
    def diagnostic(flag,prefix):
      return PetscDocStringBase.diagnosticFlag(flag,prefix=prefix)


  @DiagnosticManager.register(("section-header-missing",""),("section-header-unique",""))
  class DefaultSection(SectionBase):
    @classmethod
    def diagnostic(cls,flag):
      return super().diagnostic(flag,"")

  @DiagnosticManager.register(
    ("section-header-missing",""),
    ("section-header-unique",""),
    ("matching-symbol-name","Verify that description matches the symbol name"),
    ("missing-description","Verify that a synopsis has a description"),
  )
  class Synopsis(SectionBase):
    @classmethod
    def diagnostic(cls,flag):
      return super().diagnostic(flag,"synopsis")

    def setup(self,ds,*args,**kwargs):
      found = False
      items = []

      def genericInspector(loc,line):
        nonlocal found
        if found: return
        pre,dash,rest = line.partition("-")
        if dash and rest:
          item = pre.strip()
          items.append((ds.makeSourceRange(item,line,loc.start.line),item))
          found = True
        return

      def enumInspector(loc,line):
        genericInspector(loc,line) # find the symbol name
        lstr = line.lstrip()
        # check that "-" is in the line since some people like to use entire blocks of $'s
        # to describe a single enum value...
        if lstr.startswith("$") and "-" in lstr:
          assert len(items) # we should have already found the symbol name
          name = lstr[1:].split(maxsplit=1)[0].strip()
          items.append((ds.makeSourceRange(name,line,loc.start.line),line))
        return


      isEnum = ds.cursor.type.kind in enumTypes
      super().setup(ds,*args,inspectLine=enumInspector if isEnum else genericInspector,**kwargs)

      if isEnum:
        def checkEnumStartsWithDollar(ds,items):
          checkOptStartsWith = PetscDocString.ParameterList.checkOptStartsWith
          for key,opts in sorted(items.items()):
            if len(opts) < 1:
              mess = "number of options {} < 1, key: {}, items: {}".format(len(opts),key,items)
              raise RuntimeError(mess)
            else:
              for opt in opts:
                checkOptStartsWith(ds,opt,"Enum","$")
            return items


        params = PetscDocString.ParameterList("enum params",prefixes=("$"))
        paramLines = items[1:]
        if not paramLines: # enum has no explicit descriptions
          self.items = None
          return
        params.fill(paramLines)
        params.setup(ds,*args,parameterListPrefixCheck=checkEnumStartsWithDollar,**kwargs)
        # shuffle the enum values up
        params.items = {k+1 : v for k,v in params.items.items()}
        # reinsert the heading
        params.items[0] = items[0]
        self.items      = params
      else:
        self.items = tuple(items)
      return

  @DiagnosticManager.register(
    ("section-header-missing",""),
    ("section-header-unique",""),
    ("formatting","Verify that parameter list entries are correctly white-space formatted"),
    ("prefix","Verify that parameter list entries begin with the correct prefix"),
    ("missing-description","Verify that parameter list entries have a description"),
  )
  class ParameterList(SectionBase):
    __slots__ = "prefixes"

    def __init__(self,*args,prefixes=("+",".","-"),**kwargs):
      self.prefixes = prefixes
      super().__init__(*args,**kwargs)
      return

    @classmethod
    def diagnostic(cls,flag):
      return super().diagnostic(flag,"param-list")

    @staticmethod
    def checkIndentAllowed():
      return False

    @staticmethod
    def splitParam(text):
      """
      retrieve groups '(+)\s*([A-z,-]+)- (.*)'
      """
      stripped = text.strip()
      prefix   = stripped[0]
      rest     = stripped[1:]
      arg,sep,descr = rest.partition(" - ")
      if not sep:
        if rest.endswith("-"):
          arg = rest[:-1]
        elif "- " in rest:
          arg,_,descr = rest.partition("- ")
        # if we hit neither then there is no "-" in text, possible case of "[prefix] foo"?
      return prefix,arg.strip(),descr.lstrip()

    @staticmethod
    def prefixOffset(text,char="-"):
      stripped = text.lstrip()
      return text.find(char)+1 if stripped.startswith(char) else 0

    @classmethod
    def arglen(cls,text,char="-"):
      """
      return a length l such that text[:l] returns all text up until the end of the arg name
      """
      _,param,_ = cls.splitParam(text)
      assert param, "Could not identify a parameter in {}".format(text)
      return sum(map(len,text.partition(param)[:2]))

    @classmethod
    def checkAlignedDescriptions(cls,ds,group,maxArgLen=None,**kwargs):
      """
      Verify that the position of the '-' before the description for each argument is aligned
      to maxArgLen+1 columns
      """
      baseMess   = "Argument list entry must be formatted as ^(\S) (.*) - (.*),"
      alignDiag  = cls.diags.formatting
      splitParam = cls.splitParam

      if maxArgLen is None:
        maxArgLen = max(len(splitParam(text)[1]) for _,text,_ in group) if group else 0
      assert maxArgLen >= 0, "Negative maximum argument length {}".format(maxArgLen)

      for loc,text,_ in group:
        pre,arg,descr = splitParam(text)
        fixed         = "{} {:{width}} - {}".format(pre,arg,descr,width=maxArgLen)
        try:
          diffIndex = [i for i,(a1,a2) in enumerate(zip(text,fixed)) if a1 != a2][0]
        except IndexError:
          assert text == fixed # equal
          continue

        if diffIndex <= text.find(pre):
          mess = " ".join([
            baseMess,"prefix ('{}') is not indented to column (1)".format(pre)
          ])
        elif diffIndex <= text.find(arg):
          mess = " ".join([
            baseMess,"argument ('{}') must be 1 space from prefix ('{}')".format(arg,pre)
          ])
        else:
          mess = " ".join([
            baseMess,"description ('{}') must be 1 space from argument ('{}')".format(descr,arg)
          ])

        eloc = ds.makeSourceRange(text[diffIndex:],text,loc.end.line)
        ds.addErrorFromSourceRange(alignDiag,mess,eloc,patch=Patch(eloc,fixed[diffIndex:]))
      return

    @classmethod
    def checkOptStartsWith(cls,ds,item,descr,char):
      loc,line,_ = item
      pre,_,_    = cls.splitParam(line)
      if pre != char:
        eloc = ds.makeSourceRange(pre,line,loc.start.line)
        mess = "{} parameter list entry must start with '{}'".format(descr,char)
        ds.addErrorFromSourceRange(cls.diags.prefix,mess,eloc,patch=Patch(eloc,char))
      return

    def setup(self,ds,*args,parameterListPrefixCheck=None,**kwargs):
      subheading       = 0
      groups           = collections.defaultdict(list)
      missingDescrDiag = self.diags.missing_description
      isHeading        = ds._getIsHeading(self)

      def inspector(loc,line):
        if not line or line.isspace():
          return
        elif isHeading(line) and len(groups.keys()):
          nonlocal subheading
          subheading += 1
        lstp = line.lstrip()
        if lstp.startswith(self.prefixes):
          _,arg,descr = self.splitParam(lstp)
          if not descr:
            mess = "Parameter list entry missing a description. Expected '{} - a very useful description'".format(arg)
            ds.addErrorFromSourceRange(missingDescrDiag,mess,loc)
          groups[subheading].append((loc,line,self.arglen(line)))
        return

      def parameterListDefaultPrefixCheck(ds,items):
        checkOptStartsWith = self.checkOptStartsWith
        for key,opts in sorted(items.items()):
          lopts = len(opts)
          if lopts < 1:
            mess = "number of options {} < 1, key: {}, items: {}".format(lopts,key,items)
            raise RuntimeError(mess)
          elif lopts == 1:
            # only 1 option, should start with '.'
            checkOptStartsWith(ds,opts[0],"Solitary",".")
          else:
            # more than 1, should be "+", then however many ".", then last is "-"
            checkOptStartsWith(ds,opts[0],"First multi","+")
            for opt in opts[1:-1]:
              checkOptStartsWith(ds,opt,"Multi",".")
            checkOptStartsWith(ds,opts[-1],"Last multi","-")
        return items


      if parameterListPrefixCheck is None:
        parameterListPrefixCheck = parameterListDefaultPrefixCheck

      super().setup(ds,*args,inspectLine=inspector,**kwargs)

      self.items = parameterListPrefixCheck(ds,dict(groups))
      return

  @DiagnosticManager.register(("section-header-missing",""),("section-header-unique",""))
  class Prose(SectionBase):
    @classmethod
    def diagnostic(cls,flag):
      return super().diagnostic(flag,"prose")

    def setup(self,ds,*args,**kwargs):
      isHeading  = ds._getIsHeading(self)
      subheading = 0
      items      = {}

      def inspector(loc,line):
        if isHeading(line):
          head,_,rest = line.partition(":")
          head        = head.strip()
          if not head:
            import ipdb; ipdb.set_trace()
          if items.keys():
            nonlocal subheading
            subheading += 1
          startLine = loc.start.line
          items[subheading] = (
            (ds.makeSourceRange(head,line,startLine),head),
            [(ds.makeSourceRange(rest,line,startLine),rest)] if rest else []
          )
        elif line:
          try:
            items[subheading][1].append((loc,line))
          except KeyError:
            import ipdb; ipdb.set_trace()
        return

      super().setup(ds,*args,inspectLine=inspector,**kwargs)
      self.items = items
      return

  @DiagnosticManager.register(("section-header-missing",""),("section-header-unique",""))
  class SourceCode(SectionBase):
    @classmethod
    def diagnostic(cls,flag):
      return super().diagnostic(flag,"source-code")

    def setup(self,ds,*args,**kwargs):
      items = {}
      if self:
        startline  = self.extent.start.line
        subheading = -1
        dollars    = False

      def inspector(loc,line):
        nonlocal subheading,dollars
        lstrp = line.lstrip()
        if line.startswith((".vb","$")):
          if line.startswith("$"):
            dollars = True
          else:
            assert not dollars, "Mixing verbatim blocks and dollars?"
          subheading += 1
          items[subheading] = [dollars,loc.start.line-startline]
        elif line.startswith(".ve"):
          assert len(items[subheading]) == 2
          items[subheading].append(loc.start.line-startline+1)
        elif dollars:
          items[subheading].append(loc.start.line-startline)
          dollars = False
        return

      super().setup(ds,*args,**kwargs,inspectLine=inspector)

      self.items = items
      if 0: # TODO, think of checks to do for source code
        for blockno,(dollars,begin,end) in self.items.items():
          first,*interior,last = self.lines[begin:end]
          if dollars:
            pass
          else:
            pass
      return

  @DiagnosticManager.register(
    ("section-header-missing",""),("section-header-unique",""),
    ("formatting","Verify that inline lists are correctly white-space formatted")
  )
  class InlineList(SectionBase):
    __slots__ = "foundEmptyLine"

    def __init__(self,*args,**kwargs):
      super().__init__(*args,**kwargs)
      self.foundEmptyLine = False
      return

    @classmethod
    def diagnostic(cls,flag):
      return super().diagnostic(flag,"inline-list")

    @staticmethod
    def checkIndentAllowed():
      return False

    def clear(self,*args,**kwargs):
      super().clear(*args,**kwargs)
      self.foundEmptyLine = False
      return

    def setup(self,ds,*args,**kwargs):
      items           = []
      formatDiag      = self.diags.formatting
      makeSourceRange = ds.makeSourceRange
      alignMessBase   = "".join((
        self.transform(self.name)," values must be (1) space away from colon not ({})"
      ))
      titles = set(t.casefold() for t in self.titles)

      def inspector(loc,line):
        rest = (line.split(":",maxsplit=2)[1] if ":" in line else line).strip()
        if rest:
          if ":" not in rest:
            # try and see if this is one of the bad-egg lines where the heading is missing
            # the colon
            badTitle = next(filter(lambda t:t.casefold() in titles,rest.split()),None)
            if badTitle:
              # kind of a hack, we just erase the bad heading with whitespace so it isnt
              # picked up below in the item detection
              rest = rest.replace(badTitle," "*len(badTitle))
          startline = loc.start.line
          offset    = 0
          for sub in filter(bool,map(lambda string: string.strip(),rest.split(","))):
            subloc = makeSourceRange(sub,line,startline,offset=offset)
            offset = subloc.end.column-1
            items.append((subloc,sub))

          colonIdx = line.find(":")
          if colonIdx >= 0:
            correctOffset = colonIdx+2
            restIdx       = line.find(rest)
            if restIdx != correctOffset:
              nspaces = restIdx-correctOffset
              if restIdx > correctOffset:
                sub    = " "*nspaces
                offset = correctOffset
                fix    = ""
              else:
                sub    = ":"
                offset = 0
                fix    = " "
                import ipdb; ipdb.set_trace()
              mess = alignMessBase.format(nspaces+1)
              floc = makeSourceRange(sub,line,startline,offset=offset)
              ds.addErrorFromSourceRange(formatDiag,mess,floc,patch=Patch(floc,fix))
        return


      super().setup(ds,*args,inspectLine=inspector,**kwargs)
      self.items = tuple(items)
      return


  class Sections(object):
    __slots__ = "_sections","_findcache","_cachekey"

    def __init__(self,*args):
      assert len({s.name for s in args}) == len(args)
      sections = collections.OrderedDict()
      for section in args:
        sections[section.name] = section
      self._sections = sections
      self._resetCache()
      return

    def __getattr__(self,attr):
      sections = self._sections
      try:
        return sections[attr]
      except KeyError as ke:
        try:
          return sections[attr.replace("_"," ")]
        except KeyError:
          pass
        raise AttributeError from ke

    def __iter__(self):
      yield from self._sections.values()

    def __contains__(self,section):
      return self.registered(section)


    def _resetCache(self):
      self._cachekey  = tuple(self._sections.keys())
      self._findcache = {self._cachekey : {}}
      return


    def guessHeading(self,heading):
      return guess

    def find(self,heading):
      sections = self._sections
      cache    = self._findcache[self._cachekey]
      try:
        return sections[cache[heading]]
      except KeyError:
        pass
      sectionNames = sections.keys()
      get_matches  = difflib.get_close_matches
      try:
        match = get_matches(heading,sectionNames,n=1)[0]
        reason = "name" # delete me
      except IndexError:
        keywords = [(kw,s.name) for kw,s in self.keywords(sections=True)]
        kwOnly   = [k for k,_ in keywords]
        try:
          match = get_matches(heading,kwOnly,n=1)[0]
          reason = "keyword" # delete me
        except IndexError:
          # try if we can find a sub-word
          match = None
          # if heading splits into more than 3 params, then chances are its being mislabeled
          # as a heading anyways
          for head in heading.split(maxsplit=3):
            try:
              # higher cutoff, we have to be pretty sure of a match when using subwords,
              # because it's a lot easier for false positives
              match = get_matches(head,kwOnly,n=1,cutoff=0.8)[0]
            except IndexError:
              continue
            reason = "subword" # delete me
            break
        match = next(filter(lambda item: item[0] == match,keywords))[1] if match else None
      if match:
        print(
          "**** CLOSEST MATCH FOUND {:{}} FROM {:{}} FOR {}".format(match,max(map(len,sectionNames)),reason,len("not found"),heading)
        )
      elif heading.endswith(":") or ": " in heading:
        print(80*"*","UNHANDLED HEADING {}".format(heading),80*"*",sep="\n")
        # this should be handled
        match = "UNKNOWN"
      else:
        reason = "not found"
        print(
          "*********** DEFAULTED TO {:{}} FROM {} FOR {}".format("NOTES",max(map(len,sectionNames)),reason,heading)
        )
        # when in doubt, it's probably notes
        match = "notes"
      return sections[cache.setdefault(heading,match)]

    def registered(self,section):
      if isinstance(section,PetscDocString.SectionBase):
        return section.name in self._sections
      elif isinstance(section,str):
        return section in self._sections
      raise NotImplementedError(type(section))

    def addSection(self,section):
      assert not self.registered(section),"overwriting section"
      self._sections[section.name] = section
      self._resetCache()
      return

    def titles(self,sections=False):
      if sections:
        return ((title,section) for section in self for title in section.titles)
      return (title for section in self for title in section.titles)

    def keywords(self,sections=False):
      if sections:
        return ((keyword,section) for section in self for keyword in section.keywords)
      return (keyword for section in self for keyword in section.keywords)

    def isHeading(self,item):
      if isinstance(item,tuple):
        assert len(item) == 2
        assert isinstance(item[0],PetscSourceRange) and isinstance(item[1],str)
        text = item[1]
      elif isinstance(item,str):
        text = item
      else:
        raise NotImplementedError(type(item))
      IS_HEADING_BUT_PROBABLY_SHOULDNT_BE = -1
      NOT_HEADING   = 0
      IS_HEADING    = 1
      MAYBE_HEADING = 2
      text = text.strip()
      if not text or text.startswith(("+ ",". ","- ")):
        return NOT_HEADING
      if text.endswith(":"):
        if text.endswith(("follows:","following:","example:","instance:","one of:")):
          return IS_HEADING_BUT_PROBABLY_SHOULDNT_BE
        return IS_HEADING
      if ": " in text and " :" not in text:
        # check that all subsequent items after a ":" are letters, this ought to catch out
        # instances of ":" in random code snippets or text...
        return int(all(t.lstrip()[:1].isalpha() for t in text.split(":")[1:]))
      try:
        found = next(filter(text.casefold().startswith,(t.casefold() for t in self.titles())))
      except StopIteration:
        return NOT_HEADING
      else:
        return MAYBE_HEADING



  sections = Sections(
    Synopsis("synopsis",required=True,keywords=("Synopsis","Not Collective")),
    ParameterList("parameters",titles=("Input Parameter","Output Parameter")),
    ParameterList("options",titles=("Options Database",)),
    Prose("notes"),
    Prose("fortran notes",keywords=("Fortran",)),
    SourceCode("code",titles=("Example Usage",),keywords=("Example","Usage","Sample Usage",)),
    Prose("developer notes"),
    Prose("references"),
    InlineList("level",required=True),
    InlineList("seealso",titles=(".seealso",),required=True),
    DefaultSection("UNKNOWN",titles=("__UNKNOWN_SECTION__",)),
  )
  sowingTypes     = {"@","S","E","M"}
  clxToSowingType = {
    clx.TypeKind.FUNCTIONPROTO : ("@","functions"),
    clx.TypeKind.ENUM          : ("E","enums"),
  }
  __slots__ = "_linter","cursor","raw","extent","indent"

  def __init__(self,linter,cursor,indent=2):
    if not isinstance(linter,PetscLinter):
      raise ValueError(type(linter))
    self._linter         = linter
    self.cursor          = PetscCursor.cast(cursor)
    self.raw,self.extent = self._getSanitizedCommentAndRangeFromCursor(self.cursor)
    self.indent          = indent
    return


  @classmethod
  def diagnostic(cls,flag):
    return super().diagnosticFlag(flag)

  @classmethod
  def registerSection(cls,section):
    return cls.sections.addSection(section)

  @classmethod
  def isValidDocstring(cls,cursor,raw):
    if not raw or not isinstance(raw,str):
      return False

    # if we find sowing chars, its probably a docstring
    rsw,rew = raw.lstrip().startswith,raw.rstrip().endswith
    for char in cls.sowingTypes:
      if rsw("/*"+char) or rew(char+"*/"):
        return True

    # if we find these titles, likely this is a docstring
    if any(title.casefold() in raw.casefold() for title in cls.sections.titles()):
      return True
    print(raw)
    # if it doesn't end with _private or _internal then its very likely a docstring
    if PetscCursor.getNameFromCursor(cursor).casefold().endswith(("_private","_internal")):
      print("==========","FOUND PRIVATE/INTERNAL FUNCTION",cursor.name)
    return False

  @classmethod
  def _getSanitizedCommentAndRangeFromCursor(cls,cursor):
    if not isinstance(cursor,PetscCursor):
      raise ValueError(type(cursor))

    raw,extent = cursor.getCommentAndRange()

    if not cls.isValidDocstring(cursor,raw):
      raise ParsingError("Not a docstring")

    rawlines = raw.splitlines()
    comments = [i for i,line in enumerate(rawlines) if line.lstrip().startswith("/*")]
    extent   = PetscSourceRange.cast(extent,tu=cursor.translation_unit)
    if len(comments) > 1:
      # this handles the following case:
      #
      # /* a dummy comment that is attributed to the symbol */
      # /*
      #   the real docstring comment, note no empty line between this and the previous!
      # */
      # <the symbol>
      offset = comments[-1]
      raw    = "\n".join(rawlines[offset:])
      extent = extent.resized(lbegin=offset,cbegin=None,cend=None)
    return raw,extent

  @classmethod
  def isHeading(cls,*args,**kwargs):
    return cls.sections.isHeading(*args,**kwargs)

  @classmethod
  def _getIsHeading(cls,section):
    return getattr(section,"isHeading",cls.sections.isHeading)


  @staticmethod
  def makeErrorMessage(message,crange,numContext=2,**kwargs):
    return ":\n".join([message,crange.formatted(numContext=numContext,**kwargs)])

  def makeSourceLocation(self,lineno,col):
    return PetscSourceLocation.fromPosition(self.cursor.translation_unit,lineno,col)

  def makeSourceRange(self,token,string,lineno,offset=0):
    colBegin = string.index(token,offset)+1
    colEnd   = colBegin+len(token)
    tu       = self.cursor.translation_unit
    return PetscSourceRange.fromPositions(tu,lineno,colBegin,lineno,colEnd)

  def addErrorFromSourceRange(self,flag,error,crange,patch=None,**kwargs):
    diag = Diagnostic(flag,self.makeErrorMessage(error,crange,**kwargs),patch=patch)
    self._linter.addErrorFromCursor(self.cursor,diag)
    return


  def _checkValidCursorLinkage(self):
    """
    check that a cursor has external linkage, there is no point producing a manpage for function
    that is impossible to call
    """
    cursor = self.cursor
    # TODO, this should probably also check that the header the cursor is defined in is public
    hasIntLinkage,linkageCursorName,linkageCursor = cursor.hasInternalLinkage()
    if hasIntLinkage and not cursor.translation_unit.spelling.endswith((".h",".hpp")):
      mess = "A sowing docstring for a function with internal linkage is pointless!:\n{}\n\nNote '{}' is declared '{}' at {}".format(self.extent.formatted(numContext=2,highlight=False),cursor.displayname,linkageCursorName,PetscCursor.cast(linkageCursor))
      self._linter.addErrorFromCursor(cursor,Diagnostic(self.diags.internal_linkage,mess))
    return

  def _checkValidSowingChars(self):
    sowingType,layType = self.clxToSowingType[self.cursor.type.kind]
    # check the beginning
    splitlines  = self.raw.splitlines()
    line        = splitlines[0]
    beginSowing = line.split("/*")[1].split()
    diagName    = self.diags.sowing_chars
    try:
      beginSowing = beginSowing[0]
    except IndexError:
      mess = "Invalid comment begin line, does not contain sowing identifier. Expected '/*{}' for {}".format(sowingType,layType)
      self.addErrorFromSourceRange(
        diagName,mess,self.makeSourceRange(line,line,self.extent.start.line)
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
      mess    = "Invalid comment begin line, must only contain '/*' and sowing identifier"
      self.addErrorFromSourceRange(
        diagName,mess,restloc,patch=Patch(restloc,"\n"+(" "*self.indent)+rest)
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
        mess    = "Invalid comment end line, sowing identifier(s) do not match begin identifier(s). Expected '{}*/' found '{}*/'".format(correct,endSowing)
        patch   = Patch(self.makeSourceRange(line,line,endline),line.replace(endSowing,correct))
        self.addErrorFromSourceRange(
          diagName,mess,self.makeSourceRange(endSowing,line,endline),patch=patch
        )
    return

  def _checkValidDocstringSpacing(self):
    endLine     = self.extent.end.line+1
    cursorStart = self.cursor.extent.start
    if endLine != cursorStart.line:
      # there is at least 1 (probably empty) line between the comment end and whatever it
      # is describing
      floc = PetscSourceRange.fromLocations(self.makeSourceLocation(endLine,1),cursorStart)
      mess = "Invalid line-spacing between docstring and the symbol it describes. The docstring must appear immediately above its target"
      eloc = self.makeSourceRange("","",endLine)
      diag = self.diags.spacing
      self.addErrorFromSourceRange(diag,mess,eloc,highlight=False,patch=Patch(floc,""))
    return

  def _checkValidIndentation(self,lineno,line,lstripped):
    """
    if the line is regular (not empty, or a parameter list), check that line is
    indented correctly
    """
    linelen = len(line)
    if linelen:
      indent      = linelen-len(lstripped)
      expectedInd = 0 if line.startswith((".","+","-","$")) else self.indent
      if indent != expectedInd:
        diag = self.diags.indentation
        loc  = self.makeSourceRange(" "*indent,line,lineno)
        mess = "Invalid indentation ({}), all regular (non-empty, non-parameter, non-seealso) text must be indented to {} columns".format(indent,self.indent)
        self.addErrorFromSourceRange(diag,mess,loc,patch=Patch(loc," "*expectedInd))
    return

  def _checkValidSectionSpacing(self,prevline,lineno):
    if prevline and not prevline.isspace():
      mess = "Missing empty line between sections, must have one before this section"
      loc  = self.makeSourceRange("","",lineno)
      diag = self.diags.section_spacing
      self.addErrorFromSourceRange(diag,mess,loc,highlight=False,patch=Patch(loc,"\n"))
    return

  def _checkSectionHeaderTypo(self,line,lineno):
    assert ":" not in line, "':' in line that is ambiguously a header: {}".format(line)
    name  = line.lstrip().split(maxsplit=1)[0].rstrip()
    match = difflib.get_close_matches(name,self.sections.find(name.casefold()).titles,n=1)[0]
    mess  = "Line seems to be a section header but missing ':', did you mean '{}:'?".format(match)
    diag  = self.diags.section_header_maybe_header
    self.addErrorFromSourceRange(diag,mess,self.makeSourceRange(name,line,lineno))
    return

  def _checkSectionHeaderThatProbablyShouldNotBeOne(self,line,lineno):
    def smartTruncate(content,length=35,suffix=" [...]"):
      if len(content) <= length:
        return content
      return content[:length].rsplit(maxsplit=1)[0]+suffix

    eloc = self.makeSourceRange(":",line,lineno,offset=line.rfind(":"))
    mess = "Sowing treats all lines ending with ':' as header, are you sure '{}' qualifies? Use '\:' to escape the colon if not".format(smartTruncate(line.strip()))
    self.addErrorFromSourceRange(self.diags.section_header_fishy_header,mess,eloc)
    return


  def parse(self):
    sections = self.sections
    for s in sections:
      s.clear()
    self._checkValidCursorLinkage()
    self._checkValidDocstringSpacing()
    self._checkValidSowingChars()

    rawData     = []
    section     = sections.synopsis
    findSection = sections.find
    checkIndent = section.checkIndentAllowed()
    isHeading   = self._getIsHeading(section)
    for lineno,line in enumerate(self.raw.splitlines(),start=self.extent.start.line):
      lstrip = line.lstrip()
      if lstrip.startswith("/*") or lstrip.endswith("*/"):
        continue

      # TODO remove this, the current active section should be deciding what to do here instead
      # we shouldn't be checking indentation in verbatim blocks
      if lstrip.startswith(".vb"):
        checkIndent = False
      elif lstrip.startswith(".ve"):
        assert checkIndent is False
        checkIndent = True # note we don't need to check indentation of line with .ve
      elif lstrip.startswith("$"):
        pass # inline verbatim don't modify check flag but dont check indentation either
      elif checkIndent:
        self._checkValidIndentation(lineno,line,lstrip)

      heading = isHeading(lstrip)
      if heading > 0:
        if heading == 2:
          self._checkSectionHeaderTypo(line,lineno)
        self._checkValidSectionSpacing(rawData[-1][1] if rawData else None,lineno)
        newSection = findSection(lstrip.split(":",maxsplit=1)[0].strip().casefold())
        if newSection != section:
          if rawData:
            section.fill(rawData)
          rawData     = []
          section     = newSection
          checkIndent = newSection.checkIndentAllowed()
          isHeading   = self._getIsHeading(section)
      elif heading < 0:
        self._checkSectionHeaderThatProbablyShouldNotBeOne(line,lineno)
      rawData.append((self.makeSourceRange(line,line,lineno),line))

    if rawData:
      section.fill(rawData)
    for s in sections:
      s.setup(self)
    return self


  def getSectionHeadings(self,section):
    return list(filter(self._getIsHeading(section),section.lines))

  def checkValidSolitarySectionHeadings(self,section,headings=None):
    """
    Check that a section appears solitarily on its line, i.e. that there is no other text after
    ':'
    """
    if headings is None:
      headings = self.getSectionHeadings(section)

    diag = self.diags.section_header_solitary
    for loc,text in headings:
      _,sep,after = text.partition(":")
      assert sep
      if after.strip():
        mess = "Heading must appear alone on a line, any content must be on the next line"
        self.addErrorFromSourceRange(diag,mess,self.makeSourceRange(after,text,loc.start.line))
    return

  def checkValidSectionHeaderSpelling(self,section,headings=None,transform=None):
    """
    Check that a seciton header is correctly spelled and formatted. Sections may be found
    through fuzzy matching so this check asserts that a particular heading is actually correct
    """
    if headings is None:
      headings = self.getSectionHeadings(section)

    if transform is None:
      transform = section.transform

    diag   = self.diags.section_header_spelling
    titles = section.titles
    for loc,text in headings:
      before,sep,_ = text.partition(":")
      if not sep:
        # missing colon, but if we are at this point then we were pretty it is a header,
        # so we assume the first word is the header
        before,sep = text.split(maxsplit=1)
      assert sep
      heading = before.strip()
      if any(t in heading for t in titles):
        continue

      headingLoc = self.makeSourceRange(heading,text,loc.start.line)
      correct    = transform(heading)
      if heading != correct and any(t in correct for t in titles):
        mess = "Invalid header spelling. Expected '{}' found '{}'".format(correct,heading)
        self.addErrorFromSourceRange(diag,mess,headingLoc,patch=Patch(headingLoc,correct))
        continue

      try:
        match = difflib.get_close_matches(correct,titles,n=1)[0]
      except IndexError:
        self._linter.addWarningFromCursor(
          self.cursor,Diagnostic(diag,"Unknown section '{}'".format(heading))
        )
      else:
        mess = "Unknown section header '{}', assuming you meant '{}'".format(heading,match)
        self.addErrorFromSourceRange(diag,mess,headingLoc,patch=Patch(headingLoc,match))
    return


class PetscCursor(object):
  """
  A utility wrapper around clang.cindex.Cursor that makes retrieving certain useful properties
  (such as demangled names) from a cursor easier.
  Also provides a host of utility functions that get and (optionally format) the source code
  around a particular cursor. As it is a wrapper any operation done on a clang Cursor may be
  performed directly on a PetscCursor (although this object does not pass the isinstance() check).

  See __getattr__ below for more info.
  """
  __slots__ = "__cursor","name","typename","derivedtypename","argidx","_cache"

  def __init__(self,cursor,idx=-12345):
    if isinstance(cursor,PetscCursor):
      self.__cursor        = cursor.clangCursor()
      self.name            = cursor.name
      self.typename        = cursor.typename
      self.derivedtypename = cursor.derivedtypename
      self.argidx          = cursor.argidx if idx == -12345 else idx
      self._cache          = cursor._cache
    elif isinstance(cursor,clx.Cursor):
      self.__cursor        = cursor
      self.name            = self.getNameFromCursor(cursor)
      self.typename        = self.getTypenameFromCursor(cursor)
      self.derivedtypename = self.getDerivedTypenameFromCursor(cursor)
      self.argidx          = idx
      self._cache          = {}
    else:
      raise ValueError(type(cursor))
    return

  def __getattr__(self,attr):
    """
    Allows us to essentialy fake being a clang cursor, if __getattribute__ fails
    (i.e. the value wasn't found in self), then we try the cursor. So we can do things
    like self.translation_unit, but keep all of our variables out of the cursors
    namespace
    """
    return getattr(self.__cursor,attr)

  def __str__(self):
    return "\n".join([self.getFormattedLocationString(),self.getFormattedBlurb()])


  def _getCached(self,attr,func,*args,**kwargs):
    cache = self._cache
    return cache[attr] if attr in cache else cache.setdefault(attr,func(*args,**kwargs))

  @classmethod
  def cast(cls,cursor):
    """like numpy.asanyarray but for PetscCursors"""
    clxCursor = clx.Cursor
    if not isinstance(cursor,(clxCursor,cls)):
      raise ValueError(type(cursor))
    return cls(cursor) if isinstance(cursor,clxCursor) else cursor

  @classmethod
  def errorViewFromCursor(cls,cursor):
    """
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
    """
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
      else:
        pointees = []
      pointees = list({p.spelling: p for p in pointees}.values())
      if len(pointees) > 1:
        # sometimes array subscripts can creep in
        pointees = [c for c in pointees if c.kind not in mathCursors]
      if len(pointees) == 1:
        name = cls.getNameFromCursor(pointees[0])
    elif cursor.kind == clx.CursorKind.ENUM_DECL:
      # have a
      # typedef enum { ... } Foo;
      # so the "name" of the cursor is actually the name of the type itself
      name = cursor.type.get_canonical().spelling
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
    """
    if getNameFromCursor tries to convert **&(PetscObject)obj[i]+73 to obj then this function
    tries to extract **&(PetscObject)obj[i]+73 in the cleanest way possible
    """
    clsInstance = isinstance(cursor,cls)
    if clsInstance:
      cacheEntry = "name"
      try:
        return cursor._cache[cacheEntry]
      except KeyError:
        pass
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
    if clsInstance:
      cursor._cache[cacheEntry] = name
    return name

  @classmethod
  def getTypenameFromCursor(cls,cursor):
    """
    Try to get the most canonical type from a cursor so DM -> _p_DM *
    """
    if isinstance(cursor,cls):
      return cursor.typename
    type          = cursor.type
    pointee       = type.get_pointee()
    typeCursor    = pointee if pointee.spelling else type
    canonSpelling = typeCursor.get_canonical().spelling
    return canonSpelling if canonSpelling else typeCursor.spelling

  @staticmethod
  def getDerivedTypenameFromCursor(cursor):
    """
    Get the least canonical type form a cursor so DM -> DM
    """
    return cursor.type.spelling

  @classmethod
  def hasInternalLinkageFromCursor(cls,cursor):
    if cursor.linkage == clx.LinkageKind.INTERNAL:
      # is a static function or variable
      return True,cursor.storage_class.name,cursor.get_definition()

    hiddenVisibility = {"hidden","protected"}
    VISIBILITY_ATTR  = clx.CursorKind.VISIBILITY_ATTR
    for child in cursor.get_children():
      if child.kind == VISIBILITY_ATTR and child.spelling in hiddenVisibility:
        # is PETSC_INTERN
        return True,PetscSourceRange(child.extent).raw(tight=True),child
    return False,None,None

  def hasInternalLinkage(self):
    return self._getCached("internal_linkage",self.hasInternalLinkageFromCursor,self)

  @staticmethod
  def getRawSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,trim=False):
    return pclu.getRawSourceFromCursor(
      cursor,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,trim=trim
    )

  def getRawSource(self,**kwargs):
    return self.getRawSourceFromCursor(self,**kwargs)

  @staticmethod
  def getFormattedSourceFromCursor(cursor,nbefore=0,nafter=0,nboth=0,view=False,**kwargs):
    if cursor.kind == clx.CursorKind.FUNCTION_DECL:
      begin  = cursor.extent.start
      # -1 gives you EOL
      fnline = PetscSourceLocation.fromPosition(cursor.translation_unit,begin.line,-1)
      extent = PetscSourceRange.fromLocations(begin,fnline)
    else:
      extent = cursor.extent
    return pclu.getFormattedSourceFromSourceRange(
      extent,numBeforeContext=nbefore,numAfterContext=nafter,numContext=nboth,view=view,**kwargs
    )

  def formatted(self,**kwargs):
    return self.getFormattedSourceFromCursor(self,**kwargs)

  def view(self):
    return print(self.formatted(nboth=5))

  @staticmethod
  def getFormattedLocationStringFromCursor(cursor):
    loc = cursor.location
    return ":".join([loc.file.name,str(loc.column),str(loc.line)])

  def getFormattedLocationString(self):
    return self.getFormattedLocationStringFromCursor(self)

  @classmethod
  def getFormattedBlurbFromCursor(cls,cursor):
    cursor = cls.cast(cursor)
    return "'{}' of derived type '{}', canonical type '{}'\n{}\n".format(
      cursor.name,cursor.derivedtypename,cursor.typename,cursor.formatted(nboth=2)
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
    cxlib = clx.conf.lib
    try:
      func = getattr(cxlib,funcName)
      if (func.argtypes is None) and (func.errcheck is None):
        # if this hasn't been registered before these will be none
        raise AttributeError
    except AttributeError:
      # have to do the book-keeping ourselves since it may not be properly hooked up
      clx.register_function(cxlib,(funcName,argtypes,rettype),False)
      func = getattr(cxlib,funcName)
    return func

  @classmethod
  def findCursorReferencesFromCursor(cls,cursor):
    """
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
    func = cls.getOrRegisterClangFunction(
      "clang_Cursor_getCommentRange",[clx.Cursor],clx.SourceRange
    )
    if isinstance(cursor,cls):
      cursorRange = func(cursor.clangCursor())
    elif isinstance(cursor,clx.Cursor):
      cursorRange = func(cursor)
    else:
      raise ValueError(type(cursor))
    return cursor.raw_comment,cursorRange

  def getCommentAndRange(self):
    return self.getCommentAndRangeFromCursor(self)

  @classmethod
  def getFileFromCursor(cls,cursor):
    if isinstance(cursor,cls):
      ret = cursor._getCached("file",lambda c: c.location.file.name,cursor)
    elif isinstance(cursor,clx.Cursor):
      ret = cursor.location.file.name
    elif isinstance(cursor,clx.TranslationUnit):
      ret = cursor.spelling
    else:
      raise ValueError(type(cursor))
    return PetscPath(ret)

  def getFile(self):
    return self.getFileFromCursor(self)

  def clangCursor(self):
    """return the internal clang cursor"""
    return self.__cursor


class Patch(object):
  class Delta(object):
    __slots__ = "value","extent","offset"

    def __init__(self,value,extent,ctxlines):
      self.value  = str(value)
      self.extent = extent
      src,begin   = extent.raw(numContext=ctxlines).splitlines(True)[:ctxlines],extent.start
      self.offset = begin.offset-begin.column-sum(map(len,src))+1
      return

    def deleter(self):
      return bool(self.value == "")

    def view(self):
      before = self.extent.formatted(numContext=3,highlight=False).splitlines(True)
      after  = before.copy()
      after[3] = before[3].replace(self.extent.raw(tight=True),self.value,1)
      print("".join(difflib.unified_diff(before,after,fromfile="Original",tofile="Modified")))
      return


  __slots__ = "extent","filename","ctxlines","src","deltas","weakData","_cache"

  def __init__(self,srcRange,value,contextlines=2):
    self.extent   = PetscSourceRange.cast(srcRange)
    self.ctxlines = contextlines
    self.src      = self._makeSource()
    self.deltas   = (self.Delta(value,self.extent,self.ctxlines),)
    self.weakData = []
    self._cache   = {}
    return

  @classmethod
  def fromCursor(cls,cursor,value,**kwargs):
    return cls(cursor.extent,value,**kwargs)

  def _makeSource(self):
    return self.extent.raw(numContext=self.ctxlines)

  def _getCached(self,attr,func,*args,**kwargs):
    cache = self._cache
    return cache[attr] if attr in cache else cache.setdefault(attr,func(*args,**kwargs))

  def _contiguousExtent(self):
    """
    does my extent (which is the union of the extents of my all my deltas) have no holes?
    """
    cacheEntry = "contiguous"
    deltas     = self.deltas
    if len(deltas) == 1:
      return self._cache.setdefault(cacheEntry,True)
    return self._getCached(
      cacheEntry,all,(p.extent.overlaps(c.extent) for p,c in zip(deltas[:-1],deltas[1:]))
    )


  def discard(self):
    """
    drops the error messages corresponding to this patch from the linter
    """
    for weakElist,idx in self.weakData:
      elist = weakElist()
      if elist is not None:
        del elist[1][idx] # delete the error message
        del elist[2][idx] # delete the patch indicator
    return

  def attach(self,*args):
    """
    attach the list and index into the linter error list corresponding to this patch
    """
    self.weakData.append(args)
    return

  def dofix(self,n=1):
    newSrc   = self._makeSource()
    idxDelta = 0
    for rng,i in zip(sorted(self.deltas),range(n)):
      begin       = rng.extent.start
      end         = rng.extent.end
      beginoffset = begin.offset-rng.offset
      endoffset   = end.offset-rng.offset
      delta       = rng.value
      newSrcTemp  = "".join([newSrc[:beginoffset+idxDelta],delta,newSrc[endoffset+idxDelta:]])
      idxDelta    = len(newSrcTemp)-len(newSrc)
      newSrc      = newSrcTemp
    print(newSrcTemp)
    return newSrcTemp

  @staticmethod
  def domerge(left,right):
    a      = left.dofix()
    b      = right.dofix()
    res    = []
    deltas = []
    adelta = {d.value for d in left.deltas}
    bdelta = {d.value for d in right.deltas}
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,a,b).get_opcodes():
      print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag,i1,i2,j1,j2,a[i1:i2],b[j1:j2]))
      if tag == "equal":
        res.append(a[i1:i2])
      elif tag == "insert":
        res.append(b[j1:j2])
      elif tag == "replace":
        mine   = a[i1:i2]
        theirs = b[j1:j2]
        print("mine","'"+mine+"'","theirs","'"+theirs+"'")
        if mine in adelta:
          print("mine was in self.deltas")
          if theirs not in bdelta:
            print("theirs was not in other.deltas, adding mine")
            res.append(mine)
            deltas.append(mine)
            continue
        else:
          print("mine was not in self.deltas")
        if theirs in bdelta:
          print("theirs was in other.deltas, adding theirs")
          res.append(theirs)
          deltas.append(theirs)
          continue
        print("was in neither, taking theirs?")
        res.append(theirs)
        deltas.append(theirs)
    return res,deltas

  @staticmethod
  def mergeDiff3(mine,other):
    """
    Try to merge two "overlapping" patches using diff3 and a vile hack.

    diff3 normally may not be able to handle overlap inside a hunk, but by splitting the source
    into words (including newline and whitespace!) we can try to separate out the overlap.

    I have no idea how robust this is, so this should be used as a last resort. Try first to
    make the patches as minimal as possible. You don't have to handle complex overlap if there
    is no overlap to begin with!
    """
    def smart_split(string):
      """
      split a string into words, keeping the separators
      """
      specials = {" ","\n","\r","\t"}
      spaces   = 0
      result   = []
      was_word = False
      for char in string:
        if char in specials:
          was_word = False
          if char == " ":
            spaces += 1
          else:
            result.append(char)
        else:
          if spaces:
            result.append(" "*spaces)
            spaces = 0
          if not was_word:
            result.append("")
            was_word = True
          result[-1] += char
      return "\n".join(result)

    import tempfile
    with tempfile.NamedTemporaryFile(delete=True) as minefd,  \
         tempfile.NamedTemporaryFile(delete=True) as theirfd, \
         tempfile.NamedTemporaryFile(delete=True) as commonfd:
      minep = PetscPath(minefd.name).resolve()
      minep.write_text(smart_split(mine.collapse()))
      otherp = PetscPath(theirfd.name).resolve()
      otherp.write_text(smart_split(other.collapse()))
      commonp = PetscPath(commonfd.name).resolve()
      commonp.write_text(smart_split(mine._makeSource()))
      try:
        output = subprocessRun(
          ["diff3","-a","-m",str(minep),str(commonp),str(otherp)],
          capture_output=True,universal_newlines=True,check=True
        )
      except RuntimeError as re:
        # if we are here it means diff3 failed to merge!
        print(re)
        combined = mine.deltas+other.deltas
        argsort = sorted(range(len(combined)),key=lambda x: combined.__getitem__(x).extent)
        import ipdb; ipdb.set_trace()
        blablabla = 2
      else:
        # being here does not imply unqualified success, output may be wrong!
        merged = "".join(
          s.replace("\n","")+"\n" if s else "\n" for s in output.stdout.split("\n\n")
        )[:-1] # remove final newline
    return merged

  def isDeletionSupersetOf(self,other):
    """
    determine if any of self's deltas delete all of other's extent, in which case other is a
    pointless patch and its error messages can be discarded
    """
    oextent = other.extent
    if oextent in self.extent:
      deltas = self.deltas
      # first check if any one delta deletes all of other, then check if all deltas are deleters,
      return any(d.deleter() and oextent in d.extent for d in deltas) or \
        (self._contiguousExtent() and all(d.deleter() for d in deltas))
    return False

  def merge(self,other):
    if not isinstance(other,type(self)):
      raise ValueError(type(other))

    if self.isDeletionSupersetOf(other):
      other.discard()
      return self
    elif other.isDeletionSupersetOf(self):
      self.discard()
      return other

    assert self.src == other.src,"Need to update offset calculation to handle arbitrary src"
    assert self.ctxlines == other.ctxlines,"Need to update ctxlines to handle arbitrary src"

    self.extent = self.extent.mergeWith(other.extent)
    # uncomment when it handles arbitrary source
    # self.src    = self._makeSource()
    # fixes and ranges must be applied in order
    combined    = self.deltas+other.deltas
    argsort     = sorted(range(len(combined)),key=lambda x: combined.__getitem__(x).extent)
    self.deltas = tuple(combined[i] for i in argsort)
    self._cache = {}
    self.weakData.extend(other.weakData)
    return self

  def collapse(self):
    """Collapses a the list of fixes and produces into a modified output"""
    # Fixes probably should not overwrite each other (for now), so we error out, but this
    # is arguably a completely valid case. I just have not seen an example of it that I
    # can use to debug with yet.
    cacheEntry = "fixed"
    if cacheEntry in self._cache:
      return self._cache[cacheEntry]

    idxDelta = 0
    newSrc   = self.src = self._makeSource()
    for delta in self.deltas:
      extent      = delta.extent
      offset      = idxDelta-delta.offset
      beginoffset = extent.start.offset+offset
      endoffset   = extent.end.offset+offset
      newSrcTemp  = "".join([newSrc[:beginoffset],delta.value,newSrc[endoffset:]])
      idxDelta   += len(newSrcTemp)-len(newSrc)
      newSrc      = newSrcTemp
    assert newSrc != self.src,"Patch did not seem to do anything!"
    self._cache[cacheEntry] = newSrc
    return newSrc

  def view(self):
    for i,delta in enumerate(self.deltas):
      print("Delta:",i,"({})".format(delta))
      delta.view()
    return


class PetscLinter(object):
  """
  Object to manage the collection and processing of errors during a lint run.
  """
  __slots__ = (
    "flags","clangOpts","prefix","verbose","werror","lock","errPrefix","warnPrefix","errors",
    "warnings","patches","index"
  )

  class weaklist(list):
    __slots__ = "__weakref__"


  def __init__(self,compilerFlags,clangOptions=baseClangOptions,prefix="[ROOT]",verbose=False,werror=False,lock=None):
    self.flags      = compilerFlags
    self.clangOpts  = clangOptions
    self.prefix     = prefix
    self.verbose    = verbose
    self.werror     = werror
    self.lock       = lock
    self.errPrefix  = " ".join([prefix,85*"-"])
    self.warnPrefix = " ".join([prefix,85*"%"])
    self.index      = clx.Index.create()
    self.clear()
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


  @DiagnosticManager.register(
    ("duplicate-function","Check for duplicate function-calls on the same execution path"),
  )
  def _checkDuplicateFunctionCalls(self,processedFuncs):
    dupDiag = self._checkDuplicateFunctionCalls.diags.duplicate_function
    for pname,functionList in processedFuncs.items():
      seen = {}
      for func,scope in functionList:
        combo = [func.displayname]
        try:
          combo.extend(map(PetscCursor.getRawNameFromCursor,func.get_arguments()))
        except ParsingError:
          continue
        combo = tuple(combo)
        if combo not in seen:
          seen[combo] = (func,scope)
        elif scope >= seen[combo][1]:
          seenStart = seen[combo][0].extent.start.line
          start     = func.extent.start
          startline = start.line
          tu        = func.translation_unit
          end       = clx.SourceLocation.from_position(tu,tu.get_file(tu.spelling),startline,-1)
          patch     = Patch(PetscSourceRange.fromLocations(start,end),"")
          message   = "Duplicate function found previous identical usage:\n{}".format(
            seen[combo][0].formatted(nbefore=2,nafter=startline-seenStart)
          )
          self.addErrorFromCursor(func,Diagnostic(dupDiag,message,patch=patch))
    return

  @staticmethod
  def findFunctionCallExpr(tu,functionNames):
    """
    Finds all function call expressions in container functionNames.

    Note that if a particular function call is not 100% correctly defined (i.e. would the
    file actually compile) then it will not be picked up by clang AST.

    Function-like macros can be picked up, but it will be in the wrong 'order'. The AST is
    built as if you are about to compile it, so macros are handled before any real
    function definitions in the AST, making it impossible to map a macro invocation to
    its 'parent' function.
    """
    def walkScopeSwitch(parent,scope):
      """
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
      """
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


    lintableKinds = funcCallCursors|{clx.CursorKind.ENUM_DECL}
    cursor        = tu.cursor
    filename      = tu.spelling
    for possibleParent in cursor.get_children():
      # getting filename is for some reason stupidly expensive, so we do this check first
      if possibleParent.kind not in lintableKinds: continue
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

  @staticmethod
  def getArgumentCursors(funcCursor):
    """
    given a cursor representing a function, return a tuple of PetscCursor's of its arguments
    """
    return tuple([PetscCursor(a,i) for i,a in enumerate(funcCursor.get_arguments(),start=1)])


  def clear(self):
    """
    Resets the linter error, warning, and patch buffers.
    Called automatically before parsing a file
    """
    self.errors   = collections.OrderedDict()
    self.warnings = []
    # This can actually just be a straight list, since each linter object only ever
    # handles a single file, but use dict nonetheless
    self.patches  = {}
    return

  def parse(self,filename):
    """
    parse a file for errors
    """
    self.clear()
    if self.verbose: self.__print(self.prefix,"Processing file     ",filename)
    tu = self.index.parse(str(filename),args=self.flags,options=self.clangOpts)
    if self.verbose and tu.diagnostics:
      self.__print("\n".join({" ".join((self.prefix,d)) for d in map(str,tu.diagnostics)}))
    self.process(tu)
    return self

  @DiagnosticManager.register(("parsing-error","Generic parsing errors"))
  def process(self,tu):
    """process a translation unit for errors"""
    parsingDiag    = self.process.diags.parsing_error
    processedFuncs = collections.defaultdict(list)
    for results in self.findFunctionCallExpr(tu,set(checkFunctionMap.keys())):
      if isinstance(results,clx.Cursor):
        checkDocMap[results.kind](self,results)
        continue
      func,parent,scope = results
      try:
        checkFunctionMap[func.spelling](self,func,parent)
      except ParsingError as pe:
        self.addWarning(tu.cursor.spelling,Diagnostic(parsingDiag,str(pe)))
      processedFuncs[PetscCursor.getNameFromCursor(parent)].append((PetscCursor.cast(func),scope))
    self._checkDuplicateFunctionCalls(processedFuncs)
    return


  def addErrorFromCursor(self,cursor,diagnostic):
    """
    given a cursor attach a diagnostic error message to it, and optionally a fix
    """
    if diagnostic.disabled():
      return

    cursor   = PetscCursor.cast(cursor)
    filename = cursor.getFile()
    errors   = self.errors

    if filename not in errors:
      errors[filename] = collections.OrderedDict()

    cursorId = cursor.hash
    if cursorId not in errors[filename]:
      header = "\nERROR {}: {}\n".format(len(errors[filename]),str(cursor))
      errors[filename][cursorId] = self.weaklist([header,[],[]])

    patch          = diagnostic.patch
    havePatch      = patch is not None
    cursorIdErrors = errors[filename][cursorId]
    cursorIdErrors[1].append(diagnostic.formatMessage())
    cursorIdErrors[2].append(havePatch)

    if not havePatch:
      return # bail early

    patch.attach(weakref.ref(cursorIdErrors),len(cursorIdErrors[1])-1)
    patches = self.patches

    if filename not in patches:
      patches[filename] = [patch]
      return

    # check if this is a compound error, i.e. an additional error on the same line
    # in which case we need to combine with previous patch
    pex       = patch.extent
    pexstart  = pex.start.line
    patchList = patches[filename]
    for i,prevPatch in enumerate(patchList):
      prepex = prevPatch.extent
      if pexstart == prepex.start.line or pex.overlaps(prepex):
        # this should now be the previous patch on the same line
        patchList[i] = prevPatch.merge(patch)
        return

    patchList.append(patch) # didn't find any overlap, just append
    return

  def addWarning(self,filename,diag):
    """
    add a generic warning given a filename
    """
    if self.werror:
      return self.addErrorFromCursor(filename,diag)
    warnMsg = diag.formatMessage()
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

  def addWarningFromCursor(self,cursor,diag):
    """
    given a cursor attach a diagnostic warning message to it
    """
    if self.werror:
      return self.addErrorFromCursor(cursor,diag)
    cursor   = PetscCursor.cast(cursor)
    warnStr  = "".join(["\nWARNING {}: ".format(len(self.warnings)),str(cursor),"\n",warnMsg])
    self.warnings.append((cursor.getFile(),warnStr))
    return


  def getAllErrors(self):
    """
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
    """
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
    """
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


    def combine(filename,patches):
      fstr  = str(filename)
      diffs = []
      for p in sorted(patches,key=lambda x: x.extent.start.line):
        rn  = datetime.datetime.now().ctime()
        tmp = list(difflib.unified_diff(
          p._makeSource().splitlines(True),p.collapse().splitlines(True),
          fromfile=fstr,tofile=fstr,fromfiledate=rn,tofiledate=rn,n=p.ctxlines
        ))
        tmp[2] = re.sub(r"^@@ -([0-9,]+) \+([0-9,]+) @@",Addline(p.extent.start.line),tmp[2])
        # only the first diff should get the file heading
        diffs.append(tmp[2:] if len(diffs) else tmp)
      return filename,"".join(itertools.chain.from_iterable(diffs))

    return list(itertools.starmap(combine,self.patches.items()))


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
    """
    Various signals to indicate return type on the data queue from child processes
    """
    WARNING      = enum.auto()
    UNIFIED_DIFF = enum.auto()
    ERRORS_LEFT  = enum.auto()
    ERRORS_FIXED = enum.auto()
    EXIT_QUEUE   = enum.auto()

  def __init__(self,numWorkers,timeout=2,verbose=False,prefix="[ROOT]",**kwargs):
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
          clangLib,checkFunctionMap,classIdMap,DiagnosticManager,compilerFlags,clangOptions,
          self.verbose,werror,self.errorQueue,self.returnQueue,self,self.lock
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
    return self

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
- figure out how to handle overwriting fixits...
- figure out how to handle in-out parameters
- fix indentation check when indenting overflowing parameter list descriptions, should be indented to the dash not 2 columns
- think of checks to do for SourceCode sections
- handle stable combination of patches on single-character ranges, i.e.
patch1: insert ' ' in (1,1)
patch2: insert '\n' in (1,1)
which order should these go in?
"""
def alwaysTrue(*args,**kwargs):
  return True

def alwaysFalse(*args,**kwargs):
  return False


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
    if symbol.lstrip().startswith("$"):
      # This is special treatment for enums since they don't usually have a
      # clearly-defined "begin"
      break
    elif dash and rest:
      foundSynopsis = True
      symbol        = symbol.strip()
      if symbol != cursorName:
        diag = synopsis.diags.matching_symbol_name
        loc  = docstring.makeSourceRange(symbol,line,lineLoc.start.line)
        if len(difflib.get_close_matches(symbol,[cursorName],n=1)):
          mess  = "Docstring name '{}' does not match symbol. Assuming you meant '{}'".format(
            symbol,cursorName
          )
          patch = Patch(loc,cursorName)
        else:
          mess  = "Docstring name '{}' does not match symbol name '{}'".format(symbol,cursorName)
          patch = None
        docstring.addErrorFromSourceRange(diag,mess,loc,patch=patch)
      break
  if not foundSynopsis:
    mess = "Docstring missing synopsis. Expected '{} - a very useful description'".format(
      cursorName
    )
    diag = synopsis.diags.missing_description
    docstring.addErrorFromSourceRange(diag,mess,synopsis.extent,highlight=False)
  return

@DiagnosticManager.register(
  (PetscDocString.ParameterList.diagnostic("order"),"Verify that documentation for parameters is in order of appearance"),
  (PetscDocString.ParameterList.diagnostic("undocumented-parameter"),"Verify that all parameters for a symbol are documented"),
  (PetscDocString.ParameterList.diagnostic("extra-parameter"),"Verify that all parameters that are documented actually exist"),
  (PetscDocString.ParameterList.diagnostic("solitary-parameter"),"Verify that each parameter has its own entry"),
)
def checkDocValidParameterList(linter,docstring,params,cursorList,skipGroup=alwaysFalse,checkIndices=False,**kwargs):
  if cursorList and not params:
    undocParamDiag = checkDocValidParameterList.diags.undocumented_parameter
    linter.addErrorFromCursor(
      docstring.cursor,Diagnostic(undocParamDiag,"Symbol parameters are all undocumented")
    )
    return
  elif not cursorList:
    # check we've got no parameter docstrings, if so, we can delete them
    if params and len(params.items.values()):
      sr   = params.extent
      mess = "Found parameter docstring(s) but '{}' has no parameters".format(cursor.displayname)
      extraParamDiag = checkDocValidParameterList.diags.extra_parameter
      docstring.addErrorFromSourceRange(extraParamDiag,mess,sr,highlight=False,patch=Patch(sr,""))
    return

  solitaryParamDiag        = checkDocValidParameterList.diags.solitary_parameter
  checkAlignedDescriptions = params.checkAlignedDescriptions
  splitParam = params.splitParam
  cursorList = list(map(PetscCursor.cast,cursorList))
  argNames   = [a.name for a in cursorList]
  argSeen    = [False]*len(argNames)
  notFound   = []

  for k,group in sorted(params.items.items()):
    if skipGroup(k,group):
      continue
    indices = []
    remove  = set()
    for i,(loc,text,_) in enumerate(group):
      _,arg,_ = splitParam(text)
      asplit  = [a.strip() for a in arg.split(",")]
      if len(asplit) > 1:
        mess = "Each parameter entry must be documented separately on its own line"
        eloc = docstring.makeSourceRange(arg,text,loc.start.line)
        docstring.addErrorFromSourceRange(solitaryParamDiag,mess,eloc)
      for subarg in asplit:
        try:
          idx = argNames.index(subarg)
        except ValueError:
          notFound.append((subarg,loc))
          remove.add(i)
        else:
          argSeen[idx] = True
          indices.append(idx)
    checkAlignedDescriptions(docstring,[g for i,g in enumerate(group) if i not in remove],**kwargs)
    if 0 and checkIndices:
      # TODO what to do if the lines we swap are imperfect? This might be the straw
      # breaking the camels back, since I don't see a way to do this without 2 passes
      done = set()
      for mine,correct in zip(indices,sorted(indices)):
        if mine == correct or mine in done or correct in done:
          continue
        mloc,mtext,_ = group[mine]
        cloc,ctext,_ = group[correct]
        mloc  = mloc.extend()
        cloc  = cloc.extend()
        patch1 = Patch(mloc,ctext,combineable=False)
        print(patch1.collapse())
        patch2 = Patch(cloc,mtext,combineable=False)
        print(patch2.collapse())
        diag  = checkDocValidParameterList.diags.parameter_order
        mess  = "Parameters out of order"
        import ipdb; ipdb.set_trace()
        docstring.addErrorFromSourceRange(diag,mess,mloc,patch=patch1)
        docstring.addErrorFromSourceRange(diag,mess,cloc,patch=patch2)
        done.update({mine,correct})

  argsLeft = [name for seen,name in zip(argSeen,argNames) if not seen]
  if notFound:
    extraParamDiag    = checkDocValidParameterList.diags.extra_parameter
    get_close_matches = difflib.get_close_matches
    baseMessage       = "Extra docstring parameter '{}' not found in symbol parameter list:\n{}"
    cursor            = docstring.cursor
    for i,(arg,loc) in enumerate(notFound):
      message = baseMessage.format(arg,loc.formatted(numContext=2))
      try:
        if (len(argsLeft) == 1) and (i == len(notFound)-1):
          # if we only have 1 arg left and 1 wasn't found, chances are they are meant to
          # be the same
          match = argsLeft[-1]
        else:
          match = get_close_matches(arg,argsLeft,n=1)[0]
      except IndexError:
        pass
      else:
        matchCursor = [c for c in cursorList if c.name == match][0]
        message    += "\n\nmaybe you meant {}?".format(matchCursor.getFormattedBlurb())
        argsLeft.remove(match)
      linter.addErrorFromCursor(cursor,Diagnostic(extraParamDiag,message))

  undocParamDiag = checkDocValidParameterList.diags.undocumented_parameter
  for mess in map("Undocumented parameter '{}' not found in parameter section".format,argsLeft):
    docstring.addErrorFromSourceRange(undocParamDiag,mess,params.extent,highlight=False)
  return

def checkDocValidOptionsDatabaseKeys(linter,cursor,docstring):
  options = docstring.sections.options
  if not options:
    return

  docstring.checkValidSectionHeaderSpelling(options)
  docstring.checkValidSolitarySectionHeadings(options)

  check = options.checkAlignedDescriptions
  for _,group in sorted(options.items.items()):
    check(docstring,group)
  return

@DiagnosticManager.register(
  (PetscDocString.InlineList.diagnostic("level-casefold"),"Verify that level subheadings are lower-case"),
  (PetscDocString.InlineList.diagnostic("level-spelling"),"Verify that level subheadings are correctly spelled"),
)
def checkDocValidLevel(linter,cursor,docstring):
  level = docstring.sections.level
  if not level:
    # if no level, nothing to check here, error will already have been logged
    return

  docstring.checkValidSectionHeaderSpelling(level)

  casefoldDiag = checkDocValidLevel.diags.level_casefold
  spellingDiag = checkDocValidLevel.diags.level_spelling
  validLevels  = ("beginner","intermediate","advanced","developer","deprecated")
  expected     = ", or ".join([", ".join(validLevels[:-1]),validLevels[-1]])
  for loc,levelName in level.items:
    if levelName not in validLevels:
      locase = levelName.casefold()
      if locase in validLevels:
        mess = "Level subheading must be lowercase, expected '{}' found '{}'".format(
          locase,levelName
        )
        docstring.addErrorFromSourceRange(casefoldDiag,mess,loc,patch=Patch(loc,locase))
      else:
        match = difflib.get_close_matches(locase,validLevels,n=1)
        if match:
          match = match[0]
          mess  = "Unknown Level subheading '{}', assuming you meant '{}'".format(levelName,match)
          patch = Patch(loc,match)
        else:
          mess  = "Unknown Level subheading '{}', expected one of {}".format(levelName,expected)
          patch = None
        docstring.addErrorFromSourceRange(spellingDiag,mess,loc,patch=patch)
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
      # TODO: make a real diagnostic here
      diag = Diagnostic(spellingDiag,errorMessage,patch=Patch(delrange,""))
      linter.addErrorFromCursor(cursor,diag)
    prevloc  = loc
    prevline = line
  return

@DiagnosticManager.register(
  (PetscDocString.InlineList.diagnostic("seealso-duplicate"),"Verify that there are no duplicate entries in seealso lists"),
  (PetscDocString.InlineList.diagnostic("seealso-self-reference"),"Verify that seealso lists don't contain the current symbol name"),
)
def checkDocValidSeealso(linter,cursor,docstring):
  seealso = docstring.sections.seealso
  if not seealso:
    return

  def transform(text):
    return text.lower()

  docstring.checkValidSectionHeaderSpelling(seealso,transform=transform)

  def makeDeletionPatch(loc,text,lookBehind):
    """
    first(),    second(),      third

    Extend source range of 'second' so that deleting it yields

    first(), third
    """
    raw = loc.raw().rstrip("\n")
    col = loc.start.column-1
    # str.partition won't work here since it returns the first instance of 'sep', which in
    # our case might be the first instance of the value rather than the duplicate we just
    # found
    post = raw[col+len(text):]
    # get the number of characters between us and next alphabetical character
    cend = len(post)-len(post.lstrip(", "))
    if lookBehind:
      # look to remove comma and space the entry behind us
      pre    = raw[:col]
      cbegin = len(pre.rstrip(", "))-len(pre) # note intentionally negative value
      assert cbegin < 0
    else:
      cbegin = 0
    return Patch(loc.resized(cbegin=cbegin,cend=cend),"")


  items       = seealso.items
  lastLoc     = items[-1][0]
  itemRemain  = []
  selfRefDiag = checkDocValidSeealso.diags.seealso_self_reference
  symbolName  = PetscCursor.getNameFromCursor(cursor)
  for loc,text in items:
    if text.rstrip("()") == symbolName:
      mess = "Found self-referential seealso entry '{}'; your documentation may be good but it's not *that* good".format(text)
      docstring.addErrorFromSourceRange(
        selfRefDiag,mess,loc,patch=makeDeletionPatch(loc,text,loc == lastLoc)
      )
    else:
      itemRemain.append((loc,text))

  seen    = {}
  dupDiag = checkDocValidSeealso.diags.seealso_duplicate
  for loc,text in itemRemain:
    if text not in seen:
      seen[text] = (loc,text)
      continue

    if not text:
      import ipdb; ipdb.set_trace()
    mess = "\n\n".join((
      docstring.makeErrorMessage("Seealso entry '{}' is duplicate".format(text),loc),
      docstring.makeErrorMessage("Note first instance found here",seen[text][0],numContext=1)
    ))
    linter.addErrorFromCursor(
      cursor,Diagnostic(dupDiag,mess,patch=makeDeletionPatch(loc,text,loc == lastLoc))
    )
  return


"""utilities for checking specific types of docstrings"""
@DiagnosticManager.register(
  (PetscDocString.ParameterList.diagnostic("undocumented-parameter"),"Verify that all parameters for a symbol are documented"),
  (PetscDocString.ParameterList.diagnostic("extra-parameter"),"Verify that all documented parameters exist for a symbol"),
  (PetscDocString.ParameterList.diagnostic("fortran-interface"),"Verify that functions needing a custom fortran interface have the correct sowing indentifiers"),
)
def checkDocValidFunctionParameters(linter,cursor,docstring):
  fnargs = linter.getArgumentCursors(cursor)
  params = docstring.sections.parameters

  if fnargs and not params:
    undocParamDiag = checkDocValidFunctionParameters.diags.undocumented_parameter
    linter.addErrorFromCursor(
      cursor,Diagnostic(undocParamDiag,"Function parameters are all undocumented")
    )
    return
  elif not fnargs:
    # check we've got no parameter docstrings, if so, we can delete them
    if len(params.items.values()):
      sr   = params.extent
      mess = "Found parameter docstring(s) but '{}' has no arguments".format(cursor.displayname)
      extraParamDiag = checkDocValidFunctionParameters.diags.extra_parameter
      docstring.addErrorFromSourceRange(extraParamDiag,mess,sr,highlight=False,patch=Patch(sr,""))
    return

  docstring.checkValidSectionHeaderSpelling(params)
  docstring.checkValidSolitarySectionHeadings(params)

  requiresC    = []
  POINTER_KIND = clx.TypeKind.POINTER
  for arg in fnargs:
    canon = arg.type.get_canonical()
    kind  = canon.kind
    it    = 0
    while kind == POINTER_KIND:
      if it >= 100:
        # there is no chance that someone has a variable over 100 pointers deep, so
        # clearly something is wrong
        emess = "Ran for {} iterations (>= 100) trying to get pointer type for\n{}\n".format(
          it,arg.errorViewFromCursor(arg),"\n".join(pclu.viewAstFromCursor(arg))
        )
        raise RuntimError(emess)
      canon = canon.get_pointee()
      kind  = canon.kind
      it   += 1
    if kind in charTypes:
      requiresC.append((arg,"char"))
    elif kind in functionTypes:
      requiresC.append((arg,"function"))

  if len(requiresC) and not docstring.raw.startswith("/*@C"):
    line   = docstring.raw.split(maxsplit=1)[0]
    crange = docstring.makeSourceRange(line,line,docstring.extent.start.line)
    blame  = "\n".join("  {}. '{}' of derived type '{}' (is a {} pointer)".format(i+1,a.name,a.derivedtypename,why) for i,(a,why) in enumerate(requiresC))
    mess   = "Function requires custom fortran interface but missing 'C' from docstring header. Due to\n{}".format(blame)
    fortrInterDiag = checkDocValidFunctionParameters.diags.fortran_interface
    docstring.addErrorFromSourceRange(fortrInterDiag,mess,crange,patch=Patch(crange,line+"C"))

  checkDocValidParameterList(linter,docstring,params,fnargs)
  return

def checkDocValidEnumParameters(linter,cursor,docstring):
  synopsis = docstring.sections.synopsis
  if not synopsis:
    import ipdb; ipdb.set_trace()
    # wtf
    return

  def skipGroup(k,*args,**kwargs):
    return k == 0

  enumParams = list(map(PetscCursor,cursor.get_children()))
  checkDocValidParameterList(
    linter,docstring,synopsis.items,enumParams,skipGroup=skipGroup,checkIndices=True,char="$"
  )
  return


"""Specific 'driver' function to test a particular docstring archetype"""
def checkPetscFunctionDocString(linter,cursor):
  try:
    docstring = PetscDocString(linter,cursor).parse()
  except ParsingError as pe:
    return # error already logged with linter

  checkDocValidSynopsis(linter,cursor,docstring)
  checkDocValidFunctionParameters(linter,cursor,docstring)
  checkDocValidOptionsDatabaseKeys(linter,cursor,docstring)
  checkDocValidLevel(linter,cursor,docstring)
  checkDocValidSeealso(linter,cursor,docstring)
  return

def checkPetscEnumDocString(linter,cursor):
  try:
    docstring = PetscDocString(linter,cursor).parse()
  except ParsingError:
    return # error already logged with linter

  checkDocValidSynopsis(linter,cursor,docstring)
  checkDocValidEnumParameters(linter,cursor,docstring)
  checkDocValidLevel(linter,cursor,docstring)
  checkDocValidSeealso(linter,cursor,docstring)
  return


checkDocMap = {
  clx.CursorKind.FUNCTION_DECL : checkPetscFunctionDocString,
  clx.CursorKind.ENUM_DECL     : checkPetscEnumDocString,
}

"""utilities for checking functions"""
@DiagnosticManager.register(
  ("incompatible-function","Verify that the correct function was used for a type")
)
def addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName):
  """
  shorthand for extracting a fix from a function cursor
  """
  call = [
    c for c in funcCursor.get_children() if c.type.get_pointee().kind == clx.TypeKind.FUNCTIONPROTO
  ]
  assert len(call) == 1
  patch = Patch.fromCursor(call[0],validFuncName)
  mess  = "Incorrect use of {}(), use {}() instead".format(funcCursor.displayname,validFuncName)
  diag  = addFunctionFixToBadSource.diags.incompatible_function
  linter.addErrorFromCursor(obj,Diagnostic(diag,mess,patch=patch))
  return

def convertToCorrectPetscValidLogicalCollectiveXXX(linter,obj,objType,funcCursor=None,**kwargs):
  """
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
    addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName)
    return True
  return False

def convertToCorrectPetscValidXXXPointer(linter,obj,objType,funcCursor=None,**kwargs):
  """
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
    addFunctionFixToBadSource(linter,obj,funcCursor,validFuncName)
    return True
  return False

@DiagnosticManager.register(
  ("incompatible-type","Verify that a particular type matches the expected type")
)
def checkIsTypeXAndNotTypeY(typeX,typeY,linter,obj,objType,funcCursor=None,validFunc=None):
  """
  Check that a cursor is at least some form of derived type X and not some form of type Y
  i.e. for

  myInt **********x;

  you may check that 'x' is some form of 'myInt' instead of say 'PetscBool'
  """
  derivedname = obj.derivedtypename
  if typeX not in derivedname:
    if typeY in derivedname:
      addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
    else:
      mess = "Incorrect use of {funcName}(), {funcName}() should only be used for {}".format(
        typeX,funcName=funcCursor.displayname
      )
      diag = checkIsTypeXandNotTypeY.diags.incompatible_type
      linter.addErrorFromCursor(obj,Diagnostic(diag,mess))
  return True

def checkIsPetscScalarAndNotPetscReal(*args,**kwargs):
  return checkIsTypeXAndNotTypeY("PetscScalar","PetscReal",*args,**kwargs)

def checkIsPetscRealAndNotPetscScalar(*args,**kwargs):
  return checkIsTypeXAndNotTypeY("PetscReal","PetscScalar",*args,**kwargs)

def checkIsNotType(typename,linter,obj,funcCursor=None,validFunc=None):
  if isinstance(typename,str):
    contains = typename in obj.derivedtypename
  elif isinstance(typename,(tuple,list)):
    contains = any(t in obj.derivedtypename for t in typename)
  else:
    raise ValueError(type(typename))
  if contains:
    addFunctionFixToBadSource(linter,obj,funcCursor,validFunc)
  return True

def checkIntIsNotPetscBool(linter,obj,*args,**kwargs):
  return checkIsNotType("PetscBool",linter,obj,**kwargs)

def checkMPIIntIsNotPetscInt(linter,obj,*args,**kwargs):
  return checkIsNotType("PetscInt",linter,obj,**kwargs)

@DiagnosticManager.register(
  ("incompatible-function","Verify that the correct function was used for a type")
)
def checkIsPetscBool(linter,obj,*args,funcCursor=None,**kwargs):
  #return checkIsNotType(("PetscBool","bool"),linter,obj,**kwargs)
  if ("PetscBool" not in obj.derivedtypename) and ("bool" not in obj.typename):
    mess = "Incorrect use of {funcName}(), {funcName}() should only be used for PetscBool or bool".format(funcName=funcCursor.displayname)
    diag = checkIsPetscBool.diags.incompatible_function
    linter.addErrorFromCursor(obj,Diagnostic(diag,mess))
  return True

@DiagnosticManager.register(("incompatible-type-petscobject","Verify that a symbol is a PetscObject"))
def checkIsPetscObject(linter,obj):
  """
  Returns True if obj is a valid PetscObject, otherwise False. Automatically adds the error to the linter. Raises RuntimeError if obj is a PetscObject that isn't registered in the classIdMap.
  """
  diagName = checkIsPetscObject.diags.incompatible_type_petscobject
  if not obj.typename.startswith("_p_"):
    linter.addErrorFromCursor(
      obj,Diagnostic(diagnosticFlag,"Non-PETSc type when PETSc object expected")
    )
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
    objDecl = PetscCursor.cast(pObjType.get_declaration())
    if len(objFields) == 0:
      mess = "Object '{}' is prefixed with '_p_' to indicate it is a PetscObject but cannot determine fields. Likely the header containing definition of the object is in a nonstandard place:\n{}\n{}".format(objDecl.typename,objDecl.getFormattedLocationString(),objDecl.formatted(nafter=2))
      linter.addWarningFromCursor(obj,Diagnostic(diagnosticFlag,mess))
    else:
      mess = "Object '{}' is prefixed with '_p_' to indicate it is a PetscObject but its definition is missing a PETSCHEADER as the first struct member:\n{}\n{}".format(objDecl.typename,objDecl.getFormattedLocationString(),objDecl.formatted(nafter=2))
      linter.addErrorFromCursor(obj,Diagnostic(diagnosticFlag,mess))
  return validObject

@DiagnosticManager.register(
  ("incompatible-classid","Verify that the given classid matches the PetscObject type")
)
def checkMatchingClassid(linter,obj,objClassid):
  """
  Does the classid match the particular PETSc type
  """
  checkIsPetscObject(linter,obj)
  expected = classIdMap[obj.typename]
  name     = objClassid.name
  if name != expected:
    mess     = "Classid doesn't match. Expected '{}' found '{}'".format(expected,name)
    diagname = checkMatchingClassid.diags.incompatible_classid
    diag     = Diagnostic(diagname,mess,patch=Patch.fromCursor(objClassid,expected))
    linter.addErrorFromCursor(obj,diag)
  return

def checkTraceableToParentArgs(obj,parentArgNames):
  """
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
      if len([r for r in refsAll if r.storage_class == clx.StorageClass.STATIC]):
        # a global variable is not a function argumment, so this is unhandleable
        raise ParsingError("PETSC_CLANG_STATIC_ANALYZER_IGNORE")

    assert len(argRefs), "Could not determine the origin of cursor {}".format(obj)
    # take the first, as this is the earliest
    firstRef = argRefs[0]
    tu,line  = firstRef.translation_unit,firstRef.location.line
    srcLen   = len(firstRef.getRawSource())
    # why the following song and dance? Because you cannot walk the AST backwards, and
    # in the case that the current cursor is in a function call we need to access
    # our co-arguments to the function, i.e. "adjacent" branches since they should link
    # to (or be) in the parent functions argument list. So we have to
    # essentially reparse this line to be able to start from the top.
    lineStart = PetscSourceLocation.fromPosition(tu,line,1)
    lineEnd   = PetscSourceLocation.fromPosition(tu,line,srcLen+1)
    lineRange = PetscSourceRange.fromLocations(lineStart,lineEnd).sourceRange
    tGroup    = list(clx.TokenGroup.get_tokens(tu,lineRange))
    funcProto = [
      i for i,t in enumerate(tGroup) if t.cursor.type.get_canonical().kind in functionTypes
    ]
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
      loc = parentArgNames.index(name)
    except ValueError as ve:
      # name isn't in the parent arguments, so we raise parsing error from it
      raise ParsingError from ve
  else:
    parent = PetscCursor(parent,obj.argidx)
    # deeper into the rabbit hole
    loc = checkTraceableToParentArgs(parent,parentArgNames)
  return loc

@DiagnosticManager.register(("matching-arg-num","Verify that the given argument number matches"))
def checkMatchingArgNum(linter,obj,idx,parentArgs):
  """
  Is the Arg # correct w.r.t. the function arguments
  """
  diagName = checkMatchingArgNum.diags.matching_arg_num
  if idx.canonical.kind not in mathCursors:
    # sometimes it is impossible to tell if the index is correct so this is a warning not
    # an error. For example in the case of a loop:
    # for (i = 0; i < n; ++i) PetscValidIntPointer(arr+i,i);
    linter.addWarningFromCursor(
      idx,Diagnostic(diagName,"Index value is of unexpected type '{}'".format(idx.canonical.kind))
    )
    return
  try:
    idxNum = int(idx.name)
  except ValueError:
    linter.addWarningFromCursor(
      idx,Diagnostic(diagName,"Potential argument mismatch, could not determine integer value")
    )
    return
  parentArgNames = tuple(s.name for s in parentArgs)
  try:
    expected = parentArgs[parentArgNames.index(obj.name)]
  except ValueError:
    try:
      expected = parentArgs[checkTraceableToParentArgs(obj,parentArgNames)]
    except ParsingError as pe:
      # If the parent arguments don't contain the symbol and we couldn't determine a
      # definition then we cannot check for correct numbering, so we cannot do
      # anything here but emit a warning
      if "PETSC_CLANG_STATIC_ANALYZER_IGNORE" in pe.args:
        return
      if len(parentArgs):
        parentFunc = PetscCursor(parentArgs[0].semantic_parent)
        parentFuncName = parentFunc.name+"()"
        parentFuncSrc  = parentFunc.formatted()
      else:
        # parent function has no arguments (very likely that "obj" is a global variable)
        parentFuncName = "UNKNOWN FUNCTION"
        parentFuncSrc  = "  <could not determine parent function signature from arguments>"
      mess = "Cannot determine index correctness, parent function '{}' seemingly does not contain the object:\n{}".format(parentFuncName,parentFuncSrc)
      linter.addWarningFromCursor(obj,Diagnostic(diagName,mess))
      return
  expIdx = expected.argidx
  if idxNum != expIdx:
    errMess = "Argument number doesn't match for '{}'. Found '{}' expected '{}' from:\n{}".format(
      obj.name,str(idxNum),str(expIdx),expected.formatted()
    )
    linter.addErrorFromCursor(idx,Diagnostic(diagName,errMess,patch=Patch.fromCursor(idx,expIdx)))
  return

@DiagnosticManager.register(
  ("incompatible-type","Verify that a particular type matches the expected type")
)
def checkMatchingSpecificType(linter,obj,expectedTypeKinds,pointer,unexpectedNotPointerFunction=alwaysFalse,unexpectedPointerFunction=alwaysFalse,successFunction=alwaysTrue,failureFunction=alwaysFalse,**kwargs):
  """
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
  diagName = checkMatchingSpecificType.diags.incompatible_type
  objType  = obj.canonical.type.get_canonical()
  if pointer:
    if objType.kind in expectedTypeKinds:
      if not unexpectedNotPointerFunction(linter,obj,objType,**kwargs):
        mess = "Object of clang type {} is not a pointer. Expected pointer of one of the following types: {}".format(objType.kind,expectedTypeKinds)
        linter.addErrorFromCursor(obj,Diagnostic(diagName,mess))
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
        mess = "Object of clang type {} is a pointer when it should not be".format(objType.kind)
        linter.addErrorFromCursor(obj,Diagnostic(diagName,mess))
      return
  if objType.kind in expectedTypeKinds:
    if not successFunction(linter,obj,objType,**kwargs):
      errorMessage = "{}\nType checker successfully matched object of type {} to (one of) expected types:\n- {}\n\nBut user supplied on-successful-match hook '{}' returned non-truthy value indicating unhandled error!".format(obj,objType.kind,'\n- '.join(map(str,expectedTypeKinds)),successFunction)
      raise RuntimeError(errorMessage)
  else:
    if not failureFunction(linter,obj,objType,**kwargs):
      mess = "Object of clang type {} is not in expected types: {}".format(objType.kind,expectedTypeKinds)
      linter.addErrorFromCursor(obj,Diagnostic(diagName,mess))
  return


"""Specific 'driver' function to test a particular function archetype"""
def checkObjIdxGenericN(linter,func,parent):
  """
  For generic checks where the form is func(obj1,idx1,...,objN,idxN)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  for obj,idx in zip(funcArgs[::2],funcArgs[1::2]):
    checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidHeaderSpecificType(linter,func,parent):
  """
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
  """
  Specific check for PetscValidHeaderSpecific(obj,classid,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  obj,classid,idx = funcArgs
  checkMatchingClassid(linter,obj,classid)
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidPointerAndType(linter,func,parent,expectedTypes,failureFunction=convertToCorrectPetscValidXXXPointer,**kwargs):
  """
  Generic check for PetscValidXXXPointer(obj,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  obj,idx = funcArgs
  checkMatchingSpecificType(
    linter,obj,expectedTypes,True,failureFunction=failureFunction,funcCursor=func,**kwargs
  )
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidCharPointer(linter,func,parent):
  """
  Specific check for PetscValidCharPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,charTypes)
  return

def checkPetscValidIntPointer(linter,func,parent):
  """
  Specific check for PetscValidIntPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,intTypes,successFunction=checkIntIsNotPetscBool,validFunc="PetscValidBoolPointer")
  return

def checkPetscValidBoolPointer(linter,func,parent):
  """
  Specific check for PetscValidBoolPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,boolTypes,successFunction=checkIsPetscBool)
  return

def checkPetscValidScalarPointer(linter,func,parent):
  """
  Specific check for PetscValidScalarPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,scalarTypes,successFunction=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidRealPointer")
  return

def checkPetscValidRealPointer(linter,func,parent):
  """
  Specific check for PetscValidRealPointer(obj,idx)
  """
  checkPetscValidPointerAndType(linter,func,parent,realTypes,successFunction=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidScalarPointer")
  return

def checkPetscValidLogicalCollective(linter,func,parent,expectedTypes,failureFunction=convertToCorrectPetscValidLogicalCollectiveXXX,**kwargs):
  """
  Generic check for PetscValidLogicalCollectiveXXX(pobj,obj,idx)
  """
  funcArgs   = linter.getArgumentCursors(func)
  parentArgs = linter.getArgumentCursors(parent)

  # dont need the petsc object, nothing to check there
  _,obj,idx = funcArgs
  checkMatchingSpecificType(
    linter,obj,expectedTypes,False,failureFunction=failureFunction,funcCursor=func,**kwargs
  )
  checkMatchingArgNum(linter,obj,idx,parentArgs)
  return

def checkPetscValidLogicalCollectiveScalar(linter,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveScalar(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,scalarTypes,successFunction=checkIsPetscScalarAndNotPetscReal,validFunc="PetscValidLogicalCollectiveReal")
  return

def checkPetscValidLogicalCollectiveReal(linter,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveReal(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,realTypes,successFunction=checkIsPetscRealAndNotPetscScalar,validFunc="PetscValidLogicalCollectiveScalar")
  return

def checkPetscValidLogicalCollectiveInt(linter,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveInt(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,intTypes,successFunction=checkIntIsNotPetscBool,validFunc="PetscValidLogicalCollectiveBool")
  return

def checkPetscValidLogicalCollectiveMPIInt(linter,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveMPIInt(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,mpiIntTypes,successFunction=checkMPIIntIsNotPetscInt,validFunc="PetscValidLogicalCollectiveInt")
  return

def checkPetscValidLogicalCollectiveBool(linter,func,parent):
  """
  Specific check for PetscValidLogicalCollectiveBool(pobj,obj,idx)
  """
  checkPetscValidLogicalCollective(linter,func,parent,boolTypes,successFunction=checkIsPetscBool)
  return

def checkPetscValidLogicalCollectiveEnum(linter,func,parent):
  """
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
  """
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
  """
  Crudely tries to find libclang directory first using ctypes.util.find_library(), then llvm-config, and then finally checks a few places on macos
  """
  import ctypes.util

  llvmLibDir = ctypes.util.find_library("clang")
  if not llvmLibDir:
    try:
      output = subprocessRun(
        ["llvm-config","--libdir"],capture_output=True,universal_newlines=True,check=True
      )
      llvmLibDir = output.stdout.strip()
    except FileNotFoundError:
      # FileNotFoundError: [Errno 2] No such file or directory: 'llvm-config'
      # try to find llvmLibDir by hand
      import platform

      if platform.system().casefold() == "darwin":
        try:
          output = subprocessRun(
            ["xcode-select","-p"],capture_output=True,universal_newlines=True,check=True
          )
          xcodeDir = output.stdout.strip()
          if xcodeDir == "/Applications/Xcode.app/Contents/Developer": # default Xcode path
            llvmLibDir = "/".join([xcodeDir,"Toolchains","XcodeDefault.xctoolchain","usr","lib"])
          elif xcodeDir == "/Library/Developer/CommandLineTools":      # CLT path
            llvmLibDir = "/".join([xcodeDir,"usr","lib"])
        except FileNotFoundError:
          # FileNotFoundError: [Errno 2] No such file or directory: 'xcode-select'
          pass
  return PetscPath(llvmLibDir).resolve() if llvmLibDir else llvmLibDir

def initializeLibclang(clangDir=None,clangLib=None):
  """
  Set the required library file or directory path to initialize libclang
  """
  clxconf = clx.conf
  if not clxconf.loaded:
    clxconf.set_compatibility_check(True)
    if clangLib:
      clangLib = PetscPath(clangLib).resolve()
      clxconf.set_library_file(clangLib)
    elif clangDir:
      clangDir = PetscPath(clangDir).resolve()
      clxconf.set_library_path(clangDir)
    else:
      raise RuntimeError("Must supply either clang directory path or clang library path")
  return clangDir,clangLib

def filterCheckFunctionMap(filterChecks):
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
        petscIncludes.append(line.split("=",maxsplit=1)[1])
      elif mpiinc.search(line) or shoinc.search(line):
        mpiIncludes.append(line.split("=",maxsplit=1)[1])
      elif cxxflg.search(line):
        cxxflags.append(line.split("=",maxsplit=1)[1])
  cxxflags      = [l.strip().split() for l in cxxflags if l]
  cxxflags      = [flag for flags in cxxflags for flag in flags if flag.startswith("-std=")]
  cxxflags      = [cxxflags[-1]] if cxxflags else [] # take only the last one
  extraIncludes = [l.strip().split() for l in petscIncludes+mpiIncludes if l]
  extraIncludes = [item for sublist in extraIncludes for item in sublist if item.startswith("-I")]
  seen          = set()
  extraIncludes = [item for item in extraIncludes if not item in seen and not seen.add(item)]
  return cxxflags+extraIncludes

def getClangSysIncludes():
  """
  Get system clangs set of default include search directories.

  Because for some reason these are hardcoded by the compilers and so libclang does not have them.
  """
  output = subprocessRun(
    ["clang","-E","-x","c++","/dev/null","-v"],
    capture_output=True,check=True,universal_newlines=True
  )
  # goes to stderr because of /dev/null
  includes = output.stderr.split("#include <...> search starts here:\n")[1]
  includes = includes.split("End of search list.",maxsplit=1)[0].replace(
    "(framework directory)",""
  ).splitlines()
  return ["".join(["-I",str(PetscPath(i.strip()).resolve())]) for i in includes if i]

def buildCompilerFlags(petscDir,petscArch,extraCompilerFlags=[],verbose=False,printPrefix="[ROOT]"):
  """
  build the baseline set of compiler flags, these are passed to all translation unit parse attempts
  """
  miscFlags = [
    "-DPETSC_CLANG_STATIC_ANALYZER",
    "-xc++",
    "-Wno-empty-body",
    "-Wno-writable-strings",
    "-Wno-array-bounds",
    "-Wno-nullability-completeness",
    "-fparse-all-comments",
  ]
  petscIncludes = getPetscExtraIncludes(petscDir,petscArch)
  compilerFlags = getClangSysIncludes()+miscFlags+petscIncludes+extraCompilerFlags
  if verbose: print("\n".join([" ".join([printPrefix,"Compile flags:"]),*compilerFlags]))
  return compilerFlags

def buildPrecompiledHeader(petscDir,compilerFlags,extraHeaderIncludes=[],verbose=False,printPrefix="[ROOT]",pchClangOptions=basePCHClangOptions):
  """
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
      if not self.fname.exists():
        mess = "Filename {} does not appear to exist".format(self.fname)
        raise RuntimeError(mess)
      return

    def __enter__(self):
      self.tmp     = tempfile.NamedTemporaryFile(delete=True,suffix=self.fname.suffix)
      self.tmpPath = PetscPath(self.tmp.name).resolve()
      if not self.tmpPath.exists():
        mess = "tmpPath {} does not appear to exist".format(self.tmpPath)
        raise RuntimeError(mess)
      shutil.copy2(str(self.fname),str(self.tmpPath))
      return self

    def __exit__(self,*args,**kwargs):
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
    return [] if text is None else [l.replace(str(petscDir),".") for l in text.splitlines(True)]

  def sanitizePatchFile(text):
    # skip the diff header with file names
    return [] if text is None else text.splitlines(True)[2:]

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
      with TemporaryCopy(testFile) as tmpSrc, \
           tempfile.NamedTemporaryFile(delete=True,suffix=".patch") as tmpPatch:
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

def queueMain(clangLib,checkFunctionMapU,classIdMapU,diagMapU,compilerFlags,clangOptions,verbose,werror,errorQueue,returnQueue,fileQueue,lock):
  """
  main function for worker processes in the queue, does pretty much the same thing the
  main process would do in their place
  """
  def updateGlobals(updatedCheckFunctionMap,updatedClassIdMap,updatedDiagnosticsMngr):
    # in a function so the "globalness" doesn't leak
    global checkFunctionMap,classIdMap,DiagnosticManager
    checkFunctionMap = updatedCheckFunctionMap
    classIdMap       = updatedClassIdMap
    DiagnosticManager.disabled = updatedDiagnosticsMngr.disabled
    return

  def lockPrint(*args,**kwargs):
    if verbose:
      with lock:
        print(*args,**kwargs)
    return

  # in case errors are thrown before setup is complete
  errorPrefix = "[UNKNOWN_CHILD]"
  filename    = "QUEUE SETUP"
  printbar    = 15*"="
  try:
    updateGlobals(checkFunctionMapU,classIdMapU,diagMapU)
    proc        = mp.current_process().name
    printPrefix = proc+" --"[:len("[ROOT]")-len(proc)]
    errorPrefix = " ".join([printPrefix,"Exception detected while processing"])
    lockPrint(printPrefix,printbar,"Performing setup",printbar)
    initializeLibclang(clangLib=clangLib)
    linter = PetscLinter(compilerFlags,clangOptions=clangOptions,prefix=printPrefix,verbose=verbose,werror=werror,lock=lock)
    lockPrint(printPrefix,printbar,"Entering queue  ",printbar)
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
    lockPrint(printPrefix,printbar,"Exiting queue   ",printbar)
  except Exception:
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
  try:
    errorQueue.close()
    returnQueue.close()
  except Exception:
    pass
  return

def main(petscDir,petscArch,srcPath=None,clangDir=None,clangLib=None,verbose=False,workers=-1,checkFunctionFilter=None,patchDir=None,applyPatches=False,extraCompilerFlags=[],extraHeaderIncludes=[],testOutputDir=None,replaceTests=False,werror=False):
  """
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
      mess = "Got neither a directory or file as srcPath {}".format(srcPath)
      raise RuntimeError(mess)

  if testOutputDir is not None and not testOutputDir.exists():
    mess = "Test Output Directory {} does not appear to exist".format(testOutputDir)
    raise RuntimeError(mess)

  filterCheckFunctionMap(checkFunctionFilter)
  rootPrintPrefix   = "[ROOT]"
  compilerFlags     = buildCompilerFlags(
    petscDir,petscArch,extraCompilerFlags=extraCompilerFlags,verbose=verbose
  )

  class PrecompiledHeader(object):
    __slots__ = "pch"

    def __init__(self,*args,**kwargs):
      self.pch = buildPrecompiledHeader(*args,**kwargs)
      return

    def __enter__(self):
      return self

    def __exit__(self,*args,**kwargs):
      if verbose: print(rootPrintPrefix,"Deleting precompiled header",self.pch)
      self.pch.unlink()
      return

  with PrecompiledHeader(
      petscDir,compilerFlags,extraHeaderIncludes=extraHeaderIncludes,verbose=verbose
  ):
    warnings,errorsLeft,errorsFixed,patches = WorkerPool(
      workers,verbose=verbose
    ).setup(compilerFlags,werror=werror).walk(srcPath).finalize()

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
          ["patch",rootDir,"--strip=0","--unified","--input={}".format(patchFile)],
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
      mess = "Boolean value expected, got '{}'".format(v)
      raise argparse.ArgumentTypeError(mess)

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
  grouppetsc.add_argument("--PETSC_DIR",default=petscDir,help="if this option is unused defaults to environment variable $PETSC_DIR",dest="petscdir")
  grouppetsc.add_argument("--PETSC_ARCH",default=petscArch,help="if this option is unused defaults to environment variable $PETSC_ARCH",dest="petscarch")

  parser.add_argument("-s","--src-dir",default=defaultSrcDir,help="Alternate base directory of source tree (e.g. $SLEPC_DIR/src)",dest="src")
  parser.add_argument("-v","--verbose",type=str2bool,nargs="?",const=True,default=False,help="verbose progress printed to screen")
  filterFuncChoices = ", ".join(list(checkFunctionMap.keys()))
  parser.add_argument("--functions",nargs="+",choices=list(checkFunctionMap.keys()),metavar="FUNCTIONNAME",help="filter to display errors only related to list of provided function names, default is all functions. Choose from available function names: "+filterFuncChoices,dest="funcs")
  parser.add_argument("-j","--jobs",type=int,const=-1,default=-1,nargs="?",help="number of multiprocessing jobs, -1 means number of processors on machine")
  parser.add_argument("-p","--patch-dir",help="directory to store patches in if they are generated, defaults to SRC_DIR/../petscLintPatches",dest="patchdir")
  parser.add_argument("-a","--apply-patches",type=str2bool,nargs="?",const=True,default=False,help="automatically apply patches that are saved to file",dest="apply")
  parser.add_argument("--CXXFLAGS",nargs="+",default=[],help="extra flags to pass to CXX compiler",dest="cxxflags")
  parser.add_argument("--test",nargs="?",const="__at_src__",help="test the linter for correctness. Optionally provide a directory containing the files against which to compare patches, defaults to SRC_DIR/output if no argument is given. The files of correct patches must be in the format [path_from_src_dir_to_testFileName].out")
  parser.add_argument("--replace",type=str2bool,nargs="?",const=True,default=False,help="replace output files in test directory with patches generated")
  parser.add_argument("--werror",type=str2bool,nargs="?",const=True,default=False,help="treat all warnings as errors")

  class CheckFilter(argparse.Action):
    def __call__(self,parser,namespace,values,*args,**kwargs):
      flag = self.dest.replace(DiagnosticManager.flagprefix[1:],"",1).replace("_","-")
      if flag == "diagnostics-all":
        for diag,_ in DiagnosticManager.registered().items():
          DiagnosticManager.set(diag,values)
      else:
        DiagnosticManager.set(flag,values)
      setattr(namespace,flag,values)
      return

  groupdiag = parser.add_argument_group(title="diagnostics")
  groupdiag.add_argument("-fdiagnostics-all",type=str2bool,nargs="?",const=True,default=True,action=CheckFilter,help="disable all diagnostics")
  for diag,helpstr in sorted(DiagnosticManager.registered().items()):
    groupdiag.add_argument("-f"+diag,metavar="",type=str2bool,nargs="?",const=True,default=True,action=CheckFilter,help=helpstr)

  args = parser.parse_args()

  if args.petscdir is None:
    raise RuntimeError("Could not determine PETSC_DIR from environment, please set via options")
  if args.petscarch is None:
    raise RuntimeError("Could not determine PETSC_ARCH from environment, please set via options")

  if args.clanglib:
    args.clangdir = None

  ret = main(args.petscdir,args.petscarch,srcPath=args.src,clangDir=args.clangdir,clangLib=args.clanglib,verbose=args.verbose,workers=args.jobs,checkFunctionFilter=args.funcs,patchDir=args.patchdir,applyPatches=args.apply,extraCompilerFlags=args.cxxflags,testOutputDir=args.test,replaceTests=args.replace,werror=args.werror)
  sys.exit(ret)
