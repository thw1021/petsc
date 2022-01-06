#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:56:06 2021

@author: jacobfaibussowitsch
"""
import enum
import pathlib
import ctypes
import functools
import inspect
import weakref
import clang.cindex as clx

def verbosePrint(*args,**kwargs):
    '''filter predicate for show_ast: show all'''
    return True
def noSystemIncludes(cursor,level,**kwargs):
    '''filter predicate for show_ast: filter out verbose stuff from system include files'''
    return level != 1 or (cursor.location.file is not None and not cursor.location.file.name.startswith('/usr/include'))

def onlyFile(cursor,level,**kwargs):
  '''filter predicate to only show ast defined in file'''
  filename = kwargs['filename']
  return level != 1 or (cursor.location.file is not None and cursor.location.file.name == filename)

# A function show(level, *args) would have been simpler but less fun
# and you'd need a separate parameter for the AST walkers if you want it to be exchangeable.
class Level(int):
  """
  represent currently visited level of a tree
  """
  def view(self,*args):
    """
    pretty print an indented line
    """
    return '\t'*self+' '.join(map(str, args))
  def __add__(self,inc):
    """
    increase number of tabs and newlines
    """
    return Level(super(Level, self).__add__(inc))

def checkValidType(t):
    return t.kind != clx.TypeKind.INVALID

def fullyQualify(t):
  q = set()
  if t.is_const_qualified(): q.add('const')
  if t.is_volatile_qualified(): q.add('volatile')
  if t.is_restrict_qualified(): q.add('restrict')
  return q

def viewType(t,level,title):
  """
  pretty print type AST
  """
  retList = [level.view(title, str(t.kind),' '.join(fullyQualify(t)))]
  if checkValidType(t.get_pointee()):
    retList.extend(viewType(t.get_pointee(),level+1,'points to:'))
  return retList

def viewAstFromCursor(cursor,pred=verbosePrint,level=Level(),**kwargs):
  """
  pretty print cursor AST
  """
  retList = []
  if pred(cursor,level,**kwargs):
    retList.append(level.view(cursor.kind,cursor.spelling,cursor.displayname,cursor.location))
    if checkValidType(cursor.type):
      retList.extend(viewType(cursor.type,level+1,'type:'))
      retList.extend(viewType(cursor.type.get_canonical(),level+1,'canonical type:'))
    for c in cursor.get_children():
      retList.extend(viewAstFromCursor(c,pred=pred,level=level+1,**kwargs))
  return retList


def getRawSourceFromSourceRange(sourceRange,numBeforeContext=0,numAfterContext=0,numContext=0,trim=False,tight=False):
  numBeforeContext = numBeforeContext if numBeforeContext else numContext
  numAfterContext  = numAfterContext  if numAfterContext  else numContext
  rstart,rend = sourceRange.start,sourceRange.end
  lineBegin   = rstart.line
  lineEnd     = rend.line
  lobound     = max(1,lineBegin-numBeforeContext)
  hibound     = lineEnd+numAfterContext
  with open(rstart.file.name,"r") as fd:
    lineList = [l for i,l in enumerate(fd,start=1) if lobound <= i <= hibound]
  if tight:
    assert lineBegin == lineEnd
    # index into lineList where our actual line starts if we have context
    loidx,hiidx = lineBegin-lobound,hibound-lineBegin
    cbegin,cend = rstart.column-1,rend.column-1
    if loidx == hiidx:
      # same line, then we need to do it in 1 step to keep the indexing correct
      lineList[loidx] = lineList[loidx][cbegin:cend]
    else:
      lineList[loidx] = lineList[loidx][cbegin:]
      lineList[hiidx] = lineList[hiidx][:cend]
  # Find number of spaces to remove from beginning of line based on lowest.
  # This keeps indentation between lines, but doesn't start the string halfway
  # across the screeen
  if trim:
    minSpaces = min(len(s)-len(s.lstrip()) for s in lineList if s.replace("\n",""))
    return "\n".join([s[minSpaces:].rstrip() for s in lineList])
  return "".join(lineList)

def getRawSourceFromCursor(cursor,**kwargs):
  return getRawSourceFromSourceRange(cursor.extent,**kwargs)


def getFormattedSourceFromSourceRange(sourceRange,numBeforeContext=0,numAfterContext=0,numContext=0,view=False,highlight=True,trim=True):
  numBeforeContext  = numBeforeContext if numBeforeContext else numContext
  numAfterContext   = numAfterContext  if numAfterContext  else numContext
  begin,end         = sourceRange.start,sourceRange.end
  lineBegin,lineEnd = begin.line,end.line
  loBound  = max(1,lineBegin-numBeforeContext)
  hiBound  = lineEnd+numAfterContext
  maxWidth = len(str(hiBound))
  if highlight:
    symbolBegin  = begin.column-1
    symbolEnd    = end.column-1
    beginOffset  = max(symbolBegin,0)
    lenUnderline = max(abs(max(symbolEnd,1)-beginOffset),1)
    underline    = beginOffset*" "+lenUnderline*"^"
  with open(begin.file.name,"r") as fd:
    lineList = []
    for lineFile,line in enumerate(fd,start=1):
      if loBound <= lineFile <= hiBound:
        indicator = ">" if (lineBegin <= lineFile <= lineEnd) else " "
        prefix    = "{indicator} {lineFile: <{width}}: ".format(indicator=indicator,lineFile=lineFile,width=maxWidth)
        lineList.append((prefix,line))
        if highlight and (lineFile == lineBegin):
          lineList.append((" "*len(prefix),underline))
  # Find number of spaces to remove from beginning of line based on lowest.
  # This keeps indentation between lines, but doesn't start the string halfway
  # across the screen
  if trim:
    minSpaces = min([len(s)-len(s.lstrip(" ")) for _,s in lineList if s.replace("\n","")])
  else:
    minSpaces = 0
  lineList = [p+s[minSpaces:].rstrip() for p,s in lineList]
  srcStr   = "\n".join(lineList)
  if view:
    print(srcStr)
  return srcStr

def getFormattedSourceFromCursor(cursor,**kwargs):
  return getFormattedSourceFromSourceRange(cursor.extent,**kwargs)


def viewCursorFull(cursor):
  try:
    print("Arguments:      "," ".join([a.displayname for a in cursor.get_arguments()]))
  except AttributeError:
    pass
  try:
    print("Semantic Parent:",cursor.semantic_parent.displayname)
  except AttributeError:
    pass
  try:
    print("Lexical Parent: ",cursor.lexical_parent.displayname)
  except AttributeError:
    pass
  print("Children:       "," ".join([c.spelling for c in cursor.get_children()]))
  print("Storage Class:  ",cursor.storage_class)
  print("AST View:\n")
  print("\n".join(viewAstFromCursor(cursor)))
  return


class CXTranslationUnit(enum.IntFlag):
  """
  clang.cindex.TranslationUnit does not have all latest flags

  see: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html#gab1e4965c1ebe8e41d71e90203a723fe9
  """
  NONE                                 = 0x0
  DetailedPreprocessingRecord          = 0x01
  Incomplete                           = 0x02
  PrecompiledPreamble                  = 0x04
  CacheCompletionResults               = 0x08
  ForSerialization                     = 0x10
  SkipFunctionBodies                   = 0x40
  IncludeBriefCommentsInCodeCompletion = 0x80
  CreatePreambleOnFirstParse           = 0x100
  KeepGoing                            = 0x200
  SingleFileParse                      = 0x400
  LimitSkipFunctionBodiesToPreamble    = 0x800
  IncludeAttributedTypes               = 0x1000
  VisitImplicitAttributes              = 0x2000
  IgnoreNonErrorsFromIncludedFiles     = 0x4000
  RetainExcludedConditionalBlocks      = 0x8000


class Scope(object):
  """
  Scope encompasses both the logical and lexical reach of a callsite, and is used to
  determine if two function calls may be occur in chronological order. Scopes may be
  approximated by incrementing or decrementing a counter every time a pair of '{}' are
  encountered however it is not that simple. In practice they behave almost identically
  to sets. Every relation between scopes may be formed by the following axioms.

  - Scope A is said to be greater than scope B if one is able to get to scope B from scope A
  e.g.:
  { // scope A
    { // scope B < scope A
      ...
    }
  }
  - Scope A is said to be equivalent to scope B if and only if they are the same object.
  e.g.:
  { // scope A and scope B
    ...
  }

  One notable exception are switch-case statements. Here every 'case' label acts as its
  own scope, regardless of whether a "break" is inserted i.e.:

  switch (cond) { // scope A
  case 1: // scope B begin
    ...
    break; // scope B end
  case 2: // scope C begin
    ...
  case 2:// scope C end, scope D begin
    ...
    break; // scope D end
  }

  Semantics here are weird, as:
  - scope B, C, D < scope A
  - scope B != scope C != scope D
  """
  __slots__ = "gen","super","children"

  def __init__(self,superScope=None):
    if superScope:
      assert isinstance(superScope,Scope)
      self.gen    = superScope.gen+1
    else:
      self.gen    = 0
    self.super    = superScope
    self.children = []
    return

  def __str__(self):
    return "gen {} id {}".format(self.gen,id(self))

  def __lt__(self,other):
    assert isinstance(other,Scope)
    return not (self >= other)

  def __gt__(self,other):
    assert isinstance(other,Scope)
    return self.isChildOf(other)

  def __le__(self,other):
    assert isinstance(other,Scope)
    return not (self > other)

  def __ge__(self,other):
    assert isinstance(other,Scope)
    return (self > other) or (self == other)

  def __eq__(self,other):
    if other is not None:
      assert isinstance(other,Scope)
      return id(self) == id(other)
    return False

  def __ne__(self,other):
    return not (self == other)

  def sub(self):
    """spawn sub-scope"""
    child = Scope(self)
    self.children.append(child)
    return child

  def isParentOf(self,other):
    """self is parent of other"""
    if self == other:
      return False
    for child in self.children:
      if (other == child) or child.isParentOf(other):
        return True
    return False

  def isChildOf(self,other):
    """self is child of other, or other is parent of self"""
    return other.isParentOf(self)


@functools.total_ordering
class PetscSourceLocation(object):
  """
  A simple wrapper class to add comparison operators to clx.SourceLocations since they only
  implement __eq__()
  """

  class ClangFileNameCache(weakref.WeakKeyDictionary):
    """
    It is for whatever reason stupidly expensive to create these file objects, and clang does it
    every time you access a tu's file. So we cache them here
    """
    def getname(self,tu):
      return self.setdefault(tu,tu.get_file(tu.spelling))


  __filecache = ClangFileNameCache()
  __slots__   = "sourceLocation","translation_unit","_cache"

  def getCached(self,attr,func,*args,**kwargs):
    cache = self._cache
    return cache[attr] if attr in cache else cache.setdefault(attr,func(*args,**kwargs))

  def __init__(self,sourceLoc,tu=None):
    self.sourceLocation   = sourceLoc
    self.translation_unit = tu # store a reference to guard against GC
    self._cache           = {}
    return

  def __repr__(self):
    return "\n".join([
      "self:             {}".format(object.__repr__(self)),
      "Translation Unit: {}".format(self.translation_unit),
      "Source Location:  {}".format(self.sourceLocation)
    ])

  def __getattr__(self,attr):
    return self.getCached(attr,getattr,self.sourceLocation,attr)

  def __eq__(self,right):
    return self.sourceLocation.__eq__(self.asClangSourceLocation(right))

  def __lt__(self,right):
    if not isinstance(right,(clx.SourceLocation,type(self))):
      raise ValueError(type(right))
    lline,rline = self.line,right.line
    if lline < rline:
      return True
    if lline == rline:
      return self.column < right.column
    return False # lline > rline


  @classmethod
  def cast(cls,other):
    if isinstance(other,cls):
      return other
    elif isinstance(other,clx.SourceLocation):
      return cls(other)
    raise NotImplementedError(type(other))

  @classmethod
  def getFilenameFromTU(cls,tu):
    return cls.__filecache.getname(tu)

  @classmethod
  def fromPosition(cls,tu,line,col):
    return cls(clx.SourceLocation.from_position(tu,cls.getFilenameFromTU(tu),line,col),tu=tu)

  @classmethod
  def asClangSourceLocation(cls,other):
    if isinstance(other,clx.SourceLocation):
      return other
    elif isinstance(other,cls):
      return other.sourceLocation
    raise NotImplementedError(type(other))


@functools.total_ordering
class PetscSourceRange(object):
  """Like PetscSourceLocation but for clx.SourceRanges"""
  __slots__ = "sourceRange","translation_unit","_cache"

  def __init__(self,sourceRange,tu=None):
    self.sourceRange      = sourceRange
    self.translation_unit = tu # store a reference to guard against GC
    self._cache           = {}
    return

  def __repr__(self):
    return "\n".join([
      "self:             {}".format(object.__repr__(self)),
      "Translation Unit: {}".format(self.translation_unit),
      "Source Range:     {}".format(self.sourceRange)
    ])

  def getCached(self,attr,func,*args,**kwargs):
    cache = self._cache
    return cache[attr] if attr in cache else cache.setdefault(attr,func(*args,**kwargs))

  def __getattr__(self,attr):
    return self.getCached(attr,getattr,self.sourceRange,attr)

  def __eq__(self,other):
    return self.sourceRange.__eq__(self.asClangSourceRange(other))

  def __lt__(self,right):
    if isinstance(right,(clx.SourceRange,type(self))):
      right = right.start
    elif isinstance(right,(clx.SourceLocation,PetscSourceLocation)):
      pass
    else:
      raise NotImplementedError(type(right))
    return PetscSourceLocation.cast(self.end) <= right

  def __contains__(self,other):
    def contains(loc):
      # reimplement clx.SourceRange.__contains__() as it has a bug
      return cacheStart <= cast(loc) <= cacheEnd

    cast       = PetscSourceLocation.cast
    cacheEnd   = self._cache.setdefault("PetscSourceLocationEnd",cast(self.end))
    cacheStart = self._cache.setdefault("PetscSourceLocationStart",cast(self.start))
    if isinstance(other,(clx.SourceLocation,PetscSourceLocation)):
      return contains(other)
    if isinstance(other,(clx.SourceRange,type(self))):
      return contains(other.start) and contains(other.end)
    raise NotImplementedError(type(other))

  def __len__(self):
    return self.end.offset-self.start.offset

  @classmethod
  def cast(cls,other,**kwargs):
    if isinstance(other,cls):
      return other
    if isinstance(other,clx.SourceRange):
      return cls(other,**kwargs)
    raise NotImplementedError(type(other))

  @classmethod
  def fromLocations(cls,left,right,tu=None):
    if tu is None:
      tu = getattr(left,"translation_unit",None)
      if tu is None:
        tu = getattr(right,"translation_unit",None)
    asClangSL = PetscSourceLocation.asClangSourceLocation
    return cls(clx.SourceRange.from_locations(asClangSL(left),asClangSL(right)),tu=tu)

  @classmethod
  def fromPositions(cls,tu,lineLeft,colLeft,lineRight,colRight):
    filename = PetscSourceLocation.getFilenameFromTU(tu)
    fromPos  = clx.SourceLocation.from_position
    begin    = fromPos(tu,filename,lineLeft,colLeft)
    end      = fromPos(tu,filename,lineRight,colRight)
    return cls(clx.SourceRange.from_locations(begin,end),tu=tu)

  @classmethod
  def asClangSourceRange(cls,other):
    if isinstance(other,clx.SourceRange):
      return other
    if isinstance(other,cls):
      return other.sourceRange
    raise NotImplementedError(type(other))

  @classmethod
  def merge(cls,left,right,**kwargs):
    cast = PetscSourceLocation.cast
    start = min(cast(left.start),cast(right.start))
    end   = max(cast(left.end),cast(right.end))
    return cls.fromLocations(start,end,**kwargs)


  def mergeWith(self,other):
    return self.merge(self,other,tu=self.translation_unit)


  def overlaps(self,other):
    cast      = PetscSourceLocation.cast
    getCached = self.getCached
    return (getCached("PetscSourceLocationEnd",cast,self.end) >= cast(other.start)) and \
      (cast(other.end) >= getCached("PetscSourceLocationStart",cast,self.start))

  def resized(self,lbegin=0,lend=0,cbegin=0,cend=0):
    """
    return a resized PetscSourceRange, if the sourceRange was resized it is a new object

    lbegin - number of lines to increment or decrement self.start.lines by
    lend   - number of lines to increment or decrement self.end.lines by
    cbegin - number of columns to increment or decrement self.start.colummn by, None for BOL
    cend   - number of columns to increment or decrement self.end.colummn by, None for EOL
    """
    start,end = self.start,self.end
    if cbegin is None:
      cbegin = -start.column+1
    if lbegin+lend+cbegin == 0 and cend == 0:
      return self # nothing to do

    endcol = -1 if cend is None else end.column+cend # -1 is EOL
    return self.fromPositions(
      self.translation_unit,start.line+lbegin,start.column+cbegin,end.line+lend,endcol
    )


  def raw(self,**kwargs):
    return getRawSourceFromSourceRange(self,**kwargs)

  def formatted(self,**kwargs):
    return getFormattedSourceFromSourceRange(self,**kwargs)

  def view(self,**kwargs):
    kwargs.setdefault("numContext",5)
    return print(self.formatted(**kwargs))


CXCursorAndRangeVisitorCallBackProto = ctypes.CFUNCTYPE(
  ctypes.c_uint,ctypes.py_object,clx.Cursor,clx.SourceRange
)

class PetscCXCursorAndRangeVisitor(ctypes.Structure):
  # see https://clang.llvm.org/doxygen/structCXCursorAndRangeVisitor.html
  #
  # typedef struct CXCursorAndRangeVisitor {
  #   void *context;
  #   enum CXVisitorResult (*visit)(void *context, CXCursor, CXSourceRange);
  # } CXCursorAndRangeVisitor;
  #
  # Note this is not a  strictly accurate recreation, as this struct expects a
  # (void *) but since C lets anything be a (void *) we can pass in a (PyObject *)
  _fields_ = [
    ("context",ctypes.py_object),
    ("visit",CXCursorAndRangeVisitorCallBackProto)
  ]


class Scope(object):
  """
  Scope encompasses both the logical and lexical reach of a callsite, and is used to
  determine if two function calls may be occur in chronological order. Scopes may be
  approximated by incrementing or decrementing a counter every time a pair of '{}' are
  encountered however it is not that simple. In practice they behave almost identically
  to sets. Every relation between scopes may be formed by the following axioms.

  - Scope A is said to be greater than scope B if one is able to get to scope B from scope A
  e.g.:
  { // scope A
    { // scope B < scope A
      ...
    }
  }
  - Scope A is said to be equivalent to scope B if and only if they are the same object.
  e.g.:
  { // scope A and scope B
    ...
  }

  One notable exception are switch-case statements. Here every 'case' label acts as its
  own scope, regardless of whether a "break" is inserted i.e.:

  switch (cond) { // scope A
  case 1: // scope B begin
    ...
    break; // scope B end
  case 2: // scope C begin
    ...
  case 2:// scope C end, scope D begin
    ...
    break; // scope D end
  }

  Semantics here are weird, as:
  - scope B, C, D < scope A
  - scope B != scope C != scope D
  """
  __slots__ = "gen","super","children"

  def __init__(self,superScope=None):
    if superScope:
      assert isinstance(superScope,Scope)
      self.gen    = superScope.gen+1
    else:
      self.gen    = 0
    self.super    = superScope
    self.children = []
    return

  def __str__(self):
    return "gen {} id {}".format(self.gen,id(self))

  def __lt__(self,other):
    if isinstance(other,Scope):
      return not (self >= other)
    raise ValueError(type(other))

  def __gt__(self,other):
    if isinstance(other,Scope):
      return self.isChildOf(other)
    raise ValueError(type(other))

  def __le__(self,other):
    if isinstance(other,Scope):
      return not (self > other)
    raise ValueError(type(other))

  def __ge__(self,other):
    assert isinstance(other,Scope)
    return (self > other) or (self == other)

  def __eq__(self,other):
    if other is not None:
      if isinstance(other,Scope):
        return id(self) == id(other)
      raise ValueError(type(other))
    return False

  def __ne__(self,other):
    return not (self == other)

  def sub(self):
    """spawn sub-scope"""
    child = Scope(self)
    self.children.append(child)
    return child

  def isParentOf(self,other):
    """self is parent of other"""
    if self == other:
      return False
    for child in self.children:
      if (other == child) or child.isParentOf(other):
        return True
    return False

  def isChildOf(self,other):
    """self is child of other, or other is parent of self"""
    return other.isParentOf(self)


class PetscPath(type(pathlib.Path())):
  # inheriting pathlib.Path:
  # https://stackoverflow.com/questions/29850801/subclass-pathlib-path-fails
  def append_suffix(self,suffix):
    suffix    = str(suffix)
    dotstring = "" if suffix.startswith(".") else "."
    return self.with_suffix(dotstring.join((self.suffix,suffix)))

  def append_name(self,name):
    return self.with_name("".join((self.stem,str(name))))


class ParsingError(Exception):
  """
  Mostly to just have a custom "something went wrong when trying to perform a check" to except
  for rather than using a built-in type. These are errors that are meant to be caught and logged
  rather than stopping execution alltogether.

  This should make it so that actual errors aren't hidden.
  """
  pass


class _DiagnosticsManager(object):
  class DiagnosticMap(object):
    """
    A dict-like object that allows 'DiagnosticMap.my_diagnostic_name' to return 'my-diagnostic-name'
    """
    __slots__ = "_diags"

    def __init__(self,dlist):
      self._diags = {attr.replace("-","_") : attr for attr in dlist}
      return

    def __getattr__(self,attr):
      diags = self._diags
      try:
        return diags[attr]
      except KeyError:
        try:
          return [v for k,v in diags.items() if k.endswith(attr)][0]
        except IndexError:
          print("USING DICT GETATTR FOR ATTR",attr)
          return getattr(diags,attr)


  _registered = {}
  __slots__   = "disabled","flagprefix"

  @classmethod
  def registered(cls):
    return cls._registered

  @staticmethod
  def __expandFlag(flag):
    if not isinstance(flag,str):
      try:
        flag = "-".join(flag)
      except Exception as ex:
        raise ValueError(type(flag)) from ex
    return flag

  @classmethod
  def checkFlag(cls,flag):
    flag = cls.__expandFlag(flag)
    if flag not in cls._registered:
      mess = "Flag '{}' is not registered with {}".format(flag,cls)
      raise ValueError(mess)
    return flag

  @classmethod
  def register(cls,*args):
    def decorator(symbol):
      expandFlag = cls.__expandFlag

      if inspect.isclass(symbol):
        diagList = [(expandFlag(symbol.diagnostic(f)),h.casefold()) for f,h in args]
        wrapper  = symbol
      else:
        @functools.wraps(symbol)
        def wrapper(*args,**kwargs):
          return symbol(*args,**kwargs)

        diagList = [(expandFlag(d),h.casefold()) for d,h in args]

      assert not hasattr(wrapper,"diags"),"Object {} already has a diags attribute".format(wrapper)
      wrapper.diags = cls.DiagnosticMap(d for d,_ in diagList)
      cls._registered.update(diagList)
      return wrapper
    return decorator


  def __init__(self,flagprefix="-f"):
    self.disabled   = set()
    self.flagprefix = flagprefix if flagprefix.startswith("-") else "-"+flagprefix
    return


  def disable(self,flag):
    self.disabled.add(self.checkFlag(flag))
    return

  def enable(self,flag):
    self.disabled.discard(self.checkFlag(flag))
    return

  def set(self,flag,value):
    return self.enable(flag) if value else self.disable(flag)

  def disabledFor(self,flag):
    return self.checkFlag(flag) in self.disabled

  def enabledFor(self,flag):
    return not self.disabledFor(flag)

  def makeCommandLineFlag(self,flag):
    return self.flagprefix+self.checkFlag(flag)


DiagnosticManager = _DiagnosticsManager()

class Diagnostic(object):
  __slots__ = "flag","message","patch","clflag"

  def __init__(self,flag,message,patch=None):
    self.flag    = DiagnosticManager.checkFlag(flag)
    self.message = str(message)
    self.patch   = patch
    self.clflag  = DiagnosticManager.makeCommandLineFlag(self.flag).join((" [","]"))
    return

  def __repr__(self):
    return "\n".join((
      "flag:  {}".format(self.clflag),
      "patch: {}".format(self.patch),
      "message:\n{}".format(self.message)
    ))

  def formatMessage(self):
    message = self.message
    pos     = message.find(":")
    if pos == -1:
      ret = "".join((message,self.clflag))
    else:
      if message[pos-1].isdigit():
        import ipdb; ipdb.set_trace()
      ret = message.replace(":",self.clflag+":",1)
    return ret

  def disabled(self):
    return DiagnosticManager.disabledFor(self.flag.replace("_","-"))
