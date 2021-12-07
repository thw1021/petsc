#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:56:06 2021

@author: jacobfaibussowitsch
"""
import clang.cindex as clx
import ctypes
import functools

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


def getRawSourceFromCursor(cursor,**kwargs):
  return getRawSourceFromSourceRange(cursor.extent,**kwargs)

def getRawSourceFromSourceRange(sourceRange,numBeforeContext=0,numAfterContext=0,numContext=0,trim=False,tight=False):
  numBeforeContext = numBeforeContext if numBeforeContext else numContext
  numAfterContext  = numAfterContext  if numAfterContext  else numContext
  lineBegin = sourceRange.start.line
  lineEnd   = sourceRange.end.line
  lobound   = max(1,lineBegin-numBeforeContext)
  hibound   = lineEnd+numAfterContext
  with open(sourceRange.start.file.name,"r") as fd:
    lineList = [l for i,l in enumerate(fd,start=1) if (lobound <= i <= hibound)]
  if tight:
    import ipdb; ipdb.set_trace()
    # index into lineList where our actual line starts if we have context
    loidx           = lineBegin-lobound
    lineList[loidx] = lineList[loidx][sourceRange.start.column-1:]
    hiidx           = hibound-lineBegin
    lineList[hiidx] = lineList[hiidx][:sourceRange.end.column-1]
  # Find number of spaces to remove from beginning of line based on lowest.
  # This keeps indentation between lines, but doesn't start the string halfway
  # across the screeen
  if trim:
    minSpaces = min([len(s)-len(s.lstrip(' ')) for s in lineList if s.replace("\n","")])
    return "\n".join([s[minSpaces:].rstrip() for s in lineList])
  return "".join(lineList)


def getFormattedSourceFromCursor(cursor,**kwargs):
  return getFormattedSourceFromSourceRange(cursor.extent,**kwargs)

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


@functools.total_ordering
class PetscSourceLocation(object):
  """A simple wrapper class to add comparison operators to clx.SourceLocations since they only implement equal"""
  __slots__ = ("sourceLocation",)

  def __init__(self,sourceLoc):
    assert sourceLoc.line >= 1, "startline {} < 1".format(sourceLoc.line)
    self.sourceLocation = sourceLoc
    return

  def __getattr__(self,attr):
    return getattr(self.sourceLocation,attr)

  @classmethod
  def cast(cls,other):
    if isinstance(other,cls):
      return other
    elif isinstance(other,clx.SourceLocation):
      return cls(other)
    else:
      raise NotImplementedError

  @classmethod
  def fromPosition(cls,tu,filename,line,col):
    return cls(clx.SourceLocation.from_position(tu,filename,line,col))

  @classmethod
  def getClangSourceLocation(cls,other):
    if isinstance(other,clx.SourceLocation):
      return other
    elif isinstance(other,cls):
      return other.sourceLocation
    else:
      raise NotImplementedError

  def __eq__(self,other):
    return self.sourceLocation.__eq__(self.getClangSourceLocation(other))

  def __lt__(self,other):
    other = self.getClangSourceLocation(other)
    if self.line < other.line:
      return True
    elif self.line == other.line:
      return self.column < other.column
    else:
      return False

  def __contains__(self,other):
    return self.sourceLocation.__contains__(self.getClangSourceLocation(other))


@functools.total_ordering
class PetscSourceRange(object):
  """Like PetscSourceLocation but for clx.SourceRanges"""
  __slots__ = ("sourceRange",)

  def __init__(self,sourceRange):
    self.sourceRange = sourceRange
    return

  def __getattr__(self,attr):
    return getattr(self.sourceRange,attr)

  def __eq__(self,other):
    return self.sourceRange.__eq__(self.getClangSourceRange(other))

  def __lt__(self,other):
    raise NotImplementedError

  def __contains__(self,other):
    return self.sourceRange.__contains__(self.getClangSourceRange(other))

  @classmethod
  def cast(cls,other):
    if isinstance(other,cls):
      return other
    elif isinstance(other,clx.SourceRange):
      return cls(other)
    else:
      raise NotImplementedError

  @classmethod
  def fromLocations(cls,left,right):
    return cls(clx.SourceRange.from_locations(
      PetscSourceLocation.getClangSourceLocation(left),
      PetscSourceLocation.getClangSourceLocation(right)
    ))

  @classmethod
  def getClangSourceRange(cls,other):
    if isinstance(other,clx.SourceRange):
      return other
    elif isinstance(other,cls):
      return other.sourceRange
    else:
      raise NotImplementedError

  @classmethod
  def merge(cls,left,right):
    left  = cls.cast(left)
    right = cls.cast(right)
    if left in right:
      return right
    elif right in left:
      return left
    begin = min(
      PetscSourceLocation.cast(left.start),
      PetscSourceLocation.cast(right.start)
    ).sourceLocation
    end   = max(
      PetscSourceLocation.cast(left.end),
      PetscSourceLocation.cast(right.end)
    ).sourceLocation
    return cls.fromLocations(begin,end)

  def mergeWith(self,other):
    return self.merge(self,other)

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


def viewCursorFull(cursor):
  try:
    print("Arguments:"," ".join([a.displayname for a in cursor.get_arguments()]))
  except AttributeError:
    pass
  try:
    print("Semantic Parent:",cursor.semantic_parent.displayname)
  except AttributeError:
    pass
  try:
    print("Lexical Parent:",cursor.lexical_parent.displayname)
  except AttributeError:
    pass
  print("Children:"," ".join([c.spelling for c in cursor.get_children()]))
  print("AST View")
  print("\n".join(viewAstFromCursor(cursor)))
  return
