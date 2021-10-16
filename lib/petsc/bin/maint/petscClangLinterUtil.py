#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:56:06 2021

@author: jacobfaibussowitsch
"""
import clang.cindex as clx
import petscClangLinter

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


def getRawSourceFromCursor(cursor,numBeforeContext=0,numAfterContext=0,numContext=0,trim=False):
  return getRawSourceFromSourceRange(cursor.extent,numBeforeContext=numBeforeContext,numAfterContext=numAfterContext,numContext=numContext,trim=trim)

def getRawSourceFromSourceRange(sourceRange,numBeforeContext=0,numAfterContext=0,numContext=0,trim=False):
  numBeforeContext = numBeforeContext if numBeforeContext else numContext
  numAfterContext  = numAfterContext  if numAfterContext  else numContext
  lineno  = sourceRange.start.line
  lobound = max(1,lineno-numBeforeContext)
  hibound = lineno+numAfterContext
  with open(sourceRange.start.file.name,"r") as fd:
    lineList = [l for i,l in enumerate(fd,start=1) if (lobound <= i <= hibound)]
  # with open(sourceRange.start.file.name,"r") as fd:
  #   for lineFile,line in enumerate(fd,start=1):
  #     if lineFile >= lineno-numBeforeContext and lineFile <= lineno+numAfterContext:
  #       lineList.append(line)
  # Find number of spaces to remove from beginning of line based on lowest.
  # This keeps indentation between lines, but doesn't start the string halfway
  # across the screeen
  if trim:
    minSpaces = min([len(s)-len(s.lstrip(' ')) for s in lineList if s.replace("\n","")])
    lineList  = [s[minSpaces:].rstrip() for s in lineList]
  return "\n".join(lineList)


def getFormattedSourceFromCursor(cursor,numBeforeContext=0,numAfterContext=0,numContext=0,view=False,highlight=True,trim=True):
  return getFormattedSourceFromSourceRange(cursor.extent,numBeforeContext=numBeforeContext,numAfterContext=numAfterContext,numContext=numContext,view=view,highlight=highlight,trim=trim)

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
