#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:56:06 2021

@author: jacobfaibussowitsch
"""
import clang

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
    '''represent currently visited level of a tree'''
    def show(self,*args):
        '''pretty print an indented line'''
        print('\t'*self+' '.join(map(str, args)))
    def __add__(self,inc):
        '''increase level'''
        return Level(super(Level, self).__add__(inc))

def checkValidType(t):
    return t.kind != clang.cindex.TypeKind.INVALID

def fullyQualify(t):
    q = set()
    if t.is_const_qualified(): q.add('const')
    if t.is_volatile_qualified(): q.add('volatile')
    if t.is_restrict_qualified(): q.add('restrict')
    return q

def viewType(t,level,title):
    '''pretty print type AST'''
    level.show(title, str(t.kind),' '.join(fullyQualify(t)))
    if checkValidType(t.get_pointee()):
        viewType(t.get_pointee(),level+1,'points to:')

def viewAstRecursive(cursor,pred=verbosePrint,level=Level(),**kwargs):
    '''pretty print cursor AST'''
    if pred(cursor,level,**kwargs):
      level.show(cursor.kind,cursor.spelling,cursor.displayname,cursor.location)
      if checkValidType(cursor.type):
        viewType(cursor.type,level+1,'type:')
        viewType(cursor.type.get_canonical(),level+1,'canonical type:')
      for c in cursor.get_children():
        viewAstRecursive(c,pred=pred,level=level+1,**kwargs)
