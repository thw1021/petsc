#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:05:39 2021

@author: jacobfaibussowitsch
"""
import os
import subprocess
import clang.cindex
from petscClangParserUtil import viewAstRecursive,verbose,noSystemIncludes,onlyFile

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
      if c.location.line == 393:
        bla = 1
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
      tokens = [t.spelling for t in cursor.get_tokens() if stringOnlyAlpha(t.spelling)]
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
    'struct _p_Mat *' : 'MAT_CLASSID',
    '_p_Mat *' : 'MAT_CLASSID',
    'struct _p_Vec *' : 'VEC_CLASSID',
    '_p_Vec *' : 'VEC_CLASSID',
    'struct _p_PetscRandom *' : 'PETSC_RANDOM_CLASSID',
    '_p_PetscRandom *' : 'PETSC_RANDOM_CLASSID',
    'struct _p_PetscViewer *' : 'PETSC_VIEWER_CLASSID',
    '_p_PetscViewer *' : 'PETSC_VIEWER_CLASSID'
    }
  try:
    expectedClassid = classidMap[obj.typename]
  except KeyError:
    badSource.append("Unkown class "+str(obj))
    return badSource
  if expectedClassid != objClassid.name:
    badSource.append("Classid doesn't match. Expected '"+expectedClassid+"' found "+str(objClassid)+"'.\nUse classid for "+str(obj))
  return badSource

def checkMatchingArgNum(badSource,obj,idx,parentArgs):
  """
  Is the Arg # correct w.r.t. the function arguments
  """
  objName = obj.name
  idxstr = ''.join(t.spelling for t in idx.cursor.get_tokens())
  parentArgNames = tuple(s.name for s in parentArgs)
  try:
    matchLoc = parentArgNames.index(objName)
  except ValueError:
    badSource.append("Parent function '"+parentArgs[0].cursor.semantic_parent.spelling+"' arguments:\n  "+"\n  ".join(str(i+1)+": "+str(s) for i,s in enumerate(parentArgs))+"\nDon't contain the object "+str(obj))
    return badSource
  if int(idxstr) != parentArgs[matchLoc].argidx:
    badSource.append("Argument number doesn't match. Expected '"+str(parentArgs[matchLoc].argidx)+"' found '"+idxstr+"' instead for "+str(obj))
  return badSource

def checkPetscValidHeaderSpecific(badSource,func,parent):
  """
  Specific check for PetscValidHeaderSpecific, can be made more general
  """
  funcArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(func.get_arguments()))
  parentArgs = tuple(ArgCursor(a,i+1) for i,a in enumerate(parent.get_arguments()))
  obj,classid,idx = funcArgs
  if idx.cursor.canonical.kind != clang.cindex.CursorKind.INTEGER_LITERAL:
    badSource.append("Index value is of unexpect type "+idx.cursor.canonical.kind+" not INTEGER_LITERAL for "+str(idx))
    return badSource
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
  if idx.cursor.canonical.kind != clang.cindex.CursorKind.INTEGER_LITERAL:
    badSource.append("Index value is of unexpected type "+idx.cursor.canonical.kind+" not INTEGER_LITERAL for "+str(idx))
    return badSource
  badSource = checkMatchingArgNum(badSource,obj,idx,parentArgs)
  return badSource

def getClangSysIncludes():
  """
  Get system clangs set of default include search directories.

  Because for some reason these are hardcoded by the compilers and so libclang does not have them.
  """
  output = subprocess.run(['clang','-E','-x','c++','/dev/null','-v'],capture_output=True,check=True,universal_newlines=True)
  output.check_returncode()
  # goes to stderr because of /dev/null
  includes = output.stderr.split("#include <...> search starts here:\n")[1]
  includes = includes.split('End of search list.')[0].replace('(framework directory)','')
  includes = includes.split('\n')
  includes = ["-I"+os.path.abspath(i.strip()) for i in includes if i]
  return includes

def main(petscdir,petscarch,src,clangdir):
  checkDict = {
    "PetscValidPointer" : checkPetscValidPointer,
    "PetscValidHeaderSpecific" : checkPetscValidHeaderSpecific,
  }
  oldloc = os.getcwd()
  os.chdir(os.path.join(petscdir,'src'))

  sysincludes = getClangSysIncludes()
  flags = ['-x','c++','-include','../include/petscastfix.hpp',os.path.join('-I'+petscdir,'include'),os.path.join('-I'+petscdir,petscarch,'include'),os.path.join('-I'+os.environ['PETSC_PACK_DIR'],'include')]
  flags = sysincludes+flags
  if not clang.cindex.Config.loaded:
    clang.cindex.Config.set_compatibility_check(True)
    clang.cindex.Config.set_library_path(clangdir)
  index = clang.cindex.Index.create()
  tu = index.parse(src,args=flags)
  diags = {d.spelling : 0 for d in tu.diagnostics}
  if len(diags):
    for k in diags: print(k)
    raise RuntimeError
  badSource = []
  for func,parent in findFunctionCallExpr(tu,checkDict.keys()):
    badSource = checkDict[func.spelling](badSource,func,parent)
  for s in badSource:
    print(s)
  os.chdir(oldloc)
  return

if __name__ == "__main__":
  petscdir = os.environ['PETSC_DIR']
  petscarch = os.environ['PETSC_ARCH']
  src = os.path.join(petscdir,'src','sys','classes','random','interface','randomc.c')
  clangDir = '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/'
  #clangDir = '/Library/Developer/CommandLineTools/usr/lib/'
  main(petscdir,petscarch,src,clangDir)
