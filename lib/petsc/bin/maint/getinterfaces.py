#!/usr/bin/env python3
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $
#
# change python to whatever is needed on your system to invoke python
#
#  Processes PETSc's include/petsc*.h files to determine
#  the PETSc enums, functions and classes
#
#  Crude as all hack!
#
#  Calling sequence:
#      getinterfaces *.h
##
import os
import re
import sys
import pickle
import pathlib
import shutil

CToFortran = {'int':'integer4', 'ptrdiff_t':'PetscInt64', 'float':'PetscFortranFloat', 'int32_t':'integer4', 'double':'PetscFortranDouble',
              'short':None, 'size_t':None, 'rocblas_status':None}

classes = {}         # classes[classname] = Dict(nothing in it)
enums = {}           # enums[enumname] = List(dict(enumkey : enumvalue))
senums = {}          # like enums except strings instead of integer values for enumvalue
typedefs = {}
aliases = {}
structs = {}
files = {}           # files[filename]  = set(filenames contained in filename)
mansecs = {}         # mansec[mansecname] = set(alls submansecnames in mansecname)
submansecs = set()

regcomment  = re.compile(r'/\* [-A-Za-z _(),<>|^\*/0-9.:=\[\]\.]* \*/')
regcomment2  = re.compile(r'// [-A-Za-z _(),<>|^\*/0-9.:=\[\]\.]*')
regblank    = re.compile(r' [ ]*')

class List(list):
  """
  A subclass of list that can accept attributes.
  Should be able to be used just like a regular list.
  """
  def __new__(self, *args, **kwargs):
    return super(List, self).__new__(self, args, kwargs)

  def __init__(self, *args, **kwargs):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
      list.__init__(self, args[0])
    else:
      list.__init__(self, args)
      self.__dict__.update(kwargs)

  def __call__(self, **kwargs):
    self.__dict__.update(kwargs)
    return self

class Typedef:
    def __init__(self, *args, **kwargs):
        self.name = None
        self.mansec = 'mansec'
        self.submansec = 'submansec'
        self.file = 'file'
        self.value = None

class Dict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.mansec = 'mansec'
        self.submansec = 'submansec'
        self.file = 'file'
        self.petscobject = True
        self.opaque = True

class Set(set):
    def __init__(self, *args, **kwargs):
        set.__init__(self, *args, **kwargs)
        self.mansec = 'mansec'
        self.submansec = 'submansec'
        self.file = 'file'

def findmansec(line,mansec,submansec):
  '''Finds mansec and submansec in line'''
  if line.find(' MANSEC') > -1:
    mansec = re.sub(r'[ ]*/\* [ ]*MANSEC[ ]*=[ ]*','',line).strip('\n').strip('*/').strip()
    if mansec == line[0].strip('\n'):
      mansec = re.sub('MANSEC[ ]*=[ ]*','',line.strip('\n').strip())
    mansec = mansec.lower()
  if line.find('SUBMANSEC') > -1:
    submansec = re.sub(r'[ ]*/\* [ ]*SUBMANSEC[ ]*=[ ]*','',line).strip('\n').strip('*/').strip()
    if submansec == line[0].strip('\n'):
      submansec = re.sub('SUBMANSEC[ ]*=[ ]*','',line.strip('\n').strip())
    submansec = submansec.lower()
    if not mansec: mansec = submansec
    submansecs.add(submansec)
    if not mansec in mansecs: mansecs[mansec] = set()
    mansecs[mansec].add(submansec)
  return mansec,submansec

def getfiles(filename):
  import re

  file = os.path.basename(filename)
  files[file] = Dict()
  submansec = None
  mansec = None
  reginclude  = re.compile(r'^#include <[A-Za-z_0-9]*.h')
  f = open(filename)
  line = f.readline()
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = reginclude.search(line)
    if fl:
      line = regcomment.sub("",line)
      line = line.replace('#include <','').replace('>','').strip()
      line = line.replace('types.h','.h')
      if not line == file and os.path.isfile(os.path.join('include',line)):
        files[file][line] = None
    line = f.readline()
  files[file].mansec = mansec
  f.close()

def getenums(filename):
  import re
  regtypedef  = re.compile(r'typedef [ ]*enum')
  reg         = re.compile(r'}')
  regname     = re.compile(r'}[ A-Za-z0-9]*')

  file = os.path.basename(filename).replace('types.h','.h')
  f = open(filename)
  line = f.readline()
  submansec = None
  mansec = None
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regtypedef.search(line)
    if fl:
      struct = line
      while line:
        fl = reg.search(line)
        if fl:
          struct = regcomment.sub("",struct)
          struct = struct.replace("\\","")
          struct = struct.replace("\n","")
          struct = struct.replace(";","")
          struct = struct.replace("typedef enum","")
          struct = regblank.sub(" ",struct)

          name = regname.search(struct)
          name = name.group(0)
          name = name.replace("} ","")

          values = struct[struct.find("{")+1:struct.find("}")]
          values = values.split(',')

          ivalues = []
          for i in values:
            if i:
              if i[0] == " ": i = i[1:]
              ivalues.append(i)

          enums[name] = List(ivalues)
          if not submansec: raise RuntimeError('No SUBMANSEC in file '+filename)
          enums[name].submansec = submansec
          enums[name].mansec = mansec
          enums[name].file = file
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
  f.close()

def getsenums(filename):
  import re
  regdefine   = re.compile(r'typedef const char \*[A-Za-z]*;')
  file = os.path.basename(filename).replace('types.h','.h')
  submansec = None
  mansec = None
  f = open(filename)
  line = f.readline()
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regdefine.search(line)
    if fl:
      senum = fl.group(0)[20:-1]
      line = regblank.sub(" ",f.readline().strip())
      d = {}
      while line:
        values = line.split(" ")
        d[values[1]] = values[2]
        line = regblank.sub(" ",f.readline().strip())
      senums[senum] = Dict(d)
      if not submansec: raise RuntimeError('No SUBMANSEC in file '+filename)
      senums[senum].mansec = mansec
      senums[senum].submansec = submansec
      senums[senum].file = file
    line = f.readline()
  f.close()

def gettypedefs(filename):
  import re
  file = os.path.basename(filename).replace('types.h','.h')
  regdefine   = re.compile(r'typedef [A-Za-z0-9_]* [A-Za-z0-9_]*;')
  submansec = None
  mansec = None
  f = open(filename)
  line = f.readline()
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regdefine.search(line)
    if fl:
      typedef = fl.group(0).split()[2][0:-1];
      if typedef in typedefs:
        typedefs[typedef].name = None
      else:
        typedefs[typedef] = Typedef()
        typedefs[typedef].name = typedef
        typedefs[typedef].value = fl.group(0).split()[1]
        if not submansec: raise RuntimeError('No SUBMANSEC in file '+filename)
        typedefs[typedef].mansec = mansec
        typedefs[typedef].submansec = submansec
        typedefs[typedef].file = file
    line = f.readline()
  f.close()

def getstructs(filename):
  import re
  file = os.path.basename(filename).replace('types.h','.h')
  regtypedef  = re.compile(r'^typedef [ ]*struct {')
  reg         = re.compile(r'}')
  regname     = re.compile(r'}[ A-Za-z]*')
  submansec = None
  mansec = None
  f = open(filename)
  line = f.readline()
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regtypedef.search(line)
    opaque = True
    if fl:
      struct = line
      while line:
        fl = reg.search(line)
        if fl:
          struct = regcomment.sub("",struct)
          struct = regcomment2.sub("",struct)
          struct = struct.replace("\\","")
          struct = struct.replace("\n","")
          struct = struct.replace("typedef struct {","")
          struct = regblank.sub(" ",struct)
          struct = struct.replace("; ",";")

          name = regname.search(struct)
          name = name.group(0)
          name = name.replace("} ","")

          values = struct[struct.find("{")+1:struct.find(";}")]
          if values.find('#') > -1 or values.find('*') > -1 or values.find('][') > -1: opaque = False
          if not values.find('#') == -1: break
          values = values.split(";")
          ivalues = []
          for i in values:
            ivalues.append(i.strip())
          structs[name] = List(ivalues)
          structs[name].mansec = mansec
          structs[name].submansec = submansec
          structs[name].file = file
          structs[name].opaque = opaque  # cannot easily map to another language
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
  f.close()

def getclasses(filename):
  import re
  regclass    = re.compile(r'typedef struct _[np]_[A-Za-z_]*[ ]*\*')
  regnclass    = re.compile(r'typedef struct _n_[A-Za-z_]*[ ]*\*')
  regsemi     = re.compile(r';')
  submansec = None
  mansec = None
  file = os.path.basename(filename).replace('types.h','.h')
  f = open(filename)
  line = f.readline()
  while line:
    mansec,submansec = findmansec(line,mansec,submansec)
    fl = regclass.search(line)
    gl = regnclass.search(line)
    if fl:
      struct = line
      struct = regclass.sub("",struct)
      struct = regcomment.sub("",struct)
      struct = regblank.sub("",struct)
      struct = regsemi.sub("",struct)
      struct = struct.replace("\n","")
      classes[struct] = Dict({})
      if not submansec: raise RuntimeError('No SUBMANSEC in file '+filename)
      classes[struct].mansec = mansec
      classes[struct].submansec = submansec
      classes[struct].file = file
      if gl: classes[struct].petscobject = False
    line = f.readline()
  f.close()

def getfunctions(filename):
  import re
  regfun      = re.compile(r'EXTERN PetscErrorCode PETSC[A-Z]*_DLLEXPORT ')
  regarg      = re.compile(r'\([A-Za-z*_\[\]]*[,\)]')
  regerror    = re.compile(r'PetscErrorCode')

  rejects     = ['PetscErrorCode','DALocalFunction','...','<','(*)','(**)','off_t','MPI_Datatype','va_list','size_t','PetscStack']
  #
  # search through list BACKWARDS to get the longest match
  #
  classlist   = classes.keys()
  classlist = sorted(classlist)
  classlist.reverse()
  f = open(filename)
  line = f.readline()
  while line:
    fl = regfun.search(line)
    if fl:
      struct = line
      struct = regfun.sub("",struct)
      struct = regcomment.sub("",struct)
      struct = struct.replace("unsigned ","u")
      struct = regblank.sub("",struct)
      struct = struct.replace("\n","")
      struct = struct.replace("const","")
      struct = struct.replace(";","")
      struct = struct.strip()
      fl = regarg.search(struct)
      if fl:
        arg = fl.group(0)
        arg = arg[1:-1]
        reject = 0
        for i in rejects:
          if struct.find(i) > -1:
            reject = 1
        if  not reject:
          args = struct[struct.find("(")+1:struct.find(")")]
          args = args.split(",")
          if args == ['void']: args = []
          name = struct[:struct.find("(")]
          for i in classlist:
            if name.startswith(i):
              classes[i][name[len(i):]] = args
              break
    line = f.readline()
  f.close()

def main(petscarch):
  args = [os.path.join('include',i) for i in os.listdir('include') if i.endswith('.h')]
  for i in args:
    getfiles(i)
  for i in args:
    getenums(i)
  for i in args:
    getsenums(i)
  for i in args:
    getstructs(i)
  # these classes ONLY have static methods
  #classes['Petsc'] = Dict({})
  #classes['PetscLog'] = Dict({})
  #classes['PetscSort'] = Dict({})
  #classes['PetscStr'] = Dict({})
  #classes['PetscBinary'] = Dict({})
  #classes['PetscOptions'] = Dict({})
  #classes['PetscMalloc'] = Dict({})
  #classes['PetscToken'] = Dict({})
  for i in args:
    getclasses(i)
  for i in args:
    getfunctions(i)
  for i in args:
    gettypedefs(i)
  #file = open('classes.data','wb')
  #pickle.dump(enums,file)
  #pickle.dump(senums,file)
  #pickle.dump(structs,file)
  #pickle.dump(aliases,file)
  #pickle.dump(classes,file)
  #pickle.dump(typedefs,file)

  with open(os.path.join('doc','objects.md'),"w") as fd:
    fd.write(':html_theme.sidebar_secondary.remove: true\n')
    fd.write('::::{tab-set}\n\n')

    fd.write(':::{tab-item} PETSc objects\n')
    for i in sorted(list(classes.keys())):
      fd.write('- '+i+'\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} Typedefs to basic types\n')
    for i in sorted(list(typedefs.keys())):
      if i in ['VecScatter', 'VecScatterType']: continue
      if typedefs[i].name:
        fd.write('- '+i+' = '+typedefs[i].value+'\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} Opaque structs\n')
    for i in sorted(list(structs.keys())):
      fd.write('- '+i+' ' + str(structs[i].opaque)+'\n')
      for j in structs[i]:
        fd.write('  - '+j+'\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} Enums\n')
    for i in sorted(list(enums.keys())):
      if i in ['PetscEnum']: continue
      fd.write(':::{dropdown} '+i+'\n')
      for j in enums[i]:
         v = j.strip()
         if v.find('=') > -1 : v = v[0:v.find('=')].strip()
         fd.write('  - '+v+'\n')
      fd.write(':::\n')
    fd.write(':::\n')

    fd.write(':::{tab-item} String enums\n')
    for i in sorted(list(senums.keys())):
      fd.write('- '+i+'\n')
      for j in senums[i]:
         fd.write('  - '+j+'\n')
    fd.write(':::\n')

##########  lib/petsc/conf/bfort-petsc.txt

  with open(os.path.join('lib','petsc','conf','bfort-petsc.txt'),"w") as fd:
    for i in enums.keys():
      if i in ['PetscBool', 'PetscEnum']: continue
      fd.write("enum "+i+'\n')

    for i in ['PetscBool']:
      fd.write("native "+i+'\n')

    for i in typedefs.keys():
      if i in ['VecScatter']: continue
      fd.write("native "+i+'\n')

    for i in structs.keys():
      if not structs[i].opaque:
        fd.write("native "+i+'\n')
      else:
        fd.write("struct "+i+'\n')

    for i in classes.keys():
      fd.write("nativeptr "+i+'\n')
    for i in ['VecScatter']:
      fd.write("nativeptr "+i+'\n')

    for i in senums.keys():
      fd.write("char "+i+'\n')

###########  $PETSC_ARCH/sys/MANSEC/f90-mod/ftn-auto-interfaces/*.h

   # typedefs{} are not printed below, they are defined directly in include/petsc/finclude/petscXXXbase.h

  for i in mansecs.keys():
    d = os.path.join(petscarch,'src', i, 'f90-mod','ftn-auto-interfaces')
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)

  for i in classes.keys():
    if i in ['PetscObject']: continue
    with open(os.path.join(petscarch,'src', classes[i].mansec,'f90-mod','ftn-auto-interfaces',classes[i].file),"a") as fd:
      if not classes[i].petscobject:
        fd.write('  type t'+i+'\n')
        fd.write('    PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE\n')
        fd.write('  end type t'+i+'\n')
      else:
        fd.write('  type, extends(tPetscObject) ::  t'+i+'\n')
        fd.write('  end type t'+i+'\n')
      v = 'PETSC_NULL_'+i.upper().replace('PETSC','')
      fd.write('  '+i+', parameter :: '+v+' = t'+i+'(0)\n')
      fd.write('  '+i+', parameter :: '+v+'_ARRAY(1) = t'+i+'(0)\n')
      fd.write('  '+i+', parameter :: '+v+'_POINTER(1) = t'+i+'(0)\n')
      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::'+v+'\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::'+v+'_ARRAY\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::'+v+'_POINTER\n')
      fd.write('#endif\n')
      fd.write('\n')

  for i in enums.keys():
    if i in ['PetscBool', 'PetscEnum']: continue
    with open(os.path.join(petscarch,'src', enums[i].mansec,'f90-mod','ftn-auto-interfaces',enums[i].file),"a") as fd:
      fd.write('  type e'+i+'\n')
      fd.write('    PetscEnum:: v PETSC_FORTRAN_TYPE_INITIALIZE\n')
      fd.write('  end type e'+i+'\n\n')
      v = 'PETSC_NULL_'+i.upper().replace('PETSC','')
      fd.write('  '+i+', parameter :: '+v+' = e'+i+'(-50)\n')
      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT::'+v+'\n')
      fd.write('#endif\n')
      cnt = 0
      givenvalue = 0
      for j in enums[i]:
         if j.find('=') > -1:
           if givenvalue == -1:
             print('Some enum values for '+i+' are set but others are not set')
           v = j.replace(' = ',' = e'+i+'(') + ')'
           givenvalue = 1
         else:
           if givenvalue == 1:
             print('Some enum values for '+i+' are set but others are not set')
           v = j + ' = e'+i+'(' + str(cnt)+')'
           givenvalue = -1
         fd.write('    '+i+', parameter :: '+v+'\n')
         cnt = cnt + 1
      fd.write('\n')

      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      for j in enums[i]:
         v = j[0:j.find('=')]
         fd.write('!DEC$ ATTRIBUTES DLLEXPORT::'+v+'\n')
      fd.write('#endif\n')
      fd.write('\n')

  for i in senums.keys():
    with open(os.path.join(petscarch,'src', senums[i].mansec,'f90-mod','ftn-auto-interfaces',senums[i].file),"a") as fd:
      for j in senums[i].keys():
        fd.write('  CHARACTER(LEN=*), PARAMETER :: '+j+' = \''+senums[i][j].replace('"','')+'\'\n')
      fd.write('\n')

      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      for j in senums[i].keys():
        fd.write('!DEC$ ATTRIBUTES DLLEXPORT::'+j+'\n')
      fd.write('#endif\n')
      fd.write('\n')

  for i in structs.keys():
    if not structs[i].opaque: continue
    with open(os.path.join(petscarch,'src', structs[i].mansec,'f90-mod','ftn-auto-interfaces',structs[i].file),"a") as fd:
      fd.write('  type s'+i+'\n')
      for j in structs[i]:
        fd.write('    '+j.replace('[','(').replace(']',')')+'\n')
      fd.write('  end type s'+i+'\n')
      fd.write('\n')

  ifiles = {}
  for i in mansecs.keys():
    d = os.path.join(petscarch,'src', i, 'f90-mod','ftn-auto-interfaces')
    dd = os.path.join('../','src', i, 'f90-mod','ftn-auto-interfaces')
    args = [os.path.join(d,i) for i in os.listdir(d) if i.endswith('.h')]
    for j in args:
      if not os.path.getsize(j): os.path.remove(j)
    with open(os.path.join(d,'petscall.h'),'w') as fd:
      if not i.startswith('petsc'): f = 'petsc' + i + '.h'
      else: f = i + '.h'
      for j in files[f]:
        fd.write('#include <'+os.path.join('petsc','finclude',j)+'>\n')
        if os.path.isfile(os.path.join(d,j)) or os.path.isfile(os.path.join(d,j.replace('.h','types.h'))):
          fd.write('#include <'+os.path.join(dd,j)+'>\n')
      fd.write('#include <'+os.path.join(dd,f)+'>\n')

##########  $PETSC_ARCH/include/petsc/finclude/*.h

  d = os.path.join(os.path.join(petscarch,'include', 'petsc', 'finclude'))
  if os.path.isdir(d): shutil.rmtree(d)
  os.makedirs(d)

  for i in files.keys():
    if i.endswith('types.h'): continue
    with open(os.path.join(d, i),'w') as fd:
      dname = 'PETSC'+i.upper()[0:-2]+'DEF_H'
      fd.write('#if !defined('+dname+')\n#define '+dname+'\n\n')
      fb = os.path.join('include', 'petsc', 'finclude',i.replace('.h','base.h'))
      if os.path.isfile(fb):
        fd.write('#include "' + os.path.join('petsc','finclude',i.replace('.h','base.h')) + '"\n')
      for j in files[i]:
        fd.write('#include "' + os.path.join('petsc','finclude',j) + '"\n')
      fd.write('\n')

  for i in enums.keys():
    if i in ['PetscBool', 'PetscEnum']: continue
    with open(os.path.join(d, enums[i].file),"a") as fd:
      fd.write('#define '+i+' type(e'+i+')\n')

  for i in typedefs.keys():
    if not typedefs[i].name: continue
    value = typedefs[i].value
    if value in CToFortran:
      if not CToFortran[value]: continue
      value = CToFortran[value]
    with open(os.path.join(d, typedefs[i].file),"a") as fd:
      fd.write('#define '+i+' '+value+'\n')

  for i in structs.keys():
    with open(os.path.join(d, structs[i].file),"a") as fd:
      if structs[i].opaque:
        fd.write('#define '+ i + ' type(s'+i+')\n')
      else:
        fd.write('#define '+i+' PetscFortranAddr\n')

  for i in files.keys():
    if i.endswith('types.h'): continue
    with open(os.path.join(d, i),'a') as fd:
      fd.write('\n')

  for i in senums.keys():
    with open(os.path.join(d, senums[i].file),"a") as fd:
      fd.write('#define '+i+' CHARACTER(80)\n')

  for i in files.keys():
    if i.endswith('types.h'): continue
    with open(os.path.join(d, i),'a') as fd:
      fd.write('\n')

  for i in classes.keys():
    with open(os.path.join(d, classes[i].file),"a") as fd:
      fd.write('#define '+i+' type(t'+i+')\n')

  for i in files.keys():
    if i.endswith('types.h'): continue  
    with open(os.path.join(d, i),'a') as fd:
      fd.write('\n#endif\n')

##########  $PETSC_ARCH/sys/MANSEC/f90-mod/ftn-auto-interfaces/*.h

  for i in classes.keys():
    if i in ['PetscObject']: continue
    with open(os.path.join(petscarch,'src', classes[i].mansec,'f90-mod','ftn-auto-interfaces',classes[i].file + '90'),"a") as fd:
      fd.write('  interface operator(.ne.)\n')
      fd.write('    module procedure ' + i + 'notequals\n')
      fd.write('  end interface operator (.ne.)\n')
      fd.write('  interface operator(.eq.)\n')
      fd.write('    module procedure ' + i + 'equals\n')
      fd.write('  end interface operator (.eq.)\n\n')

    with open(os.path.join(petscarch,'src', classes[i].mansec,'f90-mod','ftn-auto-interfaces',classes[i].file + 'f90'),"a") as fd:
      fd.write('  function ' + i + 'notequals(A,B)\n')
      fd.write('    logical ' + i + 'notequals\n')
      fd.write('    type(t' + i + '), intent(in) :: A,B\n')
      fd.write('    ' + i + 'notequals = (A%v .ne. B%v)\n')
      fd.write('  end function\n')
      fd.write('  function ' + i + 'equals(A,B)\n')
      fd.write('    logical ' + i + 'equals\n')
      fd.write('    type(t' + i + '), intent(in) :: A,B\n')
      fd.write('    ' + i + 'equals = (A%v .eq. B%v)\n')
      fd.write('  end function\n')
      fd.write('#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT:: '+i+'notequals\n')
      fd.write('!DEC$ ATTRIBUTES DLLEXPORT:: '+i+'equals\n')
      fd.write('#endif\n\n')

#  for i in mansecs.keys():
#    print(i)
#    for j in mansecs[i]:
#      print('  '+j)

#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
  main(sys.argv[1])

