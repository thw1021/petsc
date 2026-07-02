#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/

#if !defined(PETSC_PYTHON_EXE)
  #define PETSC_PYTHON_EXE "python"
#endif

static PetscErrorCode PetscPythonFindExecutable(char pythonexe[], size_t len)
{
  char      pyexe[PETSC_MAX_PATH_LEN] = "";
  PetscBool flag;

  PetscFunctionBegin;
  /* get the path for the Python interpreter executable */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-python", pyexe, sizeof(pyexe), &flag));
  if (flag && pyexe[0]) {
    char *sep = NULL;
    PetscCall(PetscStrchr(pyexe, PETSC_DIR_SEPARATOR, &sep));
    if (sep) PetscCall(PetscGetFullPath(pyexe, pythonexe, len));
    else PetscCall(PetscStrncpy(pythonexe, pyexe, len));
  } else {
    /* prioritize an active virtual environment */
    PetscBool   pyenv = getenv("VIRTUAL_ENV") != NULL;
    PetscBool   conda = getenv("CONDA_PREFIX") != NULL;
    const char *pyexe = (pyenv || conda) ? "python" : PETSC_PYTHON_EXE;
    PetscCall(PetscStrncpy(pythonexe, pyexe, len));
  }
#if defined(PETSC_HAVE_POPEN)
  /* call Python to find out the full path of the Python executable */
  {
    const char cmdline[] = "-c 'import os, sys; print(os.path.abspath(sys.executable))'";
    char       command[PETSC_MAX_PATH_LEN + 64];
    char       output[PETSC_MAX_PATH_LEN + 1];
    FILE      *fp;

    PetscCall(PetscStrncpy(command, pythonexe, sizeof(command)));
    PetscCall(PetscStrlcat(command, " ", sizeof(command)));
    PetscCall(PetscStrlcat(command, cmdline, sizeof(command)));
    PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, command, "r", &fp));
    if (fgets(output, (int)sizeof(output), fp)) {
      /* remove newlines */
      char *eol = NULL;
      PetscCall(PetscStrchr(output, '\n', &eol));
      if (eol) eol[0] = 0;
      PetscCall(PetscStrncpy(pythonexe, output, len));
    }
    PetscCall(PetscPClose(PETSC_COMM_SELF, fp));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Python does not appear to have a universal way to indicate the location of Python dynamic library so try several possibilities
*/
static PetscErrorCode PetscPythonFindLibraryName(const char pythonexe[], const char attempt[], char pythonlib[], size_t pl, PetscBool *found)
{
  char  command[2 * PETSC_MAX_PATH_LEN];
  FILE *fp  = NULL;
  char *eol = NULL;

  PetscFunctionBegin;
  *found = PETSC_FALSE;
  /* call Python to find out the name of the Python dynamic library */
  PetscCall(PetscStrncpy(command, pythonexe, sizeof(command)));
  PetscCall(PetscStrlcat(command, " ", sizeof(command)));
  PetscCall(PetscStrlcat(command, attempt, sizeof(command)));
#if defined(PETSC_HAVE_POPEN)
  PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, command, "r", &fp));
  if (!fgets(pythonlib, (int)pl, fp)) {
    PetscCall(PetscPClose(PETSC_COMM_SELF, fp));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscPClose(PETSC_COMM_SELF, fp));
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: Aborted due to missing popen()");
#endif
  /* remove newlines */
  PetscCall(PetscStrchr(pythonlib, '\n', &eol));
  if (eol) eol[0] = 0;
  PetscCall(PetscTestFile(pythonlib, 'r', found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPythonFindLibrary(const char pythonexe[], char pythonlib[], size_t pl)
{
  // clang-format off
  const char *const cmdlines[] = {"-c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"),sysconfig.get_config_var(\"LDLIBRARY\")))'",
                                  "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_path(\"stdlib\"),os.path.pardir,\"libpython\"+sysconfig.get_python_version()+\".dylib\"))'",
                                  "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_path(\"stdlib\"),os.path.pardir,\"libpython\"+sysconfig.get_python_version()+\".so\"))'",
                                  "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBPL\"),sysconfig.get_config_var(\"LDLIBRARY\")))'",
                                  "-c 'import sysconfig; print(sysconfig.get_config_var(\"LIBPYTHON\"))'",
                                  "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"),\"libpython\"+sysconfig.get_python_version()+\".dylib\"))'",
                                  "-c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var(\"LIBDIR\"),\"libpython\"+sysconfig.get_python_version()+\".so\"))'"};
  // clang-format on

  PetscBool found = PETSC_FALSE;

  PetscFunctionBegin;
#if defined(PETSC_PYTHON_LIB)
  PetscCall(PetscStrncpy(pythonlib, PETSC_PYTHON_LIB, pl));
  PetscFunctionReturn(PETSC_SUCCESS);
#endif

  for (size_t i = 0; i < PETSC_STATIC_ARRAY_LENGTH(cmdlines); i++) {
    PetscCall(PetscInfo(NULL, "Looking for Python library with \"%s %s\"\n", pythonexe, cmdlines[i]));
    PetscCall(PetscPythonFindLibraryName(pythonexe, cmdlines[i], pythonlib, pl, &found));
    if (found) break;
  }
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find Python dynamic library");
  PetscCall(PetscInfo(NULL, "Python library %s found %d\n", pythonlib, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct _Py_object_t PyObject; /* fake definition */

static PyObject *Py_None = NULL;
static void (*Py_IncRef)(PyObject *);
static void (*Py_DecRef)(PyObject *);

static const unsigned long *Py_Version; /* Python 3.11 */
static const char *(*Py_GetVersion)(void);

/* Modern APIs for embedded Python initialization (Python 3.14) */
typedef struct PyInitConfig PyInitConfig;
static PyInitConfig *(*PyInitConfig_Create)(void);
static void (*PyInitConfig_Free)(PyInitConfig *);
static int (*PyInitConfig_SetInt)(PyInitConfig *, const char *, int64_t);
static int (*PyInitConfig_SetStr)(PyInitConfig *, const char *, const char *);
static int (*Py_InitializeFromInitConfig)(PyInitConfig *);

/* Legacy APIs for embedded Python initialization */
// clang-format off
typedef struct { enum { _e } _t; const char *_f, *_m; int _e; } PyStatus;
// clang-format on
static PyStatus (*_PyRuntime_Initialize)(void); /* Python 3.7 */
static wchar_t *(*Py_DecodeLocale)(const char *, size_t *);
static void (*Py_SetProgramName)(const wchar_t *);
static void (*Py_InitializeEx)(int);
static void (*PySys_SetArgv)(int, void *);

static int (*Py_IsInitialized)(void);
static void (*Py_Finalize)(void);
static PyObject *(*PySys_GetObject)(const char *);
static PyObject *(*PyObject_CallMethod)(PyObject *, const char *, const char *, ...);
static PyObject *(*PyImport_ImportModule)(const char *);

static void (*PyErr_Clear)(void);
static PyObject *(*PyErr_Occurred)(void);
static void (*PyErr_Fetch)(PyObject **, PyObject **, PyObject **);
static void (*PyErr_NormalizeException)(PyObject **, PyObject **, PyObject **);
static void (*PyErr_Display)(PyObject *, PyObject *, PyObject *);
static void (*PyErr_Restore)(PyObject *, PyObject *, PyObject *);

static void (*PyMem_RawMalloc)(void *);
static void (*PyMem_RawFree)(void *);

#define PetscDLPyLibOpen(libname)      PetscDLLibraryAppend(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, libname)
#define PetscDLPyLibSym(symbol, value) PetscDLLibrarySym(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, NULL, symbol, (void **)value)
#define PetscDLPyLibClose(comm) \
  do { \
  } while (0)

static PetscErrorCode PetscPythonLoadLibrary(const char pythonlib[])
{
  PetscFunctionBegin;
  /* open the Python dynamic library */
  PetscCall(PetscDLPyLibOpen(pythonlib));
  PetscCall(PetscInfo(NULL, "Python: loaded dynamic library %s\n", pythonlib));
  /* look required symbols from the Python C-API */
  PetscCall(PetscDLPyLibSym("_Py_NoneStruct", &Py_None));
  PetscCall(PetscDLPyLibSym("Py_IncRef", &Py_IncRef));
  PetscCall(PetscDLPyLibSym("Py_DecRef", &Py_DecRef));
  PetscCall(PetscDLPyLibSym("Py_Version", &Py_Version));
  PetscCall(PetscDLPyLibSym("Py_GetVersion", &Py_GetVersion));

  /* the PyInitConfig APIs are available since Python 3.14 */
  PetscCall(PetscDLPyLibSym("PyInitConfig_Create", &PyInitConfig_Create));
  PetscCall(PetscDLPyLibSym("PyInitConfig_Free", &PyInitConfig_Free));
  PetscCall(PetscDLPyLibSym("PyInitConfig_SetInt", &PyInitConfig_SetInt));
  PetscCall(PetscDLPyLibSym("PyInitConfig_SetStr", &PyInitConfig_SetStr));
  PetscCall(PetscDLPyLibSym("Py_InitializeFromInitConfig", &Py_InitializeFromInitConfig));

  PetscCall(PetscDLPyLibSym("_PyRuntime_Initialize", &_PyRuntime_Initialize));
  PetscCall(PetscDLPyLibSym("Py_DecodeLocale", &Py_DecodeLocale));
  PetscCall(PetscDLPyLibSym("Py_SetProgramName", &Py_SetProgramName));
  PetscCall(PetscDLPyLibSym("Py_InitializeEx", &Py_InitializeEx));
  PetscCall(PetscDLPyLibSym("PySys_SetArgv", &PySys_SetArgv));

  PetscCall(PetscDLPyLibSym("Py_IsInitialized", &Py_IsInitialized));
  PetscCall(PetscDLPyLibSym("Py_Finalize", &Py_Finalize));
  PetscCall(PetscDLPyLibSym("PySys_GetObject", &PySys_GetObject));
  PetscCall(PetscDLPyLibSym("PyObject_CallMethod", &PyObject_CallMethod));
  PetscCall(PetscDLPyLibSym("PyImport_ImportModule", &PyImport_ImportModule));
  PetscCall(PetscDLPyLibSym("PyErr_Clear", &PyErr_Clear));
  PetscCall(PetscDLPyLibSym("PyErr_Occurred", &PyErr_Occurred));
  PetscCall(PetscDLPyLibSym("PyErr_Fetch", &PyErr_Fetch));
  PetscCall(PetscDLPyLibSym("PyErr_NormalizeException", &PyErr_NormalizeException));
  PetscCall(PetscDLPyLibSym("PyErr_Display", &PyErr_Display));
  PetscCall(PetscDLPyLibSym("PyErr_Restore", &PyErr_Restore));

  PetscCall(PetscDLPyLibSym("PyMem_RawMalloc", &PyMem_RawMalloc));
  PetscCall(PetscDLPyLibSym("PyMem_RawFree", &PyMem_RawFree));

  /* XXX TODO: check that ALL symbols were there !!! */
  PetscCheck(Py_None, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_IncRef, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_DecRef, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_GetVersion, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCheck(Py_IsInitialized, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  if (Py_Version && *Py_Version >= 0x030E0000) { /* Python >= 3.14 */
    PetscCheck(PyInitConfig_Create, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
    PetscCheck(PyInitConfig_Free, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
    PetscCheck(PyInitConfig_SetInt, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
    PetscCheck(PyInitConfig_SetStr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
    PetscCheck(Py_InitializeFromInitConfig, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  } else {
    PetscCheck(Py_DecodeLocale, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
    PetscCheck(Py_SetProgramName, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
    PetscCheck(Py_InitializeEx, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  }
  PetscCheck(Py_Finalize, PETSC_COMM_SELF, PETSC_ERR_LIB, "Python: failed to load symbols from Python dynamic library %s", pythonlib);
  PetscCall(PetscInfo(NULL, "Python: all required symbols loaded from Python dynamic library %s\n", pythonlib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static char      PetscPythonExe[PETSC_MAX_PATH_LEN] = {0};
static char      PetscPythonLib[PETSC_MAX_PATH_LEN] = {0};
static PetscBool PetscBeganPython                   = PETSC_FALSE;

/*@
  PetscPythonFinalize - Finalize PETSc for use with Python.

  Level: intermediate

.seealso: `PetscPythonInitialize()`, `PetscPythonPrintError()`
@*/
PetscErrorCode PetscPythonFinalize(void)
{
  PetscFunctionBegin;
  if (PetscBeganPython) {
    if (Py_IsInitialized()) Py_Finalize();
  }
  PetscBeganPython = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPythonInitialize - Initialize Python for use with PETSc and import petsc4py.

   Input Parameters:
+  pyexe - path to the Python interpreter executable, or `NULL`.
-  pylib - full path to the Python dynamic library, or `NULL`.

  Options Database Key:
. -python exe - Initializes Python, and optionally takes a Python executable name

  Level: intermediate

.seealso: `PetscPythonFinalize()`, `PetscPythonPrintError()`
@*/
PetscErrorCode PetscPythonInitialize(const char pyexe[], const char pylib[])
{
  PyObject *module = NULL;

  PetscFunctionBegin;
  if (PetscBeganPython) PetscFunctionReturn(PETSC_SUCCESS);
  /* Python executable */
  if (pyexe && pyexe[0] != 0) {
    PetscCall(PetscStrncpy(PetscPythonExe, pyexe, sizeof(PetscPythonExe)));
  } else {
    PetscCall(PetscPythonFindExecutable(PetscPythonExe, sizeof(PetscPythonExe)));
  }
  /* Python dynamic library */
  if (pylib && pylib[0] != 0) {
    PetscCall(PetscStrncpy(PetscPythonLib, pylib, sizeof(PetscPythonLib)));
  } else {
    PetscCall(PetscPythonFindLibrary(PetscPythonExe, PetscPythonLib, sizeof(PetscPythonLib)));
  }
  /* dynamically load Python library */
  PetscCall(PetscPythonLoadLibrary(PetscPythonLib));

  /* initialize Python */
  PetscBeganPython = PETSC_FALSE;
  if (!Py_IsInitialized()) {
    static PetscBool registered = PETSC_FALSE;
    PyObject        *sys_path;
    char             path[PETSC_MAX_PATH_LEN] = {0};

    /* initialize Python */
    if (Py_Version && *Py_Version >= 0x030E0000) { /* Python >= 3.14 */
      PyInitConfig *config;
      int           retv;

      config = PyInitConfig_Create();
      PetscCheck(config, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't create initial Python configuration");
      retv = PyInitConfig_SetStr(config, "executable", PetscPythonExe);
      PetscCheck(retv == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't setup initial Python configuration");
      retv = PyInitConfig_SetInt(config, "isolated", 0);
      PetscCheck(retv == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't setup initial Python configuration");
      retv = PyInitConfig_SetInt(config, "safe_path", 0);
      PetscCheck(retv == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't setup initial Python configuration");
      retv = PyInitConfig_SetInt(config, "use_environment", 1);
      PetscCheck(retv == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't setup initial Python configuration");
      retv = PyInitConfig_SetInt(config, "user_site_directory", 1);
      PetscCheck(retv == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't setup initial Python configuration");

      PetscCall(PetscInfo(NULL, "Calling Py_InitializeFromInitConfig()\n"));
      PetscCallExternalVoid("Py_InitializeFromInitConfig", retv = Py_InitializeFromInitConfig(config));
      PetscCheck(retv == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't initialize Python");
      PetscCall(PetscInfo(NULL, "Py_InitializeFromInitConfig() called successfully\n"));
      PyInitConfig_Free(config);
    } else {
      static wchar_t wPetscPythonExe[PETSC_MAX_PATH_LEN] = {0};
      wchar_t       *wstr;

      /* set the program name to support virtual environment */
      if (_PyRuntime_Initialize) (void)_PyRuntime_Initialize(); /* for Py_DecodeLocale() */
      wstr = Py_DecodeLocale(PetscPythonExe, NULL);
      PetscCheck(wstr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't decode string '%s'", PetscPythonExe);
      PetscCall(PetscMemcpy(wPetscPythonExe, wstr, sizeof(wPetscPythonExe)));
      PyMem_RawFree(wstr);
      Py_SetProgramName(wPetscPythonExe);
      /* Py_InitializeEx() prints an error and EXITS the program if it is not successfull! */
      PetscCall(PetscInfo(NULL, "Calling Py_InitializeEx(0)\n"));
      PetscCallExternalVoid("Py_InitializeEx", Py_InitializeEx(0)); /* 0: do not install signal handlers */
      PetscCall(PetscInfo(NULL, "Py_InitializeEx(0) called successfully\n"));
      if (!PySys_GetObject("argv")) {
        /* build 'sys.argv' list */
        int   argc    = 0;
        char *argv[1] = {NULL};
        PySys_SetArgv(argc, argv);
      }
    }

    /* add PETSC_LIB_DIR in front of 'sys.path' */
    sys_path = PySys_GetObject("path");
    if (sys_path) {
      int zero = 0;
      PetscCall(PetscStrreplace(PETSC_COMM_SELF, "${PETSC_LIB_DIR}", path, sizeof(path)));
      Py_DecRef(PyObject_CallMethod(sys_path, "insert", "is", zero, (char *)path));
#if defined(PETSC_PETSC4PY_INSTALL_PATH)
      {
        char *rpath;
        PetscCall(PetscStrallocpy(PETSC_PETSC4PY_INSTALL_PATH, &rpath));
        Py_DecRef(PyObject_CallMethod(sys_path, "insert", "is", zero, rpath));
        PetscCall(PetscFree(rpath));
      }
#endif
    }
    /* register finalizer */
    if (!registered) {
      PetscCall(PetscRegisterFinalize(PetscPythonFinalize));
      registered = PETSC_TRUE;
    }
    PetscBeganPython = PETSC_TRUE;
    PetscCall(PetscInfo(NULL, "Python initialize completed\n"));
  }
  /* import 'petsc4py.PETSc' module */
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCallExternalVoid("PyImport_ImportModule", module = PyImport_ImportModule("petsc4py.PETSc"));
  PetscCall(PetscFPTrapPop());
  if (module) {
    PetscCall(PetscInfo(NULL, "Python: successfully imported module 'petsc4py.PETSc'\n"));
    Py_DecRef(module);
    module = NULL;
  } else {
    PetscCall(PetscInfo(NULL, "Python: error when importing module 'petsc4py.PETSc'\n"));
    PetscCall(PetscPythonPrintError());
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Python: could not import module 'petsc4py.PETSc', perhaps your PYTHONPATH does not contain it");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscPythonPrintError - Print any current Python errors.

  Level: developer

.seealso: `PetscPythonInitialize()`, `PetscPythonFinalize()`
@*/
PetscErrorCode PetscPythonPrintError(void)
{
  PyObject *exc = NULL, *val = NULL, *tb = NULL;

  PetscFunctionBegin;
  if (!PetscBeganPython) PetscFunctionReturn(PETSC_SUCCESS);
  if (!PyErr_Occurred()) PetscFunctionReturn(PETSC_SUCCESS);
  PyErr_Fetch(&exc, &val, &tb);
  PyErr_NormalizeException(&exc, &val, &tb);
  PyErr_Display(exc ? exc : Py_None, val ? val : Py_None, tb ? tb : Py_None);
  PyErr_Restore(exc, val, tb);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject, const char[]);
PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject, const char[]) = NULL;

/*@
  PetscPythonMonitorSet - Set a Python monitor for a `PetscObject`

  Level: developer

.seealso: `PetscPythonInitialize()`, `PetscPythonFinalize()`, `PetscPythonPrintError()`
@*/
PetscErrorCode PetscPythonMonitorSet(PetscObject obj, const char url[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(url, 2);
  if (!PetscPythonMonitorSet_C) {
    PetscCall(PetscPythonInitialize(NULL, NULL));
    PetscCheck(PetscPythonMonitorSet_C, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Couldn't initialize Python support for monitors");
  }
  PetscCall(PetscPythonMonitorSet_C(obj, url));
  PetscFunctionReturn(PETSC_SUCCESS);
}
