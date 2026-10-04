# Changes: Development

% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

```{rubric} General:
```

```{rubric} Configure/Build:
```

```{rubric} Sys:
```

- Remove `PetscDefaultCudaStream` and `PetscDefaultHipStream`; obtain the native stream from a concrete `PetscDeviceContext` with `PetscDeviceContextGetStreamHandle()` instead
- Replace implicit `NULL` contexts with `PetscDeviceContextDefault` in device-context and device-memory APIs to retain default-stream selection, or pass an explicit context. Add `PetscDeviceContextGetDefaultContext()` and petsc4py `DeviceContext.getDefault()` to capture the default context for the current device. Preserve destruction of null handles and the C-only device-interface fallback
- Add an alignment argument before the output pointer to `PetscDeviceMalloc()` and `PetscDeviceCalloc()`; pass `PETSC_DECIDE` to retain inferred alignment, or a positive power of two to request stronger alignment
- Add `PetscDeviceContextDelay()` to queue a timed host callback for testing CUDA and HIP stream ordering

```{rubric} Event Logging:
```

```{rubric} PetscViewer:
```

```{rubric} PetscDraw:
```

```{rubric} AO:
```

```{rubric} IS:
```

```{rubric} VecScatter / PetscSF:
```

```{rubric} PF:
```

```{rubric} Vec:
```

- Add `VecGetArrayAndMemTypeAsync()`, `VecGetArrayReadAndMemTypeAsync()`, and `VecGetArrayWriteAndMemTypeAsync()` to acquire vector storage for read/write, read-only, and write-only access, respectively, without waiting for CUDA and HIP transfers on the current device context

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

```{rubric} KSP:
```

```{rubric} SNES:
```

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

```{rubric} TAO:
```

```{rubric} TaoTerm:
```

```{rubric} PetscRegressor:
```

```{rubric} PetscDA:
```

```{rubric} DM:
```

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

```{rubric} FE/FV:
```

```{rubric} DMNetwork:
```

```{rubric} DMStag:
```

```{rubric} DT:
```

```{rubric} Fortran:
```
