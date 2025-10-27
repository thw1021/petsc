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

```{rubric} PetscRegressor:
```

```{rubric} DM/DA:
```

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

- Add `DMPlexSetPointNumbering()` and `DMPlexGetPointNumbering()`
- Remove ``globalPointNumbers`` argument from `DMPlexTopologyView_HDF5_Internal()` and `DMPlexLabelsView_HDF5_Internal()'
- Add `DMPlexCreatePointNumberingSF()`
- Remove `sfXC` argument from `DMPlexTopologyLoad()`, `DMPlexCoordinatesLoad()`, `DMPlexLabelsLoad()`, and `DMPlexSectionLoad_HDF5`
- Remove `sfXC` argument from `DMPlexTopologyLoad_HDF5_Internal()`, `DMPlexCoordinatesLoad_HDF5_Internal()`, `DMPlexLabelsLoad_HDF5_Internal()`, and `DMPlexSectionLoad_HDF5_Internal()`

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
