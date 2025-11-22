(ch_das)=

# PetscDAS: Data Assimilation System

The `PetscDAS` component provides infrastructure and a general API for ensemble-based data assimilation at a higher level of abstraction than purely algebraic solvers. Methods currently available include:

- {any}`sec_das_etkf`

(sec_das_usage)=

## Basic Data Assimilation Usage

`PetscDAS` supports ensemble-based data assimilation tasks: Given an ensemble of state vectors and observations, update the ensemble to better match observations while accounting for uncertainties.

Before a data assimilation system can be used, the ensemble must be initialized and observations must be provided. Every `PetscDAS` implementation provides an `Assimilate()` method to update the ensemble and a `Forecast()` method to propagate it forward in time.

Here, we introduce a simple example to demonstrate `PetscDAS` usage. The complete code is available in {ref}`ex1.c <das-ex1>`.

(das-ex1)=
:::{admonition} Listing: `src/ml/das/tests/ex1.c`
```{literalinclude} /../src/ml/das/tests/ex1.c
:language: c
:start-at: int main
:end-at: return 0;
```
:::

To create a `PetscDAS` instance, one must first call `PetscDASCreate()`:

```c
PetscDASCreate(MPI_Comm comm, PetscDAS *das);
```

To choose a data assimilation type, the user can either call

```c
PetscDASSetType(PetscDAS das, PetscDASType type);
```

or use the command-line option `-das_type <method>`; details regarding the available methods are presented in {any}`sec_das_methods`.

The application code can specify options by calling

```c
PetscDASSetFromOptions(das);
```

which interfaces with the PETSc options database.

After setting these routines and options, configure the data assimilation system:

```c
PetscDASSetEnsembleSize(das, m);           // Set ensemble size
PetscDASSetStateSize(das, n);              // Set state vector size
PetscDASSetObservationOperator(das, H);    // Set observation operator
PetscDASSetObservationErrorCovariance(das, R);  // Set observation error covariance
PetscDASSetEnsemble(das, ensemble);        // Set initial ensemble
```

Then perform data assimilation cycles:

```c
PetscDASAssimilate(das, observation);      // Update ensemble with observation
PetscDASForecast(das, model_operator);     // Propagate ensemble forward
```

Finally, destroy the `PetscDAS` context:

```c
PetscDASDestroy(PetscDAS *das);
```

(sec_das_methods)=

## Data Assimilation Methods

One can see the list of data assimilation types in Table {any}`tab-dasdefaults`.

```{eval-rst}
.. list-table:: PETSc Data Assimilation Methods
   :name: tab-dasdefaults
   :header-rows: 1

   * - Method
     - PetscDASType
     - Options Name
   * - ETKF
     - ``PETSCDASETKF``
     - ``etkf``
```

(sec_das_etkf)=

## Ensemble Transform Kalman Filter (ETKF)

The `PETSCDASETKF` (`-das_type etkf`) implementation uses the Ensemble Transform Kalman Filter algorithm to perform data assimilation. ETKF is an efficient ensemble-based method that:

- Avoids explicit storage of error covariance matrices
- Does not require tangent linear or adjoint models
- Uses Cholesky factorization for numerical stability
- Operates efficiently with sparse matrices

### Algorithm

The ETKF implementation follows Algorithm 6.4 from standard data assimilation literature:

1. Compute ensemble mean and perturbations
2. Transform ensemble to observation space
3. Compute innovation (observation minus forecast)
4. Solve analysis equations using Cholesky factorization
5. Update ensemble members
6. Forecast ensemble forward in time

### Mathematical Operations

**Cholesky-based computation**: The implementation uses Cholesky factorization to efficiently compute:
- T = (I + S^T S)^(-1) via forward and backward solves
- T^(1/2) = (L^T)^(-1) via a single backward solve

This avoids expensive eigenvalue decomposition while maintaining numerical stability.

**Sparse matrices**: All matrices use PETSc's sparse AIJ format for scalability to high-dimensional systems.

### Options

```{eval-rst}
.. list-table:: ETKF Options
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``-das_etkf_inflation``
     - Multiplicative covariance inflation factor
     - 1.0 (no inflation)
   * - ``-das_etkf_localization_radius``
     - Spatial localization radius
     - 0.0 (no localization)
```

### Example

```c
PetscDAS das;
Vec *ensemble, observation;
Mat H, R, model;
PetscInt m = 20;  // Ensemble size
PetscInt n = 100; // State dimension

// Create and configure
PetscDASCreate(PETSC_COMM_WORLD, &das);
PetscDASSetType(das, PETSCDASETKF);
PetscDASSetEnsembleSize(das, m);
PetscDASSetStateSize(das, n);

// Set observation operator and error covariance
PetscDASSetObservationOperator(das, H);
PetscDASSetObservationErrorCovariance(das, R);

// Initialize ensemble
// ... create ensemble vectors ...
PetscDASSetEnsemble(das, ensemble);

// Data assimilation cycle
for (int k = 0; k < num_steps; k++) {
  // Get observation for current time
  // ... generate or read observation[k] ...
  
  // Assimilate observation
  PetscDASAssimilate(das, observation);
  
  // Forecast to next time step
  PetscDASForecast(das, model);
}

PetscDASDestroy(&das);
```

## Implementation Details

### Ensemble Storage

Ensembles are stored as arrays of `Vec` objects:
```c
Vec *ensemble;  // Array of m state vectors
```

Each ensemble member is a separate `Vec`, allowing standard PETSc vector operations.

### Gaussian Random Numbers

The implementation includes a utility function to generate Gaussian random numbers from PETSc's uniform random number generator using the Box-Muller transform:

```c
VecSetGaussianRandom(Vec v, PetscRandom rctx, PetscReal mean, PetscReal stddev);
```

### Matrix Operations

The ETKF implementation uses:
- **Sparse AIJ matrices** for observation operators and error covariances
- **Dense matrices** for ensemble perturbations (typically small: n x m where m << n)
- **Cholesky factorization** via `PCCHOLESKY` for solving linear systems
- **Matrix-free operations** where possible to minimize memory usage

## Performance Considerations

1. **Ensemble size**: Typically m = 20-100 members provide good balance between accuracy and computational cost
2. **Sparse matrices**: Use sparse AIJ format for scalability
3. **Cholesky vs eigenvalue decomposition**: Cholesky factorization is approximately 3x faster
4. **Parallelization**: Ensemble forecasts are embarrassingly parallel

## References

- Algorithm 6.4 ETKF from "Data Assimilation: A Mathematical Introduction" by Asch, Bocquet, and Nodet
- Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. Physica D, 230(1-2), 112-126.