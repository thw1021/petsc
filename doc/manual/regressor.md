(ch_regressor)=

# PetscRegressor: Regression Solvers

The `PetscRegressor` component provides some basic infrastructure and a general API for supervised
machine learning tasks at a higher level of abstraction than a purely algebraic "solvers" view.
Methods are currently available for

- {any}`sec_regressor_linear`
- {any}`sec_regressor_nlls`

Note that by "regressor" we mean an algorithm or implementation used to fit and apply a regression
model, following standard parlance in the machine-learning community.
Regressor here does NOT mean an independent (or predictor) variable, as it often does in the
statistics community.

(sec_regressor_usage)=

## Basic Regressor Usage

`PetscRegressor` supports supervised learning tasks:
Given a matrix of observed data $X$ with size $n_{samples}$ by $n_{features}$,
predict a vector of "target" values $y$ (of size $n_{samples}$), where the $i$th entry of $y$
corresponds to the observation (or "sample") stored in the $i$th row of $X$.
Traditionally, when the target consists of continuous values this is called "regression",
and when it consists of discrete values (or "labels"), this task is called "classification";
we use `PetscRegressor` to support both of these cases.

Before a regressor can be used to make predictions, the model must be fitted using an initial set of training data.
Once a fitted model has been obtained, it can be used to predict target values for new observations.
Every `PetscRegressor` implementation provides a `Fit()` and a `Predict()` method to support this workflow.
Fitting (or "training") a model is a relatively computationally intensive task that generally involves solving an
optimization problem (often using `TAO` solvers) to determine the model parameters, whereas making predictions
(or performing "inference") is generally much simpler.

Here, we introduce a simple example to demonstrate `PetscRegressor` usage.
Please read {any}`sec_regressor_solvers` for more in-depth discussion.
The code presented {any}`below <regressor-ex3>` solves an ordinary linear
regression problem, with various options for regularization.

In the simplest usage of a regressor, the user provides a training (or "design") matrix
(`Mat`) and a target vector (`Vec`) against which to fit the model.
Once the regressor is fitted, the user can then obtain a vector of predicted values for a set of new observations.

PETSc's default method for solving regression problems is ordinary least squares,
`REGRESSOR_LINEAR_OLS`, which is a sub-type of linear regressor,
`PETSCREGRESSORLINEAR`.
By "linear" we mean that the model $f(x, \theta)$ is linear in its coefficients $\theta$
but not necessarily linear in its features $x$.
Models that depend nonlinearly on their parameters are handled by `PETSCREGRESSORNLLS`,
described in {any}`sec_regressor_nlls`.

Note that data creation, option parsing, and cleaning stages are omitted here for
clarity. The complete code is available in {ref}`ex3.c <regressor-ex3>`.

(regressor-ex3)=
:::{admonition} Listing: `src/ml/regressor/tests/ex3.c`
```{literalinclude} /../src/ml/regressor/tests/ex3.c
:prepend: '#include <petscregressor.h>'
:start-at: int main
:end-at: PetscFinalize
:append: return 0;}
```
:::

To create a `PetscRegressor` instance, one must first call `PetscRegressorCreate()`:

```
PetscRegressorCreate(MPI_Comm comm, PetscRegressor *regressor);
```

To choose a regressor type, the user can either call

```
PetscRegressorSetType(PetscRegressor regressor, PetscRegressorType type);
```

or use the command-line option `-regressor_type <method>`; details regarding the
available methods are presented in {any}`sec_regressor_solvers`.
The application code can specify the options used by underlying linear,
nonlinear, and optimization solver methods used in fitting the model
by calling

```
PetscRegressorSetFromOptions(regressor);
```

which interfaces with the PETSc options database and enables convenient
runtime selection of the type of regression algorithm and setting various
various solver or problem parameters.
This routine can also control all inner solver options in the `KSP`, and `Tao`
modules, as discussed in {any}`ch_ksp`, {any}`ch_tao`.

After having set these routines and options, the user can fit (or "train") the regressor
by calling

```
PetscRegressorFit(PetscRegressor regressor, Mat X, Vec y);
```

where `X` is training data, and `y` is target values.
Finally, after fitting the regressor, the user can compute model
predictions, that is, perform inference, for a data matrix of unlabeled observations
using the fitted regressor:

```
PetscRegressorPredict(PetscRegressor regressor, Mat X, Vec y_predicted);
```

Finally, after the user is done using the regressor,
the user should destroy its `PetscRegressor` context with

```
PetscRegressorDestroy(PetscRegressor *regressor);
```

(sec_regressor_solvers)=

## Regression Solvers

One can see the list of regressor types in Table
{any}`tab-regressordefaults`. Currently, we support two types, `PETSCREGRESSORLINEAR`
for models that are linear in their coefficients and `PETSCREGRESSORNLLS` for nonlinear
least squares curve fitting, although we plan to add several others in the near future.
`PETSCREGRESSORNLLS` is built only when PETSc is configured for real scalars.

```{eval-rst}
.. list-table:: PETSc Regressor
   :name: tab-regressordefaults
   :header-rows: 1

   * - Method
     - PetscRegressorType
     - Options Name
   * - Linear
     - ``PETSCREGRESSORLINEAR``
     - ``linear``
   * - Nonlinear least squares
     - ``PETSCREGRESSORNLLS``
     - ``nlls``
```

If the particular method being employed is one that supports regularization,
the user can set regularizer's weight via

```
PetscRegressorSetRegularizerWeight(PetscRegressor regressor, PetscReal weight);
```

or with the option `-regressor_regularizer_weight weight`.
The weight defaults to 1.0, which is usually the default in scikit-learn, and both
`PETSCREGRESSORLINEAR` and `PETSCREGRESSORNLLS` pass it on to the underlying `TAOBRGN`
solver. `REGRESSOR_LINEAR_OLS` overrides this with a weight of 0.0, since ordinary least
squares is by definition unregularized, but `PETSCREGRESSORNLLS` does not; see
{any}`sec_regressor_nlls`.

(sec_regressor_linear)=

## Linear regressor

The `PETSCREGRESSORLINEAR` (`-regressor_type linear`) implementation
constructs a linear model to reduce the sum of squared differences
between the actual target values ("observations") in the dataset and the target
values estimated by the fitted model.
By default, bound-constrained regularized Gauss-Newton `TAOBRGN` is used to solve the underlying optimization problem.

Currently, linear regressor has three types, which are described
in Table {any}`tab-lineartypes`.

```{eval-rst}
.. list-table:: Linear Regressor types
   :name: tab-lineartypes
   :header-rows: 1

   * - Linear method
     - ``PetscRegressorLinearType``
     - Options Name
   * - Ordinary
     - ``REGRESSOR_LINEAR_OLS``
     - ``ols``
   * - Lasso
     - ``REGRESSOR_LINEAR_LASSO``
     - ``lasso``
   * - Ridge
     - ``REGRESSOR_LINEAR_RIDGE``
     - ``ridge``
```

If one wishes, the user can (when appropriate) use `KSP` to solve the problem, instead of `Tao`,
via

```
PetscRegressorLinearSetUseKSP(PetscRegressor regressor, PetscBool flg);
```

or with the option `-regressor_linear_use_ksp <true,false>`.

Calculation of the intercept (also known as the "bias" or "offset") is performed
separately from the rest of the model fitting process, because data sets are often
already mean-centered and because it is generally undesirable to regularize the
intercept term.
By default, this step is omitted; if the user wishes to compute the intercept,
this can be done by calling

```
PetscRegressorLinearSetFitIntercept(PetscRegressor regressor, PetscBool flg);
```

or by specifying the option `-regressor_linear_fit_intercept <true,false>`.

For a fitted regression, one can obtain the intercept and
a vector of the model coefficients from a linear regression model via

```
PetscRegressorLinearGetCoefficients(PetscRegressor regressor, Vec *coefficients);
PetscRegressorLinearGetIntercept(PetscRegressor regressor, PetscScalar *intercept);
```

(sec_regressor_nlls)=

## Nonlinear least squares regressor

The `PETSCREGRESSORNLLS` (`-regressor_type nlls`) implementation fits a user-supplied model
$f(X, p)$ to the observed target values $y$ by minimizing

$$
\min_p \| f(X, p) - y \|_2^2
$$

over the vector of model parameters $p$.
Unlike `PETSCREGRESSORLINEAR`, which requires the model to be linear in its coefficients,
the model may depend on its parameters in an arbitrary nonlinear fashion; this is the
"curve fitting" task performed by, for instance, `scipy.optimize.curve_fit`.
Because the underlying solvers operate on real residuals, `PETSCREGRESSORNLLS` is available
only when PETSc is configured for real scalars.

The model is supplied as a callback registered with

```
PetscRegressorNLLSSetFunction(PetscRegressor regressor, Vec f, PetscRegressorNLLSFunctionFn *fn, void *ctx);
```

where `fn` has the calling sequence

```
PetscErrorCode fn(PetscRegressor regressor, Mat X, Vec p, Vec f, void *ctx);
```

Two conventions here differ from what one might expect based on the rest of PETSc.
First, the callback computes the model values $f(X, p)$ rather than the residual
$f(X, p) - y$, and the implementation forms the residual internally; this follows the
convention of `scipy.optimize.curve_fit` instead of that of `SNESSetFunction()` or
`TaoSetResidualRoutine()`.
Second, the data matrix `X` is passed to the callback as an argument, mirroring
`PetscRegressorFit(regressor, X, y)` and `PetscRegressorPredict(regressor, X, y)`, so there
is no need to stash it in the user context; while fitting, `X` is the training matrix given
to `PetscRegressorFit()`, and while predicting it is the matrix given to
`PetscRegressorPredict()`.
The `f` argument is an optional template for the model output, which must have the same
layout as the target vector; if it is `NULL`, a vector is duplicated from the target during
`PetscRegressorSetUp()`.

An initial guess for the parameters is required, and must be set before fitting with

```
PetscRegressorNLLSSetInitialParameters(PetscRegressor regressor, Vec p0);
```

The length of `p0` determines the number of model parameters.

The Jacobian of the model with respect to the parameters, $\partial f/\partial p$, can
optionally be provided using

```
PetscRegressorNLLSSetJacobian(PetscRegressor regressor, Mat J, Mat Jpre, PetscRegressorNLLSJacobianFn *fn, void *ctx);
```

If no Jacobian function is set, the Jacobian is approximated by finite differences, at a
cost of one additional model evaluation per parameter.
If `J` is `NULL`, a dense matrix of the appropriate size is allocated during
`PetscRegressorSetUp()`, which is convenient for the small parameter counts typical of curve
fitting; for large numbers of parameters, preallocate a sparse matrix and pass it to this
routine.

Once `PetscRegressorFit()` has been called, the fitted parameters can be obtained from

```
PetscRegressorNLLSGetParameters(PetscRegressor regressor, Vec *p);
```

This vector is owned by the regressor and must not be destroyed by the user.
Calling `PetscRegressorPredict()` evaluates the model function at the fitted parameters for
a new set of observations.

The optimization problem is solved with `TAOBRGN`, whose options are reached using the
`-regressor_nlls_` prefix, appended to any options prefix set on the regressor itself;
for example, `-regressor_nlls_tao_monitor` or `-regressor_nlls_tao_max_it 100`.
As noted in {any}`sec_regressor_solvers`, the regularizer weight defaults to 1.0 and is
handed to `TAOBRGN`, so a fit performed without setting it is regularized.
An unregularized nonlinear least squares fit is obtained by setting the weight to zero:

```
PetscRegressorSetRegularizerWeight(regressor, 0.0);
```

The example {ref}`below <regressor-ex4>` fits the three-parameter exponential model
$f(x; p) = p_0 e^{p_1 x} + p_2$ to data generated from known parameter values, supplying an
analytic Jacobian. Running it with `-use_analytic_jacobian false` instead exercises the
finite-difference Jacobian.

(regressor-ex4)=
:::{admonition} Listing: `src/ml/regressor/tests/ex4.c`
```{literalinclude} /../src/ml/regressor/tests/ex4.c
:prepend: '#include <petscregressor.h>'
:start-at: static PetscErrorCode Model
:end-at: PetscFinalize
:append: return 0;}
```
:::
