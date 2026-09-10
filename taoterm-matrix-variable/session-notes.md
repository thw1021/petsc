# Matrix-valued unknowns in Tao: session notes (2026-09-03)

Branch: `hsuh/taoterm-prox-port-v1` (TaoTerm proximal maps, TAOFB forward-backward solver).
Question: users want `0.5*||A X - B||_F^2 + rowTV(X)` (later NMF), but Tao only takes a Vec. How to pass a matrix X? How do other packages do it?

## Files in this directory

| File | What it is | Status |
|---|---|---|
| `dense_rowtv.c` | X dense M x N, data term `0.5||A X - B||_F^2`, regularizer row-TV. X lives in the Tao solution Vec; a persistent MATDENSE placeholder borrows the Vec's storage via `MatDensePlaceArray()` for every matrix op. | Compiles. The TV term does not exist yet: the two `TaoTermTV1D*` lines are under `#if defined(TAOTERM_HAVE_TV1D)`, with `TAOTERML1` as a stand-in so the plumbing runs today. |
| `coo_sparse_l1.c` | X sparse with a fixed pattern. The unknown is the Vec of nonzero values (local length = number of COO entries this rank supplies). X is refilled each evaluation with `MatSetValuesCOO()`; gradient entries gathered with two `VecScatter`s. | Compiles and runs. |
| `makefile` | `make PETSC_DIR=<worktree> PETSC_ARCH=arch-prox-debug dense_rowtv coo_sparse_l1` | |

Both examples take `-M`, `-N`, `-lambda`, `-check_gradient` (central-difference check of the shell term), `-no_reg` (smooth term only, for classical Tao solvers).

## Verification actually done

- Both compile against `arch-prox-debug` in this worktree with no warnings.
- `-check_gradient`: hand-coded gradient vs central difference agrees to ~1e-9 relative, on 1 rank and on 2 (dense) / 3 (COO) ranks.
- Results are bit-identical across rank counts, also for larger sizes with `-malloc_debug`. The COO run on 3 ranks uses a naive row split that differs from PETSc's ownership, so remote entries are exercised.
- `-no_reg -tao_type lmvm`: smooth term alone drives the objective to ~1e-14 (dense) and the residual to ~4e-8 (COO) in ~25-30 iterations. This proves the reshape plumbing (placed arrays + reused products) and the COO refill path are correct.
- TAOFB with the L1 stand-in stops at objectives far above the known optimum (dense: 185.6, optimum below 0.3; COO: 9.4, optimum below 0.1) and reports `CONVERGED_GTTOL`. Not investigated further; the user does not care about the L1 stand-in. Worth a look when the TV term exists, since the same solver will drive it.

## Conclusions from the discussion

### 1. Vec-vs-Mat is the wrong axis
- Tao solvers only use vector-space algebra on the solution (axpy, dot, norm, pointwise). They never index into it. A Vec whose storage is a dense matrix, or a VecNest of such views, satisfies Tao fully.
- `vec(X)` is a relabeling, not a copy. MATMPIDENSE is itself one specific vec(X): column-major within the rank, rows distributed, leading dimension = local rows. Choosing Mat over Vec picks a layout, it does not remove the locality question.
- The real locality questions: which axis is contiguous locally, which axis is split across ranks, on GPU which axis adjacent threads stride over. Row-TV and column-TV answer all three oppositely, so no single layout serves both. GPU twist: thread-per-line Condat coalesces when lines are interleaved, the opposite of contiguous lines.
- Therefore the **term owns the layout**, via a shape setter (`TaoTermTV1DSetShape(term, M, N, axis)`) or a DM. Not the container.

### 2. How other packages do it
- CVXPY: matrix variables vectorized column-major internally; locality irrelevant (interior point).
- ODL: shape on the space, product spaces for tuples; prox acts on space elements. Closest to Vec + DM.
- ProximalOperators.jl / ProximalAlgorithms.jl: `prox!` on arrays of any shape, tuples for products. Solver sees only vector-space ops. Has `TotalVariation1D`.
- JAX/jaxopt, PyTorch: pytrees; solver never sees a flat vector.
- PyProximal, TFOCS: TV = L1 composed with a shaped gradient operator, iterative prox (PyProximal TV uses Beck-Teboulle FGP).
- proxTV (Barbero-Sra): `tv1d`, `tv2d`, `tvNd`; anisotropic ND TV via stacked 1D proxes. Same naming as recommended below.
- PETSc/SLEPc precedent: DMDA + `DMDAVecGetArray` for grid-shaped vectors; SLEPc BV for "Mat that is a set of Vecs" with zero-copy both ways; `VecNest` / `DMComposite` for product spaces.
- Uniform lesson: nobody makes the solver shape-aware. Shape lives in the function objects.

### 3. Is `0.5||A X - B||_F^2 + rowTV(X)` a real problem?
- Yes, but only when the columns of X are separate unknowns sharing one forward operator (data term is `(I kron A)`, block diagonal). Row-TV then couples adjacent columns.
- Real instances: dynamic / multi-channel imaging (columns = frames, row-TV = temporal TV, GRASP-style MRI, multi-energy CT); hyperspectral unmixing (SUnSAL-TV, Iordache/Bioucas-Dias/Plaza; leads to TV-NMF); fused lasso across ordered tasks; multi-trace seismic.
- NOT the tomography script (`~/code/overleaf/prox-latency/papers/tomography_acgn_routing_sim.py`): its data term is a full quadratic on vec(X). For one 2D image, the realistic matrix forms are denoising (A = I) or separable blur `A_c X A_r^T`; general tomography stays on vec(X); rowTV + colTV there is the Barbero-Sra anisotropic split.
- Layout flips with the instance: tall X (pixels x frames) wants row distribution = PETSc's default dense layout, row-TV fully local; wide X (endmembers x pixels) wants column distribution (store X^T), row-TV needs a one-column halo.

### 4. Two routes for passing X to Tao
- **Route B (used in `dense_rowtv.c`), for linear-algebra data terms:** X is the solution Vec of local length Mloc*N. Persistent placeholders `X`, `G` (MATDENSE) get their storage swapped to Tao's vectors each call with `MatDensePlaceArray()` / `MatDenseResetArray()`. Products `R = A X`, `G = A^T R` are created once with `MAT_INITIAL_MATRIX` and reused with `MAT_REUSE_MATRIX`. Reverse direction if the user already holds X as a Mat: `MatDenseGetArray` + `VecCreateMPIWithArray` (check `MatDenseGetLDA` == local rows). GPU: device place-array variants, memtype-aware getters, `MatCreateDenseFromVecType`.
- **Route A (DMDA), for grid/stencil/ray data terms:** `DMDACreate2d` -> `DMCreateGlobalVector` -> `TaoSetSolution`; callbacks use `DMGlobalToLocal`, `DMDAGetCorners`, `DMDAVecGetArray` (`x[j][i]`). Existing Tao examples: `src/tao/bound/tutorials/jbearing2.c`, `plate2.c`, `src/tao/unconstrained/tutorials/minsurf2.c`, `eptorsion2.c`. For local row-TV pick the process grid with one process along the TV axis. DMDA vectors cannot be reinterpreted as a dense Mat (different ordering).
- Gap found: `TaoTermSetSolutionTemplate()` keeps only layout and VecType, so vectors created through the term carry no DM (`VecGetDM` is empty). Vectors from `VecDuplicate` of a DMDA vector do carry it (composed-object list is duplicated). Conclusion: the TV term must store its own DM/shape, not rely on the solution Vec.
- Other in-tree precedents for the reshape: `src/ml/da/tutorials/ex1.c` (ensemble as MatDense, `MatDenseGetColumnVec`), `src/ksp/ksp/tutorials/ex76.c` (`MatCreateDenseFromVecType` + `KSPMatSolve`), `src/mat/tutorials/ex19.c` (memtype guarantee), `src/dm/tutorials/ex22.c` (`VecCreateMPIWithArray` on a DMDA slice).

### 5. Sparse unknowns
- Value sparsity is not storage sparsity. TV solutions are piecewise constant (full support); L1 solutions are sparse only at the optimum, iterates are dense. No proximal package stores iterates sparsely; lasso codes exploit sparsity via active-set/screening over dense storage.
- Dynamically changing pattern breaks the vector space (no axpy/dot between iterates), PETSc preallocation, and every term. Do not do it.
- Legitimate case: fixed known pattern (edge weights on a graph, entries of a stencil operator). Unknown = Vec of nonzero values; push into the operator with `MatSetPreallocationCOO` once + `MatSetValuesCOO` per evaluation (`coo_sparse_l1.c`). Any rank may supply any entry; supplying by row owner makes the residual gather a local permutation.
- Huge unknown -> factorize (W H), not sparsify. That is NMF / matrix completion; the sparse object there is the data B. PETSc has no SDDMM (dense-dense product sampled on a sparse pattern); that kernel would be needed for completion-style residuals.
- DMDA has nothing to do with sparsity: dense grid storage + halo; it generates sparse operators (`DMCreateMatrix`), not sparse unknowns.

### 6. NMF (later)
- Do not push NMF through a generic Tao solver on a flattened (W, H). With H fixed, rows of W decouple into m independent r-dim NNLS problems sharing one r x r Gram; HALS/ANLS/AO-ADMM live on BLAS3 and per-row micro-solves (sklearn, nimfa, PLANC all hand-roll; PLANC's contribution is the communication schedule on a 2D process grid).
- Fit for Tao: factors as MatDense; variable presented as VecNest of zero-copy views (SLEPc BV pattern); Tao gives monitors, convergence tests, the existing box term for nonnegativity; subproblem solvers stay in matrix form. Beyond 3 tensor axes store dims/strides in the term; do not build a tensor type.

### 7. Naming
- Use `TAOTERMTV1D` with an axis option (`-taoterm_tv1d_axis k`, axis as an index into the shape, not a rows/columns enum since that depends on layout). "1D" describes the TV, not the variable (proxTV, ProximalOperators.jl convention).
- Anisotropic 2D TV = rowTV + colTV = sum of two TV1D terms along different axes; `TAOTERMSUM` already expresses it, no new type.
- Isotropic 2D/ND TV (ROF, Euclidean norm of the per-pixel gradient) is the default "TV" in imaging (scikit-image, ODL, PyProximal); its prox needs an inner iterative solver with its own tolerances. Different algorithm -> different type later. Leave the bare `TAOTERMTV` name free for it.

### 8. Solver notes
- TAOFB handles one prox term. `LS + rowTV + colTV` needs a three-operator splitting (Davis-Yin) or the paper's routing scheme; the two TV terms want transposed decompositions, and the scatter between them is the routing cost the tomography script measures. Build it as a VecScatter from two DMDA natural orderings when the splitting engine exists; v1 of the TV term should just error if its axis is split across ranks.
- Layout-agnostic alternative already available: mapped L1 term via `TaoTermSumAddTerm(..., map = difference operator)` for PDHG / Condat-Vu style methods. No closed-form prox, works under any decomposition.

## Suggested order of work
1. `TAOTERMTV1D`: term-owned shape (or DM), axis option, Condat prox on each local line, `PetscCheck` that the axis is not split across ranks.
2. Mapped-L1 TV form for objective evaluation and primal-dual methods.
3. Transposing scatter between row-local and column-local layouts when the splitting engine lands.
4. Look at why TAOFB stalls on the L1 stand-in before trusting it on the TV term.
