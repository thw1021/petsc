# Temporal-TV reconstruction: PETSc implementation decisions and schedule

Date: September 11, 2026.

This is the new execution plan for `ACGN-PREDICTIVE-CODESIGN-PLAN.md`. It supersedes the task ordering and architectural recommendations of the old `SC26-PETSC-PLAN.md` for this project, without modifying that file. All new API names below are proposals, not interfaces already present in PETSc. This planning task changes no implementation code.

## 1. Decisions and deliverables

| Topic | Recommended decision |
|---|---|
| Application | Dynamic reconstruction with partitioned least squares and spatial-row, spatial-column, and temporal anisotropic TV |
| TV interface | One `TAOTERMTV1D` type, configured by shape/DM, axis, ownership, and boundary convention; no separate temporal-TV type |
| Term communicators | Preserve subgroup-local `TaoTerm` evaluation contracts; expose them on the Tao communicator through an explicit redistribution wrapper |
| Solver interface | Proposed `TAOSPLIT`, selected by `-tao_type split`, with a method selector initially supporting ACGN |
| Numerical algorithm | Fixed coefficients during each solve, generated externally and validated in PETSc |
| First parallel implementation | Deterministic CPU/MPI executor with explicit ownership transfers; include a two-rank term early |
| First GPU implementation | Device-resident buffers, correct real operators, and measured MPI/NCCL/NVSHMEM executions |
| Optimization order | Remove data movement and synchronization mistakes before optimizing kernels; then overlap, grouping, and optional graph replay |
| Predictive research | Offline shortlist prediction using actual convergence data at the benchmark dimensions; online racing remains optional |

The hardest required interface work is the communicator/layout boundary. Treat it as a separately reviewed deliverable with tests, not as a few relaxed communicator checks inside the sum implementation.

## 2. Source baseline inspected

The proximal worktree is `/home/hsuh/petsc/.claude/worktrees/taoterm-prox`, at `fc5c4316ba1` (`TV`, September 10). The transport worktree is at `abd936076bd` (`SF-NCCL measurement campaign`, September 9). These are inspection snapshots, not proposed upstream bases.

The user identifies the proximal branch as building on the mature `hsuh/feature-taoterm-improvement-v4` work expected to merge into main. Use that lineage and avoid reconstructing its improvements in the older transport branch. Reconcile against the actual merged commit when it becomes available; a merge date is not assumed.

Observed in the proximal worktree:

- `TaoTermProximalMap()` exists and has a generalized regularizer argument, with explicit aliasing semantics.
- `TaoTermSetLipschitz()` and `TaoTermGetLipschitz()` exist; zero means unknown.
- `TAOFB` exists, recognizes exactly `f_` and `g_` prefixes for its roles, and currently rejects nonidentity outer maps.
- `TaoAddTerm()` requires the term, parameters, and optional matrix map to share the Tao communicator.
- `TAOTERMSUM` validates full layouts and packs parameters using same-communicator factories and nested vectors.
- `TaoTermSetSolutionTemplate()` stores layout and vector type through a vector factory. It does not preserve all DM/shape semantics of the supplied vector.
- `taoterm-matrix-variable/dense_rowtv.c` sketches `TaoTermTV1DSetShape()` but uses an L1 stand-in when the proposed TV type is unavailable.
- `taoterm-matrix-variable/dmda_tv.c` is an objective/layout sketch for grid TV, not an implemented directional-TV prox.
- The worktree has pre-existing local changes to that examples folder. Leave them intact.

The examples' session notes report a concerning early `TAOFB` convergence result. Later FB fixes exist in the branch history, and the current code computes a gradient-mapping residual. Reproduce the issue before classifying it as an outstanding bug; do not assume either that the old report remains current or that subsequent fixes resolved it.

Observed in the transport worktree:

- SF-NCCL is implemented and measured; it is not a future backend to write from scratch.
- SF-NVSHMEM put/get, graph probes, and device-side prototypes exist.
- Current CUPM SF link setup reads the current `PetscDeviceContext`. The old plan's blanket claim that every SF link only uses the default stream is outdated. NVSHMEM still has setup/helper paths using `PetscDefaultCudaStream`, and link reuse across contexts needs auditing.
- The execution probes generally emulate operator work. Their timing results are infrastructure evidence, not measured dynamic-TV solver convergence.

## 3. Temporal TV: preserve the term model, extend the shape contract

### 3.1 Mathematical model

Use a field $X(i,j,t)$ of shape $(n_x,n_y,n_t)$ and solve

$$
\min_X\sum_{s=1}^{m}\sum_t\frac{1}{2\nu_{st}}\|B_{st}X_t-b_{st}\|^2
+\lambda_x\|D_xX\|_1
+\lambda_y\|D_yX\|_1
+\lambda_t\|D_tX\|_1.
$$

Here a shard may own a subset of measurements in each frame; absent frame/shard contributions are omitted. All normalizations $\nu_{st}$ and regularization scales must be explicit. Start with uniformly spaced frames, nonperiodic finite differences, real double precision, and constant nonnegative regularization weights.

Do not add arbitrary regularization only to satisfy a solver theorem. Establish existence for the selected instance family. For the small reference tests, a full-column-rank stacked measurement operator is a simple sufficient choice. Real acquisition geometries need their own existence and conditioning discussion.

The temporal regularizer is

$$
h_t(X)=\sum_{i,j}\sum_{t=0}^{n_t-2}|X(i,j,t+1)-X(i,j,t)|.
$$

Its Euclidean prox is one independent 1D TV solve per spatial pixel. This is the same operator family as row and column TV, applied along another axis. A separate term type is justified only for materially different mathematics, such as motion-compensated differences or a different time penalty.

Uniform spacing factors can be incorporated into the term scale. Nonuniform temporal weights require a weighted-TV algorithm and explicit weights; they are a later extension, not something the unweighted Condat kernel silently implements. Periodic temporal TV also requires different boundary handling.

### 3.2 Proposed user interface

Keep the solution a `Vec`. Shape belongs to the TV term and its DM/layout metadata, not to the Tao solver. Use axis indices rather than names that depend on matrix storage order.

Proposed minimal setters:

```c
PetscErrorCode TaoTermTV1DSetShape(TaoTerm, PetscInt, const PetscInt[]);
PetscErrorCode TaoTermTV1DSetAxis(TaoTerm, PetscInt);
PetscErrorCode TaoTermTV1DSetDM(TaoTerm, DM);
```

The shape setter specifies dimension count and extents. Shape-only mode initially has a documented contiguous sequential layout. DM mode obtains distributed ownership and indexing from a supported DMDA and stores its own reference to the DM. The setter combination must reject contradictory shape/DM descriptions.

For a 3D DMDA with natural index $I=i+n_x(j+n_y t)$, use axis 0 for horizontal differences, axis 1 for vertical differences, and axis 2 for temporal differences. That natural index labels physical unknowns; it is not necessarily the PETSc global Vec numbering on a particular process grid.

The dense matrix prototype also remains useful: $X\in\mathbb R^{N_{\rm pixels}\times n_t}$, with rows distributed and all frames retained for owned pixels. Temporal chains then remain local and are strided in the column-major local storage. Supporting this mode requires an explicit dense-layout/stride adapter; the extents alone cannot distinguish it from contiguous row-major storage. Prefer a small documented adapter to a general tensor-object API.

Initial implementation order:

1. Sequential shape mode and DMDA mode with complete local lines.
2. Dense column-major adapter if needed by the chosen projector or example.
3. Internal redistribution to complete-line ownership for a split axis.

The current sketch `TaoTermTV1DSetShape(term, M, N, axis)` has the right intent, but its fixed two-dimensional signature and implicit layout are too narrow for the complete space-time example. Extend the draft before publishing it, while retaining the `TAOTERMTV1D` concept.

### 3.3 Exact use of the proximal prototype

The inspected API computes

$$
\operatorname*{arg\,min}_x\{\alpha f(x;p)+\beta g(x;q)\}.
$$

With `reg == NULL`, $g(x;q)=\tfrac12\|x-q\|^2$. Therefore, for an unscaled directional-TV term and external weight $\lambda_i$, the ACGN call is

```c
TaoTermProximalMap(term, params, gamma_i * lambda_i, NULL, argument, 1.0, output);
```

This yields $\operatorname{prox}_{\gamma_i\lambda_i h_i}(\text{argument})$. Apply $\lambda_i$ exactly once. The `q` argument is the proximal center, while `p` is data/parameters belonging to the term. A TV term with geometry stored internally can use `TAOTERM_PARAMETERS_NONE` and `p == NULL`.

Implement the Euclidean regularizer case first and reject unsupported generalized regularizers. A generic nonquadratic `reg` is not covered by the ACGN resolvent theorem just because the function signature accepts it.

Honor the existing aliasing promise, including `output == argument`. If the line solver cannot operate safely in place, allocate reusable scratch during setup. Do not change the public aliasing contract to accommodate a kernel.

### 3.4 Distributed lines and data fidelity

A one-cell halo suffices to evaluate adjacent differences in the objective. It does not suffice to evaluate the exact prox of a line crossing ranks: the proximal problem couples the whole line. Correct the old session note's suggestion that a one-column halo alone solves a distributed row-TV prox.

Start by choosing a DMDA process grid unsplit along the active axis. Then add a cached redistribution to an axis-local decomposition, solve complete lines, and redistribute back. This keeps all three TV terms mathematically exact while allowing a common canonical problem layout.

Do not model general dynamic tomography as $AX$ unless every frame actually shares the same measurement operator. Use $B_{st}$ or a frame-aware shell term when view geometry varies in time. Check forward/adjoint consistency and put the Lipschitz bound on the common-variable gradient, including normalization and outer scaling. For a block-diagonal-in-time shard, a valid bound is the maximum frame-block bound; summing shard bounds is safe for the aggregate gradient but can be looser.

## 4. Communicator-aware terms: a wrapper with an explicit layout bridge

### 4.1 Why relaxing existing checks is insufficient

Simply accepting a subgroup in `TaoAddTerm()` or skipping local-size comparisons in `TAOTERMSUM` leaves unresolved:

- Nonparticipants have no valid local handle to the subgroup term or its parameter vectors.
- The term's local ownership cannot be inferred from global size alone.
- A parent `VecNest` cannot serve as an implicit pack of arbitrary subgroup vectors.
- Objective scalars may already be reduced over the subgroup and must not be counted once per member.
- Callbacks, finite differences, viewers, destruction, and parameter updates also have communicator contracts.
- Different DMDA or dense layouts can have equal sizes but different physical index meanings.

Preserve the mature same-communicator sum implementation. Make the distribution boundary an object with defined operations.

### 4.2 Recommended architecture

Introduce a proposed parent-communicator `TaoTerm` wrapper, provisionally `TAOTERMREDISTRIBUTE`:

```text
Tao and canonical Vec                    parent communicator
    |
    +-- ordinary TaoAddTerm() --> redistribution wrapper on parent
                                      |
                                      +-- layout bridge and membership metadata
                                      +-- local inner TaoTerm on subgroup
                                      +-- local inner parameters on subgroup
```

Every parent rank owns the wrapper. Only subgroup members own the inner term and parameters. The inner term continues to require same-communicator inputs/outputs, so `TaoTermCompute*()` and `TaoTermProximalMap()` retain their current validation.

The wrapper presents the canonical parent solution layout, and stores the inner parameters internally. Initially it advertises no externally packed parameters and has an explicit subgroup-parameter update operation. This avoids feeding foreign vectors to the mature sum parameter machinery. Document that generic derivatives with respect to these encapsulated parameters are unsupported until an explicit bridge is implemented.

The wrapper is useful outside ACGN: any consumer of an available objective, gradient, Hessian action, or supported prox can evaluate it through the parent interface. Implement and advertise only the operations it actually supports. For example, forward a Hessian action when implemented, but do not claim an assembled Hessian or silently materialize one.

### 4.3 Public construction contract to review in week 1

The constructor/configuration sequence must receive:

- Parent communicator and canonical solution template.
- An inner term and parameter vector on participating ranks; `NULL` on nonparticipants.
- An explicit mapping from each owned inner solution entry to a canonical physical unknown.
- A stable term name/prefix and any setup options for the bridge.

The constructor is collective on the parent. Validate that the inner communicator is an intracommunicator whose group is a subgroup of the parent, including reordered membership. Translate ranks during setup instead of assuming local subgroup ranks equal parent ranks. Validate one coherent inner communicator instance per logical term; separate `PETSC_COMM_SELF` terms on several ranks are separate terms, not one accidental replicated term.

Exact function names and whether construction is one call or several setters are a week-1 API-review deliverable. The key user experience is fixed: construct a local term, wrap its placement, and add the parent wrapper with ordinary `TaoAddTerm()`.

Use a parent-communicator `PetscSF` with local buffer access for the bridge. Do not assume a conventional `VecScatter` directly accepts arbitrary parent/subgroup Vec pairs. DMDA bridges should use natural-order/AO information to map physical entries; dense adapters need their explicit local-to-canonical map.

### 4.4 Mathematical bridge semantics

For the first version, require each subgroup's distributed solution vector to contain exactly one copy of every canonical variable, possibly permuted. Let $R_i$ denote that ownership change and permutation. It is an isometry of the global vector space, not an arbitrary rectangular application matrix:

$$
\widetilde h_i(x)=h_i(R_ix),\qquad
\nabla\widetilde h_i(x)=R_i^\top\nabla h_i(R_ix),
$$

$$
\operatorname{prox}_{\gamma\widetilde h_i}(v)
=R_i^\top\operatorname{prox}_{\gamma h_i}(R_iv).
$$

These identities justify the ordinary objective, gradient, and Euclidean-prox wrapper. A gather to one hosting rank is valid when that rank owns the whole inner vector; a distributed subgroup vector is valid when its ownership partitions the whole vector. Input replication inside a cooperative projector is an implementation detail of that term and must not multiply the mathematical norm or objective.

Initially reject partial-variable or overlapping inner solution maps at this wrapper boundary. Such maps need different proximal semantics. The projector $B_{st}$ belongs inside the inner smooth term, or in a supported same-communicator mathematical map; it is not the redistribution map $R_i$.

### 4.5 Evaluation, scalar accounting, and parameters

A generic wrapper evaluation is collective on the parent:

1. Move the canonical input to the subgroup layout.
2. Members call the inner operation collectively on their subgroup; nonmembers do not call it.
3. Move vector outputs to the canonical parent layout, or publish a scalar result.

A scalar returned collectively by the inner term is contributed by one designated subgroup leader to a parent reduction. Summing that scalar from every subgroup rank would multiply the objective by the subgroup size. A gradient assembled from disjoint owned inner entries is moved back once; accumulation across distinct terms happens at the sum/solver level.

Reference the inner term and its parameters on members. Parameter replacement is coordinated through a parent-collective wrapper operation, with subgroup-local handles supplied by members, and invalidates the relevant cached state. Changes affecting geometry or norms also invalidate the algorithm design. Changing a least-squares right-hand side can reuse geometry-dependent bounds, but still changes objective state.

### 4.6 Lifetime, errors, and collective order

- Use deterministic parent registration and destruction order. Destroy or release inner objects collectively on their subgroup before releasing the communicator they require.
- Do not rely on arbitrary last-reference destruction to happen simultaneously across an overlapping set of communicators.
- Validate membership, index coverage, duplicated indices, sizes, capabilities, and option consistency during setup. Aggregate detectable local setup errors before entering a later collective that would otherwise strand other ranks.
- Support disjoint and overlapping subgroups, but serialize operations on overlapping groups initially. All members must see the same collective order.
- Do not promise recovery from arbitrary rank failure or a rank-local runtime error inside MPI/NCCL collectives. The goal is deterministic valid execution and early detection of configuration errors.
- Viewer calls stay on the viewer's communicator. Parent `TaoView()` should print gathered metadata rather than pass a parent viewer into a subgroup object.
- Cloning, reset, changing `TaoType`, and repeated solves must not retain stale bridge or subgroup references.

### 4.7 Efficient splitting execution without abandoning the wrapper

The generic wrapper implementation is the correctness path. Executing it literally for every ACGN operation would route each vector through the canonical parent layout and introduce unnecessary barriers.

The splitting solver should compile the registered wrappers into an execution plan. It can retain $x_i$, $w_i$, and gradients at their hosting layouts and route directly between producers and consumers. Reuse the wrapper's validated membership/layout metadata and inner-operation entry points rather than maintaining an unrelated solver-owned term registry.

Start with private execution-plan hooks. Introduce a public split-phase term API only if a concrete second consumer needs it. The planner schedules parent-SF operations consistently across parent ranks, then invokes subgroup computations in an order that respects both the mathematical DAG and overlapping communicator collectives. Generic wrapper calls may be used for infrequent objective evaluations and reference checks.

Synchronize the public `Tao` solution into its canonical layout before user monitors, convergence callbacks, and final return. If objective reporting at each iteration costs extra forwards or redistribution, record and charge it; never report a sum of objectives at different slot iterates as the objective at the returned solution.

### 4.8 Required communicator tests

Test on CPU before GPU:

- One-rank parent, full-parent inner term, one-rank subgroup, and a two-rank subgroup.
- Noncontiguous and reordered parent rank membership.
- Disjoint subgroups and overlapping groups such as ranks `{0,1}` and `{1,2}`.
- Empty local ownership, inactive parent ranks, repeated parameter updates, reset, and destruction.
- Different valid DMDA decompositions representing the same physical field.
- Objective values counted once; gradient adjoint identity for the bridge; prox equality under permutation and redistribution.
- Missing/duplicate semantic indices, inconsistent registration, and unsupported capabilities rejected during setup.

Do not postpone all gang support until the GPU phase. A two-rank inner term is necessary to validate the abstraction itself.

## 5. Tao solver interface: describe the operation to the user

### 5.1 Name and scope

Recommend `TAOSPLIT` with `-tao_type split`. `TAOOS` is defensible but less discoverable; `TAONOS` does not communicate a clear contract. Keep ACGN visible as the chosen method, not as the only possible top-level solver type.

The initial solver contract is a sum of common-variable smooth terms and proximable terms. It does not promise to unify every ADMM, primal-dual, stochastic, or nonconvex splitting. Existing `TAOFB` and `TAOADMM` should retain their behavior. Share proven utility code where appropriate, without forcing those solvers into a new hierarchy.

Proposed configuration:

```text
-tao_type split
-tao_split_method acgn
-tao_split_design_file design.json
-tao_split_execution eager
-tao_view
```

Initially `-tao_split_method acgn` is the only implementation. Later methods may share the same problem contract when justified. Proposed execution choices are `-tao_split_execution (eager|graph)`, with graph available only when the compiled plan passes its capability checks. Transport remains an execution choice, using the SF/backend options where appropriate; it is not encoded in the mathematical method name.

### 5.2 Explicit term roles

Keep `TaoAddTerm()` for objective composition. Add a proposed `TaoSplitSetTermRole()` operation keyed by the registered term prefix/name, with roles corresponding to direct gradient evaluation and proximal evaluation. Prefix spelling has no mathematical meaning.

The user registers `data0_`, `data1_`, `space_x_`, `space_y_`, and `time_`, then assigns the data terms to forward evaluation and the TV terms to backward evaluation. This avoids extending `TAOFB`'s current hard-coded `f_`/`g_` convention to a many-term solver.

Role selection must be explicit when a term supports both operations. A gradient capability alone is not a proof of convexity or a valid Lipschitz constant. Require the ACGN assumptions and usable bounds in setup, and reject unsupported roles or generalized prox regularizers. For this workload all outer weights are nonnegative.

### 5.3 Algorithm design and execution plan

Keep three internal records distinct:

| Record | Contents |
|---|---|
| Problem registration | Terms, roles, weights, parameters, shape, and wrapper placement metadata |
| Method design | Slot order, gradient routing, coefficients, stepsizes, validity diagnostics, and normalization |
| Execution plan | Ownership, transfers, grouping, streams, buffers, dependencies, and capture state |

The first PETSc engine accepts a generated design and validates it. Python handles large candidate enumeration and predictive search. Avoid embedding an optimizer or adding a public first-class design object until an actual use case requires that API; an internal record plus a versioned input file is sufficient initially.

The file must bind coefficients to stable term identities and geometry/norm provenance. A permutation of slots cannot silently swap shard data. Freeze coefficients during a solve. Surface the selected method, design hash, stepsizes, term placement, and actual transports in `TaoView()`.

### 5.4 Stopping and state

ACGN lifted-state convergence is not identical to the existing `TAOFB` gradient-mapping test. Give the new solver its own documented residual semantics and connect them to Tao's monitor/history/convergence mechanisms.

Use small independent references first. Establish a deployable common optimality measure before performance claims across algorithms; do not select candidates solely by unscaled $\|\mathcal Lx\|$, which changes when the design changes. If an application supplies a primal-dual-gap or validated KKT callback, its communication and computation enter the cost model.

Specify initialization explicitly. The certified reference implementation can use $w^0=0$; arbitrary primal warm starts need a documented lift to admissible state rather than ignoring `TaoSetSolution()` contents. Full continuation checkpoints must store the design and lifted state, not only the final primal vector. Do not apply `TAOFB`'s adaptive or accelerated update rules automatically to ACGN.

## 6. GPU work: more than a TV kernel

### 6.1 Baseline requirements

Before custom fused kernels, complete:

1. Correct GPU TV operations, including aliasing, short lines, endpoints, strides, and precision limitations.
2. A real GPU projector/adjoint implementation and an adjoint test. Use existing PETSc matrix backends where suitable.
3. Persistent device buffers for state, inputs, routing lanes, transposes, and operator workspaces.
4. Device-aware array access throughout the hot path, with no accidental host copies or host norm evaluation between every task.
5. Correct operation-to-stream association, explicit data dependencies, and audited SF link reuse.
6. Setup-time communicator/backend creation and buffer registration, with lifetime rules spanning asynchronous execution.
7. Per-term and per-transfer logging, separate setup/solve stages, and traces that expose actual overlap.
8. A costed stopping protocol and rank-consistent termination.

Keep input/output vector ownership and PETSc object state correct when borrowing dense arrays or caching device pointers. Do not keep a pointer past a restore operation without a documented lifetime guarantee.

### 6.2 Evidence from the existing campaign

The September 9 nonblocking-stream ACGN skeleton measurements in `NVSHMEM-PERF-NOTES.md`, section 29.4, report:

| Placement and payload | MPI | SF-NCCL | SF-NVSHMEM put |
|---|---:|---:|---:|
| One node, 64 KB | 148 microseconds | 93 microseconds | 109 microseconds |
| One node, 4 MB | 318 microseconds | 377 microseconds | 301 microseconds |
| Two nodes, 2+2 ranks, 64 KB | 190 microseconds | 165 microseconds | 183 microseconds |
| Two nodes, 2+2 ranks, 4 MB | 927 microseconds | 1060 microseconds | 854 microseconds |

These are execution-skeleton measurements with emulated operator costs. They support prioritizing certain experiments, not assuming the same ranking for dynamic-TV reconstruction. The recorded allreduce arm still used host `MPI_Allreduce`; enabling SF-NCCL did not replace all collectives in the application.

The tuned ring results also show topology-dependent crossover: NCCL wins small messages, while the large-message advantage often favors MPI. Treat latency/size thresholds as machine-and-pattern dependent rather than a fixed universal rule.

### 6.3 Where NCCL can help this problem

| Work in the solver | Potential benefit | Required evidence |
|---|---|---|
| Distribution of a small evaluation-point vector | Lower exposed latency and stream-ordered execution | Measure actual fan-out and recipient set |
| Small branch-gradient contributions | Reduced host waiting; useful grouping | Preserve branch readiness; avoid waiting on unrelated late data |
| Small complete-line redistribution | Lower startup cost for many peers/messages | Measure transpose pattern, not only a ring |
| Cooperative gradient reduction | Device-resident reduction and potential graph integration | Measure gang size, payload, and overlap with local operator work |
| Repeated small state updates/collectives | Fewer host synchronization points | Include actual reduction implementation and stop protocol |
| Frozen repeated iteration body | CUDA graph replay can amortize launch overhead | All operations, not just SF, must be capture-compatible |

Small scalar stopping reductions benefit only if they actually remain on the device and use an appropriate collective path. SF-NCCL byte transport does not automatically replace `VecNorm()` or an explicit `MPI_Allreduce()`.

For the main field size, $B=8n_xn_yn_t$ bytes in double precision. A $16\times16\times32$ field is 64 KB, whereas a $256\times256\times32$ field is 16 MB. Large space-time reconstructions can quickly leave the latency-dominated regime. Small messages may still occur for subdomains or reductions, but no NCCL speedup should be promised from the word "temporal" alone.

Group transfers by a compatible readiness/completion window, not mechanically by an entire drawn stage. A larger group can reduce launches while delaying an otherwise ready branch. NVIDIA documents that mixing streams within a group imposes dependencies across those streams [1]. Grouping is an execution-design choice to measure.

NCCL supports capture of collective and point-to-point operations, but capture and replay participation must be consistent across involved ranks [2]. Test the installed version and communicator lifecycle. Prefer one GPU per process for the initial executor.

### 6.4 Where NVSHMEM can help

NVSHMEM has two separate opportunities:

- **Existing SF transport:** stream-aware one-sided movement with signaling. The campaign's 4 MB skeletons favored the put path, so include it in bulk routing and redistribution comparisons as well as small-message tests.
- **Specialized device-driven execution:** producer/consumer signals and device-issued movement can remove host orchestration from very fine-grained repeated work. This is a later specialization, not a prerequisite for useful SF-NVSHMEM support.

Start with the existing host-enqueued SF path. Validate communicator eligibility, all-rank allocation/setup order, buffer symmetry where required, put/get completion, and reuse protection. Ineligible subgroup exchanges should take a clear supported fallback; do not silently label the run NVSHMEM. Parent-communicator routing can use the eligible parent SF where its contract is satisfied.

A persistent kernel cannot invoke arbitrary PETSc host APIs or cuSPARSE calls as though they were device functions. The existing emulation kernel does not solve this integration problem. Only pursue a device-resident fused iteration after identifying a real operator set with device-callable implementations and demonstrating an orchestration bottleneck worth the additional work.

Do not create a general symmetric-heap Vec type solely to support the first solver. Begin with transport-owned symmetric staging buffers. A new allocation mode needs independent lifetime, collectivity, resizing, and ownership design and is a later deliverable if staging is measured to dominate.

### 6.5 Device contexts and graph work

The execution plan owns task/stream scheduling initially. Use existing `PetscDeviceContext` operations and explicit dependency events. Audit current-context use and cached links before proposing a new per-term or per-SF public context API. The key is a defined execution contract, not the mere existence of a setter.

First prove that compute and communication use the intended nonblocking streams and that cached links still behave correctly when contexts change. Keep a conservative single-context path. Overlap must be visible in a trace and must improve the measured critical path; two streams competing for HBM need not help.

Graph replay is a conditional work package. Preallocate, warm up, and resolve lazy initialization before capture. Check the actual operator implementations for forbidden host interactions. Device-side data-dependent loops, such as a TV kernel's own loop, are not inherently a capture problem; host-side decisions that change the launched workflow are.

Start with a fixed chunk of iterations and a stopping check between chunks. All ranks agree on chunk count and termination. Report delayed stopping and any extra iterations. Use the same checking policy in controlled transport comparisons or report the policy difference explicitly.

### 6.6 Setup cost and break-even

The campaign records approximately 2.5–3 seconds of NCCL startup/lazy connection cost in a particular solver setup. This is a measurement to revisit, not a universal constant. A warmed microsecond-level gain may lose on a cold one-off solve.

For two executions of the same numerical algorithm, a simple break-even estimate is

$$
N_{\rm break-even}
=\frac{T_{\rm setup,new}-T_{\rm setup,base}}
{C_{\rm base}-C_{\rm new}},
$$

when the denominator is positive. Report both cold total time and warmed iteration/solve time. An argument that startup is amortized must identify the actual repeated-use workload. Do not merely exclude startup from all results, as the old campaign plan suggested.

## 7. Concrete schedule

### 7.1 Planning assumptions

Use one experienced primary implementer and intermittent GPU allocations as the baseline. The dates below start Monday, September 14, 2026. They are effort estimates, not a promise tied to an unspecified SC26 deadline. No submission or release-clearance date has been supplied.

Target a first valid distributed GPU solve in weeks 6–7, a calibrated candidate evaluation in weeks 8–10, and hardened results/documentation by week 12. A second contributor can advance transport validation or the Python pipeline independently, but the dates do not assume an extra contributor.

### 7.2 Week-by-week plan

| Week | Dates | Primary work | Reviewable exit condition |
|---|---|---|---|
| 1 | Sep 14–18 | Pin branch bases; rerun prox/FB regressions; settle TV shape and redistribution-wrapper contracts; settle solver roles/name | Written API contract, buildable integration base, regression status, and a tiny canonical space-time fixture |
| 2 | Sep 21–25 | Implement CPU `TAOTERMTV1D`, sequential and complete-line DMDA modes; build temporal phantom and forward/adjoint fixture | All three directional proxes match independent small references; endpoints, scale, and aliasing checked |
| 3 | Sep 28–Oct 2 | Implement proposed splitting Tao shell with explicit roles, design loading/checks, and serial ACGN iteration | Actual spatial-plus-temporal-TV problem converges to the independent reference; monitor and returned solution agree |
| 4 | Oct 5–9 | Implement parent wrapper, semantic index bridge, objective/gradient/prox operations, and subgroup parameters | One-rank and two-rank inner terms produce the same mathematical outputs through the parent wrapper |
| 5 | Oct 12–16 | Harden overlapping-communicator order/lifetime; compile direct term-to-term routing; add complete-line redistribution | Four-rank CPU solve agrees with serial to tolerance for multiple placements, including a two-rank term |
| 6 | Oct 19–23 | Port the actual TV operations to GPU; use a GPU projector/adjoint; persistent buffers; first multi-GPU MPI baseline | Device-resident real-operator solve with correct stopping; initial traffic and memory accounting |
| 7 | Oct 26–30 | Audit contexts/link reuse; enable measured SF-NCCL and NVSHMEM executions; validate a cooperative gradient | Same design and target on all transports; actual backend identified; cold/warm timings and a timeline explain differences |
| 8 | Nov 2–6 | Improve transfer grouping and overlap; attempt bounded graph replay if capture audit passes; connect executor traces to Stage-2 model | Measured legal execution plans and clear error budget; eager path remains complete if graph capture is blocked |
| 9 | Nov 9–13 | Calibrate full operator/pattern catalog; generate dynamic-TV candidate pool; measure numerical convergence at actual sizes | Held-out execution-model checks and a dataset linking exact design, problem, period, iterations, and time to target |
| 10 | Nov 16–20 | Implement simple predictive shortlist rules/model; compare against fastest-period and diverse/random shortlists | Untouched-test shortlist quality with equal evaluation budgets and all feature costs reported |
| 11 | Nov 23–27 | Stress placements, precision/scale, repeated solves, and memory; prepare reproducible result artifacts and figures | Final comparisons have a common accuracy criterion, documented failures/timeouts, and reproducible provenance |
| 12 | Nov 30–Dec 4 | Reserve for failures, review, documentation, MR revisions, and final result checks | Reviewable MR stack and evidence-backed narrative; unsupported performance claims removed |

The working critical path is TV correctness → serial numerical engine → communicator wrapper/direct routing → real GPU solve → calibrated timings and convergence data → predictive evaluation. Communication architecture gets two dedicated weeks because its testing and lifetime work are substantial.

Weeks 4–5 can begin earlier if another contributor is available after the week-1 contract. Early GPU allocations can validate existing transports without waiting for the TV implementation, but those measurements do not replace the real-solver milestone.

### 7.3 Milestone gates and scope control

| Milestone | Required evidence | If it fails |
|---|---|---|
| M0, end W1 | Prox contract understood; upstream base pinned; regression results reproducible | Repair or isolate the base before adding APIs |
| M1, end W3 | Serial dynamic-TV solve validated independently | Stop performance work on that numerical path |
| M2, end W5 | Communicator/layout invariance and correct scalar accounting | Keep parent-wrapper path as reference; fix routing before GPU optimization |
| M3, end W7 | Real distributed GPU solve on supported transports | Publish narrower transport coverage; do not substitute AXPY convergence claims |
| M4, end W9 | Stage-2 timing model validated on held-out executions | Retain more executions in shortlists and use direct timing for finalists |
| M5, end W10 | Predictive shortlist beats simple budget-matched alternatives or has characterized limits | Report algorithm/execution tradeoffs without claiming a successful selector |

If a delivery deadline is earlier than this schedule, preserve M1–M3 and a small measured algorithm comparison. Cut graph replay, large candidate pools, and predictive automation before cutting communicator correctness or common-accuracy validation. Confirm actual deadline constraints before treating the calendar as a publication commitment.

### 7.4 First five working days

1. Pin the two branch snapshots and the upstream-v4 ancestry; record existing user changes; run the proximal and FB regression cases relevant to the new term.
2. Write a small explicit $(n_x,n_y,n_t)$ fixture with semantic indices, frame operators, normalization, and a trusted solution procedure. Select the common stopping target.
3. Review the three TV setters, supported layout modes, Euclidean-prox contract, and axis boundary behavior against that fixture.
4. Review a parent/subgroup wrapper scenario with noncontiguous members and a two-rank term; trace input, gradient, scalar, parameter, and destruction operations.
5. Freeze the first public surface and the private solver records; prepare the CPU TV work package and the serial solver skeleton against those exact contracts.

## 8. Merge-request and verification sequence

Keep the mature term improvements and their upstream merge independent of this project's larger solver changes. Develop on an integration branch based on the proximal worktree lineage, and bring reviewed transport changes in separately. Do not merge campaign binaries/results into an upstream MR.

Suggested MR units:

1. Proximal API/Box/Lipschitz/FB work already on the prototype branch, reconciled with the v4 merge and independently checked.
2. CPU `TAOTERMTV1D`, shape/DM support, objective/prox tests, and user documentation.
3. Redistribution wrapper and communicator/layout contract, with CPU tests independent of any ACGN solver.
4. Proposed `TAOSPLIT` interface, roles, serial ACGN implementation, design validation, and small tutorial.
5. Distributed execution plan and complete-line redistribution, using the wrapper's established metadata.
6. GPU term implementations, context/stream correctness changes, and targeted GPU tests.
7. Remaining reviewed NVSHMEM/NCCL robustness changes; the existing backend implementation is reused, not rewritten.
8. Optional graph execution and narrowly justified fusion work after measured evidence.

Exact MR boundaries may change with upstream review, but each should have a runnable correctness story. Run the relevant CPU tests, formatting, source checks, and docs/lint where available; run GPU/multi-node tests on allocated hardware and state any unavailable verification explicitly. Add narrow test cases to PETSc's normal TEST harness instead of relying only on campaign scripts.

## 9. Additional work that must not fall through the gaps

- **Canonical indexing:** physical index identity across DMDA decompositions and dense views is a required data-model feature, not a cosmetic reshape.
- **Initialization and restart:** define how primal input and lifted state interact; changing a design invalidates an incompatible checkpoint.
- **Accuracy:** choose the common criterion before collecting training labels. Do not fit a predictor to iteration counts from incomparable residuals.
- **Safe norms:** do not copy the FB fallback based on two random points into the fixed-coefficient ACGN certificate path. It is not a safe global Lipschitz bound.
- **Scalability:** retain a capacity model for $n$ lifted vectors, outputs, $m$ gradients, transpose buffers, and communication lanes. Start with a one-GPU-per-term execution only where the full vector fits.
- **Counter overcounting:** profile one logical gradient/prox regardless of subgroup size, and count subgroup-reduced objective values once.
- **Backend robustness:** optimized builds must have a well-defined collective backend choice under mixed memory types; setup validation and fallback rules deserve their own tests. Avoid assuming debug-only guards provide a production contract.
- **Reproducibility:** record code/design hashes, geometry, norms, dtype, GPU placement, backend eligibility, environment, cold setup, warm time, stopping policy, and failures.
- **Prediction:** use real temporal-TV runs at each claimed size. The old box example and its five iteration counts remain regression fixtures only.
- **Boundary of scope:** weighted/motion-compensated temporal TV, generic inexact-prox theory, persistent device-callable full solvers, online racing, and a general tensor type are not required for the first completion milestone.

## 10. References to inspected sources

### Local files

- `ACGN-PREDICTIVE-CODESIGN-PLAN.md`: research scope and detailed ACGN/cost-model definitions.
- `/home/hsuh/petsc/.claude/worktrees/taoterm-prox/include/petsctaoterm.h`: current proximal function type and declarations.
- `/home/hsuh/petsc/.claude/worktrees/taoterm-prox/src/tao/term/interface/taoterm.c`: proximal validation and solution-template behavior.
- `/home/hsuh/petsc/.claude/worktrees/taoterm-prox/src/tao/interface/taosolver.c`: `TaoAddTerm()` communicator contract.
- `/home/hsuh/petsc/.claude/worktrees/taoterm-prox/src/tao/term/impls/sum/taotermsum.c`: layout setup and nested parameter packing.
- `/home/hsuh/petsc/.claude/worktrees/taoterm-prox/src/tao/proximal/impls/fb/fb.c`: current FB role, map, and residual behavior.
- `/home/hsuh/petsc/.claude/worktrees/taoterm-prox/taoterm-matrix-variable/session-notes.md`, `dense_rowtv.c`, and `dmda_tv.c`: proposed TV and matrix/grid layout examples.
- `src/vec/is/sf/impls/basic/cupm/sfcupm_impl.hpp`: current-context stream selection at SF link setup.
- `src/vec/is/sf/impls/basic/nccl/sfnccl.cu` and `nvshmem/sfnvshmem.cu`: current backend restrictions, streams, and protocols.
- `NVSHMEM-PERF-NOTES.md`, sections 29.4–29.6, and `nvshmem-tools/results-20260909-sfnccl.txt`: recorded skeleton, graph, startup, and ring measurements.
- `ACGN-NCCL-PLAN.md`: earlier interpretation of the transport campaign. Treat its stage-fusion and startup guidance as hypotheses to refine using the requirements above.

### External documentation checked for this plan

1. NVIDIA, [NCCL CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html): stream-ordered operations and the dependencies introduced by grouping multiple streams.
2. NVIDIA, [Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html): capture/replay participation and implementation restrictions. Validate against the version actually installed on the target machine.
