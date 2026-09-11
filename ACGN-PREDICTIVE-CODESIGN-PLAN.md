# Predictive co-design of ACGN splitting and GPU execution

Status: research and implementation plan, September 11, 2026. This document specifies future work; it does not claim that the complete workflow or PETSc solver already exists.

## 1. Objective and scope

Given a convex reconstruction problem, an admissible family of splitting algorithms, and a particular GPU machine, select a small set of candidates likely to give good measured time to a common accuracy target.

The workflow has three stages:

1. **ACGN candidate generation:** construct concrete numerical algorithms satisfying the ACGN convergence conditions.
2. **Execution design and cost estimation:** turn each algorithm into feasible GPU executions and estimate their per-iteration costs.
3. **Predictive selection and validation:** combine execution estimates with inexpensive numerical information to shortlist candidates, then assess the shortlist using actual solves.

The mathematical role of ACGN is to constrain the search to valid algorithms. The performance role of the cost model is to estimate execution time. Neither by itself predicts iteration counts.

The initial project is predictive and empirical. Deriving new worst-case PEP bounds, proving optimality over the full ACGN family, solving a global mixed-integer semidefinite program, and proving a generally successful online solver selector are outside its required scope. Existing certificates may be used as predictive features if they prove useful, but they are optional.

The primary research question is:

> Can a small, hardware-aware shortlist retain algorithms close to the best measured time to tolerance, using substantially less evaluation than an exhaustive search?

A stronger deployment question is evaluated separately:

> On a new problem, does selection plus the final solve cost less than immediately running a strong default?

These questions have different costs and evidence requirements. Offline experiments can answer the first even when online selection does not pay for itself.

## 2. Problem class and benchmark

### 2.1 Common-variable convex problems

Consider

$$
\min_{x\in\mathbb R^d} F(x)
=\sum_{s=1}^{m}f_s(x)+\sum_{i=1}^{n}h_i(x).
$$

Assume:

- Each $f_s$ is convex and differentiable, with a valid positive Lipschitz upper bound $L_s$ for its gradient.
- Each $h_i$ is proper, lower semicontinuous, and convex, and has an available proximal operator.
- The inclusion below has a solution. For an optimization model, verify the subdifferential sum qualification needed to identify minimizers with inclusion solutions. Finite-valued directional-TV penalties and quadratic losses satisfy the usual finite-dimensional qualification.
- Operator evaluations satisfy the accuracy assumptions of the chosen convergence result. The starting implementation targets direct proximal evaluations; approximate inner solves require a separately justified error policy.

Write $\mathcal A_i=\partial h_i$ and $\mathcal C_s=\nabla f_s$. The inclusion is

$$
0\in\sum_{i=1}^{n}\mathcal A_i(x)+\sum_{s=1}^{m}\mathcal C_s(x).
$$

All candidates solve this same problem, with fixed regularization parameters and the same accuracy requirement. Different iterates, iteration counts, and limit points in a nonunique solution set are acceptable.

### 2.2 Preferred benchmark: dynamic tomography

A useful extension of the existing row/column-TV example is a sequence of images $X\in\mathbb R^{n_r\times n_c\times n_t}$:

$$
\min_X\sum_{s=1}^{m}\frac{1}{2\nu_s}\|\mathcal B_sX-b_s\|^2
+\lambda_R\|D_RX\|_1
+\lambda_C\|D_CX\|_1
+\lambda_T\|D_TX\|_1.
$$

Here $\mathcal B_s$ is a linear measurement shard and $\nu_s>0$ is an explicit normalization, such as its measurement count. Setting $\nu_s=1$ gives unnormalized least squares. The three regularizers express spatial and temporal piecewise smoothness. Temporal regularization has an application rationale in dynamic tomography; this particular equation is the proposed benchmark specialization, not a verbatim reproduction of a cited application paper [4].

Each directional prox is a batch of independent one-dimensional TV problems along its axis. These are exact directional decompositions of anisotropic TV. They do not implement isotropic TV. Different axes create different chain lengths, memory strides, and ownership requirements. Those differences should determine measured costs; do not impose artificial operator-cost ratios.

Without a box constraint, existence should be verified rather than inherited from compactness. For example, full column rank of the stacked measurement operator makes the quadratic loss coercive. Other acquisition models may require a separate existence argument or a physically justified constraint.

An alternative single-image benchmark adds $\mu\|Wx\|_1$ to row and column TV, where $W$ is a square orthogonal wavelet transform. Its prox is

$$
\operatorname{prox}_{\gamma\mu\|W\cdot\|_1}(v)
=W^\top\operatorname{soft}_{\gamma\mu}(Wv).
$$

Orthogonality is essential to this formula; a redundant analysis transform generally needs a different proximal implementation. Wavelet and finite-difference sparsity are established reconstruction priors [5].

### 2.3 Decomposition and fusion

Initially, the decomposition into smooth shards and nonsmooth terms is an input. Automated selection of whether to fuse terms is optional future work.

Include obvious exact simplifications among the baselines. In particular, a uniform box can be absorbed into directional anisotropic TV by clipping the TV-prox output [6]. Keeping it separate is valid, but any claimed advantage must survive comparison with that simplified formulation.

A general term interface is broader than the algorithm's mathematical contract. A prox for $h$ does not automatically supply a cheap prox for $h(Bx)$. Smooth mapped terms can use the chain rule; backward terms must expose their actual common-variable prox or use a justified reformulation.

## 3. Stage 1: detailed ACGN characterization

### 3.1 What is being characterized

ACGN gives a constructive family of averaged, frugal, minimal-lifting splitting operators [1]. Frugal means each backward operator is evaluated through one resolvent and each forward operator directly once per logical iteration. Minimal lifting concerns the number of independent vectors retained by the abstract fixed-point iteration, not all runtime buffers or replicated GPU storage.

We use the sufficiency direction: construct a member of the family and inherit its convergence result. We do not claim to enumerate every admissible splitting. The dimension qualification in the full characterization, $d\ge 2n+m-1$, concerns necessity/completeness; it is not needed to validate an individual construction through sufficiency.

### 3.2 Forward-operator constants

Convexity and $L_s$-Lipschitz continuity of $\nabla f_s$ give $1/L_s$-cocoercivity:

$$
\langle\mathcal C_s(x)-\mathcal C_s(y),x-y\rangle
\ge\frac{1}{L_s}\|\mathcal C_s(x)-\mathcal C_s(y)\|^2.
$$

For the normalized least-squares shard,

$$
\mathcal C_s(x)=\frac{1}{\nu_s}B_s^\top(B_sx-b_s),
\qquad L_s=\frac{\|B_s\|_2^2}{\nu_s}.
$$

A safe upper bound may replace the exact norm. For an explicit matrix,

$$
\widehat L_s=\frac{\|B_s\|_1\|B_s\|_\infty}{\nu_s}
\ge L_s.
$$

For a nonnegative projector, its row and column sums can be obtained through forward and adjoint applications to all-ones vectors. This shortcut does not apply to arbitrary signed operators. An ordinary power-iteration Rayleigh quotient is usually a lower estimate of the maximum eigenvalue, not a certified upper bound; multiplying it by an arbitrary safety factor does not create a deterministic certificate.

Compute valid bounds during setup and freeze the resulting algorithm during a solve. Cache them by geometry, discretization, sharding, normalization, and operator implementation. Changes to $b_s$ alone do not change the least-squares Lipschitz constants. A zero operator can be removed or assigned a positive conservative bound.

### 3.3 Parameters and their meaning

Use $n$ proximal slots and $m$ forward operators. A slot permutation assigns physical term identities to the slots.

| Parameter | Dimensions | Meaning |
|---|---|---|
| $H$ | $n\times m$ | Weights routing each forward output into later proximal arguments |
| $K$ | $m\times n$ | Weights combining proximal outputs into each forward evaluation point |
| $\mathcal L$ | $n\times n$ | Symmetric coupling used to update the lifted state |
| $S$ | $n\times n$ | Coupling whose diagonal determines stepsizes and lower triangle determines direct proximal dependencies |
| $\vartheta$ | Scalar | Relaxation, initially restricted to $(0,1)$ |

Normalization requires

$$
H^\top\mathbf1_n=\mathbf1_m,
\qquad K\mathbf1_n=\mathbf1_m.
$$

At consensus, each row of $K$ therefore evaluates its gradient at the common point, and each column of $H$ distributes that gradient with total weight one. The underlying sum of operators remains unchanged.

The theory does not require all entries of $H$ and $K$ to be nonnegative. Initially restrict them to simple nonnegative patterns to simplify generation, execution, and interpretation. That is a deliberate restriction of the family.

### 3.4 Causality

Every gradient must be evaluated using already available proximal outputs, then consumed only by later proxes. After ordering gradient columns, this can be expressed through nondecreasing cutoffs $F_i$:

$$
F_1=0,\qquad F_n=m,
\qquad H_{is}=0\text{ if }s>F_i,
\qquad K_{si}=0\text{ if }s\le F_i.
$$

A simpler generator can represent each gradient by a stage $q_s\in\{1,\ldots,n-1\}$: it uses slots at most $q_s$ and feeds slots strictly greater than $q_s$. Grouping gradients by stage recovers the cutoff representation.

For $n=3$, the existing four choices per shard are:

| Choice | Evaluation point | Consumers |
|---|---|---|
| `p2` | $x_1$ | Slot 2 only |
| `p3` | $x_1$ | Slot 3 only |
| `split` | $x_1$ | Slots 2 and 3, with weights $\eta$ and $1-\eta$ |
| `late` | $x_2$ | Slot 3 only |

The late choice uses another current-iteration point. It is not a stale-gradient method. Altering these choices changes the numerical algorithm; that is permitted because all candidates retain the same fixed-point problem.

### 3.5 Constructive certificate

Define

$$
J_n=I_n-\frac1n\mathbf1_n\mathbf1_n^\top,
\qquad
D=\frac1{\sqrt2}(H-K^\top)\operatorname{diag}(L_1,\ldots,L_m)^{1/2}.
$$

Require symmetric $S,\mathcal L$ satisfying

$$
S\mathbf1_n=0,\qquad S_{ii}>0,
\qquad\mathcal L\mathbf1_n=0,
\qquad\mathcal L\succeq\delta J_n
$$

for a positive connectivity margin $\delta$, together with

$$
\begin{bmatrix}
S-\mathcal L&D\\
D^\top&I_m
\end{bmatrix}\succeq0.
$$

The Schur complement gives $S-\mathcal L-DD^\top\succeq0$. Normalization implies $D^\top\mathbf1_n=0$. Thus the remaining PSD slack also annihilates $\mathbf1_n$ and can be factored as $PP^\top$ with $P\in\mathbb R^{n\times(n-1)}$ and $P^\top\mathbf1_n=0$. Likewise, $\mathcal L=MM^\top$ for a full-column-rank $M\in\mathbb R^{n\times(n-1)}$.

This recovers the original constructive parameterization

$$
S=MM^\top+PP^\top+DD^\top.
$$

The initial generator sets $P=0$ and constructs

$$
S=\mathcal L+DD^\top.
$$

The Schur block then has the explicit factorization

$$
\begin{bmatrix}D\\I_m\end{bmatrix}
\begin{bmatrix}D^\top&I_m\end{bmatrix},
$$

so no SDP solve is needed for this part. The choice $P=0$ is a restriction, not an optimality claim. Nonzero slack may change both stepsizes and dependency support.

### 3.6 Executable iteration

Define

$$
\gamma_i=\frac{2}{S_{ii}},\qquad
\Gamma=\operatorname{diag}(\gamma_1,\ldots,\gamma_n),\qquad
L_S=-\operatorname{slt}(S).
$$

Initialize $w^0$ with $\sum_iw_i^0=0$, for example $w^0=0$. One iteration is

$$
x_i^{k+1}=\operatorname{prox}_{\gamma_i h_i}
\left(\gamma_i\left[
w_i^k-\sum_{j<i}S_{ij}x_j^{k+1}
-\sum_{s=1}^{m}H_{is}\mathcal C_s\left(\sum_{j=1}^{n}K_{sj}x_j^{k+1}\right)
\right]\right),
$$

followed by

$$
w^{k+1}=w^k-\vartheta\mathcal Lx^{k+1}.
$$

Only forward evaluations with nonzero routing weights need to be present when a particular prox is evaluated. Causality ensures their inputs are already available. Each forward output is computed once and reused by its consumers.

The lifted state satisfies $w=Mz$ and stores $n$ vectors constrained to a zero-sum subspace, equivalent to $n-1$ independent abstract state vectors. This representation avoids executing a factorization of $\mathcal L$. It does not mean the full distributed implementation uses only $n$ vectors: outputs, gradients, staging lanes, and workspaces are additional storage.

At a fixed point, $\mathcal Lx=0$ implies consensus. Summing the resolvent optimality conditions and using the zero-sum and normalization identities recovers the desired inclusion. The ACGN sufficiency result then supplies convergence under the stated assumptions [1]. It does not supply a generally useful iteration count or a hardware-time guarantee.

### 3.7 Coupling families and a worked routing example

A basic connected coupling is

$$
\mathcal L=\alpha J_n,\qquad\alpha>0.
$$

For three slots, consider three early split gradients and one late gradient:

$$
H=\begin{bmatrix}
0&0&0&0\\
1/2&1/2&1/2&0\\
1/2&1/2&1/2&1
\end{bmatrix},\qquad
K=\begin{bmatrix}
1&0&0\\
1&0&0\\
1&0&0\\
0&1&0
\end{bmatrix}.
$$

Let $G=DD^\top$, $a=\alpha/3$, and $b=G_{23}$. A family that removes the direct slot-2-to-slot-3 coupling is

$$
\mathcal L=\begin{bmatrix}
2a&-a&-a\\
-a&a+b&-b\\
-a&-b&a+b
\end{bmatrix}.
$$

Its eigenvalues are $0$, $3a$, and $a+2b$. It is PSD-connected exactly when $a>0$ and $a+2b>0$. Since $\mathcal L_{23}=-G_{23}$, constructing $S=\mathcal L+G$ gives $S_{23}=0$.

For this routing, $b=(L_1+L_2+L_3)/8-L_4/2$, which can be negative. Positive off-diagonal entries of $\mathcal L$ are therefore possible and do not invalidate the construction; the PSD and nullspace conditions are the relevant checks.

Removing $S_{23}$ does not make slots 2 and 3 independent: the late gradient still uses $x_2$ and feeds slot 3. Name this family by the removed coefficient, rather than implying complete branch independence.

The existing count $3!\,4^4\,2=3072$ describes permutations, four routing choices per shard, and two coupling templates at a fixed split weight. Scalar grids enlarge this collection, and feasibility checks remove some entries. This is not the number of all ACGN algorithms and should not be extrapolated into an unsupported general counting formula.

### 3.8 Generator output and numerical checks

For every retained candidate, record:

- Term identities, slot permutation, shard identities, evaluation stages, and normalization.
- Full $H,K,S,\mathcal L,\Gamma,\vartheta$, not just a symbolic routing name.
- Lipschitz bounds and their provenance.
- Construction method, scalar choices, support pattern, and numerical validation results.
- Exact logical dependency graph and a stable identifier including numerical coefficients.

Checks include symmetry, both normalization identities, causality, zero row sums, positive diagonals, and eigenvalues on $\mathbf1_n^\perp$. Test the reduced matrix $U^\top\mathcal LU$, where $U$ has orthonormal columns spanning $\mathbf1_n^\perp$, to distinguish the intended zero eigenvalue from loss of connectivity.

Use scale-aware tolerances and preserve construction identities structurally. Floating-point checks support a theorem-backed construction but are not an interval-arithmetic proof. Never drop a small coefficient from the execution graph without changing and revalidating the numerical design. A symbolic cancellation can justify an exact zero; numerical proximity alone cannot.

Carry the selected scalar parameters into Stage 2. The current prototype's hard-coded scale-2 feasibility check can discard a design that was feasible at another screened scale. This must be corrected in a future implementation rather than copied into the new workflow.

## 4. Stage 2: detailed per-iteration execution model

### 4.1 Separate numerical algorithms from executions

For each fixed numerical algorithm $a$, construct a catalog of legal executions $e\in\mathcal E(a,\mathsf M)$ on machine $\mathsf M$. Execution choices include term placement, rank gangs, data layout, primitive selection, stream assignment, graph capture, and communication grouping.

These choices should preserve the numerical iteration apart from floating-point effects. If an implementation changes the operator, stepsizes, evaluation point, or effective accuracy, it is not merely another transport choice.

The Stage-2 optimization is

$$
e_a^*\in\operatorname*{arg\,min}_{e\in\mathcal E(a,\mathsf M)}
\widehat C(a,e;\mathsf M),
$$

subject to correctness, memory, and resource constraints. $\widehat C$ is a predicted steady-state iteration period, not a predicted time to solution. Return several executions when uncertainty makes their differences insignificant.

### 4.2 Compile the algebra into tasks and dependencies

The graph must include all work needed by the iteration:

| Task | Dependencies and interpretation |
|---|---|
| Evaluation-point assembly $y_s=\sum_jK_{sj}x_j$ | Needs every referenced current-iteration output and any transfers or conversions |
| Gradient $g_s=\mathcal C_s(y_s)$ | Includes the forward/adjoint implementation and its internal communication |
| Proximal argument assembly | Combines local state, lower-triangular $S$ contributions, and $H$-weighted gradients |
| Proximal evaluation | Consumes the assembled argument in its required layout |
| State update | Computes each owned component of $\mathcal Lx$ and updates $w$ |
| Movement and synchronization | Transfers, packing, unpacking, events, signals, completion, and buffer-reuse protection |

Dependencies arise from $K$, $H$, the strict lower triangle of $S$, and $\mathcal L$. Cross-iteration dependencies connect state updates to the next uses of those state components. Buffer reuse can add further dependencies even when the mathematical outputs are no longer needed locally.

An algebraic edge is a data requirement, not necessarily one network message. Multiple consumers may share a received vector. Contributions may be aggregated before transmission. A collective may replace many algebraic edges. Conversely, a logical node-to-node transfer may require local packing, a GPU-to-gateway transfer, a network transfer, and redistribution at its destination.

Construct communication tasks after selecting ownership and implementation. Charge an aggregate transfer once, while preserving its producer dependencies and the weights applied to its contents.

### 4.3 Dense matrices can have inexpensive implementations

For $\mathcal L=\alpha J_n$,

$$
(\mathcal Lx)_i=\alpha\left(x_i-\frac1n\sum_jx_j\right).
$$

This can be implemented by an appropriate reduction and distribution of a sum, plus local vector operations. It need not generate pairwise all-to-all exchanges. Which collective is appropriate depends on where the outputs and state components live.

Similarly, repeated columns or rows in $H$ can expose reusable partial sums. An all-shard reduction is legal only when the required weighted combinations can be reconstructed from the reduced quantities. Gradients evaluated at different points cannot silently be replaced by a gradient at a common point.

Search over these algebraic implementations before assigning costs. Counting nonzero coefficients is an inadequate communication model.

### 4.4 Operator implementation catalog

Profile each operator for the actual problem dimensions and implementation configuration. A catalog entry should include:

- Operator identity, input/output layout, ownership, and precision.
- GPU/rank participants and required workspace.
- Compute duration, host submission cost, and variability across representative inputs.
- Internal communication and synchronization.
- Whether timing includes argument assembly, layout conversion, and allocation.
- Graph-capture compatibility and any approximation or correctness restrictions.

For least squares, distinguish two sparse matrix-vector products from a matrix-free projector, a precomputed normal matrix, and a cooperative shard implementation. They have different arithmetic, storage, and communication costs. A normal-matrix evaluation on a tiny CPU model is not automatically representative of a large GPU projector.

For a cooperative gradient, include partial-gradient reduction and its layout in the operator cost or explicitly expose those internal tasks. Never count that reduction in both places.

Directional TV depends on chain length, number of independent chains, strides, and the algorithm's input-dependent control flow. A complete line stored locally permits a direct line solve. A line crossing rank boundaries requires a valid distributed line algorithm or redistribution; independent local solves with a fixed halo generally change the proximal operator.

Measure alternatives such as strided access, local transpose, distributed transpose, replication, and cooperative execution. Colocation avoids a network transfer but does not make local copies or conversions free.

### 4.5 Communication calibration

A first approximation for a primitive $r$ is

$$
\tau_r(B)\approx\alpha_r+\frac{B}{\beta_r},
$$

where $B$ is payload in bytes, $\alpha_r$ is an effective startup cost, and $\beta_r$ is effective bandwidth. Actual calibration should use measured tables or piecewise models because protocol thresholds and topology can create discontinuities.

Index the measurements by payload, participants, placement, direction, transport, memory type, and execution mode. Include at least:

- Point-to-point transfers, bidirectional exchanges, and fan-out/fan-in.
- Relevant reductions, allreduces, broadcasts, and redistribution patterns.
- Within-node and across-node placement.
- Host-staged MPI, GPU-aware MPI/UCX, PetscSF-NVSHMEM, PetscSF-NCCL, and direct grouped/graph/device implementations when available.
- Isolated transfers and representative overlap with compute and other transfers.

Record software versions, environment settings, GPU/NIC affinity, and evidence that the requested backend actually executed. A silent MPI fallback is not an NVSHMEM or NCCL measurement.

Distinguish end-to-end exposed exchange timings from raw network timings. If an exchange measurement already includes packing, launch, and synchronization, do not add those costs again. Conversely, raw bandwidth cannot substitute for the exposed cost of an SF exchange.

These transport paths are separate catalog entries. The existing device-resident NVSHMEM prototype and an SF-NVSHMEM exchange have different launch and synchronization structures, even though both use NVSHMEM.

### 4.6 Host and device timelines

Model submission and execution separately where they can overlap. A CPU call can enqueue GPU work and return before completion. A Begin/End pair can expose overlap, but only when useful independent work is actually scheduled between the calls and the stream dependencies permit it.

For NCCL, account for the selected group's actual participants, send/receive order, streams, and completion dependencies. A group can introduce a broader wait than an individual logical edge. Do not universally assume that an early-posted receive can be consumed late without blocking intervening same-stream work.

For NVSHMEM, include data readiness, arrival signaling, completion semantics, and acknowledgment or other buffer-reuse protection. A producer must not overwrite a slot while a consumer still needs it. Include any proxy progress requirements in measured execution behavior.

CUDA graph replay can reduce host submission overhead but retains the graph's dependencies and consumes device resources. Persistent kernels have additional residency and progress constraints; they require their own validated model.

### 4.7 Resource constraints and scheduling

Start with an explicit task duration $\tau_v$ and start time $t_v$ for every task. With communication represented as tasks, an ordinary dependency is

$$
t_v\ge t_u+\tau_u\qquad\text{for }u\to v.
$$

Do not add a second edge-transfer cost when the transfer is already an intervening task. With unlimited resources and a single acyclic iteration, longest-path evaluation gives the makespan. Real executions also require resource constraints.

For an exclusive resource $r$, a capacity model is

$$
\sum_{v:\,t_v\le t<t_v+\tau_v}q_{vr}\le Q_r\qquad\text{for all }t,
$$

where $q_{vr}$ is task demand and $Q_r$ is capacity. Concrete GPU-gang identities matter: two tasks each using two GPUs may overlap on disjoint pairs, but not if their required GPU sets intersect. Per-node GPU counts alone cannot express that distinction.

Represent stream order, host submission, GPU occupancy where relevant, copy engines, and network ingress/egress. Bandwidth sharing can be approximated conservatively through serialized resources initially, then refined by calibrated concurrency patterns. Two streams do not imply unrestricted overlap, because their kernels may compete for HBM bandwidth or execution units.

Begin with a small finite placement/primitive catalog and deterministic scheduling. Use enumeration and a discrete-event simulator before introducing a large solver. For a fixed duration/resource model, a MILP or constraint-programming formulation can improve scheduling. Its optimum is an optimum of that model, not proof of physically optimal GPU execution.

### 4.8 Repeated iterations and steady state

A single-iteration makespan is not always the repeated execution period. Let $t_v^{(k)}$ be the start of task $v$ in logical iteration $k$. A periodic model uses

$$
t_v^{(k)}=\phi_v+kC.
$$

For a dependency $u^{(k)}\to v^{(k+\ell)}$, feasibility requires

$$
\phi_v+\ell C\ge\phi_u+\tau_u.
$$

All periodically repeated resource conflicts and buffer lifetimes must also be respected. Two instantiated iterations are sufficient only with an argument excluding conflicts spanning more periods. Otherwise use modulo scheduling or sufficient unrolling with explicit checks; finite unrolling alone is not a proof for arbitrary repetitions.

For a simulator, estimate steady-state period from completion markers after warmup and verify stability as the run length increases. Preserve the executor's actual cross-iteration ordering. Neither impose a global iteration barrier absent from the implementation nor assume pipelining that the implementation cannot execute.

### 4.9 Outputs and validation of Stage 2

Return a predicted period, a feasible execution description, memory use, a task timeline, the critical path, and identified bottlenecks. Include variability estimates based on repeated measurements and held-out model errors; do not label them formal confidence bounds without a statistical construction.

Validate at increasing scope:

1. Individual operator and communication measurements.
2. Small patterns involving fan-out, aggregation, overlap, and shared resources.
3. Full execution graphs for several held-out algorithms and placements.
4. Actual numerical iterations using real operators.

Assess multiplicative timing error, pairwise ranking accuracy, and whether the model preserves near-best candidates. Calibration and evaluation data must be separated. Near-ties smaller than the observed model error should remain ties for shortlist purposes.

AXPY-based executors are useful for validating transport and scheduling behavior. They do not establish the cost or convergence of the final tomography solver. Never reuse small-image iteration counts at larger GPU dimensions as if they were measured large-problem convergence.

## 5. Stage 3: predictive selection, with explicit limits

### 5.1 Unknown iteration counts

For candidate $(a,e)$, define the actual time to the common target by

$$
T_\varepsilon(a,e)
=T_{\mathrm{setup}}(a,e)
+\sum_{k=0}^{N_\varepsilon(a,e)-1}t_k(a,e)
+T_{\mathrm{stopping}}(a,e).
$$

The simpler approximation $T_\varepsilon\approx T_{\mathrm{setup}}+N_\varepsilon C$ is appropriate only when iteration costs stabilize and stopping overhead is accounted for. In exact arithmetic, execution choices preserving the algorithm should preserve $N_\varepsilon$; finite precision can introduce differences that must be checked.

Operator durations and Lipschitz upper bounds do not determine $N_\varepsilon$. Short residual traces also do not guarantee the eventual ranking. All candidates may converge to the same objective value while having different threshold-crossing times. Hyperparameter-tuning methods can suggest allocation heuristics, but their guarantees do not automatically transfer to this objective [7,8].

### 5.2 Initial predictive target: shortlist quality

Develop the selector offline on a declared distribution of reconstruction instances. Candidate features can include:

- Stage-2 cost estimates and bottleneck summaries.
- Routing and coupling structure, stepsizes, connectivity margins, and safe norm bounds.
- Inexpensive problem features, such as dimension, chain lengths, measurement geometry, and clearly labeled spectral estimates.
- Optional short numerical traces, including transient behavior, at a prescribed and charged budget.
- Optional existing PEP descriptors, only if their cost and predictive value justify inclusion.

Start with transparent ranking rules or a simple predictor before complex surrogate models. Predicting which candidates belong in a good shortlist may be easier and more useful than predicting exact completion times.

Let the selector return $\mathcal S_q$ with at most $q$ candidates from a declared pool $\mathcal A_{\mathrm{pool}}$. Offline, measure

$$
\operatorname{shortlist\ ratio}
=\frac{\min_{a\in\mathcal S_q}T_\varepsilon(a)}
{\min_{a\in\mathcal A_{\mathrm{pool}}}T_\varepsilon(a)}.
$$

Here each algorithm is paired with its selected execution. The denominator is an empirical oracle over the evaluated pool, not the optimum over all ACGN designs. Obtaining that denominator may require expensive research runs; those runs are not deployment selection costs.

Report the probability or empirical frequency of retaining a candidate within a prescribed factor of the pool's best, across held-out instances. Report failures, timeouts, and the cost of generating the shortlist. Treat incomplete runs as censored or failed observations, not exact convergence times equal to the timeout.

### 5.3 Data splits and accuracy

Separate model fitting, validation/tuning, and final testing. Where generalization across size or geometry is claimed, split by those attributes rather than only by random right-hand sides of one geometry. Timing-model calibration also needs independent validation.

Use the same mathematical stopping criterion across algorithms. On small problems, reference-solution error is useful when the reference is trustworthy and uniqueness is addressed. For deployable runs, select and validate an appropriate optimality measure, such as a computable primal-dual gap. A small consensus residual does not automatically certify a comparable objective gap or reconstruction error.

The stopping implementation must be part of the benchmark and execution model: evaluation frequency, reductions, feasibility checks, and any extra forward/adjoint applications all cost time. Test several tolerances because early rankings can differ from tight-tolerance rankings.

### 5.4 Optional online selection

Do not make online racing a prerequisite of the project. Add it only after offline experiments show that limited observations help on unseen instances.

If tested, preserve each candidate's own state and fixed coefficients. Continue the chosen candidate from its checkpoint. Do not splice lifted states or alternate numerical updates across algorithms without a separate convergence analysis.

Measure

$$
T_{\mathrm{deployment}}
=T_{\mathrm{features}}+T_{\mathrm{selection}}+T_{\mathrm{continuation}}
$$

against immediately running a strong default. Include setup, trial execution, checkpoint memory, switching, and contention under the same GPU resource budget. Prior training costs should also be disclosed and any amortization justified.

If prediction fails, the outcome remains useful: the project can provide calibrated execution plans and an empirical map of which algorithm structures perform well. A claim of automatic per-instance acceleration would then be withheld.

## 6. PETSc integration and experimental controls

The numerical solver, execution planner, and transport implementation should remain distinguishable components. This enables two controlled comparisons:

- Hold the numerical algorithm fixed and compare execution choices.
- Hold the hardware budget and problem fixed and compare numerical algorithms by actual time to tolerance.

The current worktree contains term composition and SF transport work; the proposed integrated ACGN solver and proximal-term support remain implementation tasks described in `SC26-PETSC-PLAN.md`.

Account for current backend restrictions when building the execution catalog. SF-NVSHMEM eligibility is tied to a communicator identical or congruent to `PETSC_COMM_WORLD`, with additional data/memory rules. SF-NCCL has communicator/device eligibility checks of its own. Term-subcommunicator placement therefore needs explicit routing and cannot assume every subgroup exchange automatically uses the requested backend.

For initial numerical validation, use CPU executions and small independent references, then distributed CPU, then GPU transports. Check coefficient and operator agreement, preserved evaluation points, zero-sum state drift, and comparable stopping behavior. Do not require bitwise agreement across all reduction orders when an appropriate numerical tolerance is the meaningful criterion.

Robust NVSHMEM/NCCL support is an independent deliverable. A transport can be correct and useful even if it loses for a particular payload or schedule. Positive backend identification and end-to-end numerical checks are required before attributing a solver result to that transport.

## 7. Work packages and decision gates

| Work package | Deliverable | Decision gate |
|---|---|---|
| Benchmark specification | Fixed convex model, decomposition, dimensions, operator contracts, and accuracy measure | Every operator and assumption is well defined |
| ACGN generator | Finite candidate catalog with coefficients, validation records, and logical graphs | Independent small-problem numerical checks pass |
| Hardware catalog | Operator and communication measurements with layouts and provenance | Costs have consistent boundaries and no double counting |
| Execution planner | Feasible task graphs, placement choices, estimated periods, and memory requirements | Held-out full-graph timing and ranking errors are understood |
| Numerical executor | Actual gradients and proxes executed through the planned dependencies | Time-to-tolerance measurements are for the same problem and criterion |
| Offline predictor | Small shortlist using bounded-cost features | Near-best candidates are retained on untouched instances |
| Optional online selector | Trials and continuation with preserved state | Total deployment time improves over a strong default |

Baselines should include a strong fixed ACGN configuration, collective-compatible execution, fastest-predicted-iteration selection, a simple diverse or random shortlist, and a credible application solver. Include obvious fused formulations when relevant. Compare shortlist methods with the same evaluation budget.

The initial paper can claim convergence-backed candidate construction, a validated GPU cost model, measured algorithm/execution tradeoffs, and predictive shortlist quality. A worst-case optimal solver, reliable exact iteration-count prediction, or universally beneficial online selector is not implied by those results.

## 8. Existing artifacts and references

### Local starting points

- `/home/hsuh/sc26-poster/codesign_misdp_statement.tex`: factor-free certificate, three-slot example, and limitations of the full formulation.
- `/home/hsuh/prox-latency/papers/acgn_hpc_note_v2.tex`: ACGN formulation and benchmark considerations.
- `/home/hsuh/prox-latency/experiments/toy_design_enum.py`: restricted candidate generator and small numerical iteration.
- `/home/hsuh/prox-latency/experiments/harness/schedule.py`: existing event-driven cost-model prototype.
- `/home/hsuh/prox-latency/experiments/harness/machine_janus.py`: measured-machine catalog prototype.
- `/home/hsuh/prox-latency/experiments/rescreen_measured.py` and `rescreen_finals.py`: prior screening/scoring workflow, with the limitations discussed above.
- `SC26-PETSC-PLAN.md`: proposed solver and term integration.
- `src/vec/is/sf/impls/basic/nvshmem/sfnvshmem.cu` and `src/vec/is/sf/impls/basic/nccl/sfnccl.cu`: transport implementations to inspect when declaring execution eligibility.
- `nvshmem-tools/acgnrun.c`, `acgnrun-nccl.c`, and `acgnbench-nvdev2.cu`: execution prototypes; their emulated operator work is not a completed reconstruction solve.

### Papers

1. Åkerman, Chenchene, Giselsson, and Naldi. [Splitting the Forward–Backward Algorithm: A Full Characterization](https://arxiv.org/html/2504.10999v1). Source of the constructive admissibility and convergence conditions.
2. Bassett and Barkley. [Optimal Design of Resolvent Splitting Algorithms](https://arxiv.org/html/2407.16159v2). Related design and timing framework; it does not directly instantiate the GPU scheduling and predictive selection problem specified here.
3. Barkley and Bassett. [Coupled Adaptable Backward–Forward–Backward Resolvent Splitting Algorithm](https://arxiv.org/html/2505.13927v1). Related treatment of selection-operator structure; not a general prox-composition rule for arbitrary matrices.
4. Goethals et al. [Dynamic CT Reconstruction With Improved Temporal Resolution for Scanning of Fluid Flow in Porous Media](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031365). Application motivation for temporal regularization.
5. Lustig, Donoho, and Pauly. [Sparse MRI: The Application of Compressed Sensing for Rapid MR Imaging](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.21391). Application motivation for transform and finite-difference sparsity.
6. Xu and Noo. [A Sequential Solution for Anisotropic Total Variation Image Denoising With Interval Constraints](https://pmc.ncbi.nlm.nih.gov/articles/PMC5779866/). Uniform-bound fusion relevant to the earlier box example.
7. Jamieson and Talwalkar. [Non-stochastic Best Arm Identification and Hyperparameter Optimization](https://proceedings.mlr.press/v51/jamieson16.pdf). Resource-allocation ideas and limitations of inference from convergent traces.
8. Li et al. [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://www.jmlr.org/papers/v18/16-558.html). Optional early-stopping heuristic inspiration; not a ready-made theorem for fastest ACGN threshold crossing.
