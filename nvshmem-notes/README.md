# How four GPUs learned to talk: a field guide from a real measurement campaign

These notes retell, in tutorial form, a measurement campaign carried out on the ALCF
"Janus" cluster (nodes with 4x NVIDIA H100 GPUs) in August 2026. The goal of the campaign
was to answer one question for an SC26 poster:

> When an optimization algorithm is split across four GPUs, what is the fastest way for
> those GPUs to exchange data -- and does the answer change depending on *who* drives the
> communication: the CPU, a library, or the GPUs themselves?

Along the way we used, measured, broke, and fixed: GPU-aware MPI, NVSHMEM (both its
host and device APIs), NCCL, CUDA graphs, cooperative kernels, and two compute nodes'
worth of networking. Every number in these notes is a real measurement; every design
decision is explained from first principles.

**Audience.** You know C and roughly what MPI is (`mpirun`, ranks, `MPI_Send`). You have
never programmed a GPU or used an HPC cluster's GPUs. Everything else is defined when it
first appears.

**How to read.** In order. Each chapter builds vocabulary the next one uses.

| chapter | contents |
| --- | --- |
| [01 - The machine](01-the-machine.md) | what a GPU node actually is; host vs device; kernels, streams, and the ~2 us law that governs everything |
| [02 - The problem](02-the-problem.md) | the algorithm we are running; its communication graph; how to benchmark honestly |
| [03 - How GPUs talk](03-how-gpus-talk.md) | the three transports: GPU-aware MPI, NVSHMEM, NCCL -- who initiates, who waits |
| [04 - PetscSF's NVSHMEM path and the 2 us law](04-petscsf-nvshmem-and-the-2us-law.md) | why the library's NVSHMEM support loses to MPI; why `<<<1,1>>>` kernels exist; two failed optimizations and the law they proved |
| [05 - NCCL and fused groups](05-nccl-fused-groups.md) | batching a whole communication stage into one kernel; the experiment that killed our favorite hypothesis |
| [06 - CUDA graphs](06-cuda-graphs.md) | recording work once and replaying it; the hidden stream that refuses to be recorded; what "iterate until converged" means for a frozen recording |
| [07 - Device-side NVSHMEM](07-device-side-nvshmem.md) | making the GPUs communicate with no CPU at all; signals, epochs, double buffers, and four protocol revisions |
| [08 - Multi-node and GPUDirect](08-multinode-and-gpudirect.md) | why crossing between nodes is a different world; the kernel module whose absence blocked everything; how to diagnose a fabric; the epilogue where the module arrived and half our diagnosis turned out wrong |
| [09 - Results and lessons](09-results-and-lessons.md) | the complete measured matrix; what was deliverable vs showcase; the general lessons |

**Primary sources.** These notes are the narrative version of the working documents in
the parent directory: `NVSHMEM-PERF-NOTES.md` (all measurements), `NVSHMEM-BUILD-NOTES.md`
(build and environment), `NVSHMEM-NEXT-STEPS.md` (the plan and its predictions),
`TAOTERM-WISHLIST.md` (API consequences), and the benchmark sources in `../nvshmem-tools/`.
When a chapter quotes a number, the section reference points there.

**A note on honesty.** Several of our predictions were wrong, one optimization made
things worse before it made them better, and the flashiest benchmark is explicitly
labeled a showcase rather than a deliverable. That is deliberate: the point of these
notes is to show how measurement-driven work actually proceeds, dead ends included.
