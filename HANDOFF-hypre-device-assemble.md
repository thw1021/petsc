# Handoff: hypre device IJ assembly, and two open device-matrix questions

Three items about hypre and PETSc device matrices, found while making a MOOSE Kokkos
p-multigrid preconditioner assemble its coarsest level directly as a `MATHYPRE` operator in
device memory. The first is fixed and submitted upstream and needs only shepherding. The
second and third are open and need a machine that can build PETSc with CUDA and a
device-enabled hypre, which is why they are written down rather than finished.

None of this is recorded in the MOOSE repository. Apart from the upstream pull request, this
file is the only description of it.

## 1. Fixed and submitted: device IJ assembly reads a null off-diagonal column map

**Symptom.** `MatSetValues` into a device-resident `MATHYPRE` matrix that was preallocated with
`MatSetPreallocationCOO` aborts on more than one process:
`hypre_Memcpy warning: copy N bytes from (nil)`, then `cudaErrorIllegalAddress`. N is the
rank's off-diagonal column count times `sizeof(HYPRE_BigInt)` in every case measured.

**Root cause**, from a backtrace planted in `hypre_Memcpy` rather than from reading:
`hypre_IJMatrixAssembleParCSRDevice` passes `hypre_ParCSRMatrixDeviceColMapOffd(par_matrix)`
into `hypre_CSRMatrixSplitDevice_core`, which reaches `hypre_CSRMatrixMergeColMapOffd` and
copies device-to-device from that pointer. A ParCSR built on the host carries `col_map_offd` on
the host only, which is the state PETSc's coordinate preallocation leaves it in, so the pointer
is null while `num_cols_offd` is nonzero.

**Fix.** One call to `hypre_ParCSRMatrixCopyColMapOffdToDevice` ahead of the split. The
`MatGetValues` path in the same hypre file already makes exactly that call before reading the
same field; the assemble path does not. The call is idempotent, being a no-op when the device
copy already exists.

**State.** Submitted as [hypre-space/hypre#1625](https://github.com/hypre-space/hypre/pull/1625)
from branch `fix-ij-device-assemble-colmap-offd`, fork remote `lindsayad/hypre`. Verified at
one, two and four ranks with the expected norms, and hypre's own `ij` driver under BoomerAMG is
unchanged on host and device at one and two ranks. The file is identical between `v3.2.0` and
`master`, so that verification transfers to the branch the PR targets. The project asks
contributors for an `AI-assisted` label which could not be set from outside the organization;
the disclosure is in the PR body and **the label still wants adding by hand**.

**What is left:** respond to review, and add that label.

## 2. Open, needs a GPU: entry reads of a device MATHYPRE matrix disagree

Reading a device-resident `MATHYPRE` matrix one entry at a time with `MatGetValues` returns
values that disagree with an independently computed reference by about 0.68, from polynomial
order five upward, where the same operator read through a host `MATSEQAIJ`/`MATMPIAIJ` copy
agrees with that reference to 4.4e-16. Each such read also costs a device allocation.

What is established, and what is not:

- The operator itself is right. Its action measured against a matrix-free operator through a
  host copy agrees to 2.7e-15, and the solve converges in the same iteration count to the same
  residual as the AIJ path.
- So the copy reads correctly and the direct read did not, at that sparsity, for a reason nobody
  has pinned down.
- **"Entry reads of a device hypre matrix are wrong" is not the mechanism.** A standalone
  reproducer that reads back every entry of the locally owned rows with `MatGetValues` after a
  coordinate fill finds no error, at one, two and four ranks, with host and with device values
  alike. Do not repeat the stronger claim.

The gap between those two is the whole of the open question: something about the real operator's
sparsity or fill history, not about device entry reads in general. Reproducing it needs an
operator of that shape, so the first job is to find the smallest one that shows the
disagreement.

## 3. Open, host-testable: MatConvert to the MATAIJ alias returns zeros silently

`MatConvert` to the `MATAIJ` alias from a device-resident hypre source resolves the alias to
`MATSEQAIJCUSPARSE` or `MATMPIAIJCUSPARSE` (`src/mat/impls/hypre/mhypre.c`, in
`MatConvert_HYPRE_AIJ`). That is deliberate. The defect is what happens next: multiplying that
device matrix by host vectors **produces zeros with no error**, which reads exactly like a wrong
assembly and cost several rounds of misdirected investigation here.

The fix worth proposing is the diagnostic, not the conversion: that product should raise rather
than return zeros. Naming `MATSEQAIJ` or `MATMPIAIJ` explicitly is the workaround, and is what
callers wanting a host matrix must do today.

This one needs a device build to reproduce but the change itself is in host-side error checking.

## The reproducer

Item 1 reproduces in about twenty lines with only `petscmat.h` and `-lpetsc`, no MOOSE and no
hypre headers: preallocate a tridiagonal pattern with `MatSetPreallocationCOO`, fill the
diagonal block with `MatSetValues`, assemble, multiply. This is the order
`src/mat/tutorials/ex18.c` itself uses, so it is sanctioned rather than novel. Give it options
to switch the preallocation, the memory binding, the matrix type, and whether the declared
pattern couples neighbouring ranks:

| arguments | np = 1 | np = 2 |
| --- | --- | --- |
| `-prealloc coo` | passes | **aborts** |
| `-prealloc coo -hostbound` | passes | passes |
| `-prealloc coo -mat_type aij` | passes | passes |
| `-prealloc coo -uncoupled` | passes | passes |
| `-prealloc hypre -write_offdiag` | passes | passes |

Four things that table pins down, and they are what make the diagnosis rather than guesses:

1. The coordinate interface is not involved beyond the preallocation. No `MatSetValuesCOO` runs
   at all.
2. Binding the matrix to the host makes it pass, so it is the device path.
3. `-uncoupled` drops the entries coupling neighbouring ranks from the *declared* pattern, so no
   rank has an off-diagonal block, and it passes. Together with the size of the null-source copy
   this identifies `hypre_ParCSRMatrixColMapOffd`. Note the block has to **exist** and does not
   have to be written: the first row never writes a coupling column and still aborts. So the
   shortest statement of the bug is that a coordinate preallocation naming any inter-rank
   coupling is enough to break a later `MatSetValues`.
4. The same writes over the same sparsity pass under `MatHYPRESetPreallocation` and abort under
   `MatSetPreallocationCOO`. That last row needs the off-diagonal writes because
   `MatHYPRESetPreallocation` materializes no off-diagonal block until off-diagonal values
   arrive, which the norm shows directly: 2 for the block-diagonal operator against sqrt(2) for
   the coupled one. A coordinate preallocation materializes that block from the pattern
   regardless.

Confirmed against PETSc `main` as well as a pinned 3.25.4, using a minimal build with CUDA and
hypre alone. That build was independent in the ways that might have mattered: hypre 3.2.0 rather
than 3.1.0, no `--enable-gpu-aware-mpi`, and no `PETSC_HAVE_HYPRE_MIXEDINT`, so none of those
options is the cause. Building a minimal PETSc to check took about twenty minutes and was worth
more than reading the source for whether a fix had landed, which is how this started.

## Dead ends, so they are not repeated

For item 1 these were all measured or read and are **not** involved: `MAT_SORTED_FULL`, the
`MatHYPRE_AttachCOOMat` aliasing of the delegate matrix's arrays into the ParCSR, and the
`hypre_AuxParCSRMatrix` that `HYPRE_IJMatrixAssemble` destroys. Three rounds of reading PETSc's
`mhypre.c` did not find the cause and one planted backtrace did; instrument earlier.

A separate PETSc defect found in the same investigation is already fixed and submitted, and is
**not** part of this: `MatSetPreallocationCOO_MPIAIJ` does not leave hash-table assembly mode,
so a later `MatSetValues` into an off-diagonal column runs against a reduced column index space
and fails with `Column too large`. That is
[PETSc merge request 9644](https://gitlab.com/petsc/petsc/-/merge_requests/9644). Keep the two
apart: conflating them makes the hypre report unreadable.

## Why MOOSE no longer depends on item 1

MOOSE hit this because a p-multigrid level's operator carries the identity on its constrained
rows, and that identity was written with `MatSetValues` after the coordinate assembly had filled
the rest. It now writes those rows from a device loop over the level's own rows, into the same
coordinate buffer the element loop fills. That is the better arrangement regardless: it removes a
host loop over the local degrees of freedom, its `MatSetValues` calls and a second matrix
assembly from the preconditioner's setup, and leaves the whole of the level's operator assembled
on the device. Only locally owned rows are written, since a ghosted row's entries are summed into
the owner and writing the identity twice would double it.

So nothing downstream is blocked on the hypre fix landing. It is worth landing for other users.
