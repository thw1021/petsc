# PETSc

PETSc is a C library for parallel numerical computation using MPI. The codebase is primarily C, with Python bindings in `src/binding/petsc4py/`.

This file must be self-contained. Do not rely on linked Markdown files being read automatically. The essential repo guidance is embedded below.

## Project Layout

- `src/<package>/` - source organized by package: `vec`, `mat`, `ksp`, `snes`, `ts`, `dm`, `tao`, `sys`
- `src/<package>/impls/` - concrete implementations of each package's abstract type
- `src/<package>/interface/` - public API for each package
- `src/<package>/tests/` and `src/<package>/tutorials/` - tests and examples
- `include/` - public headers such as `petsc*.h`
- `include/petsc/private/` - private headers such as `*impl.h`
- `src/binding/petsc4py/` - Python bindings and packaging logic
- `config/` - configure, build, and test harness generation
- `doc/` - user and developer documentation

## Core Working Rules

- Preserve PETSc style and naming conventions.
- Keep edits minimal and local to the requested change.
- Read nearby code and match existing patterns in the package you are modifying before introducing a new one.
- Avoid unnecessary code duplication. Prefer reusing or extending nearby logic when it keeps behavior clear and local.
- If `.codegraph/` exists, follow `.agents/skills/codegraph/SKILL.md` before navigating, modifying, or reviewing PETSc C, C++, or Python code.
- When higher-level APIs depend on the same inference or selection policy, centralize that policy in one private helper that returns any provenance or strategy its callers need; do not duplicate the decision tree across those APIs.
- Do not add speculative abstractions or broad refactors unless explicitly requested.
- When a procedure, script, makefile rule, configuration step, test tool, or documentation requires avoidable workarounds or obscures how it works, identify the concrete problem and suggest the smallest useful improvement, with its file path and expected benefit. Apply fixes needed for the requested task; keep broader suggestions separate from required fixes and formal MR findings. Do not expand the patch or post unrelated suggestions externally unless requested.
- Do not add shadow state or extra validation in petsc4py to compensate for invariants that the PETSc C API cannot express or validate. Validate only requirements needed to safely marshal Python-owned data, then call the PETSc API directly.
- mpi4py is not a dependency of petsc4py; do not make it a required build or runtime dependency. Tests may use mpi4py, but must guard its import with `try`/`except ImportError` and skip only the tests that require it when it is unavailable. The remaining test suite must still load and run.
- When a caller needs object metadata, such as the length of a returned array, use an existing PETSc accessor. If the API does not expose the required information, add or extend a focused accessor at the layer that owns it instead of reconstructing internal layouts or duplicating implementation logic in callers or petsc4py. Document the returned information, including the size, ownership, and lifetime of any returned data. If only the documentation is unclear, clarify the existing API instead of adding one.
- If you modify source, check whether test blocks, expected output files, or documentation need corresponding updates. For public interfaces, include headers and examples in that check.
- The user selects `PETSC_ARCH`. If it has not already been supplied for the task, ask for it before configuring or building PETSc or petsc4py, running tests that depend on them, importing petsc4py, or running a PETSc executable; never infer or change it unless asked to do so.
- Pass it to commands that perform those actions. Targets that manage a dedicated default architecture, such as `make docs`, are exempt when using that default.
- Architecture-independent checks, such as skill validation and pure Python helper tests that do not load PETSc or petsc4py, do not require `PETSC_ARCH`.
- A documentation audit or review does not authorize `make docs`; run it only when the user explicitly requests it or approves it.
- Never call `PetscFinalize()` inside an `if (...)` block, including early-return patterns like `if (flag) { ...; PetscFinalize(); return 0; }`. Arrange control flow so finalization happens exactly once on every normal exit path.

## Development Skills

Use the skill for the current task. If your tool does not discover skills automatically, read the
corresponding file directly:

- Configure or reconfigure PETSc: `.agents/skills/petsc-configure/SKILL.md`.
- Build PETSc or petsc4py: `.agents/skills/petsc-build/SKILL.md`.
- Select, run, debug, or update tests: `.agents/skills/petsc-test/SKILL.md`.
- Format source or run source checks: `.agents/skills/petsc-lint/SKILL.md`.
- Audit or build PETSc or petsc4py documentation: `.agents/skills/petsc-docs/SKILL.md`.
- Review a local branch: `.agents/skills/review-branch/SKILL.md`.

Read only the skills needed for the work. Read a relevant skill when it first applies, unless
its instructions are already available in the current context. Reuse those instructions throughout
the task. Reread only when the file has changed, the needed guidance is no longer available in
context, or a specific uncertainty requires checking it. For a targeted clarification, read only
the relevant section. Apply the same principle to `AGENTS.md` and supporting references.

Explain relevant verification results and limitations that affect correctness, confidence, or
next steps. Include exact file paths, commands, and architecture or configuration details when
needed to understand or reproduce a result.

## Human-Facing Writing

- Use clear, natural, grammatically correct English in all human-facing text, including responses, MR titles and descriptions, commit messages, GitLab comments, review reports, documentation, and user-facing messages.
- Treat the reader's time as limited. Include information that helps the intended reader understand the result, assess its consequences or evidence, make a decision, or take action. Omit routine process narration, repeated context, boilerplate assurances, and statements about unchanged or unperformed work unless they serve one of those purposes or were requested.
- Instructions to perform work do not automatically require reporting every step. Report verification results when they help the intended reader assess the change or decide what to do. Explain what the results establish and any material limitations, using concrete descriptions that make sense without knowledge of the agent's tools or working process.
- Write complete sentences in prose. Titles, labels, and concise list items may be fragments. Avoid unexplained shorthand, telegraphic notes, and mechanical phrasing.
- Lead with the result and why it matters. Scale the detail to the task's complexity and the reader's needs. For MR descriptions and review comments, provide enough context and evidence for a reviewer who has not read the conversation; brevity must not hide material risks or uncertainty.
- Do not assume readers know agent-specific terminology, formats, or features. Explain unfamiliar concepts briefly at first use and state why they matter to the task or change. Include the essential explanation in the text and add useful links to authoritative sources for further detail, so readers can understand the point without following the links.
- Check relevance, repetition, spelling, grammar, and readability before presenting or posting text. Remove sentences whose omission would not reduce the reader's understanding or ability to act. Preserve exact API names, commands, diagnostics, and other technical identifiers when quoting them.

## Documentation And Code comments

- Write technical documentation for peer mathematicians and software engineers. Follow ASD-STE100 when it does not conflict with established PETSc terminology or mathematical precision.
- Prefer self-explanatory code. Add comments or docstrings only to explain non-obvious behavior, correctness constraints, or durable design rationale, or when PETSc documentation conventions require them. Do not narrate operations, explain obvious code, summarize edits, record implementation history, or describe removed and rejected approaches.
- Preserve existing comments unless a change makes them inaccurate or obsolete.
- Add an entry to `doc/changes/dev.md` for new features or API changes; otherwise, add one only when requested. Preserve entries that predate the task. All other Markdown files in `doc/changes/` are immutable.
- In Markdown, describe designs conceptually. When a code block reproduces PETSc source beyond a standalone function prototype, such as a structure, macro definition, or function body, use a narrowly anchored `literalinclude` with `:start-at:` and `:end-at:`. Keep illustrative code, pseudocode, and standalone function prototypes in fenced code blocks. In standalone function prototypes, retain parameter names that surrounding text references.
- Do not present nonexistent, obsolete, or deprecated symbols as current API. Mention them only when historical, migration, or compatibility context requires it.

## PETSc Naming And API Conventions

- Public function names use capitalized components, for example `KSPSolve()` and `MatGetOrdering()`.
- Enum constants and macros are uppercase with underscores, for example `MAT_FINAL_ASSEMBLY`.
- Private PETSc functions generally end in `_Private` or include an implementation suffix such as `MatMult_SeqAIJ`.
- Implementation functions should start with the interface operation name and then the implementation name, for example `KSPSolve_GMRES()`.
- Options database keys are lowercase with underscores and usually mirror the setter name without `set`, for example `-ksp_gmres_restart`.
- Options-line syntax (applies in **all** documentation — `/*@ @*/` blocks, `/*MC M*/` blocks, `.md` chapters, and prose comments):
  - Enumerated values use `(choice1|choice2|choice3)`, **never** `<a, b>` or `{a, b}`.
  - Free-form arguments use plain words (e.g. `radius`, `size`, `name`), **never** `<radius>` and never backticks around the arg.
  - Inside the `Options Database Keys:` linter block (the bullet entries `+ -opt val - desc`), the option-name is **bare** — no backticks. Example: `. -petscda_letkf_localization_type (none|gaspari_cohn|gaussian|boxcar) - select the localization kernel`.
  - In inline prose elsewhere (Notes blocks, `.md` chapters, `/*MC M*/` body), wrap the option in backticks as code: `` `-petscda_type name` ``, `` `-log_view` ``. This matches the convention used throughout `doc/manual/` and PETSc's docstring prose.
- Function typedef names should end in `Fn`.
- `MPI_Comm_size()` → local `size`; `MPI_Comm_rank()` → `rank`. No prefixed variants (`comm_size`, `nprocs`). If `size` is taken, rename the other local.
- Reserve the `_p_` prefix for struct tags associated with PETSc objects, such as `_p_Mat`; other struct tags must not use it.

## PETSc Data Type Rules

- Use `PetscInt` for most indices and array lengths.
- Use `PetscCount` for sizes or counts that may exceed `PetscInt`.
- Use `size_t` for memory sizes in bytes, not logical array lengths.
- Do not silence narrowing warnings with blind casts. Use PETSc cast helpers such as `PetscIntCast()` when converting to narrower integer types.
- Prefer PETSc MPI wrappers that accept PETSc count types when large counts may be involved.

## C Coding Style

- Header prototypes should not include parameter names, but function typedef declarations should.
- The declaration block at the top of a routine or nested scope is one contiguous group: variables grouped by type (all `PetscInt`s adjacent, all `PetscReal`s adjacent, etc.), no mixed pointer arities on a single line, no blank lines or section comments splitting the block. Initialize in the declaration when practical. Exactly one blank line separates the block from the first statement, including `PetscFunctionBegin`/`PetscFunctionBeginUser` at routine scope.
- In PETSc tutorials and tests, `main()` and all functions returning `PetscErrorCode` must begin with `PetscFunctionBeginUser` after declarations.
- Functions that begin with `PetscFunctionBegin` must return with `PetscFunctionReturn(...)` or `PetscFunctionReturnVoid()`, not raw `return`.
- For `PetscErrorCode` functions, return `PetscFunctionReturn(PETSC_SUCCESS)` on success.
- Wrap PETSc calls with `PetscCall(...)`. For external library calls, use the appropriate PETSc wrapper such as `PetscCallExternal()` or package-specific variants.
- Omit braces around any `if`, `else if`, or `else` branch whose body is a single statement.
- Do not leave commented-out code or dead `#ifdef` blocks in source files.
- Use `/* ... */` for multiline comments and `// ...` for short single-line comments.
- Do not decorate multiline comments with leading `*` on each line.
- Always append `()` to function names when mentioning them in comments, for example `MatAssemblyEnd_MPIAIJ()`.
- Follow C90-style declarations at the start of their enclosing block. Prefer declaring variables used only within a genuinely new nested `{ ... }` scope at the beginning of that scope. The only other allowed exception is a loop index in a `for (...)` initializer. Do **not** sprinkle `const T x = ...;` lines between statements, including after an early-return guard.

## Error Handling And PETSc Idioms

- Most PETSc functions return `PetscErrorCode`.
- In petsc4py public Python-callable methods, validate user arguments with explicit exceptions, not `assert`, because optimized Python can remove assertions. Internal implementation code, including `.pxi` helpers and callback trampolines, may use `assert`.
- Use `PetscFunctionBegin`/`PetscFunctionBeginUser` and `PetscFunctionReturn(...)` consistently.
- Check object validity and arguments using the usual PETSc validation macros when working in code paths that already use them.
- Reuse existing PETSc utility routines and macros before adding custom helpers.
- Do not wrap `PetscCheck()` in an outer `if (...)` when the condition can be expressed directly in the check. Prefer a single guard such as `PetscCheck(!use_mms || sw->Ax == sw->Ay, ...)` over `if (use_mms) PetscCheck(sw->Ax == sw->Ay, ...)`.
- Do not call `MatAssemblyBegin()`/`MatAssemblyEnd()` after `MatDenseRestoreArray*()` or `MatDenseRestoreColumnVec*()`. The Get/Restore pair is the assembled write path for dense matrices — the matrix stays assembled across it. Adding "just to be safe" assembly is wrong, not defensive. Assembly is only needed after `MatSetValues()`-style entry, where deferred stashing actually requires a flush.

## Kokkos / Device Code

For an unreachable guard (a `default:` arm or "can't happen" branch) inside a `KOKKOS_INLINE_FUNCTION`, use `Kokkos::abort("message")`. `SETERRQ`/`SETERRABORT` are not device-callable.

When a persistent workspace view is processed in chunks, only build a `Kokkos::subview` for the active range when a consumer actually reads the view extent (e.g. `KokkosBatched::TeamVectorGMRES` infers batch size from `view.extent(0)`). If every kernel is bounded by an explicit count parameter (`RangePolicy(0, n_active)`, or a function arg like `n_batch`), pass the full-capacity view directly — the subview adds no safety and obscures intent.

## Docstring Conventions (`/*@ ... @*/`)

`petsclinter` enforces docstring formatting. See the `petsc-lint` skill for commands and dependencies.

- **Section order.** Sections in `/*@ ... @*/` always appear in this order:
  1. One-line synopsis (`FunctionName - one-line description`)
  2. Collectivity (`Collective`, `Logically Collective`, `Not Collective`, `Asynchronous`)
  3. `Input Parameter(s):`
  4. `Output Parameter(s):`
  5. `Options Database Key(s):`
  6. `Level:`  ← always before Notes
  7. `Notes:` / `Note:`
  8. `Example Usage:`
  9. `Fortran Notes:`
  10. `.seealso:`

Two recurring traps the linter catches:

- **Param-list alignment.** In `Input Parameters:` / `Output Parameters:` blocks, every entry's `-` must sit exactly one space past the longest valid argument name. With args `da, xyz, bd, H` (longest is `xyz`), the correct form is:
  ```
  + da  - the `PetscDA` context
  . xyz - array of coordinate vectors
  . bd  - array of periodic-domain extents
  - H   - the observation operator
  ```
  Continuation lines for a multi-line description must be indented to line up under the description (i.e., the column right after `- `), not under the argument name.

- **Stray paragraphs in `Notes:`.** A bare paragraph that starts with a capitalized word and no trailing colon can be misparsed as a section header (`-fdoc-section-header-maybe-header`). Keep follow-up sentences in the same paragraph as the existing Notes text (no blank line between them), or rephrase so the line cannot look like a heading.

When in doubt, pattern-match against existing well-formatted docstrings in the same file.

## Merge Request Expectations

- All changes are expected to arrive through GitLab merge requests.
- Keep diffs reviewable and focused.
- For source changes, select and run relevant checks and tests according to the guidance in the development skills.
- If relevant verification cannot run in the current environment, explain the resulting gap and its effect on confidence in the change. Do not list inapplicable checks.

## Anti-Patterns (MUST avoid when writing or reviewing)

### PetscFinalize inside conditional

WRONG — finalization inside an early-return `if`:
```c
if (test_spatial_order) {
  PetscCall(TestSpatialOrder(comm, &sw));
  PetscCall(PetscFinalize());
  return 0;
}
```
RIGHT — finalization on the single exit path:
```c
if (test_spatial_order) PetscCall(TestSpatialOrder(comm, &sw));
else PetscCall(RunForwardModel(comm, &sw));
PetscCall(PetscFinalize());
return 0;
```

### Braces on single-statement if/else

WRONG:
```c
if (radius <= 0.0) {
  return 0.0;
}
```
RIGHT:
```c
if (radius <= 0.0) return 0.0;
```

This also applies to `else` blocks paired with multi-statement `if` — check each branch independently:
WRONG:
```c
  if (type == TYPE_A) {
    stmt1;
    stmt2;
  } else {
    SETERRQ(comm, PETSC_ERR_SUP, "unsupported");
  }
```
RIGHT:
```c
  if (type == TYPE_A) {
    stmt1;
    stmt2;
  } else SETERRQ(comm, PETSC_ERR_SUP, "unsupported");
```

## Key References

External links are for human convenience only; do not assume linked Markdown files will be ingested automatically.

- Development docs: https://petsc.org/main/overview/
- Release docs: https://petsc.org/release/overview/
- GitLab project: https://gitlab.com/petsc/petsc
