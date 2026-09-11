# Docstring Conventions (`/*@ ... @*/`)

`petsclinter` enforces docstring formatting. See [petsc-lint](../skills/petsc-lint/SKILL.md) for commands and dependencies.

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
