---
name: review-mr-post
description: Review a PETSc GitLab merge request and post the findings back as inline DiffNote comments (with apply-able suggestion blocks where possible). Use when the user asks to "post review comments on MR <N>", "leave inline comments on the MR", or "review and post" — anything that should land on GitLab, not stdout.
argument-hint: <MR_IID | empty for current branch>
---

Adhere to @AGENTS.md while reviewing and drafting comments.

## Identify and fetch
Follow @../review-mr/identify.md (Sections 1–3) to fetch the merge request, check for drift, and repeat its warnings.

## Review
Follow @../review-mr/review-procedure.md (Sections 4–7) to classify and verify findings. Then continue below to filter and post.

## 8. Filter findings
Only post findings that have a **concrete, actionable fix** (a code change the author can apply). Do NOT post:
- Informational or observational notes ("just noting...", "no issue, but...")
- Findings that acknowledge correctness but flag theoretical fragility
- Style nits with no specific suggested change
- Comments on code that was not changed in the MR (surrounding context is for understanding, not reviewing)

Each posted comment opens a discussion thread the author must resolve — avoid noise.

## 9. Line number mapping
Each comment anchors to `line`, the line number in the **new** version of the file.
- For **new files**: the line number in the file itself.
- For **modified files**: parse the `@@` hunk headers in `DIFF_FILE` to map correctly.
- **Only comment on lines that are part of the MR diff.** Do not comment on unchanged code that happens to be near the diff.

## 10. Use GitLab suggestion blocks for concrete fixes
When a comment has a specific code fix, end the body with a suggestion block so the author can click "Apply suggestion":

````
```suggestion:-0+0
corrected line here
```
````

- `-0+0` means replace just the target line (the `line` in the finding). Use `-N+M` to expand the range to N lines before and M lines after the target line.
- **CRITICAL:** The suggestion body **replaces the entire selected range**. You MUST reproduce every line in the range, not just the changed ones. For example, `suggestion:-2+0` selects 3 lines (2 before + the target); the body must contain all 3 lines (with your edits applied). Omitting unchanged lines will **delete them**. When in doubt, prefer `suggestion:-0+0` targeting a single line.
- To insert a new line after the target, use `suggestion:-0+0` and include both the original target line and the new line.
- To delete a line, use an empty suggestion block.
- Only use suggestions for concrete fixes. Use plain comments for design/architectural feedback.

## 11. Post inline comments as DiffNotes
Write the findings to `mr-<MR_IID>-findings.json` as a list of objects, each with `file` (the new-side path in the MR diff — the `b/` path of its `diff --git` header, even where the file is renamed), `line` (Section 9) and `body` (Section 10):

```json
[
  {"file": "src/ts/interface/ts.c", "line": 412, "body": "Missing `PetscCall()`.\n\n```suggestion:-0+0\n  PetscCall(TSSetUp(ts));\n```"}
]
```

Then run `python3 lib/petsc/bin/maint/ai_review_post.py <MR_IID> mr-<MR_IID>-findings.json` as a single shell command, from the PETSc repository root (also write the findings file there, so CI collects it). It anchors every comment to the `diff_refs` in `mr-<MR_IID>-meta.json`, demotes suggestion blocks GitLab would render incorrectly, and confirms each response is a DiffNote. Add `--dry-run` to validate the findings file without posting.

## 12. Verify
- Report `POSTED_OK` and `POSTED_FAILED`, and quote every `FAILED` line.
- Verify that the ai-review.html report was written, as instructed in Section 7.
