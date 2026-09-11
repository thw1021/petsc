---
name: review-mr
description: Review code changes in a PETSc GitLab merge request and report findings to stdout, and to ai-review.html. Use when the user asks to "review this MR", "review MR <number>", or "look at MR !N", and wants the review printed (not posted as comments).
argument-hint: <MR_IID | empty for current branch>
---

Reviews the **remote MR state**, not local `HEAD`.

## Identify and fetch
Follow @identify.md (Sections 1–2) to fetch the merge request, check for drift, and repeat its warnings.

## Review
Follow @review-procedure.md (Sections 3–5) to read the diff, classify findings, verify each one,
and compose the report.

## Write report
Always write the titled report to `ai-review.html`, including when there are no findings. Add a
footnote with the agent, version, model, and effort level actually used for this review, the
current date and time, `MR_IID`, `CI_PIPELINE_ID`, and `CI_JOB_ID`. Omit unavailable values; use
`$CLAUDE_EFFORT` only when Claude performed the review; do not infer the agent or model from
installed executables or unrelated environment settings.
