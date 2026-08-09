Run each command as a separate Bash call with no shell metacharacters (no `$(...)`, pipes, `;`, `&&`/`||`, redirections, here-docs).

### 1. Identify the MR — resolve `<MR_IID>`
- Number given (e.g. `8786`) → use it.
- Diff file given → ask user for source branch, then `glab mr list --source-branch <branch>`.
- Nothing given → `git branch --show-current` (if empty, ask); then `glab mr list --source-branch <branch>`.

If `glab mr list` returns 0 MRs, stop and report. If >1, ask which IID.

### 2. Get MR metadata
1. `glab api "projects/:id/merge_requests/<MR_IID>"` — record `sha` as `<MR_HEAD_SHA>` and `source_branch`.
2. `glab api "projects/:id/merge_requests/<MR_IID>/changes"` — diff payload.

#### Public discussion fallback
When the request concerns existing review comments, retrieve the MR discussions as well. If `glab` is unavailable or the anonymous GitLab REST discussions endpoint returns `401`, use GitLab's public Rails endpoint:

`curl --fail --silent --show-error --header 'X-Requested-With: XMLHttpRequest' 'https://gitlab.com/<namespace>/<project>/-/merge_requests/<MR_IID>/discussions.json?notes_filter=1&per_page=100&persist_filter=false'`

- `notes_filter=1` returns comments without most system-note noise; use `notes_filter=0` when the activity history matters.
- Parse each discussion's `resolved` state and each note's `author.username`, `created_at`, `note`, and `position`.
- Do not interpret a `401` from `/api/v4/projects/.../discussions` as proof that authentication is required; try the public Rails endpoint before asking the user for credentials.

### 3. Drift check
`<source_branch>` comes from the GitLab API and is trusted (GitLab validates branch names; PETSc convention narrows further).
- `git show-ref --verify --quiet refs/heads/<source_branch>` — if non-zero, skip.
- Else `git rev-parse <source_branch>`; if it differs from `<MR_HEAD_SHA>`, warn that local and MR head diverge and recommend `/review-branch`.
