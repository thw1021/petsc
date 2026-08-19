#!/usr/bin/env python3
"""
Fetch and prepare the diff reviewed by the review skills in .claude/skills/.

  ai_review_fetch.py mr [IID]      remote merge request state; needs glab
  ai_review_fetch.py branch [REF]  local REF against origin/main or origin/release

Both modes write the <prefix>-diff.txt that is reviewed, in which the bodies of
.out reference outputs are omitted, then print a KEY=value summary on stdout.
Lines starting with WARNING: report conditions that do not stop the review.
The mr mode also writes mr-IID-meta.json, holding the shas that inline review
comments are anchored to and the pre-rename path of every file it renames.

Exit status: 0 success, 1 failure, 3 the caller must pick a merge request.
argparse exits 2 on a usage error, so 2 never means the review may continue.
"""
import os
import re
import sys
import json
import argparse
import subprocess
import urllib.parse

EXIT_OK        = 0
EXIT_FAIL      = 1
EXIT_AMBIGUOUS = 3

DEFAULT_TIMEOUT = 120

OUT_MARKER = ' [.out reference; body omitted]'

IID_RE  = re.compile(r'^[0-9]+$')
REF_RE  = re.compile(r'^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$')
OUT_RE  = re.compile(rb'\.out"?$')
KEEP_RE = re.compile(rb'^(new file mode|deleted file mode|rename from|rename to|similarity index) ')

def die(msg, code=EXIT_FAIL):
  sys.stderr.write('%s: error: %s\n' % (os.path.basename(sys.argv[0]), msg))
  raise SystemExit(code)

def warn(msg):
  print('WARNING: ' + msg)

def emit(key, value):
  print('%s=%s' % (key, value))

def flatten(raw):
  """Decode captured output to a single line, for embedding in a message."""
  return raw.decode('utf-8', 'replace').strip().replace('\n', ' ')

def run(cmd, timeout, check=True):
  """Run cmd, always as an argument list so no shell quoting or expansion applies."""
  try:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
  except FileNotFoundError:
    die('%s is not in PATH' % cmd[0])
  except subprocess.TimeoutExpired:
    die('%s timed out after %d seconds' % (' '.join(cmd), timeout))
  if check and proc.returncode:
    die('%s exited %d\n%s' % (' '.join(cmd), proc.returncode, proc.stderr.decode('utf-8', 'replace').strip()))
  return proc.returncode, proc.stdout, proc.stderr

def glab_json(path, timeout):
  _, out, _ = run(['glab', 'api', path], timeout)
  try:
    return json.loads(out)
  except ValueError:
    die('%s did not return JSON:\n%s' % (path, out[:400].decode('utf-8', 'replace')))

def strip_out_bodies(raw):
  """Replace the body of every .out file in raw with a marker on its header line."""
  kept     = []
  skipping = False
  for line in raw.split(b'\n'):
    if line.startswith(b'diff --git '):
      skipping = bool(OUT_RE.search(line))
      if skipping:
        kept.append(line + OUT_MARKER.encode())
        continue
    if skipping:
      if KEEP_RE.match(line): kept.append(line)
      continue
    kept.append(line)
  return b'\n'.join(kept)

def scan(diff):
  """Return the 'diff --git' header lines and those whose body is binary."""
  headers  = []
  binaries = []
  current  = None
  for raw_line in diff.split(b'\n'):
    line = raw_line.decode('utf-8', 'surrogateescape')
    if line.startswith('diff --git '):
      current = line
      headers.append(line)
    elif current and (line == 'GIT binary patch' or (line.startswith('Binary files ') and line.endswith(' differ'))):
      if current not in binaries: binaries.append(current)
  return headers, binaries

def build(raw, diff_path):
  """Write the reviewed diff and confirm the filter kept every file of the fetched one."""
  filtered = strip_out_bodies(raw)
  headers, binaries = scan(raw)
  kept, _           = scan(filtered)
  if not headers: die('the fetched diff has no "diff --git" header; nothing to review')
  if len(kept) != len(headers): die('the fetched diff covers %d files but %s covers %d' % (len(headers), diff_path, len(kept)))
  with open(diff_path, 'wb') as fd: fd.write(filtered)
  emit('DIFF_FILE', diff_path)
  emit('FILES', len(headers))
  emit('LINES', filtered.count(b'\n'))
  for header in binaries: emit('BINARY_FILE', header[len('diff --git '):])
  return headers

def resolve_iid(args):
  if args.iid is not None:
    if not IID_RE.match(args.iid): die('invalid merge request IID %r' % args.iid)
    return args.iid
  _, out, _ = run(['git', 'branch', '--show-current'], args.timeout)
  branch = out.decode('utf-8', 'replace').strip()
  if not branch: die('HEAD is detached; pass the merge request IID explicitly', EXIT_AMBIGUOUS)
  query = 'projects/:id/merge_requests?state=opened&source_branch=' + urllib.parse.quote(branch, safe='')
  mrs   = glab_json(query, args.timeout)
  if not mrs: die('no open merge request has source branch %s' % branch)
  if len(mrs) > 1:
    die('source branch %s has %d open merge requests (%s); pass one explicitly'
        % (branch, len(mrs), ' '.join('!%s' % mr['iid'] for mr in mrs)), EXIT_AMBIGUOUS)
  return str(mrs[0]['iid'])

def cmd_mr(args):
  iid       = resolve_iid(args)
  meta_path = 'mr-%s-meta.json' % iid
  changes   = glab_json('projects/:id/merge_requests/%s/changes?access_raw_diffs=true' % iid, args.timeout)
  refs      = changes.get('diff_refs') or {}
  missing   = [key for key in ('iid', 'sha', 'source_branch') if not changes.get(key)]
  missing  += ['diff_refs.' + key for key in ('base_sha', 'head_sha', 'start_sha') if not refs.get(key)]
  if missing: die('merge request !%s is missing %s' % (iid, ', '.join(missing)))

  listed = changes.get('changes') or []
  meta   = {key: changes.get(key) for key in ('iid', 'sha', 'source_branch', 'target_branch', 'title', 'web_url', 'diff_refs')}
  # A comment on a renamed file must be anchored to the path on both sides of the rename.
  meta['renames'] = {c['new_path']: c['old_path'] for c in listed if c.get('renamed_file') and c.get('old_path') and c.get('new_path')}
  with open(meta_path, 'w') as fd: json.dump(meta, fd, indent=2, sort_keys=True)

  code, raw, err = run(['glab', 'api', 'projects/:id/merge_requests/%s/raw_diffs' % iid], args.timeout, check=False)
  if code or not raw.strip():
    if code: warn('the raw_diffs endpoint exited %d (%s); falling back to glab mr diff' % (code, flatten(err) or 'no error output'))
    else:    warn('the raw_diffs endpoint returned nothing; falling back to glab mr diff')
    _, raw, _ = run(['glab', 'mr', 'diff', iid, '--raw'], args.timeout)
  if not raw.strip(): die('the diff of !%s is empty' % iid)

  emit('MR_IID', iid)
  emit('SOURCE_BRANCH', changes['source_branch'])
  emit('MR_HEAD_SHA', changes['sha'])
  emit('BASE_SHA', refs['base_sha'])
  emit('HEAD_SHA', refs['head_sha'])
  emit('START_SHA', refs['start_sha'])
  emit('META_FILE', meta_path)
  headers = build(raw, 'mr-%s-diff.txt' % iid)

  # The file list from the changes endpoint is fetched separately from the diff itself,
  # so it detects a diff that GitLab truncated.
  if changes.get('overflow'): warn('GitLab reports overflow=true, so the file list checked against the diff is itself incomplete')
  counted = str(changes.get('changes_count') or '')
  if counted.endswith('+'): warn('GitLab reports changes_count=%s, so the file list checked against the diff is capped' % counted)
  if len(headers) < len(listed):
    die('the fetched diff covers %d of the %d files changed by !%s; the diff is truncated' % (len(headers), len(listed), iid))
  for change in listed:
    old, new = change.get('old_path'), change.get('new_path')
    if 'diff --git a/%s b/%s' % (old, new) in headers: continue
    if 'diff --git "a/%s" "b/%s"' % (old, new) in headers: continue
    warn('changed file not matched in the diff, confirm it is covered: %s' % (new or old))

  # Always the fully qualified ref: git rev-parse resolves a bare name to a same-named tag first.
  head = 'refs/heads/' + changes['source_branch']
  code, _, _ = run(['git', 'show-ref', '--verify', '--quiet', head], args.timeout, check=False)
  if code:
    emit('DRIFT', 'unknown')
  else:
    _, out, _ = run(['git', 'rev-parse', head], args.timeout)
    emit('DRIFT', 'no' if out.decode().strip() == changes['sha'] else 'yes')

def cmd_branch(args):
  src = args.ref
  if not REF_RE.match(src): die('invalid ref %r' % src)
  for ref in ('origin/main', 'origin/release'):
    code, _, _ = run(['git', 'rev-parse', '--verify', '-q', ref], args.timeout, check=False)
    if code:
      run(['git', 'fetch', '-q', '--no-tags', 'origin', '+release:refs/remotes/origin/release', '+main:refs/remotes/origin/main'], args.timeout)
      break
  for ref in ('origin/main', 'origin/release'):
    code, _, _ = run(['git', 'rev-parse', '--verify', '-q', ref], args.timeout, check=False)
    if code: die('%s does not resolve after fetching from origin' % ref)

  _, out, _  = run(['git', 'merge-base', 'origin/main', src], args.timeout)
  forked     = out.decode().strip()
  code, _, _ = run(['git', 'merge-base', '--is-ancestor', forked, 'origin/release'], args.timeout, check=False)
  if code > 1: die('git merge-base --is-ancestor %s origin/release exited %d' % (forked, code))
  dest = 'origin/release' if code == 0 else 'origin/main'
  # origin/main can lag origin/release, so the commit that picked dest is not always the base of the diff.
  _, out, _ = run(['git', 'merge-base', dest, src], args.timeout)
  base      = out.decode().strip()

  _, raw, _ = run(['git', 'diff', '%s...%s' % (dest, src)], args.timeout)
  if not raw.strip(): die('%s...%s is empty; nothing to review' % (dest, src))
  _, out, _ = run(['git', 'diff', '--shortstat', '%s...%s' % (dest, src)], args.timeout)
  shortstat = out.decode('utf-8', 'replace').strip()
  _, out, _ = run(['git', 'rev-parse', src], args.timeout)

  emit('SRC', src)
  emit('SRC_SHA', out.decode().strip())
  emit('DEST', dest)
  emit('MERGE_BASE', base)
  emit('SHORTSTAT', shortstat)
  build(raw, 'branch-review.txt')

def main():
  # SUPPRESS keeps the subparser from overwriting a --timeout given before the mode.
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument('--timeout', type=int, default=argparse.SUPPRESS, metavar='SECONDS',
                      help='limit on each git or glab invocation (default: %d)' % DEFAULT_TIMEOUT)
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common])
  modes  = parser.add_subparsers(dest='mode', required=True)
  mode   = modes.add_parser('mr', parents=[common], help='fetch a merge request diff from GitLab')
  mode.add_argument('iid', nargs='?', help='merge request IID (default: the open merge request for the current branch)')
  mode.set_defaults(func=cmd_mr)
  mode = modes.add_parser('branch', parents=[common], help='diff a local ref against origin/main or origin/release')
  mode.add_argument('ref', nargs='?', default='HEAD', help='ref to review (default: HEAD)')
  mode.set_defaults(func=cmd_branch)
  args = parser.parse_args()
  if not hasattr(args, 'timeout'): args.timeout = DEFAULT_TIMEOUT
  args.func(args)
  return EXIT_OK

if __name__ == '__main__':
  sys.exit(main())
