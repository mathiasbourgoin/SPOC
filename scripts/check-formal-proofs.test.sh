#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Covering test for scripts/check-formal-proofs.sh (backlog-229).
#
# FORMAL_ALLOW_DIRTY=1 was documented as bypassing the pre-flight dirty-tree
# refusal. It did that, but the per-project `restore_tree` EXIT trap still ran
# unconditionally afterwards, and `git checkout -- .` inside it cannot tell a
# developer's own uncommitted edits from build output — so a tree the flag had
# just been told to tolerate got destroyed at exit anyway. It already happened
# once to a real agent's edits during this campaign.
#
# This test never runs the real Rocq toolchain (that would need
# rocq/rocq-prover and take ~50s, and is out of scope for a fast covering
# test). Instead it substitutes a stub `rocq` on PATH that satisfies
# `--version`, then fails as soon as `rocq makefile` is invoked -- late enough
# to have installed the `restore_tree` trap being tested, early enough to need
# nothing else from the toolchain. The stub also appends a line to a tracked
# file to stand in for the real script's own documented drift (extraction
# rewriting ~68 .ml/.mli files), so the "did the trap wrongly restore/fail to
# restore" question can be asked without a real build.
#
# Three cases, both polarities of the flag:
#
#   * flag=1,   pre-existing dirty tree  -> nothing is discarded (backlog-229)
#   * flag unset, clean tree             -> build-induced drift IS restored,
#                                            i.e. the default behaviour this
#                                            trap exists for is unchanged
#   * flag unset, pre-existing dirty tree -> pre-flight still refuses outright
#                                            and never reaches the trap at all
#
# D-55 adds the extracted-model drift cases (E..H below): the gate now
# regenerates EVERY committed extracted model and byte-compares it (after
# canonicalisation), so a stale committed model, an extraction driver missing
# from _CoqProject, a committed model nothing regenerates, and an unnamed
# driver that produces no committed model must each fail with a message that
# says which. The assertions are on the message text, not only the exit code:
# every one of these paths exits 1, and a gate that fails for the wrong reason
# (a missing tool, a syntax error) would satisfy an exit-code check
# vacuously.
#
# Every fixture is a throwaway `git init` under mktemp, never a real formal/
# project and never a real worktree. Per the project's own recorded hazard
# (`git -C ""` silently operates on the caller's cwd instead of failing), every
# fixture directory is asserted non-empty before it is ever passed to a git
# command, and all git invocations below use `cd "$dir" && git ...` in a
# subshell rather than `git -C`, so an empty variable cannot silently retarget
# a command at this script's own working tree.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE_SRC="$SCRIPT_DIR/check-formal-proofs.sh"
[ -x "$GATE_SRC" ] || { echo "FAIL: $GATE_SRC not found or not executable"; exit 2; }

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/check-formal-proofs-test.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT

pass=0
fail=0

# require_nonempty <name> <value> -- refuse to let an empty fixture path near
# a git command. The incident this guards against (`git -C ""` operating on
# the caller's own cwd instead of failing) is specifically a TMPROOT going
# empty: `mktemp -d` can fail with its output still captured by a bare `$(...)`
# under this script's `set -uo pipefail` (no `-e`), so TMPROOT="" is reachable
# in practice, not just in principle. Every other call site here checks a
# value built by concatenating TMPROOT with a fixed literal (a case name, or a
# path returned by make_fixture) and so cannot be empty once TMPROOT itself has
# been checked -- those calls are defense-in-depth against a future edit
# breaking that construction, not each guarding against their own reachable
# incident.
require_nonempty() {
  if [ -z "$2" ]; then
    echo "FATAL: $1 is empty; refusing to run git against an unknown directory." >&2
    exit 2
  fi
}

require_nonempty TMPROOT "$TMPROOT"

# stub rocq: enough to get check-formal-proofs.sh past its toolchain probe and
# into one project's subshell, where it appends a drift line to a BUILD-OWNED
# tracked file -- extraction/model.ml, matching the real script's own
# ':(glob)**/extraction/*.ml' restore pattern, standing in for extraction
# rewriting real .ml/.mli files -- and then fails, which is exactly where the
# trap under test fires, without compiling anything. driftfile.txt (below) is
# a SEPARATE, non-build-owned tracked file standing in for a developer's own
# source edit; the two must be distinguishable for this test to say anything
# about the discriminating restore, so they are never the same file.
FAKEBIN="$TMPROOT/fakebin"
mkdir -p "$FAKEBIN"
cat > "$FAKEBIN/rocq" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  --version)
    echo "stub rocq 0.0.0 (test double, not the real toolchain)"
    ;;
  makefile)
    if [ -n "${STUB_MAKE_BODY:-}" ]; then
      # Free-form recipe for the "build": lets a case model a build that does
      # NOT recreate a model, or one that produces an extra file.
      printf 'all:\n\t%s\n' "$STUB_MAKE_BODY" > CoqMakefile
      exit 0
    fi
    if [ -n "${STUB_REGENERATED_MODEL:-}" ]; then
      # D-55 mode: emit a CoqMakefile whose build "re-extracts" the model with
      # the content the test chose, then let the gate carry on to its
      # extraction comparison. (The stub's `check` below fails afterwards, so
      # the gate always ends non-zero; the assertions are on what it printed.)
      printf 'all:\n\tprintf "%%s\\n" "$$STUB_REGENERATED_MODEL" > extraction/model.ml\n' > CoqMakefile
      exit 0
    fi
    # Stand-in for the real script's documented drift: a rebuild rewrites a
    # BUILD-OWNED tracked file (here, the extraction output). Then fail,
    # before anything is actually compiled -- all this test needs is for the
    # subshell's EXIT trap to fire.
    echo "stub-build-drift" >> extraction/model.ml
    exit 1
    ;;
  *)
    exit 1
    ;;
esac
STUB
chmod +x "$FAKEBIN/rocq"

# make_fixture <name> -- a minimal repo with one formal/ project, laid out
# exactly as scripts/check-formal-proofs.sh expects (script one directory
# below repo root, project directory holding a _CoqProject with a `-R theories
# <Name>` line). Returns the repo root on stdout.
make_fixture() {
  local dir="$TMPROOT/$1"
  require_nonempty "fixture dir ($1)" "$dir"
  mkdir -p "$dir/scripts" "$dir/formal/testproj/theories" "$dir/formal/testproj/extraction"
  cp "$GATE_SRC" "$dir/scripts/"
  cp "$SCRIPT_DIR/canonicalize-extraction.py" "$dir/scripts/"
  chmod +x "$dir/scripts/check-formal-proofs.sh"
  printf -- '-R theories Test\nextraction/Extract.v\n' > "$dir/formal/testproj/_CoqProject"
  printf 'Extraction "model.ml" nothing.\n' > "$dir/formal/testproj/extraction/Extract.v"
  printf 'committed\n' > "$dir/formal/testproj/driftfile.txt"
  printf 'committed\n' > "$dir/formal/testproj/extraction/model.ml"
  # A TRACKED .lia.cache, standing in for the real repo's
  # formal/convergence-safety/.lia.cache (backlog-229's second recurrence: the
  # trap's own `rm -f .lia.cache .nia.cache` is unconditional and ran before
  # this fix folded the two cache names into BUILD_OWNED_GLOBS, so a tracked
  # cache was deleted every run and never restored, regardless of whether the
  # stub build below touches it at all).
  printf 'committed-lia-cache\n' > "$dir/formal/testproj/.lia.cache"
  ( cd "$dir" && git init -q . && git config user.email t@t.t && git config user.name t \
    && git add -A && git commit -q -m init ) >/dev/null 2>&1
  echo "$dir"
}

# run_gate <dir> <extra-env...> -- invoke the real gate with the stub rocq
# ahead of any real one on PATH.
run_gate() {
  local dir="$1"; shift
  require_nonempty "run_gate dir" "$dir"
  ( cd "$dir" && env "$@" PATH="$FAKEBIN:$PATH" ./scripts/check-formal-proofs.sh ) >/tmp/check-formal-proofs-test-out.$$ 2>&1
  local rc=$?
  cat /tmp/check-formal-proofs-test-out.$$
  rm -f /tmp/check-formal-proofs-test-out.$$
  return $rc
}

echo "check-formal-proofs.sh covering test (backlog-229)"

# --- Case A: FORMAL_ALLOW_DIRTY=1, tree already had an uncommitted edit -----
# The exact regression: a developer's own uncommitted work (outside the
# build-owned set) must survive. Separately -- this is the #229 fix itself,
# not just the flag -- the build-owned artefact the stub rewrote must NOT be
# left dirty, or a following unflagged run is refused for drift this run
# caused (the absorbing state the discriminating restore exists to remove).
d="$(make_fixture case-a)"
require_nonempty "case-a dir" "$d"
printf 'committed\nunstaged-user-edit\n' > "$d/formal/testproj/driftfile.txt"
out="$(run_gate "$d" FORMAL_ALLOW_DIRTY=1)"
content="$(cat "$d/formal/testproj/driftfile.txt")"
owned_after="$(cat "$d/formal/testproj/extraction/model.ml")"
if printf '%s' "$content" | grep -qF 'unstaged-user-edit'; then
  echo "  PASS: case-a: FORMAL_ALLOW_DIRTY=1 preserves the developer's uncommitted edit" \
       "(outside the build-owned set)"
  pass=$((pass + 1))
else
  echo "  FAIL: case-a: uncommitted edit was discarded despite FORMAL_ALLOW_DIRTY=1"
  echo "    driftfile.txt now contains:"
  printf '%s\n' "$content" | sed 's/^/      /'
  echo "    gate output:"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi
if [ "$owned_after" = "committed" ]; then
  echo "  PASS: case-a: the build-owned artefact the stub rewrote was restored" \
       "even under FORMAL_ALLOW_DIRTY=1 -- the next unflagged run is not bricked"
  pass=$((pass + 1))
else
  echo "  FAIL: case-a: build-owned extraction/model.ml was left dirty" \
       "(got: $(printf '%s' "$owned_after" | tr '\n' '|')) -- this is exactly" \
       "the absorbing state #229's fix was supposed to remove"
  fail=$((fail + 1))
fi
if printf '%s' "$out" | grep -qF 'build-owned tracked file' \
   && printf '%s' "$out" | grep -qF 'only the build-owned paths above were touched'; then
  echo "  PASS: case-a: gate names the mechanism -- restored the build-owned set," \
       "left your other edits alone -- not merely the outcome"
  pass=$((pass + 1))
else
  echo "  FAIL: case-a: expected output to name both the build-owned restore and" \
       "the flag's narrower scope; got:"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

# --- Case B: FORMAL_ALLOW_DIRTY unset, tree starts CLEAN --------------------
# Pins the other polarity: the trap's default job -- discarding build-induced
# drift on a tree that started clean -- must be untouched by this fix.
d="$(make_fixture case-b)"
require_nonempty "case-b dir" "$d"
before="$(cat "$d/formal/testproj/extraction/model.ml")"
out="$(run_gate "$d")"
after="$(cat "$d/formal/testproj/extraction/model.ml")"
if [ "$after" = "$before" ]; then
  echo "  PASS: case-b: default behaviour (flag unset) still restores build-induced drift"
  pass=$((pass + 1))
else
  echo "  FAIL: case-b: drift was left in place with FORMAL_ALLOW_DIRTY unset --" \
       "the default (documented, load-bearing) restore regressed"
  echo "    expected: $(printf '%s' "$before" | tr '\n' '|')"
  echo "    got:      $(printf '%s' "$after" | tr '\n' '|')"
  fail=$((fail + 1))
fi
# M4a: content-equality alone cannot distinguish "the trap restored the
# drift" from "the trap never ran at all" -- both leave before == after, and a
# mutation that suppresses the stub's drift line satisfies this assertion
# vacuously (case-0, no drift was ever produced). Pin the trap's own restore
# NOTE, printed only when it actually found and restored build-owned drift
# (checked-formal-proofs.sh's restore_tree), so a mutation that removes the
# drift can no longer pass by removing the thing being restored along with it.
if printf '%s' "$out" | grep -qF 'restoring them'; then
  echo "  PASS: case-b: gate's output names the restore -- not merely a" \
       "before/after match that a no-op trap would also produce"
  pass=$((pass + 1))
else
  echo "  FAIL: case-b: expected the gate to report restoring build-owned" \
       "drift; got:"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

# --- Case C: FORMAL_ALLOW_DIRTY unset, tree already had an uncommitted edit -
# The pre-flight refusal itself must still fire, and fire BEFORE the trap-
# bearing subshell is ever entered -- confirming the flag's off-state is
# exactly as before this fix, not just "less destructive than it was".
d="$(make_fixture case-c)"
require_nonempty "case-c dir" "$d"
printf 'committed\nunstaged-user-edit\n' > "$d/formal/testproj/driftfile.txt"
out="$(run_gate "$d")"
rc=$?
content="$(cat "$d/formal/testproj/driftfile.txt")"
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qF '::error::formal/ has uncommitted changes' \
   && printf '%s' "$content" | grep -qF 'unstaged-user-edit'; then
  echo "  PASS: case-c: flag unset + dirty tree still refuses outright (exit $rc), edit untouched"
  pass=$((pass + 1))
else
  echo "  FAIL: case-c: expected an outright refusal naming the dirty tree, got exit $rc"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi


# --- Case D: a TRACKED .lia.cache must survive the trap's own unconditional -
# `rm -f .lia.cache .nia.cache`, and a clean tree after must not brick the
# NEXT unflagged run. This is backlog-229's second recurrence: the same class
# of gap ("restore everything the trap touches"), this time a 1-file door
# instead of a 145-file one, and caused by an unchecked "untracked residue"
# assumption rather than a wrong restore mechanism.
d="$(make_fixture case-d)"
require_nonempty "case-d dir" "$d"
before_cache="$(cat "$d/formal/testproj/.lia.cache")"
out1="$(run_gate "$d")"
after_cache="$(cat "$d/formal/testproj/.lia.cache" 2>/dev/null || echo "<MISSING>")"
status_after=""
if [ -n "$d" ]; then
  status_after="$(cd "$d" && git status --porcelain -- formal)"
fi
if [ "$after_cache" = "$before_cache" ]; then
  echo "  PASS: case-d: the trap's own unconditional cache rm did not leave" \
       "the tracked .lia.cache deleted -- it was restored"
  pass=$((pass + 1))
else
  echo "  FAIL: case-d: tracked .lia.cache was not restored (got: $after_cache)"
  fail=$((fail + 1))
fi
if ! printf '%s' "$status_after" | grep -q '^ D'; then
  echo "  PASS: case-d: formal/ shows no lingering ' D' entry for the cache" \
       "after the flagged run"
  pass=$((pass + 1))
else
  echo "  FAIL: case-d: formal/ still shows a deleted-tracked-file entry:"
  printf '%s\n' "$status_after" | sed 's/^/      /'
  fail=$((fail + 1))
fi
# Second run, flag unset: the pre-flight's dirty check must NOT refuse. The
# stub still fails the actual build (rocq makefile always exits 1), so a
# nonzero exit is expected either way -- the thing under test is that the
# refusal is never the dirty-tree one, which would mean case-d's own gate run
# left drift the pre-flight has to react to.
out2="$(run_gate "$d")"
if ! printf '%s' "$out2" | grep -qF '::error::formal/ has uncommitted changes'; then
  echo "  PASS: case-d: the following unflagged run is not refused for drift" \
       "the previous run caused"
  pass=$((pass + 1))
else
  echo "  FAIL: case-d: the next unflagged run was refused -- the cache" \
       "restore left drift behind:"
  printf '%s\n' "$out2" | sed 's/^/      /'
  fail=$((fail + 1))
fi

# assert_output <case> <output> <fixed string> <description>
assert_output() {
  if printf '%s' "$2" | grep -qF -- "$3"; then
    echo "  PASS: $1: $4"
    pass=$((pass + 1))
  else
    echo "  FAIL: $1: $4 -- expected the output to contain: $3"
    printf '%s\n' "$2" | sed 's/^/      /'
    fail=$((fail + 1))
  fi
}

# --- Case E: a STALE committed extracted model must fail the gate (D-55) ----
# The stub "build" re-extracts different content from what is committed. The
# gate must say so, name the file and the project, show the diff, and give the
# regeneration command -- not merely exit non-zero.
d="$(make_fixture case-e)"
require_nonempty "case-e dir" "$d"
out="$(run_gate "$d" STUB_REGENERATED_MODEL=regenerated-differs)"
assert_output case-e "$out" 'differs from the committed copy' \
  "a stale committed model is reported as such"
assert_output case-e "$out" 'formal/testproj/extraction/model.ml' \
  "the message names the project and the stale file"
assert_output case-e "$out" '-committed' \
  "the diff shows the committed side"
assert_output case-e "$out" '+regenerated-differs' \
  "the diff shows the regenerated side"
assert_output case-e "$out" 'Regenerate: cd formal/testproj && rocq makefile' \
  "the message gives the regeneration command"
if printf '%s' "$out" | grep -qF 'extracted models match their theories'; then
  echo "  FAIL: case-e: gate claimed the models match"
  fail=$((fail + 1))
else
  echo "  PASS: case-e: gate did not claim the stale model matches"
  pass=$((pass + 1))
fi

# --- Case F: an up-to-date model passes the comparison (the negative of E) ---
# Without this, case E could be satisfied by a gate that fails EVERY model. The
# stub re-extracts exactly what is committed; the gate must reach and print its
# success line for the comparison (it then dies in the stub's kernel check,
# which is not under test).
d="$(make_fixture case-f)"
require_nonempty "case-f dir" "$d"
out="$(run_gate "$d" STUB_REGENERATED_MODEL=committed)"
assert_output case-f "$out" 'extracted models match their theories: testproj' \
  "a model that matches its regenerated extraction passes the comparison"

# --- Case G: a driver missing from _CoqProject must fail (D-55) -------------
# This was the state of all twelve type-safety drivers: present on disk, never
# compiled by CI, so their models could not be drift-checked.
d="$(make_fixture case-g)"
require_nonempty "case-g dir" "$d"
printf -- '-R theories Test\n' > "$d/formal/testproj/_CoqProject"
( cd "$d" && git add -A && git commit -q -m "drop driver from _CoqProject" ) >/dev/null 2>&1
out="$(run_gate "$d" STUB_REGENERATED_MODEL=committed)"
assert_output case-g "$out" 'is an extraction driver but is not listed' \
  "a driver absent from _CoqProject is rejected"
assert_output case-g "$out" 'formal/testproj/extraction/Extract.v' \
  "the message names the driver"

# --- Case H: a committed model no driver regenerates must fail ---------------
d="$(make_fixture case-h)"
require_nonempty "case-h dir" "$d"
printf 'orphan\n' > "$d/formal/testproj/extraction/orphan.ml"
( cd "$d" && git add -A && git commit -q -m "add orphan model" ) >/dev/null 2>&1
out="$(run_gate "$d" STUB_REGENERATED_MODEL=committed)"
assert_output case-h "$out" "no driver under formal/testproj/extraction/ extracts 'orphan.ml'" \
  "a committed model that nothing regenerates is rejected"

# --- Case I: a driver that produces no committed model, and is not named ----
# in NOT_DRIFT_CHECKED, must fail: the exemption list may not grow silently.
d="$(make_fixture case-i)"
require_nonempty "case-i dir" "$d"
printf 'Extraction Library Foo.\n' > "$d/formal/testproj/extraction/Other.v"
printf -- '-R theories Test\nextraction/Extract.v\nextraction/Other.v\n' > "$d/formal/testproj/_CoqProject"
( cd "$d" && git add -A && git commit -q -m "add unnamed library driver" ) >/dev/null 2>&1
out="$(run_gate "$d" STUB_REGENERATED_MODEL=committed)"
assert_output case-i "$out" 'produces no committed model and is not named in NOT_DRIFT_CHECKED' \
  "an unnamed driver with no committed model is rejected"

# --- Case J: a failed build puts the committed model back ------------------
# The gate deletes every committed model before the build (so that finding it
# afterwards proves regeneration). The default stub fails at `rocq makefile`,
# i.e. AFTER that deletion; the restore trap must bring the model back, or the
# next run would start from a tree missing a tracked file. (This asserts the
# restore, not the "not recreated" detection -- that is case K.)
d="$(make_fixture case-j)"
require_nonempty "case-j dir" "$d"
out="$(run_gate "$d")"
content="$(cat "$d/formal/testproj/extraction/model.ml" 2>/dev/null || echo "<MISSING>")"
if [ "$content" = "committed" ]; then
  echo "  PASS: case-j: the committed model deleted for the build is restored after a failed build"
  pass=$((pass + 1))
else
  echo "  FAIL: case-j: committed model not restored (got: $content)"
  fail=$((fail + 1))
fi

# --- Case K: a model the full rebuild does not recreate must fail ------------
# The build succeeds but its driver no longer extracts the committed model. The
# committed copy was deleted beforehand, so absence is detectable; if the gate
# compared against the working tree it would pass on a stale file.
d="$(make_fixture case-k)"
require_nonempty "case-k dir" "$d"
out="$(run_gate "$d" STUB_MAKE_BODY=true)"
assert_output case-k "$out" 'formal/testproj/extraction/model.ml does not exist after a full' \
  "a committed model that the build does not recreate is rejected, by name"
if printf '%s' "$out" | grep -qF 'extracted models match their theories'; then
  echo "  FAIL: case-k: gate claimed the models match"
  fail=$((fail + 1))
else
  echo "  PASS: case-k: gate did not claim the models match"
  pass=$((pass + 1))
fi

# --- Case L: a build that produces an uncommitted model must fail ------------
d="$(make_fixture case-l)"
require_nonempty "case-l dir" "$d"
out="$(run_gate "$d" "STUB_MAKE_BODY=printf 'committed\\n' > extraction/model.ml; printf 'x\\n' > extraction/extra.ml")"
assert_output case-l "$out" 'the build produced formal/testproj/extraction/extra.ml, which is not' \
  "an extracted model that is not committed is rejected, by name"

# --- Case M: a NOT_DRIFT_CHECKED entry naming a missing driver must fail -----
# The gate's one exemption names formal/codegen-ptx/extraction/Extract.v. If
# that project exists but the driver does not, the entry exempts nothing.
d="$(make_fixture case-m)"
require_nonempty "case-m dir" "$d"
mkdir -p "$d/formal/codegen-ptx"
printf 'x\n' > "$d/formal/codegen-ptx/placeholder.txt"
( cd "$d" && git add -A && git commit -q -m "codegen-ptx without its driver" ) >/dev/null 2>&1
out="$(run_gate "$d")"
assert_output case-m "$out" "NOT_DRIFT_CHECKED names 'formal/codegen-ptx/extraction/Extract.v', which does not" \
  "an exemption for a nonexistent driver is rejected"

# (The final model-count pin, EXPECTED_DRIFT_CHECKED_MODELS, runs only after
# every project has passed rocq check and the ledger generator; the stub cannot
# get that far, so it is exercised by the real gate run, not here.)

echo ""
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
