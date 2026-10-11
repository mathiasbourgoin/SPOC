#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Machine-check the Rocq proofs under formal/ (task #45), and the repository's
# claims about them (task #95).
#
# WHAT THIS REPLACES
#
# CI's only enforcement of the "0 admits" guarantee was a grep over the .v
# sources, and the then-committed .vo files were never re-verified on a branch. A grep
# is a lexical check pretending to be a proof: it cannot tell whether the files
# still COMPILE, whether a tactic silently stopped closing its goal, or whether a
# compiled .vo actually corresponds to the .v next to it. A .vo could be stale,
# hand-edited, or built from different sources, and every gate in the repository
# would have stayed green.
#
# This script does the two things only the toolchain can do:
#
#   1. rebuilds every .v FROM SCRATCH (any leftover .vo files are deleted first,
#      so they cannot vouch for themselves), and
#   2. runs `rocq check`, the standalone kernel re-checker, over the results.
#      `rocq check` re-verifies every proof term against the kernel without
#      trusting the elaborator that produced it, and prints the axioms each
#      module depends on.
#
# and then (#95) it derives each project's proof-ledger.json from that build and
# fails if the committed one differs, if the kernel found a project-local axiom
# formal/axiom-allowlist.txt does not sanction, or if a proof-notes.json
# annotates a theorem that does not exist. Proving the proofs and proving that
# what the repository SAYS about them is still true are two different claims;
# before #95 only the first was checked.
#
# The grep gate in .github/workflows/ci.yml is kept as well: it is instant, it
# runs without a Rocq toolchain, and it catches an `admit.` in review before this
# heavier job gets to it. The two are complementary — the grep is the fast lexical
# tripwire, this is the proof.
#
# MEASURED COST (Rocq 9.1.1, 16-core workstation, cold: all .vo deleted)
#   rocq c   (all three projects)   ~11 s total
#   rocq check                       codegen-ptx 17.1 s
#                                    convergence-safety 11.3 s
#                                    type-safety 11.0 s
#   ------------------------------------------------------------------
#   total                           ~50 s of CPU work.
# That is cheap enough that there is no argument for keeping the grep-only
# arrangement. The toolchain is NOT added to ci/Dockerfile: the official
# rocq/rocq-prover image already ships the exact version these proofs were
# developed against, so the proof job runs in that container instead of growing
# the main image by a Rocq build.
# ---------------------------------------------------------------------------
set -euo pipefail

# How many formal/ projects must be found and checked. Pinned so that a project
# silently disappearing (a moved _CoqProject, a bad merge) fails the gate
# instead of quietly shrinking its scope.
EXPECTED_PROJECTS=3

# How many committed extracted models (.ml files under formal/*/extraction)
# must be regenerated and compared. Pinned for the same reason: a model that
# silently drops out of the comparison (a deleted driver, a renamed output, a
# driver removed from _CoqProject) must fail the gate rather than shrink it.
# 1 (codegen-ptx layout) + 1 (codegen-ptx shift) + 1 (convergence-safety)
# + 12 (type-safety).
EXPECTED_DRIFT_CHECKED_MODELS=15

# Extraction drivers whose output is deliberately NOT drift-checked. Each entry
# is "<project>/<driver path>|<reason>". A driver that is neither here nor the
# producer of a committed model fails the gate, so the exemption list cannot
# silently grow: it is named, with its reason, and printed on every run.
NOT_DRIFT_CHECKED=(
  "formal/codegen-ptx/extraction/Extract.v|\
\`Extraction Library\` writes five .ml files into the project root; they are \
not tracked (backlog-188), no dune stanza compiles them and nothing links \
them (they open Datatypes/List/Nat, which only exist if the Stdlib is \
extracted alongside). There is no committed copy to compare against. See \
formal/codegen-ptx/STATUS.md, 'Extraction'."
)

cd "$(dirname "$0")/.."
# Absolute, because the per-project steps run in a subshell that has cd'd into
# formal/<project> and still need to invoke scripts/ from the repository root.
ROOT=$(pwd)

# Writability pre-flight.
#
# This script rebuilds in place: it deletes any leftover .vo files and regenerates
# CoqMakefile next to the sources. In CI that happens inside a container whose
# user need not own the mounted checkout — rocq/rocq-prover runs as `rocq`
# (uid 1000) while a GitHub-hosted runner owns the workspace as `runner`
# (uid 1001), and a uid-1000 process cannot write a uid-1001 0755 directory.
# Without this probe the first symptom is an EACCES from `rm` several steps
# later, which reads like a broken proof rather than a broken mount.
#
# This cannot be caught by running the script locally: a developer's checkout is
# already owned by the user running it. That is precisely why it is asserted.
probe=".formal-write-probe.$$"
if ! : > "$probe" 2>/dev/null; then
  echo "::error::$(pwd) is not writable by uid $(id -u). This script rebuilds \
the proofs in place. In CI, mount a checkout owned by the container's user (see \
the formal-proofs job in .github/workflows/ci.yml)."
  exit 1
fi
rm -f "$probe"

# Refuse to run on a formal/ tree that already has uncommitted changes.
#
# Step 6 restores the tree with `git checkout -- .` because the build rewrites
# ~68 extracted .ml/.mli files. That restore cannot tell build output from work
# in progress, so on a dirty tree it would silently destroy a developer's
# uncommitted edits. Refusing here is what makes the restore safe: after this
# check, any drift observed later must have been produced by the build.
#
# FORMAL_ALLOW_DIRTY=1 bypasses this refusal, and by itself that would reopen
# exactly the loss described above — this was filed as backlog #229 after it
# happened. The flag does NOT change what the restore_tree trap below touches:
# in every case it restores only the set of paths the build itself owns and
# regenerates (the .vo/.vok/.vos/.glob/CoqMakefile* artefacts step 1 deletes,
# and the extracted .ml/.mli files step 3 rewrites) — a set that is knowable in
# advance and disjoint from a developer's own source edits, so restoring it
# never discards anything the flag is meant to protect. What the flag changes
# is scope, not mechanism: without it, this pre-flight also requires formal/ to
# have been clean on entry, so "restore the build-owned set" and "restore
# everything" are the same operation; with it, entry may be dirty, and the
# trap restores only the build-owned set, leaving whatever else you had dirty
# exactly as it was. Either way, the BUILD ITSELF (`make`, extraction,
# `rocq makefile`) can still overwrite a build-owned path directly, before the
# trap ever runs — the flag protects your edits from the trap, not from a
# build that regenerates a file you happened to have hand-edited.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  dirty=$(git status --porcelain -- formal | grep -v '^?? ' || true)
  if [ -n "$dirty" ] && [ "${FORMAL_ALLOW_DIRTY:-0}" != "1" ]; then
    echo "::error::formal/ has uncommitted changes to tracked files. This \
script rebuilds in place, and the per-project restore_tree trap (below) \
restores the build-owned artefacts it rewrites — .vo/.vok/.vos/.glob, \
CoqMakefile*, and the extracted .ml/.mli files — which would discard your \
edits to those same paths. Commit or stash first, or set FORMAL_ALLOW_DIRTY=1 \
to run anyway — with the flag set, entry is allowed to be dirty and the trap \
still restores only the build-owned set, leaving any of your edits outside it \
untouched; edits INSIDE that set are not protected, from the trap or from the \
build that regenerates them."
    printf '%s\n' "$dirty" | sed 's/^/    /'
    exit 1
  fi
fi

if command -v rocq >/dev/null 2>&1; then
  ROCQ=rocq
elif command -v coqc >/dev/null 2>&1; then
  # Rocq 9 renamed the driver; fall back for a Coq 8.x toolchain, which will
  # most likely fail to compile these sources and say so loudly.
  ROCQ=""
else
  echo "::error::no rocq/coqc on PATH. This gate needs a real proof checker — \
that is the entire point of #45. Run it in the rocq/rocq-prover container (see \
.github/workflows/ci.yml)."
  exit 1
fi

echo "== toolchain"
if [ -n "$ROCQ" ]; then
  rocq --version
else
  coqc --version
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "::error::no python3 on PATH. scripts/gen-proof-ledger.py derives each \
project's proof-ledger.json from the toolchain, and without it the ledger \
comparison and the axiom allowlist (#95) cannot run at all — which is the \
state this gate was written to end. rocq/rocq-prover:9.1.1 ships python3."
  exit 1
fi

projects=$(find formal -mindepth 1 -maxdepth 1 -type d | sort)
if [ -z "$projects" ]; then
  echo "::error::no formal/ projects found — this gate would pass vacuously."
  exit 1
fi

# Generated ledgers land OUTSIDE the project directories, deliberately. The
# per-project `restore_tree` trap runs `git checkout -- .` inside the project,
# so a ledger written next to the sources would be reverted to the committed
# copy before it could be compared against it — the comparison would pass by
# comparing the committed file with itself. Writing elsewhere is what makes the
# drift check able to fail.
LEDGER_OUT=$(mktemp -d)
# Canonicalised copies of the committed extracted models, taken BEFORE the build
# deletes them (see step 0 below), and the list of models that were compared.
# Outside the project directories for the same reason as LEDGER_OUT: the
# restore_tree trap would otherwise revert them before they could be compared.
MODEL_SNAP=$(mktemp -d)
trap 'rm -rf "$LEDGER_OUT" "$MODEL_SNAP"' EXIT
: > "$MODEL_SNAP/checked.list"

checked=0

for proj in $projects; do
  [ -f "$proj/_CoqProject" ] || continue
  echo
  echo "== $proj"

  logical=$(grep -oP '^-R\s+theories\s+\K\S+' "$proj/_CoqProject" || true)
  if [ -z "$logical" ]; then
    echo "::error::$proj/_CoqProject has no '-R theories <LogicalName>' line; \
cannot determine the logical path to check."
    exit 1
  fi

  (
    cd "$proj"

    # Restore the working tree on EVERY exit path, installed BEFORE the first
    # destructive step.
    #
    # `make -f CoqMakefile` re-runs extraction, and the extracted .ml/.mli files
    # are committed in ocamlformat-formatted form while extraction emits them
    # raw. A full run therefore rewrites ~68 tracked files with semantically
    # identical but differently formatted output, and also leaves a regenerated
    # CoqMakefile behind. Harmless on a throwaway CI checkout, destructive for
    # anyone running this documented gate on their own tree.
    #
    # As a trap rather than a final step because the interesting exits are the
    # early ones: a failed compile or a failed kernel re-check used to leave the
    # committed .vo files deleted and the tree rewritten, and — now that this
    # script refuses to start on a dirty formal/ — that residue would block the
    # very next run with a message about uncommitted changes the developer never
    # made. Restoring unconditionally keeps a failed run repeatable.
    #
    # WHAT GETS RESTORED, and why it is narrower than "the whole tree" (#229):
    #
    # An earlier version restored with a bare `git checkout -- .`, reasoned to
    # be safe because the dirty-tree pre-flight guaranteed the tree started
    # clean. That reasoning broke the moment FORMAL_ALLOW_DIRTY=1 let entry be
    # dirty: `git status` at exit then reports BOTH the build's own rewrites
    # AND whatever uncommitted edits the developer already had, `git checkout
    # -- .` cannot tell them apart, and the fix shipped for #229 responded by
    # having the flag disable the restore ENTIRELY — which meant the six
    # mutating steps below (the `find -exec rm`, `rocq makefile`, `make`, and
    # for codegen-ptx the canonicalize step) stayed applied under the flag:
    # 145 tracked build artefacts deleted or rewritten and never put back. That
    # left ` D `/` M ` entries that survive the pre-flight's `grep -v '^?? '`,
    # so the very next UNFLAGGED run refused on drift the developer never
    # made — an absorbing state, the exact failure this trap exists to
    # prevent, one flagged run away.
    #
    # The actual fix does not need to classify the diff (build drift vs.
    # pre-existing dirt look identical as a `git status` line either way) —
    # it only needs the PATH SET, which this script already knows because it
    # is the one deleting and regenerating those paths: the artefacts step 1
    # finds (.vo/.vok/.vos/.glob/*.CoqMakefile.d/CoqMakefile/CoqMakefile.conf)
    # and the extracted .ml/.mli files under extraction/ that `make` and (for
    # codegen-ptx) canonicalize-extraction.py rewrite. That set is disjoint
    # from a developer's own source edits (theories/*.v, _CoqProject, ...) by
    # construction — nothing outside it is ever written by this script — so
    # restoring exactly that set is safe UNCONDITIONALLY, flag or no flag:
    #
    #   * it never discards a developer's own edits, because those live
    #     outside the build-owned set and this restore never touches outside
    #     it;
    #   * it never leaves the build-owned set in an absorbing dirty state,
    #     because the build-owned set is exactly what gets restored, flag or
    #     no flag.
    #
    # BUILD_OWNED_GLOBS below must name the same suffixes as the `find` in
    # step 1 (kept as two lists rather than one shared array so that pathspec
    # syntax and `find -name` syntax do not have to share a representation —
    # but that also means the two can drift, so a change to one's file types
    # must be mirrored in the other) plus the extraction outputs step 3/3b
    # rewrite. Restoring per-pattern, not in one `git checkout` call: a glob
    # pathspec that matches nothing tracked makes `git checkout` exit non-zero
    # and apply NONE of the pathspecs given alongside it (verified against a
    # throwaway fixture) — fatal here under `set -e` outside this subshell,
    # and silently a no-op restore if swallowed as one call. One `checkout`
    # per pattern means an unmatched pattern (e.g. no `*.CoqMakefile.d` this
    # project) only skips itself.
    #
    # The .lia/.nia caches the `lia`/`nia` tactics drop next to the sources
    # are NOT always untracked residue: `git ls-files formal` shows
    # formal/convergence-safety/.lia.cache committed and tracked (since
    # untracked and ignored, formal/.gitignore; `.nia.cache` was untracked — the same filename can be either, project to
    # project). A bare unconditional `rm -f` on a tracked cache is exactly
    # the #229 hazard in miniature: it deletes a tracked path and never
    # restores it, so the next unflagged run refuses on drift nobody made.
    # They are folded into BUILD_OWNED_GLOBS below instead, so a tracked
    # cache is restored the same way as every other build-owned path, and
    # deleting an untracked one is still a no-op restore (nothing tracked to
    # check out, and each pattern's failure is independent of the rest).
    BUILD_OWNED_GLOBS=(
      ':(glob)**/*.vo' ':(glob)**/*.vok' ':(glob)**/*.vos' ':(glob)**/*.glob'
      ':(glob)**/*.CoqMakefile.d' ':(glob)**/CoqMakefile' \
      ':(glob)**/CoqMakefile.conf' \
      ':(glob)**/extraction/*.ml' ':(glob)**/extraction/*.mli' \
      '.lia.cache' '.nia.cache'
    )
    restore_tree() {
      rm -f .lia.cache .nia.cache
      if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 0
      fi
      drift=$(git status --porcelain -- "${BUILD_OWNED_GLOBS[@]}" 2>/dev/null \
               | grep -v '^?? ' || true)
      if [ -n "$drift" ]; then
        n=$(printf '%s\n' "$drift" | wc -l)
        echo "  NOTE: the build rewrote $n build-owned tracked file(s)" \
             "(.vo/.vok/.vos/.glob, CoqMakefile*, extraction output, and/or" \
             ".lia.cache/.nia.cache); restoring them."
        for pat in "${BUILD_OWNED_GLOBS[@]}"; do
          git checkout -- "$pat" 2>/dev/null || true
        done
        remaining=$(git status --porcelain -- "${BUILD_OWNED_GLOBS[@]}" \
                     2>/dev/null | grep -v '^?? ' || true)
        if [ -n "$remaining" ]; then
          echo "  WARNING: could not restore all build-owned paths; still" \
               "dirty:"
          printf '%s\n' "$remaining" | sed 's/^/    /'
        fi
      fi
      if [ "${FORMAL_ALLOW_DIRTY:-0}" = "1" ]; then
        echo "  NOTE: FORMAL_ALLOW_DIRTY=1 — only the build-owned paths above" \
             "were touched; any pre-existing uncommitted edits of yours" \
             "outside that set are left exactly as they were. Edits INSIDE" \
             "that set are not protected, from this trap or from the build" \
             "step that regenerated them."
      fi
    }
    trap restore_tree EXIT

    # 0. Extracted models: static checks and snapshot (finding D-55).
    #
    #    Until D-55 only ONE of the fourteen committed extracted models
    #    (PtxLayout) was compared with a fresh extraction. The other thirteen
    #    -- twelve type-safety models and the convergence model -- are what the
    #    conformance suites run as their oracle, and nothing checked that they
    #    still corresponded to the theories: type-safety's extraction drivers
    #    were not even in _CoqProject, so CI never compiled them, and
    #    TypeSafetyModel.ml had in fact gone stale (it lacked TRecord and
    #    TVariant). A conformance suite that passes against a stale model
    #    proves the production code agrees with something the theorems are no
    #    longer about.
    #
    #    The rule is now uniform and project-independent:
    #
    #      * every extraction driver (extraction/*.v) is listed in _CoqProject,
    #        so the from-scratch build below runs it;
    #      * every driver either produces a committed model or is named in
    #        NOT_DRIFT_CHECKED with its reason;
    #      * every committed model (extraction/*.ml, *.mli) is produced by a
    #        driver, is DELETED before the build (so that finding it again
    #        afterwards proves the build regenerated it, rather than the
    #        committed copy vouching for itself), and is compared after the
    #        build, both sides canonicalised.
    #
    #    The snapshot is taken here, before step 1, from the working tree, so no
    #    git is needed to make the comparison (the earlier check required a
    #    usable git directory and had to fail when it was missing).
    snap="$MODEL_SNAP/$(basename "$PWD")"
    mkdir -p "$snap"
    shopt -s nullglob
    drivers=(extraction/*.v)
    models=(extraction/*.ml extraction/*.mli)
    shopt -u nullglob

    all_outputs=""
    for driver in "${drivers[@]}"; do
      if ! grep -qxF "$driver" _CoqProject; then
        echo "::error::$proj/$driver is an extraction driver but is not listed \
in $proj/_CoqProject, so the from-scratch build never runs it and the model it \
produces cannot be drift-checked (this was the state of all twelve type-safety \
drivers before D-55). Add '$driver' to $proj/_CoqProject."
        exit 1
      fi
      all_outputs="$all_outputs
$(grep -oP 'Extraction\s+"\K[^"]+' "$driver" || true)"
    done

    for model in "${models[@]}"; do
      base=$(basename "$model")
      stem=${base%.mli}
      stem=${stem%.ml}
      if ! printf '%s\n' "$all_outputs" | grep -qxF "$stem.ml"; then
        echo "::error::$proj/$model is a committed extracted model but no \
driver under $proj/extraction/ extracts '$stem.ml'. A committed model nothing \
regenerates cannot be drift-checked and is exactly the stale-oracle hazard \
D-55 closed. Delete it, or add the driver that produces it."
        exit 1
      fi
      cp "$model" "$snap/$base"
      python3 "$ROOT/scripts/canonicalize-extraction.py" "$snap/$base"
    done

    for driver in "${drivers[@]}"; do
      exempt_reason=""
      for entry in "${NOT_DRIFT_CHECKED[@]}"; do
        if [ "${entry%%|*}" = "$proj/$driver" ]; then
          exempt_reason=${entry#*|}
        fi
      done
      if [ -n "$exempt_reason" ]; then
        echo "  NOT drift-checked (named exemption): $driver -- $exempt_reason"
        continue
      fi
      owns=0
      for model in "${models[@]}"; do
        stem=$(basename "$model"); stem=${stem%.mli}; stem=${stem%.ml}
        if grep -qP "Extraction\s+\"$stem\.ml\"" "$driver"; then owns=1; fi
      done
      if [ "$owns" -eq 0 ]; then
        echo "::error::$proj/$driver produces no committed model and is not \
named in NOT_DRIFT_CHECKED in scripts/check-formal-proofs.sh. Either commit \
the model it extracts (so the gate compares it) or add the driver to \
NOT_DRIFT_CHECKED with the reason it cannot be compared."
        exit 1
      fi
    done

    # Delete the committed models: finding them after the build then proves the
    # build regenerated them. restore_tree puts the committed copies back.
    for model in "${models[@]}"; do rm -f "$model"; done

    # 1. Delete every build artefact (none is tracked: formal/.gitignore). A .vo that is never rebuilt is
    #    an unverified binary blob, and re-checking it would only prove it is
    #    self-consistent, not that it matches the .v beside it.
    #
    #    The generated makefile and its dependency files go too, and that is not
    #    tidiness — it is the stale-generated-artefact hazard this script exists
    #    to catch, biting the script itself. The formerly tracked .CoqMakefile.d files
    #    hard-code /usr/lib/ocaml/rocq-runtime/rocqworker, the path of a
    #    system-packaged Rocq. That path exists on a distro install and does not
    #    exist in the opam-based rocq/rocq-prover image, so reusing them fails
    #    the build with "No rule to make target" before a single proof is
    #    checked.
    find . \( -name '*.vo' -o -name '*.vok' -o -name '*.vos' \
              -o -name '*.glob' -o -name '*.CoqMakefile.d' \
              -o -name 'CoqMakefile' -o -name 'CoqMakefile.conf' \) \
         -exec rm -f {} +

    # 2. Regenerate the makefile from _CoqProject. Never reuse the committed
    #    CoqMakefile: see above.
    if [ -n "$ROCQ" ]; then
      rocq makefile -f _CoqProject -o CoqMakefile
    else
      coq_makefile -f _CoqProject -o CoqMakefile
    fi

    # 3. Compile. Any Admitted/admit that the grep gate missed, any broken
    #    proof, any missing dependency fails here.
    make -f CoqMakefile -j"$(nproc)"

    # 3b. Extraction drift (task #46, generalised to every model by D-55).
    #
    #     The `make` above re-ran every extraction driver. Each regenerated
    #     model must equal the committed copy snapshotted in step 0, because
    #     the conformance suites run those committed copies: without this
    #     check the chain from a theory to the thing actually tested is only
    #     as good as whoever last remembered to regenerate, and an edit to the
    #     theory leaves a stale model in the tree while the suite stays green
    #     testing something the theorems are no longer about. That is the
    #     hand-mirror defect #46 removed, reappearing one level up.
    #
    #     Deliberately NOT extended to Extract.v's five outputs in the
    #     codegen-ptx project root: `Extraction Library` output that is
    #     untracked, uncompiled and linked by nothing has no committed copy to
    #     compare. It is a named entry in NOT_DRIFT_CHECKED, printed in step 0.
    #
    #     CANONICALISED BEFORE COMPARING, and that is not tidiness. The first CI
    #     run of this check failed on a diff whose entire content was:
    #
    #       -  { cl_tag = tag; cl_leaves = (flattens (tag :: []) payoff …);
    #       -    cl_payload_size =
    #       +  { cl_tag = tag; cl_leaves =
    #       +    (flattens (tag :: []) payoff …); cl_payload_size =
    #
    #     Identical tokens, different line breaks. Rocq's extractor pretty-prints
    #     through OCaml's Format, whose breaking decisions differ between the
    #     OCaml the PROVER was built with — 5.5.0 for a local distro install,
    #     4.14.2 inside rocq/rocq-prover:9.1.1. A byte-compare of raw extractor
    #     output is therefore a check on which machine ran the extractor: it
    #     would have fired on every CI run forever, while being unable to
    #     distinguish that from the drift it exists to catch. A gate that cries
    #     wolf is a gate the next person in a hurry deletes.
    #
    #     ocamlformat is not an option here: the extraction happens in the Rocq
    #     image, which ships none (verified), and moving the comparison to a job
    #     that has it would mean moving the extraction to a job that has no
    #     Rocq. scripts/canonicalize-extraction.py needs only python3, which the
    #     image does have. Both sides go through it, so what is compared is the
    #     model, not its wrapping. It follows that committed models are stored
    #     canonicalised and listed in .ocamlformat-ignore: ocamlformat would
    #     change tokens (parentheses), not just wrapping.
    #
    #     A missing regenerated file FAILS rather than skips. An earlier version
    #     of this check was written as one `if ... && ...; then` with no else,
    #     and running it in a container that could not see the git directory
    #     made the whole check disappear: the run reported OK having compared
    #     nothing. A drift gate that silently opts out when its inputs are
    #     missing is the failure mode this script exists to remove.
    for snapped in "$snap"/*; do
      base=$(basename "$snapped")
      [ "$base" = "checked.list" ] && continue
      if [ ! -f "extraction/$base" ]; then
        echo "::error::$proj/extraction/$base does not exist after a full \
rebuild. A driver is meant to produce it, and it is committed so that the \
conformance tests can link it; without a regenerated copy there is nothing to \
compare. Check that its driver in $proj/extraction/ is listed in \
$proj/_CoqProject and still extracts '${base%.mli}'."
        exit 1
      fi
      python3 "$ROOT/scripts/canonicalize-extraction.py" "extraction/$base"
      if ! diff -u --label "committed/$proj/extraction/$base" \
                   --label "regenerated/$proj/extraction/$base" \
                   "$snapped" "extraction/$base"; then
        echo "::error::the regenerated extracted model $proj/extraction/$base \
differs from the committed copy. The diff above is in TOKENS, not whitespace: \
both sides went through scripts/canonicalize-extraction.py. The theory (or its \
extraction driver) changed without the model being regenerated, so everything \
that links this model is checking production against a stale oracle. \
Regenerate: cd $proj && rocq makefile -f _CoqProject -o CoqMakefile && \
make -f CoqMakefile && python3 ../../scripts/canonicalize-extraction.py \
extraction/$base, then commit it."
        exit 1
      fi
      case "$base" in
        *.ml) echo "$proj/extraction/$base" >> "$MODEL_SNAP/checked.list" ;;
      esac
    done
    shopt -s nullglob
    for produced in extraction/*.ml extraction/*.mli; do
      if [ ! -f "$snap/$(basename "$produced")" ]; then
        echo "::error::the build produced $proj/$produced, which is not \
committed. A model nothing compares is not an oracle: commit it (canonicalised) \
so the gate can check it from now on."
        exit 1
      fi
    done
    shopt -u nullglob
    echo "  extracted models match their theories: $(basename "$PWD")"

    # 4. Kernel re-check. This is the load-bearing step: it re-verifies the
    #    proof terms independently of the elaborator that built them, and
    #    reports the axioms they rest on.
    #    `set +e` around the capture is load-bearing, not defensive habit.
    #    `out=$(rocq check ...)` is a plain assignment in a command body, not an
    #    `if` condition, so `set -e` applies to it in full. `rocqchk` exits
    #    non-zero when it finds an error, so on a REAL kernel-check failure the
    #    subshell died on this very line — before the diagnostics below could
    #    run — and because `2>&1` had redirected the checker's entire output
    #    into `$out`, that output was discarded with it. The job still failed
    #    (no false pass), but reported none of the context this step exists to
    #    provide. Untested until now because the committed proofs all pass.
    set +e
    if [ -n "$ROCQ" ]; then
      out=$(rocq check -R theories "$logical" theories/*.vo 2>&1)
    else
      out=$(coqchk -R theories "$logical" theories/*.vo 2>&1)
    fi
    rc=$?
    set -e
    printf '%s\n' "$out" | tail -3
    # Two independent conditions, both fatal: a non-zero exit from the checker,
    # and the absence of its success marker. Neither implies the other — a
    # checker that dies on a missing .vo exits non-zero without ever printing a
    # verdict, and a truncated or partial run can exit 0 having checked nothing.
    #
    # Substring match on the variable, NOT `printf ... | grep -q`: grep -q exits
    # on its first match and closes the pipe, printf takes SIGPIPE (141), and
    # `set -o pipefail` propagates that as a failure. With `rocq check` output
    # running to thousands of lines this fires reliably, and the check would
    # report a failure on a run where every proof passed.
    case "$out" in
      *"Modules were successfully checked"*) marker=1 ;;
      *) marker=0 ;;
    esac
    if [ "$rc" -ne 0 ] || [ "$marker" -ne 1 ]; then
      echo "::error::kernel re-check FAILED for $proj (exit $rc, success marker \
present: $marker)"
      printf '%s\n' "$out"
      exit 1
    fi

    # 5. Derive this project's proof ledger from the build that just happened.
    #
    #    This REPLACES a grep over `$out` for lines matching /axiom/i, which
    #    reported ~2200 lines per project because `rocq check`'s trace names
    #    every Stdlib module it loads and several of them are called
    #    `...Uint63Axioms` / `...FloatAxioms`. It printed module names, never an
    #    axiom, and was labelled "assumptions reported by the kernel checker".
    #
    #    gen-proof-ledger.py parses the CONTEXT SUMMARY that `rocq check -o`
    #    prints — the actual transitive assumption closure — and cross-checks it
    #    against theories/*.glob. It fails loudly if the summary is missing or
    #    its format changed, so a parse that finds nothing cannot be mistaken
    #    for a project with no axioms.
    #
    #    Written to $LEDGER_OUT, not next to the sources: see the comment where
    #    LEDGER_OUT is created.
    python3 "$ROOT/scripts/gen-proof-ledger.py" . \
      -o "$LEDGER_OUT/$(basename "$PWD").json"

    # 6. The working tree is put back by the restore_tree EXIT trap installed
    #    at the top of this subshell, so that it also runs on the failure paths
    #    above. See the comment there.
  )
  checked=$((checked + 1))
done

echo
# A gate that passes when it found nothing to check is the exact failure mode
# this script was written to remove. Moving the three _CoqProject files aside
# used to yield "OK: 0 formal project(s)" and exit 0.
if [ "$checked" -eq 0 ]; then
  echo "::error::no formal/ project had a _CoqProject — nothing was verified. \
This gate must never pass vacuously."
  exit 1
fi
if [ "$checked" -ne "$EXPECTED_PROJECTS" ]; then
  echo "::error::expected $EXPECTED_PROJECTS formal projects, checked $checked. \
If a project was added or removed on purpose, update EXPECTED_PROJECTS at the \
top of this script."
  exit 1
fi

compared=$(wc -l < "$MODEL_SNAP/checked.list")
if [ "$compared" -ne "$EXPECTED_DRIFT_CHECKED_MODELS" ]; then
  echo "::error::expected $EXPECTED_DRIFT_CHECKED_MODELS extracted models to be \
regenerated and compared, compared $compared:"
  sed 's/^/    /' "$MODEL_SNAP/checked.list"
  echo "If a model was added or removed on purpose, update \
EXPECTED_DRIFT_CHECKED_MODELS at the top of this script."
  exit 1
fi
echo "OK: $compared extracted model(s) regenerated from their theories and compared."

# ---------------------------------------------------------------------------
# Ledger drift, axiom allowlist, note anchors (task #95).
#
# Everything above proves the proofs. This proves that what the repository SAYS
# about the proofs is still true — the half that #45 deliberately left open,
# because the three hand-written proof-ledger.json files disagreed with each
# other and there was no trustworthy expected value to compare against.
#
# Three checks, each failing for a different reason:
#
#   drift     — the committed ledger differs from what this build produced.
#               Someone edited it, or changed the proofs without regenerating.
#   allowlist — the kernel found a project-local axiom nobody sanctioned, or an
#               entry in formal/axiom-allowlist.txt no proof depends on any more.
#               This is the check the "reported for review rather than enforced"
#               note used to stand in for.
#   anchors   — proof-notes.json names a theorem that does not exist. The old
#               ledgers carried one such entry (an anchor pointing at a PAIR of
#               lemmas under a name Rocq never saw); nothing could detect it.
#
# Runs after the loop, so the per-project restore_tree traps have already put
# the working tree back and the committed ledgers are the committed ledgers.
# ---------------------------------------------------------------------------
echo
echo "== ledger, axioms, anchors"
python3 "$ROOT/scripts/check-proof-ledger.py" --generated "$LEDGER_OUT"

# Shipped-artefact vs model-only split (backlog-201). Runs against the SAME
# freshly generated ledgers as the check above, so the split is derived from
# this from-scratch rebuild rather than from whatever the committed
# proof-ledger.json files happen to say (that equivalence is what the drift
# check above already established, but this keeps the two gates independent
# of each other rather than one trusting the other's output silently).
echo
echo "== shipped-artefact / model-only split"
python3 "$ROOT/scripts/check-production-link.py" --generated "$LEDGER_OUT"

echo
echo "OK: $checked formal project(s) rebuilt from source and kernel-re-checked."
