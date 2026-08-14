# shellcheck shell=bash
#
# Sourced by every sbatch script. Two jobs:
#
#   1. Stamp the exact commit into the job log, so any result can be traced back
#      to code without guesswork. This is what docs/PROVENANCE_ESSD.md had to be
#      reconstructed by hand for; doing it up front costs nothing.
#
#   2. Refuse to run from a dirty checkout. The Sherlock clone is a *run target*,
#      not a place to edit code — local + GitHub are the source of truth. Edits
#      made here are invisible to everyone, are not backed up, and silently
#      decouple published numbers from the repo that supposedly produced them.
#
# Failure philosophy, learned from job 39051098 (which this file killed):
#   - a DIRTY TREE is fatal — that is the guardrail.
#   - INABILITY to check (git too old, git missing, weird FS) is a WARNING.
#     The job's purpose is science; the stamp must never be the reason a
#     3-hour allocation dies 30 seconds in.
#
# Portability: Sherlock compute nodes carry an ancient git. `git -C` needs
# 1.8.5+, so we cd in a subshell instead. Assume nothing newer than ~1.8.
#
# Untracked files are tolerated — stray outputs and editor droppings do not
# change what the job executes. Only modifications to *tracked* files are fatal.
#
# Escape hatch, for when you knowingly want to test an uncommitted change:
#     LV_ALLOW_DIRTY=1 sbatch engine/training/<script>.sh
#
# Usage, after cd-ing to the repo:
#     source engine/sherlock_preflight.sh

lv_preflight() {
    local repo_dir="${1:-$PWD}"
    local sha branch subject dirty

    echo "--- provenance ---"
    echo "repo    : $repo_dir"

    if ! command -v git >/dev/null 2>&1; then
        echo "tree    : UNKNOWN — no git on this node; provenance not recorded" >&2
        echo "------------------"
        return 0
    fi

    # Old-git-safe: run everything from inside the repo, no `git -C`.
    if ! sha=$(cd "$repo_dir" && git rev-parse HEAD 2>/dev/null); then
        echo "tree    : UNKNOWN — git could not read $repo_dir; provenance not recorded" >&2
        echo "------------------"
        return 0
    fi

    branch=$(cd "$repo_dir" && git rev-parse --abbrev-ref HEAD 2>/dev/null)
    subject=$(cd "$repo_dir" && git log -1 --pretty=%s 2>/dev/null)
    echo "branch  : ${branch:-unknown}"
    echo "commit  : $sha"
    echo "subject : ${subject:-unknown}"

    # If status itself fails, warn and continue — do not kill the job.
    if ! dirty=$(cd "$repo_dir" && git status --porcelain --untracked-files=no 2>/dev/null); then
        echo "tree    : UNKNOWN — git status failed; dirty-check skipped" >&2
        echo "------------------"
        return 0
    fi

    if [ -n "$dirty" ]; then
        echo "tree    : DIRTY"
        echo "$dirty" | sed 's/^/          /'
        if [ "${LV_ALLOW_DIRTY:-0}" != "1" ]; then
            cat >&2 <<EOF

PREFLIGHT FAILED: tracked files are modified in the Sherlock checkout.

This clone is for running code, not editing it. Commit and push from your
local machine, then here:

    cd "$repo_dir"
    git fetch origin
    git reset --hard origin/<branch>

To discard what is here (check it first with 'git diff'):

    git checkout -- .

To run anyway, knowing the results will not match any commit:

    LV_ALLOW_DIRTY=1 sbatch <script>

EOF
            return 1
        fi
        echo "          (LV_ALLOW_DIRTY=1 -- results do NOT correspond to $sha)"
    else
        echo "tree    : clean"
    fi
    echo "------------------"
}
