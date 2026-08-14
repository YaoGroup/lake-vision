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
#      We already lost time to this: two stashes sat on the Sherlock clone for
#      months, one of them holding the only copy of the SLURM-array preprocessing
#      code that actually ran in March 2026.
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

    if ! git -C "$repo_dir" rev-parse --git-dir >/dev/null 2>&1; then
        echo "PREFLIGHT: $repo_dir is not a git repo; cannot record provenance." >&2
        return 1
    fi

    local sha branch dirty
    sha=$(git -C "$repo_dir" rev-parse HEAD)
    branch=$(git -C "$repo_dir" rev-parse --abbrev-ref HEAD)
    dirty=$(git -C "$repo_dir" status --porcelain --untracked-files=no)

    echo "--- provenance ---"
    echo "repo    : $repo_dir"
    echo "branch  : $branch"
    echo "commit  : $sha"
    echo "subject : $(git -C "$repo_dir" log -1 --pretty=%s)"

    if [ -n "$dirty" ]; then
        echo "tree    : DIRTY"
        echo "$dirty" | sed 's/^/          /'
        if [ "${LV_ALLOW_DIRTY:-0}" != "1" ]; then
            cat >&2 <<EOF

PREFLIGHT FAILED: tracked files are modified in the Sherlock checkout.

This clone is for running code, not editing it. Commit and push from your
local machine, then here:

    git -C "$repo_dir" fetch origin
    git -C "$repo_dir" checkout <branch>
    git -C "$repo_dir" reset --hard origin/<branch>

To discard what is here (check it first -- 'git -C "$repo_dir" diff'):

    git -C "$repo_dir" checkout -- .

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
