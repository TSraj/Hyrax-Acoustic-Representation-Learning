#!/bin/bash
# Chain the species layer probe so a timeout costs nothing.
#
# Each link runs the full 6-cell array. Cells that already finished exit in
# seconds; a cell interrupted mid-extraction resumes from its per-split cache.
# So link N+1 continues exactly where link N stopped.
#
#   ./submit_species_layers_chain.sh        # 2 links
#   ./submit_species_layers_chain.sh 3
#
# --dependency=afterany: the next link starts whether the previous finished,
# timed out, or failed. A timeout is the expected path, not an error.

set -euo pipefail

LINKS=${1:-2}
SCRIPT=run_phase3_species_layer_probe.sh

[[ -f "$SCRIPT" ]] || { echo "FATAL: $SCRIPT not found -- run from the project root"; exit 1; }
if ! (( LINKS >= 1 && LINKS <= 6 )); then
    echo "FATAL: LINKS must be 1-6 (got '$LINKS')"; exit 1
fi

mkdir -p logs
echo "submitting a $LINKS-link chain of $SCRIPT"
echo

PREV=""
for ((i = 1; i <= LINKS; i++)); do
    if [[ -z "$PREV" ]]; then
        JOB=$(sbatch --parsable "$SCRIPT")
    else
        JOB=$(sbatch --parsable --dependency=afterany:"$PREV" "$SCRIPT")
    fi
    echo "  link $i/$LINKS  job $JOB${PREV:+  (starts after $PREV)}"
    PREV=$JOB
done

echo
echo "each link is 6 cells x 8 h max; finished cells are skipped instantly."
echo
echo "  squeue -u $USER"
echo "  grep -a 'RESULT\|BEST layer\|already exists' logs/species_layers_*.out"
