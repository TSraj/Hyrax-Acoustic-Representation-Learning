#!/bin/bash
# Submit a CHAIN of LoRA jobs so a 24 h cluster limit stops being a deadline.
#
# Each link runs run_phase3_lora_finetune.sh. The trainer writes full resume
# state every epoch, so link N+1 continues from wherever link N was killed. A
# link that finds a DONE marker exits in seconds, so extra links cost nothing.
#
#   ./submit_lora_chain.sh hubert_base 2
#   ./submit_lora_chain.sh xls_r 3
#
# Run BOTH commands back to back: the two models are independent jobs and will
# occupy separate GPUs, so they progress in parallel.
#
# --dependency=afterany means the next link starts whether the previous one
# succeeded, timed out, or failed -- which is the point: a timeout is not an
# error condition here, it is the expected path.
#
# Watch progress with:
#   squeue -u $USER
#   tail -f logs/lora_species7_*.out
#
# Cancel the whole chain with:
#   scancel <first job id>   # then scancel the rest, or scancel -u $USER

set -euo pipefail

MODEL=${1:-}
LINKS=${2:-3}
SCRIPT=run_phase3_lora_finetune.sh

case "$MODEL" in
    hubert_base|xls_r) ;;
    *) echo "FATAL: first argument must be hubert_base or xls_r (got '${MODEL:-empty}')"
       echo "  usage: ./submit_lora_chain.sh <model> [links]"; exit 1 ;;
esac

[[ -f "$SCRIPT" ]] || { echo "FATAL: $SCRIPT not found -- run from the project root"; exit 1; }

if ! (( LINKS >= 1 && LINKS <= 10 )); then
    echo "FATAL: LINKS must be between 1 and 10 (got '$LINKS')"
    exit 1
fi

mkdir -p logs

echo "submitting a $LINKS-link chain for $MODEL"
echo

PREV=""
for ((i = 1; i <= LINKS; i++)); do
    if [[ -z "$PREV" ]]; then
        JOB=$(sbatch --parsable "$SCRIPT" "$MODEL")
    else
        JOB=$(sbatch --parsable --dependency=afterany:"$PREV" "$SCRIPT" "$MODEL")
    fi
    echo "  link $i/$LINKS  job $JOB${PREV:+  (starts after $PREV)}"
    PREV=$JOB
done

echo
echo "chain submitted. Each link is capped at 24 h; together they cover up to"
echo "$((LINKS * 24)) h. Links after completion exit immediately."
echo
echo "  squeue -u $USER"
echo "  tail -f logs/lora_species7_*.out"
echo
echo "First epoch's runtime is printed in the log. Multiply by 16 to see whether"
echo "$LINKS links are enough, and add more with the same command if not."
