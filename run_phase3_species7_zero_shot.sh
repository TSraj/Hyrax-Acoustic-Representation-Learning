#!/bin/bash
#SBATCH --job-name=phase3_species7_zs
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/phase3_species7_zs_%A_%a.out
#SBATCH --error=logs/phase3_species7_zs_%A_%a.err

# Phase A / Step A4 - frozen-encoder zero-shot baselines on the 7-CLASS species
# task (hyrax excluded).
#
# WHY THESE ARE NEEDED. The existing baselines (xls_r 0.7194, hubert_base
# 0.8635) are 8-CLASS numbers. Every 7-class adaptation run needs a same-label-
# space frozen baseline to be measured against, and the --baseline-f1 curve
# annotation is wrong without one.
#
# ############################################################################
# # READ THIS BEFORE PUTTING THESE NUMBERS IN A TABLE
# #
# # These are GENUINE 7-WAY results: the classifier has 7 outputs and the
# # encoder never sees hyrax. Chance is 1/7 = 0.1429.
# #
# # They are NOT comparable to the 'f1_7' / 'test_f1_macro_7cls' columns in
# # phase3_11_lora_sweep_analysis.py and phase3_15_paper_figures.py. Those are
# # 7 classes SCORED OUT OF AN 8-WAY MODEL - macro-F1 recomputed after dropping
# # the 2-test-file hyrax class from an 8-output classifier. Chance there is
# # 1/8 = 0.125, the model had an 8th logit competing for probability mass, and
# # its encoder was adapted on hyrax audio.
# #
# # Different label spaces. Different chance levels. Different training data.
# # NEVER place them in the same column, and NEVER compute a delta between
# # them. Wherever both appear, the caveat must appear with them.
# ############################################################################
#
# ISOLATION. Reads the 7-class manifest dir, writes to zero_shot_species7/.
# The existing outputs/phase3/zero_shot/species_id/ 8-class results are not
# touched, and the 8-class species sweep is NOT being re-run.
#
# METHOD-NEUTRAL. These baselines describe the FROZEN encoder, so they are the
# reference point for every adaptation method on this task (LoRA and the
# first-4-layers port alike). Nothing here is specific to either.
#
# NOTE ON WINDOWING. phase3_03 windows internally with its own defaults and
# does not share phase3_10's float16 cache, so A3 is not a prerequisite for
# this job. The two can run concurrently.
#
# SUBMIT:
#   sbatch run_phase3_species7_zero_shot.sh
#   bash run_phase3_species7_zero_shot.sh --list     # show the job table

set -e

MODELS=("wav2vec2_base" "wav2vec2_base_960h" "hubert_base" "xls_r" "wavlm" "ecapa_tdnn")
N_JOBS=${#MODELS[@]}

if [ "${1:-}" = "--list" ]; then
    echo "Job table: $N_JOBS tasks  ->  --array=0-$((N_JOBS - 1))%2"
    for i in "${!MODELS[@]}"; do
        printf "%5d  %s\n" "$i" "${MODELS[$i]}"
    done
    exit 0
fi

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

export HF_HOME=$WORK/hf_cache/huggingface

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate
mkdir -p logs

MANIFEST_DIR="outputs/phase3/manifests_species7"
OUT_ROOT="outputs/phase3/zero_shot_species7"

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$N_JOBS" ]; then
    echo "ERROR: array index $IDX is outside the job table (0-$((N_JOBS - 1)))."
    exit 1
fi
MODEL="${MODELS[$IDX]}"
OUT_DIR="$OUT_ROOT/$MODEL"

echo "========================================"
echo "PHASE A - 7-CLASS ZERO-SHOT (frozen encoder)"
echo "Array task : $IDX / $((N_JOBS - 1))"
echo "Model      : $MODEL"
echo "Manifest   : $MANIFEST_DIR/species_id.json  (7 classes, hyrax EXCLUDED)"
echo "Output     : $OUT_DIR"
echo "Chance     : 1/7 = 0.1429"
echo "Started    : $(date)"
echo "========================================"
echo ""
echo "!! GENUINE 7-WAY numbers. NOT comparable to the 8-class task's 'f1_7' /"
echo "!! 'test_f1_macro_7cls' columns, which are 7 classes scored out of an"
echo "!! 8-WAY model (chance 1/8, encoder adapted on hyrax audio). Different"
echo "!! label spaces - never same-column them, never delta them."
echo ""

if [ ! -f "$MANIFEST_DIR/species_id.json" ]; then
    echo "ERROR: 7-class manifest not found: $MANIFEST_DIR/species_id.json"
    exit 1
fi

if [ -f "$OUT_DIR/results.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP: $OUT_DIR/results.json already exists. Set FORCE=1 to re-run."
    exit 0
fi

mkdir -p "$OUT_DIR"

# phase3_03 reads num_classes and the label map straight from the manifest and
# accepts both overrides, so it needs no code change for the 7-class task.
python scripts/phase3_03_zero_shot_evaluation.py \
    --model "$MODEL" \
    --task species_id \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$OUT_DIR" \
    --log-tag species7

# Stamp the caveat and the label space into the results file itself, so the
# number can never be read without them.
python - "$OUT_DIR/results.json" "$MANIFEST_DIR/species_id.json" <<'PY'
import json, sys
res_path, man_path = sys.argv[1], sys.argv[2]
with open(res_path) as f:
    res = json.load(f)
with open(man_path) as f:
    man = json.load(f)

assert res['num_classes'] == 7, f"expected 7 classes, got {res['num_classes']}"
assert 'hyrax' not in res['class_names'], "hyrax present in class_names"

res['label_space'] = '7way_hyrax_excluded'
res['chance_f1_macro'] = 1.0 / 7
res['excluded_species'] = man.get('excluded_species')
res['comparability_note'] = man['comparability_note']
with open(res_path, 'w') as f:
    json.dump(res, f, indent=2)

t = res['test_metrics']
print(f"\nStamped label_space=7way_hyrax_excluded into {res_path}")
print(f"  7-way test macro-F1 : {t['f1_macro']:.4f}")
print(f"  7-way test accuracy : {t['accuracy']:.4f}")
print(f"  chance (1/7)        : {1/7:.4f}")
print("\n  " + res['comparability_note'])
PY

echo ""
echo "========================================"
echo "Task $IDX ($MODEL) complete: $(date)"
echo "Use these as --baseline-f1 for 7-class adaptation runs (any method)."
echo "========================================"
