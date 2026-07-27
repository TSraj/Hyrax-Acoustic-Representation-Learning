#!/bin/bash
#SBATCH --job-name=hyrax_windowed
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --output=logs/hyrax_windowed_%j.out
#SBATCH --error=logs/hyrax_windowed_%j.err

# Run windowed zero-shot evaluation for hyrax_id tasks
# 6 models × 2 tasks = 12 runs with 5s windowing

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

echo "========================================"
echo "HYRAX WINDOWED ZERO-SHOT EVALUATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

MODELS=("wav2vec2_base" "wav2vec2_base_960h" "hubert_base" "xls_r" "wavlm" "ecapa_tdnn")
TASKS=("hyrax_id" "hyrax_id_session_holdout")

echo ""
echo "Running 12 evaluations (6 models × 2 tasks) with 5s windowing..."
echo ""

for task in "${TASKS[@]}"; do
    echo "========================================"
    echo "TASK: $task"
    echo "========================================"

    for model in "${MODELS[@]}"; do
        echo ""
        echo "--- Model: $model ---"
        python scripts/phase3_03_zero_shot_evaluation.py --model $model --task $task
    done
done

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  outputs/phase3/zero_shot/hyrax_id/*/results.json"
echo "  outputs/phase3/zero_shot/hyrax_id/session_holdout/*/results.json"
