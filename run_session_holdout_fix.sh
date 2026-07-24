#!/bin/bash
#SBATCH --job-name=session_holdout_fix
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/session_holdout_fix_%j.out
#SBATCH --error=logs/session_holdout_fix_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

echo "Regenerating manifests with class_weights fix..."
python scripts/phase3_02_create_manifests.py

echo ""
echo "Running session_holdout task for all models..."
for model in wav2vec2_base wav2vec2_base_960h hubert_base xls_r wavlm ecapa_tdnn; do
    echo "  Model: $model"
    python scripts/phase3_03_zero_shot_evaluation.py --model $model --task hyrax_id_session_holdout
done

echo ""
echo "Session holdout fix complete!"
