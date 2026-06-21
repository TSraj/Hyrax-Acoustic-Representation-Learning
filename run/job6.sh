#!/bin/bash -l

#SBATCH --job-name=p2-samplerate
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tanver.s.raj@fau.de

# -----------------------------------------------------------
# Environment
# -----------------------------------------------------------

unset SLURM_EXPORT_ENV
cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.12-conda
source venv/bin/activate

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# -----------------------------------------------------------
# Stage 6: Sampling Rate Experiment (ResNet-18 on mel spectrograms)
# Tests: Picidae + Wetlands Bird (both datasets)
# Needs:   outputs/phase2/manifests/  (from job1)
# Outputs: outputs/phase2/sampling_rate_experiment/{picidae,wetlands_bird}/
# -----------------------------------------------------------

echo "========================================"
echo "Stage 6: Sampling Rate Experiment"
echo "Started: $(date)"
echo "========================================"

echo "Running sampling rate experiment on Picidae dataset..."
python3 scripts/phase2_06_sampling_rate_experiment.py --dataset picidae
echo ">>> Picidae experiment complete: $(date)"

echo ""
echo "Running sampling rate experiment on Wetlands Bird dataset..."
python3 scripts/phase2_06_sampling_rate_experiment.py --dataset wetlands_bird
echo ">>> Wetlands Bird experiment complete: $(date)"

echo ""
echo "Stage 6 complete: $(date)"
