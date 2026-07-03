#!/bin/bash
#SBATCH --job-name=pre_download_models
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/pre_download_%j.out
#SBATCH --error=logs/pre_download_%j.err

# Pre-download all models to HF cache to prevent parallel download conflicts

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

echo "Starting pre-download at $(date)"
echo "HF_TOKEN set: ${HF_TOKEN:0:10}..."
echo ""

# Set HF token from environment variable
# Before running: export HF_TOKEN=your_token_here
# Or set in ~/.bashrc: export HF_TOKEN=hf_your_token

# Use unbuffered Python for immediate output
python -u << 'PYEOF'
import sys
import os
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
log("="*80)
log("PRE-DOWNLOADING MODELS TO CACHE")
log("="*80)

# Check HF token
token = os.getenv('HF_TOKEN')
if token:
    log(f"✓ HF_TOKEN found: {token[:10]}...")
else:
    log("⚠ WARNING: HF_TOKEN not set - downloads may be rate-limited")

log("")

from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    WavLMModel
)
import torch

models = {
    "wav2vec2_base": "facebook/wav2vec2-base",
    "wav2vec2_base_960h": "facebook/wav2vec2-base-960h",
    "xls_r": "facebook/wav2vec2-xls-r-300m",
    "wavlm": "microsoft/wavlm-base-plus",
}

for i, (name, model_id) in enumerate(models.items(), 1):
    log(f"[{i}/4] {name}: Starting download from {model_id}")
    try:
        log(f"  → Downloading model...")
        if "wavlm" in name:
            model = WavLMModel.from_pretrained(model_id, trust_remote_code=False)
        else:
            model = Wav2Vec2Model.from_pretrained(model_id, trust_remote_code=False)

        log(f"  → Downloading processor...")
        processor = Wav2Vec2Processor.from_pretrained(model_id, trust_remote_code=False)

        log(f"  ✓ {name} downloaded successfully")

        # Cleanup
        del model, processor
        torch.cuda.empty_cache()

    except Exception as e:
        log(f"  ✗ {name} failed: {str(e)[:100]}")
        sys.exit(1)

    log("")

log("="*80)
log("NOTE: ECAPA-TDNN (speechbrain) downloads on first use")
log("All transformers models cached successfully!")
log("="*80)
PYEOF

echo ""
echo "Pre-download completed at $(date)"
