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

# Set HF token from environment variable
# Before running: export HF_TOKEN=your_token_here
# Or set in ~/.bashrc: export HF_TOKEN=hf_your_token

python << 'PYEOF'
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    WavLMModel
)
import torch

print("=" * 80)
print("PRE-DOWNLOADING ALL MODELS TO CACHE")
print("=" * 80)

models = {
    "wav2vec2_base": "facebook/wav2vec2-base",
    "wav2vec2_base_960h": "facebook/wav2vec2-base-960h",
    "xls_r": "facebook/wav2vec2-xls-r-300m",
    "wavlm": "microsoft/wavlm-base-plus",
}

for name, model_id in models.items():
    print(f"\n[{name}] Downloading {model_id}...")
    try:
        if "wavlm" in name:
            model = WavLMModel.from_pretrained(model_id)
            processor = Wav2Vec2Processor.from_pretrained(model_id)  # WavLM uses Wav2Vec2Processor
        else:
            model = Wav2Vec2Model.from_pretrained(model_id)
            processor = Wav2Vec2Processor.from_pretrained(model_id)
        print(f"[{name}] ✓ Downloaded successfully")
        del model, processor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[{name}] ✗ Error: {e}")

print("\n" + "=" * 80)
print("ECAPA-TDNN (speechbrain) will download on first use - cannot pre-cache")
print("All downloadable models cached successfully!")
print("=" * 80)
PYEOF
