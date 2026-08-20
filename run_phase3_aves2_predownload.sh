#!/bin/bash
#SBATCH --job-name=aves2_predownload
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/aves2_predownload_%j.out
#SBATCH --error=logs/aves2_predownload_%j.err

# Pre-download AVES 2 (EAT) so the probe jobs never hit the network.
#
# This model pulls TWO artefacts from TWO different caches, which is why the
# existing pre_download_models.sh does not cover it:
#
#   $HF_HOME/hub/models--worstchan--EAT-base_epoch30_pretrain
#       the EAT backbone (~343 MB) AND its remote modelling code, which is
#       fetched with trust_remote_code=True
#   $ESP_CACHE_HOME/esp-aves2-eat-bio-*.safetensors
#       the AVES 2 bio-pretrained weights (~358 MB), resolved from
#       hf://EarthSpeciesProject/esp-aves2-eat-bio via fsspec -- NOT the HF hub
#       cache, so warming the hub alone is not enough
#
# QUOTA: the second cache is NOT controlled by HF_HOME. avex resolves it as
#   ESP_CACHE_HOME if set, else Path.home()/".cache"/"esp"   (avex/utils/utils.py)
# so on a cluster where HOME is quota-limited, setting HF_HOME alone still drops
# ~358 MB into HOME. Both variables are exported below, and Slurm does not
# reliably source .bashrc, so they are set here rather than assumed.
#
# It also runs one real forward pass, which is what actually proves the
# transformers>=5 shim works on this node's package versions rather than just
# that the bytes are on disk.
#
# RUN THIS ONCE, and wait for it to finish, before submitting the probe array.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJECT_DIR"

# ---------------------------------------------------------------- caches
# Slurm does not reliably source .bashrc, so an interactive-shell HF_HOME does
# not survive into the job. Set both caches explicitly, and refuse to run
# rather than silently defaulting to a quota-limited HOME.
if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so the model caches would land on HOME."
    echo "       Set WORK (or edit HF_HOME/ESP_CACHE_HOME below) and resubmit."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export ESP_CACHE_HOME=${ESP_CACHE_HOME:-$WORK/hf_cache/esp}
mkdir -p "$HF_HOME" "$ESP_CACHE_HOME"

# HuggingFace serves large files through Xet (cas-server.xethub.hf.co), which
# this cluster's compute nodes cannot reach -- small .py files download fine,
# then the weights fail with "CAS Client Error". Fall back to the classic CDN.
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

echo "=============================================================="
echo "AVES 2 (EAT) PRE-DOWNLOAD"
echo "=============================================================="
echo "node           : $(hostname)"
echo "started        : $(date)"
echo "HF_HOME        : $HF_HOME"
echo "ESP_CACHE_HOME : $ESP_CACHE_HOME"
echo ""

case "$HF_HOME:$ESP_CACHE_HOME" in
    */home/hpc/*|/home/hpc/*)
        echo "REFUSING: a cache path is under /home/hpc (quota-limited)."
        exit 1 ;;
esac
echo "free space on \$WORK:"
df -h "$WORK" | tail -1
echo ""

module load cuda 2>/dev/null || true

# ---------------------------------------------------------------- venv
# avex requires torch>=2.5. The project venv is older than that, and upgrading
# it in place would change the torch underneath every already-published number
# in this repo. So AVES gets its OWN venv, on $WORK (a torch install is ~3 GB
# and HOME is quota-limited). The project venv is never touched.
#
# The AVES results are a separate output tree that recomputes none of the
# existing cells, so an isolated environment costs nothing in comparability
# beyond the probe itself -- a full-batch linear fit, where the torch version
# is not a meaningful source of difference.
AVEX_VENV=${AVEX_VENV:-$WORK/venv_avex}

# pip caches downloaded wheels in ~/.cache/pip by default. The torch wheels
# alone are several GB, which lands straight on a quota-limited HOME.
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORK/pip_cache}
mkdir -p "$PIP_CACHE_DIR"

if [[ ! -x "$AVEX_VENV/bin/python" ]]; then
    echo "creating AVES venv at $AVEX_VENV (one-off, a few minutes)"
    python -m venv "$AVEX_VENV"
    source "$AVEX_VENV/bin/activate"
    pip install --upgrade pip -q
    # let pip pick a consistent torch/torchvision pair >= avex's floor
    pip install "torch>=2.5" torchvision
    pip install avex
    # repo-side deps used by phase3_24 / _28 / _29
    pip install numpy scipy scikit-learn librosa soundfile pyyaml tqdm transformers
else
    echo "reusing AVES venv at $AVEX_VENV"
    source "$AVEX_VENV/bin/activate"
fi

# Keep this check to what actually matters: both packages import. Anything
# fancier (reaching for avex's version attribute) risks failing on a cosmetic
# detail and killing a job whose environment is perfectly fine.
python - <<'PYEOF' || { echo "FATAL: avex venv is not usable"; exit 1; }
import torch
import avex
print(f"avex venv OK: torch {torch.__version__}, avex imported")
PYEOF

python -u - <<'PYEOF'
import time

import numpy as np
import torch

t0 = time.time()
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)

import sys
sys.path.insert(0, "scripts")
from phase3_28_avex_extractor import (AVEX_MODEL_ID, CANVAS_SECONDS, NUM_LAYERS,
                                      AvesLayerExtractor)


class L:
    def info(self, m): print(f"  {m}", flush=True)
    def warning(self, m): print(f"  WARN {m}", flush=True)


print(f"downloading + loading {AVEX_MODEL_ID} ...", flush=True)
ex = AvesLayerExtractor("aves2_eat_bio", None, L(), batch_size=2)

# a real forward pass: proves the shim, the hooks and the layer mapping all
# work on THIS node, not merely that the files are cached
rng = np.random.default_rng(0)
out = ex.embed_many([rng.normal(0, 0.05, 16000).astype(np.float32),
                     rng.normal(0, 0.05, 40000).astype(np.float32)])
ex.close()

assert out.shape == (2, NUM_LAYERS, 768), f"unexpected shape {out.shape}"
assert np.isfinite(out).all(), "non-finite embeddings"
print(f"\nforward OK: {out.shape}, {NUM_LAYERS} layers, canvas {CANVAS_SECONDS:.2f}s",
      flush=True)
print(f"elapsed {time.time() - t0:.0f}s", flush=True)
PYEOF

echo ""
echo "cache locations now warm:"
du -sh "$HF_HOME/hub/models--worstchan--EAT-base_epoch30_pretrain" 2>/dev/null \
    || echo "  (EAT backbone not found - check for errors above)"
du -sh "$ESP_CACHE_HOME" 2>/dev/null \
    || echo "  (AVES weights not found - check for errors above)"

# Prove where the bytes actually landed, rather than trusting the env vars.
echo ""
echo "resolved repo paths (all must be OFF /home/hpc):"
python - <<'PYEOF'
import sys
from huggingface_hub import scan_cache_dir
bad = []
for r in scan_cache_dir().repos:
    print(f"  {r.repo_id}  ->  {r.repo_path}")
    if "/home/hpc/" in str(r.repo_path):
        bad.append(r.repo_id)
import os, glob
for p in glob.glob(os.path.join(os.environ.get("ESP_CACHE_HOME", ""), "*.safetensors")):
    print(f"  esp-aves2 weights  ->  {p}")
    if "/home/hpc/" in p:
        bad.append(p)
if bad:
    print(f"\nFATAL: these landed on quota-limited HOME: {bad}", file=sys.stderr)
    sys.exit(1)
print("\nall caches are off /home/hpc")
PYEOF

echo ""
echo "finished: $(date)"
echo "Safe to submit run_phase3_aves2_zeroshot.sh now."
