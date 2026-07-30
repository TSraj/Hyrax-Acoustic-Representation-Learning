#!/usr/bin/env python3
"""
Phase 3 - SQ4 step 1: extract acoustic predictors per individual.

Unit of analysis is the individual from the Phase 2 pooled 69-way task. For each
individual we sample up to --max-files of that individual's own recordings and
summarise their acoustic properties.

Features are computed at the file's NATIVE sample rate, not at 16 kHz. The
models only ever receive 16 kHz audio, so any energy above 8 kHz is discarded by
resampling - `frac_energy_above_8k` measures exactly how much of a species'
signal is thrown away before the encoder sees it, which is a candidate
explanation for transfer failure and would be invisible if we measured at 16k.

Outputs one row per individual to acoustic_predictors.csv.
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

DATASETS = ['anuraset', 'bengalese_finch', 'macaque', 'marmoset',
            'picidae', 'wetlands_bird', 'zebra_finch']

# Coarse taxonomy: 7 datasets against N~45 would overfit; 3 levels is tractable.
TAXON = {
    'anuraset': 'amphibian',
    'bengalese_finch': 'bird',
    'picidae': 'bird',
    'wetlands_bird': 'bird',
    'zebra_finch': 'bird',
    'macaque': 'primate',
    'marmoset': 'primate',
}


def resolve(file_path):
    p = Path(file_path)
    if not p.exists() and not str(file_path).startswith('outputs/'):
        p = Path("Data") / file_path
    return p


def file_features(path, max_seconds=30.0):
    """Acoustic summary of one recording, at its native sample rate."""
    audio, sr = librosa.load(str(path), sr=None, mono=True, duration=max_seconds)
    if len(audio) < 512:
        return None

    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    power = S ** 2
    total = power.sum()
    if total <= 0:
        return None

    centroid = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    roll95 = float(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95).mean())
    roll05 = float(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.05).mean())

    # Energy the 16 kHz pipeline discards
    above = float(power[freqs > 8000].sum() / total) if (freqs > 8000).any() else 0.0

    # SNR proxy: loud frames vs quiet frames. There is no ground-truth noise
    # floor for these corpora, so this is an estimate, not a measurement.
    frame_e = power.sum(axis=0)
    hi = np.percentile(frame_e, 95)
    lo = np.percentile(frame_e, 10)
    snr = float(10 * np.log10((hi + 1e-12) / (lo + 1e-12)))

    return {
        'centroid_hz': centroid,
        'rolloff95_hz': roll95,
        'bandwidth_hz': roll95 - roll05,
        'frac_energy_above_8k': above,
        'snr_proxy_db': snr,
        'duration_s': len(audio) / sr,
        'native_sr': sr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/phase2V2/manifests/pooled_manifest.json")
    ap.add_argument("--out", default="outputs/phase3/sq4/acoustic_predictors.csv")
    ap.add_argument("--max-files", type=int, default=20,
                    help="Recordings sampled per individual")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    per = defaultdict(list)
    for split in ['train', 'val', 'test']:
        for it in man.get(split, []):
            per[it['individual']].append(it['file'])

    rng = np.random.default_rng(args.seed)
    rows = []

    for ind in tqdm(sorted(per), desc="individuals"):
        files = sorted(set(per[ind]))
        if len(files) > args.max_files:
            files = [files[i] for i in sorted(rng.choice(len(files), args.max_files, replace=False))]

        feats, failed = [], 0
        for f in files:
            p = resolve(f)
            try:
                r = file_features(p)
                if r:
                    feats.append(r)
                else:
                    failed += 1
            except Exception:
                failed += 1

        if not feats:
            print(f"  WARNING: no usable audio for {ind}")
            continue

        fdf = pd.DataFrame(feats)
        dataset = next(d for d in DATASETS if ind.startswith(d + '_'))
        row = {'individual': ind, 'dataset': dataset, 'taxon': TAXON[dataset],
               'n_files_used': len(feats), 'n_files_failed': failed,
               'n_files_total': len(per[ind])}
        for c in ['centroid_hz', 'rolloff95_hz', 'bandwidth_hz',
                  'frac_energy_above_8k', 'snr_proxy_db', 'duration_s']:
            row[c] = fdf[c].mean()
        row['duration_s_median'] = fdf['duration_s'].median()
        row['native_sr'] = int(fdf['native_sr'].mode().iloc[0])
        rows.append(row)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(out)} individuals)")
    print(out.groupby('dataset')[['centroid_hz', 'bandwidth_hz',
                                  'frac_energy_above_8k', 'snr_proxy_db',
                                  'duration_s']].mean().round(3).to_string())


if __name__ == "__main__":
    main()
