#!/usr/bin/env python3
"""
Phase 3 - Step 42: the layer plot, from the phase3_41 per-layer CV sweep.

One line per encoder: pooled UAR (5-fold by recording, mean of 5 repeats) at
every layer, with the layer shown as a percentage of depth (0% = the first
hidden state, 100% = the last) so a 13-layer and a 49-layer model share one
axis. Small multiples by family, because 15 lines in one panel cannot be told
apart: speech monolingual (8), speech multilingual (4), and bioacoustic (2).
The bioacoustic panel is also the SPECTROGRAM panel -- AVES 2 EAT reads mel
patches, every speech model reads the raw waveform -- so the panels answer
"do waveform and spectrogram models peak at different depths" directly.
ECAPA-TDNN has a single embedding, not a layer stack; it is drawn as a dashed
reference level in every panel, not as a line.

WHAT IT WRITES (default: outputs/phase3/FINAL/13_cv_layer_curves)
-----------------------------------------------------------------
  cv_layers_<audio>.png      the plot, 300 DPI, one per audio version
  cv_layer_peaks_<audio>.csv one row per encoder: input type, layers, peak
                             layer and its depth, UAR and AUC at the peak,
                             UAR at the last layer
  cv_layer_curves_<audio>.csv every encoder x layer, long format

The peak is the best layer ON THE TEST FOLDS. It describes where the
information sits; it is not a headline score (phase3_37's inner-CV-selected
number is).

USAGE
-----
    python scripts/phase3_42_layer_plot.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 300
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcdc"
# reference categorical order, assigned in fixed order within each panel
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

PANELS = [
    ("Speech, monolingual (waveform input)",
     ["wav2vec2_base", "wav2vec2_base_960h", "hubert_base", "hubert_large",
      "wavlm", "wavlm_large", "data2vec_base", "unispeech_sat"]),
    ("Speech, multilingual (waveform input)",
     ["xls_r", "xls_r_1b", "mms_300m", "mhubert_147"]),
    ("Bioacoustic (spectrogram input)",
     ["aves2_eat_all", "aves2_eat_bio"]),
]
BASELINE = "ecapa_tdnn"
INPUT = {m: "waveform" for _, ms in PANELS[:2] for m in ms}
INPUT.update({"aves2_eat_all": "spectrogram", "aves2_eat_bio": "spectrogram",
              "ecapa_tdnn": "spectrogram"})
FAMILY = {m: f for f, ms in zip(["monolingual", "multilingual", "bioacoustic"],
                                [p[1] for p in PANELS]) for m in ms}
FAMILY[BASELINE] = "speaker-ID (baseline)"
LABEL = {
    "wav2vec2_base": "wav2vec2 Base", "wav2vec2_base_960h": "wav2vec2 Base-960h",
    "hubert_base": "HuBERT Base", "hubert_large": "HuBERT Large",
    "wavlm": "WavLM Base+", "wavlm_large": "WavLM Large",
    "data2vec_base": "data2vec Base", "unispeech_sat": "UniSpeech-SAT Base+",
    "xls_r": "XLS-R 300M", "xls_r_1b": "XLS-R 1B", "mms_300m": "MMS 300M",
    "mhubert_147": "mHuBERT-147", "aves2_eat_all": "AVES 2 EAT (all)",
    "aves2_eat_bio": "AVES 2 EAT (bio)", "ecapa_tdnn": "ECAPA-TDNN",
}
AUDIO_LABEL = {"original": "Original audio", "bioda": "BIODA denoised audio"}


def depth(layer, n_layers):
    return 100.0 * layer / (n_layers - 1) if n_layers > 1 else 100.0


def load(result_dir):
    out = {}
    for f in sorted(Path(result_dir).glob("cv_layers_by_file_5fold_*.json")):
        j = json.load(open(f))
        if len(j["layers"]) != j["n_layers"]:
            raise SystemExit(f"{f.name}: {len(j['layers'])} of {j['n_layers']} "
                             f"layers -- incomplete run")
        out[j["model"]] = j
    expected = {m for _, ms in PANELS for m in ms} | {BASELINE}
    missing = expected - set(out)
    if missing:
        raise SystemExit(f"{result_dir}: missing {sorted(missing)}")
    return out


def tables(runs):
    peaks, long = [], []
    for model in [m for _, ms in PANELS for m in ms] + [BASELINE]:
        j = runs[model]
        n = j["n_layers"]
        for l in j["layers"]:
            long.append({"model": model, "encoder": LABEL[model],
                         "input": INPUT[model], "layer": l["layer"],
                         "depth_pct": round(depth(l["layer"], n), 1),
                         "uar": l["uar_mean"], "uar_sd": l["uar_sd"],
                         "auc": l["auc_macro_ovr_mean"],
                         "f1_macro": l["f1_macro_mean"]})
        best = max(j["layers"], key=lambda r: (r["uar_mean"], -r["layer"]))
        last = j["layers"][-1]
        peaks.append({
            "encoder": LABEL[model], "model": model, "family": FAMILY[model],
            "input": INPUT[model], "n_layers": n,
            "peak_layer": best["layer"],
            "peak_depth_pct": round(depth(best["layer"], n), 1),
            "peak_uar": best["uar_mean"], "peak_auc": best["auc_macro_ovr_mean"],
            "last_layer_uar": last["uar_mean"],
            "drop_peak_to_last": round(best["uar_mean"] - last["uar_mean"], 4),
        })
    return peaks, long


def plot(runs, audio, path):
    chance = runs[BASELINE]["chance_uar"]
    base_uar = runs[BASELINE]["layers"][0]["uar_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, (title, models) in zip(axes, PANELS):
        for i, m in enumerate(models):
            j = runs[m]
            x = [depth(l["layer"], j["n_layers"]) for l in j["layers"]]
            y = [l["uar_mean"] for l in j["layers"]]
            ax.plot(x, y, color=SERIES[i], lw=2, label=LABEL[m], zorder=3)
            k = int(np.argmax(y))
            # the peak, ringed in the surface colour so it reads on overlap
            ax.plot(x[k], y[k], marker=MARKERS[i], ms=8, color=SERIES[i],
                    mec="white", mew=1.5, zorder=4)
        ax.axhline(base_uar, color=MUTED, lw=1.2, ls="--", zorder=2)
        ax.axhline(chance, color=MUTED, lw=1, ls=":", zorder=2)
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.legend(loc="best", fontsize=8, frameon=False,
                  labelcolor=INK, handlelength=1.6)
        ax.set_xlabel("layer depth (0% = first hidden state, 100% = last layer)",
                      fontsize=9, color=MUTED)
    # reference-line labels once, at the left edge of the first panel
    axes[0].text(1, base_uar, f"ECAPA-TDNN {base_uar:.3f}", fontsize=8,
                 color=MUTED, va="bottom")
    axes[0].text(1, chance, f"chance {chance:.3f}", fontsize=8,
                 color=MUTED, va="bottom")
    axes[0].set_ylabel("UAR, 18 individuals\n(5-fold by recording, mean of 5 repeats)",
                       fontsize=9, color=MUTED)
    lo = min(chance, base_uar) - 0.03
    hi = max(l["uar_mean"] for j in runs.values() for l in j["layers"]) + 0.05
    axes[0].set_ylim(max(0, lo), hi)
    fig.suptitle(f"Hyrax individual ID by layer depth - frozen encoders, "
                 f"{AUDIO_LABEL[audio]}  (marker = peak layer)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", default=["original", "bioda"])
    ap.add_argument("--results-root", default="outputs/phase3")
    ap.add_argument("--output-dir",
                    default="outputs/phase3/FINAL/13_cv_layer_curves")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        runs = load(Path(args.results_root) / f"cv_{audio}" / "layers")
        peaks, long = tables(runs)
        write_csv(peaks, out / f"cv_layer_peaks_{audio}.csv")
        write_csv(long, out / f"cv_layer_curves_{audio}.csv")
        plot(runs, audio, out / f"cv_layers_{audio}.png")
        print(f"\n{AUDIO_LABEL[audio]}: peak layer per encoder")
        for p in sorted(peaks, key=lambda r: -r["peak_uar"]):
            print(f"  {p['encoder']:22s} {p['input']:11s} layer {p['peak_layer']:2d}"
                  f"/{p['n_layers'] - 1:<2d} ({p['peak_depth_pct']:5.1f}%)  "
                  f"UAR {p['peak_uar']:.4f}  AUC {p['peak_auc']:.4f}")
        for kind in ("waveform", "spectrogram"):
            d = [p["peak_depth_pct"] for p in peaks
                 if p["input"] == kind and p["model"] != BASELINE]
            print(f"  {kind:11s} peaks: median depth {np.median(d):.0f}% "
                  f"(range {min(d):.0f}-{max(d):.0f}%, n={len(d)})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
