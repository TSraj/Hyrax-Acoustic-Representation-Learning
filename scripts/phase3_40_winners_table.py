#!/usr/bin/env python3
"""
Phase 3 - Step 40: the encoder table and the two fine-tuning winners.

THE SELECTION RULE (fixed 2026-10-11, BEFORE the AUC rerun came back)
--------------------------------------------------------------------
  data      Original audio, leave-one-session-out cross-validation
  metric    UAR (unweighted average recall over the 18 individuals),
            pooled out-of-fold predictions
  winners   1. the best SPEECH encoder: highest UAR among the 12 monolingual
               and multilingual self-supervised models (ECAPA-TDNN is the
               speaker-ID baseline, not a candidate)
            2. the better AVES encoder: aves2_eat_all vs aves2_eat_bio,
               higher UAR
  ties      equal UAR at 4 decimals -> higher AUC
  AUC       a cross-check only. If the AUC order disagrees with the UAR
            order, that is reported next to the pick; it does NOT change it.

Fixed in advance because WavLM Large and AVES 2 EAT (all) are within 0.001
UAR and the order flips on BIODA audio. Choosing the rule after seeing the
numbers would let any of four pairs be argued for.

WHAT IT WRITES (into the CV output directory)
---------------------------------------------
  encoder_table.csv   all 15 encoders: Encoder | Best layer | UAR | AUC,
                      plus the share of folds that chose that layer
  winners.csv         the two picks, the runner-up in each pool, and
                      whether AUC agrees
  encoder_table.md    the same table, with the X0 / O1 note, to send on

USAGE
-----
    python scripts/phase3_40_winners_table.py
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

AUDIO, PROTOCOL = "cv_original", "session_loso"
SPEECH = ["wav2vec2_base", "wav2vec2_base_960h", "hubert_base", "hubert_large",
          "wavlm", "wavlm_large", "data2vec_base", "unispeech_sat",
          "xls_r", "xls_r_1b", "mms_300m", "mhubert_147"]
AVES = ["aves2_eat_all", "aves2_eat_bio"]
BASELINE = ["ecapa_tdnn"]
LABEL = {
    "wav2vec2_base": "wav2vec2 Base", "wav2vec2_base_960h": "wav2vec2 Base-960h",
    "hubert_base": "HuBERT Base", "hubert_large": "HuBERT Large",
    "wavlm": "WavLM Base+", "wavlm_large": "WavLM Large",
    "data2vec_base": "data2vec Base", "unispeech_sat": "UniSpeech-SAT Base+",
    "xls_r": "XLS-R 300M", "xls_r_1b": "XLS-R 1B", "mms_300m": "MMS 300M",
    "mhubert_147": "mHuBERT-147", "aves2_eat_all": "AVES 2 EAT (all)",
    "aves2_eat_bio": "AVES 2 EAT (bio)", "ecapa_tdnn": "ECAPA-TDNN",
}
LEAKY_NOTE = ("X0 and O1 were each recorded in a single session, so their "
              "test bouts share a session with their training bouts. Their "
              "per-individual scores are optimistic, and they are included "
              "in UAR and AUC above.")


def load(result_dir, model):
    j = json.load(open(Path(result_dir) / f"cv_{PROTOCOL}_{model}.json"))
    s = j["summary"]
    if s.get("pooled_auc_mean") is None:
        raise SystemExit(f"{model}: no AUC in the result -- this is the old "
                         f"run; rerun run_phase3_cv.sh first")
    layers = [l for r in j["repeats"] for l in r["layers_chosen"]]
    layer, n = Counter(layers).most_common(1)[0]
    return {"model": model, "encoder": LABEL[model],
            "pool": ("speech" if model in SPEECH else
                     "aves" if model in AVES else "baseline"),
            "best_layer": layer, "layer_share": round(n / len(layers), 2),
            "uar": s["pooled_uar_mean"], "auc": s["pooled_auc_mean"],
            "f1_macro": s["pooled_f1_macro_mean"],
            "leaky": sorted(j.get("split_exceptions", {}))}


def pick(rows):
    """Rank by UAR, AUC only to break an exact tie at 4 decimals."""
    by_uar = sorted(rows, key=lambda r: (-r["uar"], -r["auc"]))
    by_auc = sorted(rows, key=lambda r: -r["auc"])
    win, runner = by_uar[0], by_uar[1]
    return {"pool": win["pool"], "winner": win["model"],
            "winner_uar": win["uar"], "winner_auc": win["auc"],
            "runner_up": runner["model"], "runner_up_uar": runner["uar"],
            "runner_up_auc": runner["auc"],
            "uar_margin": round(win["uar"] - runner["uar"], 4),
            "auc_agrees": by_auc[0]["model"] == win["model"],
            "auc_top": by_auc[0]["model"]}


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default=f"outputs/phase3/{AUDIO}")
    ap.add_argument("--output-dir",
                    default="outputs/phase3/FINAL/12_cross_validation")
    args = ap.parse_args()

    rows = [load(args.result_dir, m) for m in SPEECH + AVES + BASELINE]
    rows.sort(key=lambda r: (-r["uar"], -r["auc"]))
    winners = [pick([r for r in rows if r["pool"] == p])
               for p in ("speech", "aves")]
    win_ids = {w["winner"] for w in winners}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = [{"rank": i + 1, "encoder": r["encoder"], "model": r["model"],
              "pool": r["pool"], "best_layer": r["best_layer"],
              "layer_share": r["layer_share"], "uar": r["uar"],
              "auc": r["auc"], "f1_macro": r["f1_macro"],
              "winner": r["model"] in win_ids}
             for i, r in enumerate(rows)]
    write_csv(table, out / "encoder_table.csv")
    write_csv(winners, out / "winners.csv")

    leaky = sorted({x for r in rows for x in r["leaky"]})
    md = ["# Encoder table: Original audio, leave-one-session-out CV", "",
          "| Encoder | Best layer | UAR | AUC |", "|---|---|---|---|"]
    for r in table:
        name = f"**{r['encoder']}**" if r["winner"] else r["encoder"]
        md.append(f"| {name} | {r['best_layer']} ({r['layer_share']:.0%} of folds) "
                  f"| {r['uar']:.4f} | {r['auc']:.4f} |")
    md += ["", "Chance UAR = 1/18 = 0.0556; chance AUC = 0.5. Best layer = the "
           "layer most often chosen by the inner 3-fold CV across the 15 outer "
           "folds. Bold = fine-tuning winner.", "",
           f"**Note:** {LEAKY_NOTE}" if leaky == ["O1", "X0"] else
           f"**Note:** single-session individuals: {', '.join(leaky)}", "",
           "## Winners (rule fixed before the results: UAR, AUC as cross-check)", ""]
    for w in winners:
        md.append(f"- **{w['pool']}**: {LABEL[w['winner']]} (UAR {w['winner_uar']:.4f}, "
                  f"AUC {w['winner_auc']:.4f}); runner-up {LABEL[w['runner_up']]} "
                  f"(UAR {w['runner_up_uar']:.4f}), margin {w['uar_margin']:.4f}. "
                  + ("AUC agrees." if w["auc_agrees"] else
                     f"AUC disagrees: {LABEL[w['auc_top']]} has the higher AUC."))
    (out / "encoder_table.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {out}/encoder_table.csv, winners.csv, encoder_table.md")


if __name__ == "__main__":
    main()
