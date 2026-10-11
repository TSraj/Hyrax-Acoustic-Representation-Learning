#!/usr/bin/env python3
"""
Phase 3 - Step 39: cross-validation vs the single-split experiments.

Puts experiments 10 (single split, 15 models), 11 (same, after the AVES 2
loading fix) and 12 (cross-validation) side by side, one row per
model x audio x protocol.

READ THE RAW F1 WITH CARE
-------------------------
The single splits and the CV do not score the same task. The single splits
keep only the individuals that fit one held-out split (10 classes by file,
8 by session); the CV keeps all 18. Chance moves from 0.10 / 0.125 to 0.056,
so raw F1 drops for that reason alone. The columns that compare like with like are:
  *_lift   F1 divided by chance
  *_rank   rank among the 15 models within the same audio x protocol
The single-split layer is also the best layer on the test split, while the CV
chooses its layer inside the training folds, so the single-split F1 is
optimistic on top of that.

WHAT IT WRITES (into the CV output directory)
---------------------------------------------
  cv_vs_single_split.csv        one row per model x audio x protocol
  cv_vs_single_split_ranks.csv  one row per audio x protocol: Spearman rank
                                correlation 11 vs 12 and the top model in each

USAGE
-----
    python scripts/phase3_39_cv_vs_single_split.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np

FINAL = Path("outputs/phase3/FINAL")
PROTOCOL_OF_SPLIT = {"file": "by_file_5fold", "session": "session_loso"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def single_split(path):
    """(model, audio, protocol) -> scores, from a summary_wide.csv."""
    out = {}
    for r in read_csv(path):
        protocol = PROTOCOL_OF_SPLIT[r["split"]]
        for audio in ("bioda", "original"):
            out[(r["model"], audio, protocol)] = {
                "n_classes": int(r["n_classes"]),
                "chance": float(r["chance_f1"]),
                "f1": float(r[f"bout_{audio}_f1"]),
                "layer": int(r[f"bout_{audio}_layer"]),
                "f1_agg": float(r[f"agg_{audio}_f1"]),
            }
    return out


def ranks(values):
    """Rank 1 = highest; ties share the lower rank."""
    order = sorted(values, key=values.get, reverse=True)
    return {k: 1 + sum(values[o] > values[k] for o in order) for k in values}


def spearman(a, b):
    keys = sorted(a)
    x = np.array([a[k] for k in keys], float)
    y = np.array([b[k] for k in keys], float)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp10", default=FINAL / "10_audio_comparison_15models/summary_wide.csv")
    ap.add_argument("--exp11", default=FINAL / "11_audio_comparison_avesfix/summary_wide.csv")
    ap.add_argument("--cv-summary", default=FINAL / "12_cross_validation/cv_summary.csv")
    ap.add_argument("--output-dir", default=FINAL / "12_cross_validation")
    args = ap.parse_args()

    e10, e11 = single_split(args.exp10), single_split(args.exp11)
    cv = {(r["model"], r["audio"], r["protocol"]): r for r in read_csv(args.cv_summary)}

    missing = set(cv) ^ set(e11)
    if missing:
        raise SystemExit(f"model/audio/protocol mismatch between 11 and CV: {sorted(missing)}")

    rows, rank_rows = [], []
    for audio in ("original", "bioda"):
        for protocol in ("session_loso", "by_file_5fold"):
            keys = sorted(k for k in cv if k[1:] == (audio, protocol))
            r10 = ranks({k: e10[k]["f1"] for k in keys})
            r11 = ranks({k: e11[k]["f1"] for k in keys})
            r12 = ranks({k: float(cv[k]["f1_macro"]) for k in keys})
            for k in sorted(keys, key=r12.get):
                c, s10, s11 = cv[k], e10[k], e11[k]
                f12, ch12 = float(c["f1_macro"]), float(c["chance"])
                rows.append({
                    "model": k[0], "label": c["label"], "family": c["family"],
                    "audio": audio, "protocol": protocol,
                    "n_classes_single": s11["n_classes"], "n_classes_cv": int(c["n_individuals"]),
                    "chance_single": s11["chance"], "chance_cv": ch12,
                    "f1_exp10": s10["f1"], "f1_exp11": s11["f1"], "f1_exp12_cv": f12,
                    "f1_cv_sd_folds": float(c["f1_sd_across_folds"]),
                    "f1_cv_cohort100": float(c["f1_cohort100"]),
                    "lift_exp11": round(s11["f1"] / s11["chance"], 2),
                    "lift_exp12_cv": round(f12 / ch12, 2),
                    "rank_exp10": r10[k], "rank_exp11": r11[k], "rank_exp12_cv": r12[k],
                    "rank_change_11_to_12": r11[k] - r12[k],
                    "layer_exp11": s11["layer"], "layer_exp12_cv_modal": int(c["modal_layer"]),
                    "f1_agg_exp11": s11["f1_agg"], "f1_agg_exp12_cv": float(c["f1_aggregated"]),
                    "exp10_differs_from_11": s10["f1"] != s11["f1"],
                })
            top = lambda r: min(r, key=r.get)[0]
            rank_rows.append({
                "audio": audio, "protocol": protocol,
                "spearman_rank_11_vs_12": round(spearman(r11, r12), 3),
                "top_exp11": top(r11), "top_exp12_cv": top(r12),
                "top3_exp11": " / ".join(k[0] for k in sorted(keys, key=r11.get)[:3]),
                "top3_exp12_cv": " / ".join(k[0] for k in sorted(keys, key=r12.get)[:3]),
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "cv_vs_single_split.csv")
    write_csv(rank_rows, out / "cv_vs_single_split_ranks.csv")
    print(f"wrote {out / 'cv_vs_single_split.csv'} ({len(rows)} rows)")
    print(f"wrote {out / 'cv_vs_single_split_ranks.csv'}")


if __name__ == "__main__":
    main()
