#!/usr/bin/env python3
"""
Phase 3 - Step 38: the cross-validation report.

Reads the per-cell JSONs written by phase3_37 and produces the tables the
supervisor asked for: fold-averaged scores for every encoder, both audio
versions, both protocols, plus a per-individual breakdown so the thin-data
animals are visible instead of buried in a macro average.

WHAT IT WRITES
--------------
  cv_summary.csv                one row per model x audio x protocol: pooled
                                macro-F1 with its SD, accuracy, the
                                aggregated-by-recording score, the >=100-bout
                                cohort score, and the layer the inner folds
                                chose
  cv_family_<protocol>.csv      the same rows grouped into monolingual /
                                multilingual / bioacoustic, both audio
                                versions side by side
  cv_per_individual_<protocol>.csv
                                model x individual F1, with each animal's
                                sessions, recordings and leaky_split flag
  cv_individual_difficulty_<protocol>.csv
                                one row per individual, averaged over the 15
                                encoders: which animals are actually hard,
                                and how much data each stands on
  cv_family_<protocol>.png      family comparison, 300 DPI
  cv_individuals_<protocol>.png per-individual F1, 300 DPI, leaky animals
                                marked

TWO NUMBERS, NOT ONE
--------------------
`f1_macro` is the mean over all 18 individuals. `f1_cohort100` is the mean
over the 11 with at least 100 bouts, computed from the SAME 18-way
predictions. Both are reported because the first is dominated by classes with
16-38 bouts and the second hides that those animals exist. Quoting either one
alone invites the obvious objection.

THE DUPLICATE GUARD
-------------------
Two models that produce identical pooled metrics are the same encoder, not two
encoders. That happened here once already -- avex silently failed to load the
AVES 2 checkpoints and eat_bio and eat_all ran the same backbone for eight
cells -- and nothing in the tables said so. This refuses to build a report
where it recurs.

USAGE
-----
    python scripts/phase3_38_cv_report.py \
        --result-dir outputs/phase3/cv_bioda \
        --result-dir outputs/phase3/cv_original \
        --output-dir outputs/phase3/FINAL/12_cross_validation
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 300
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dcdcdc"

AUDIO_FROM_SUBDIR = {"BIODA/denoised": "bioda", "Audio": "original"}
AUDIO_LABEL = {"bioda": "BIODA denoised", "original": "Original"}
PROTOCOL_LABEL = {"session_loso": "leave-one-session-out",
                  "by_file_5fold": "5-fold by recording"}

FAMILY = {
    "wav2vec2_base": "monolingual", "wav2vec2_base_960h": "monolingual",
    "hubert_base": "monolingual", "hubert_large": "monolingual",
    "wavlm": "monolingual", "wavlm_large": "monolingual",
    "data2vec_base": "monolingual", "unispeech_sat": "monolingual",
    "xls_r": "multilingual", "xls_r_1b": "multilingual",
    "mms_300m": "multilingual", "mhubert_147": "multilingual",
    "aves2_eat_bio": "bioacoustic", "aves2_eat_all": "bioacoustic",
    "ecapa_tdnn": "speaker-ID (baseline)",
}
LABEL = {
    "wav2vec2_base": "wav2vec2 Base", "wav2vec2_base_960h": "wav2vec2 Base-960h",
    "hubert_base": "HuBERT Base", "hubert_large": "HuBERT Large",
    "wavlm": "WavLM Base+", "wavlm_large": "WavLM Large",
    "data2vec_base": "data2vec Base", "unispeech_sat": "UniSpeech-SAT Base+",
    "xls_r": "XLS-R 300M", "xls_r_1b": "XLS-R 1B", "mms_300m": "MMS 300M",
    "mhubert_147": "mHuBERT-147", "aves2_eat_bio": "AVES 2 EAT (bio)",
    "aves2_eat_all": "AVES 2 EAT (all)", "ecapa_tdnn": "ECAPA-TDNN",
}
FAMILY_ORDER = ["monolingual", "multilingual", "bioacoustic",
                "speaker-ID (baseline)"]
FAMILY_COLOUR = {"monolingual": "#0072B2", "multilingual": "#E69F00",
                 "bioacoustic": "#009E73",
                 "speaker-ID (baseline)": "#999999"}


def load_cells(result_dirs, logger=print):
    """-> {(model, audio, protocol): result}. Missing dirs are reported."""
    cells, skipped = {}, []
    for d in result_dirs:
        d = Path(d)
        if not d.exists():
            skipped.append(f"{d} (missing)")
            continue
        for f in sorted(d.glob("cv_*.json")):
            with open(f) as fh:
                s = json.load(fh)
            audio = AUDIO_FROM_SUBDIR.get(s.get("audio_subdir"))
            if audio is None:
                skipped.append(f"{f.name} (unknown audio_subdir "
                               f"{s.get('audio_subdir')!r})")
                continue
            key = (s["model"], audio, s["protocol"])
            if key in cells:
                skipped.append(f"{f} (duplicate of {key}, kept the first)")
                continue
            cells[key] = s
    for s in skipped:
        logger(f"  skipped: {s}")
    return cells


def check_distinct(cells, logger=print):
    """No two models may report the same numbers -- see the module docstring."""
    groups = defaultdict(dict)
    for (model, audio, protocol), s in cells.items():
        sig = (s["summary"]["pooled_f1_macro_mean"],
               s["summary"]["pooled_accuracy_mean"],
               s["summary"]["aggregated_f1_macro_mean"])
        groups[(audio, protocol)].setdefault(sig, []).append(model)

    bad = [(a, p, ms) for (a, p), sigs in groups.items()
           for ms in sigs.values() if len(ms) > 1]
    for a, p, ms in bad:
        logger(f"  FATAL: {a}/{p}: {sorted(ms)} report identical pooled "
               f"metrics -- they are the same encoder")
    if bad:
        raise SystemExit(
            "Refusing to build a report in which two models are the same "
            "encoder. Check that each model's checkpoint actually loaded.")


def modal_layer(s):
    """The layer the inner folds chose most often, and how often."""
    picks = [l for r in s["repeats"] for l in r["layers_chosen"]]
    if not picks:
        return None, 0.0
    layer, n = Counter(picks).most_common(1)[0]
    return int(layer), round(n / len(picks), 3)


def summary_rows(cells):
    rows = []
    for (model, audio, protocol), s in sorted(cells.items()):
        su = s["summary"]
        layer, share = modal_layer(s)
        rows.append({
            "family": FAMILY.get(model, "?"),
            "model": model,
            "label": LABEL.get(model, model),
            "audio": audio,
            "protocol": protocol,
            "n_individuals": s["num_classes"],
            "chance": s["chance"],
            "n_bouts": s["n_bouts"],
            "n_folds": s["n_folds"],
            "n_repeats": s["n_repeats"],
            "modal_layer": layer,
            "modal_layer_share": share,
            "f1_macro": su["pooled_f1_macro_mean"],
            "f1_macro_sd_repeats": su["pooled_f1_macro_std"],
            "f1_sd_across_folds": su["fold_f1_std_mean"],
            "accuracy": su["pooled_accuracy_mean"],
            "f1_cohort100": su["pooled_f1_cohort100_mean"],
            "f1_aggregated": su["aggregated_f1_macro_mean"],
            "f1_aggregated_sd": su["aggregated_f1_macro_std"],
        })
    return rows


def per_individual_rows(cells, protocol):
    """model x individual, F1 averaged over repeats."""
    rows = []
    for (model, audio, proto), s in sorted(cells.items()):
        if proto != protocol:
            continue
        acc = defaultdict(list)
        meta = {}
        for r in s["repeats"]:
            for pi in r["per_individual"]:
                acc[pi["individual"]].append(pi)
                meta[pi["individual"]] = pi
        for ind, entries in sorted(acc.items()):
            m = meta[ind]
            rows.append({
                "protocol": protocol,
                "audio": audio,
                "family": FAMILY.get(model, "?"),
                "model": model,
                "label": LABEL.get(model, model),
                "individual": ind,
                "f1": round(float(np.mean([e["f1"] for e in entries])), 4),
                "precision": round(float(np.mean([e["precision"]
                                                  for e in entries])), 4),
                "recall": round(float(np.mean([e["recall"]
                                               for e in entries])), 4),
                "support_bouts": m["support_bouts"],
                "n_sessions": m["n_sessions"],
                "n_recordings": m["n_recordings"],
                "leaky_split": m["leaky_split"],
            })
    return rows


def difficulty_rows(pi_rows):
    """One row per individual x audio, averaged over the encoders.

    This is the table that answers 'which hyraxes are actually hard, and is it
    because the model cannot tell them apart or because they have 16 bouts'.
    """
    by = defaultdict(list)
    for r in pi_rows:
        by[(r["audio"], r["individual"])].append(r)
    out = []
    for (audio, ind), rs in sorted(by.items()):
        f1s = [r["f1"] for r in rs]
        best = max(rs, key=lambda r: r["f1"])
        out.append({
            "protocol": rs[0]["protocol"],
            "audio": audio,
            "individual": ind,
            "support_bouts": rs[0]["support_bouts"],
            "n_sessions": rs[0]["n_sessions"],
            "n_recordings": rs[0]["n_recordings"],
            "leaky_split": rs[0]["leaky_split"],
            "f1_mean_over_models": round(float(np.mean(f1s)), 4),
            "f1_sd_over_models": round(float(np.std(f1s)), 4),
            "f1_best": round(best["f1"], 4),
            "best_model": best["model"],
            "n_models": len(rs),
        })
    out.sort(key=lambda r: (r["audio"], r["f1_mean_over_models"]))
    return out


def write_csv(rows, path, cols=None):
    if not rows:
        return None
    cols = cols or list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    return path


def family_table(rows, protocol):
    """One row per model, both audio versions side by side."""
    by_model = defaultdict(dict)
    for r in rows:
        if r["protocol"] == protocol:
            by_model[r["model"]][r["audio"]] = r
    out = []
    for model, per_audio in by_model.items():
        row = {"family": FAMILY.get(model, "?"), "model": model,
               "label": LABEL.get(model, model)}
        for audio in ("original", "bioda"):
            a = per_audio.get(audio)
            for k, src in [("layer", "modal_layer"), ("f1", "f1_macro"),
                           ("f1_sd", "f1_sd_across_folds"),
                           ("acc", "accuracy"),
                           ("f1_cohort100", "f1_cohort100"),
                           ("f1_agg", "f1_aggregated")]:
                row[f"{audio}_{k}"] = a[src] if a else None
        o, b = row.get("original_f1"), row.get("bioda_f1")
        row["delta_original_minus_bioda"] = (round(o - b, 4)
                                             if o is not None and b is not None
                                             else None)
        out.append(row)
    out.sort(key=lambda r: (FAMILY_ORDER.index(r["family"])
                            if r["family"] in FAMILY_ORDER else 99,
                            -(r["original_f1"] or 0)))
    return out


def plot_family(fam_rows, protocol, chance, path):
    rows = [r for r in fam_rows if r.get("original_f1") is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 2.0))
    ypos = np.arange(len(rows))[::-1]
    h = 0.38
    for i, (key, hatch, alpha) in enumerate([("original_f1", None, 1.0),
                                             ("bioda_f1", "///", 0.55)]):
        vals = [r[key] or 0 for r in rows]
        err = [r["original_f1_sd" if i == 0 else "bioda_f1_sd"] or 0
               for r in rows]
        ax.barh(ypos + (h / 2 if i == 0 else -h / 2), vals, height=h,
                xerr=err, error_kw=dict(lw=0.8, ecolor=MUTED, capsize=2),
                color=[FAMILY_COLOUR.get(r["family"], MUTED) for r in rows],
                alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0.5)
    ax.axvline(chance, color=MUTED, ls=":", lw=1)
    ax.text(chance, len(rows) - 0.2, f" chance {chance:.3f}", color=MUTED,
            fontsize=8, va="top")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['label']}" for r in rows], fontsize=9)
    ax.set_xlabel("pooled macro-F1 over 18 individuals "
                  "(solid = Original, hatched = BIODA); bars are SD across folds")
    ax.set_title(f"Cross-validated hyrax ID - {PROTOCOL_LABEL.get(protocol, protocol)}",
                 fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOUR[f])
               for f in FAMILY_ORDER]
    ax.legend(handles, FAMILY_ORDER, fontsize=8, frameon=False,
              loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_individuals(diff_rows, protocol, chance, path):
    rows = [r for r in diff_rows if r["audio"] == "original"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(rows) + 2.0))
    ypos = np.arange(len(rows))[::-1]
    colours = ["#D55E00" if r["leaky_split"] else "#0072B2" for r in rows]
    ax.barh(ypos, [r["f1_mean_over_models"] for r in rows],
            xerr=[r["f1_sd_over_models"] for r in rows],
            error_kw=dict(lw=0.8, ecolor=MUTED, capsize=2),
            color=colours, height=0.62)
    ax.axvline(chance, color=MUTED, ls=":", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{r['individual']}  ({r['support_bouts']} bouts, "
         f"{r['n_sessions']}s/{r['n_recordings']}r)" for r in rows],
        fontsize=8)
    ax.set_xlabel("macro-F1 averaged over the 15 encoders, Original audio "
                  "(bars = SD across encoders)")
    ax.set_title(f"Per-individual difficulty - "
                 f"{PROTOCOL_LABEL.get(protocol, protocol)}\n"
                 f"orange = train and test share a recording/session "
                 f"(optimistic)", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", action="append", required=True)
    ap.add_argument("--output-dir",
                    default="outputs/phase3/FINAL/12_cross_validation")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reading:")
    cells = load_cells(args.result_dir)
    if not cells:
        raise SystemExit("no cv_*.json found in the given directories.")
    check_distinct(cells)

    models = sorted({k[0] for k in cells})
    protocols = sorted({k[2] for k in cells})
    audios = sorted({k[1] for k in cells})
    print(f"  {len(cells)} cells: {len(models)} models x {len(audios)} audio "
          f"x {len(protocols)} protocols")
    missing = [(m, a, p) for m in models for a in audios for p in protocols
               if (m, a, p) not in cells]
    if missing:
        print(f"  INCOMPLETE: {len(missing)} cells absent, e.g. {missing[:4]}")

    unknown = [m for m in models if m not in FAMILY]
    if unknown:
        raise SystemExit(f"no family assigned to {unknown}; add it rather "
                         f"than letting an encoder appear ungrouped.")

    rows = summary_rows(cells)
    write_csv(rows, out_dir / "cv_summary.csv")
    print(f"wrote {out_dir / 'cv_summary.csv'}")

    for protocol in protocols:
        chance = next(s["chance"] for k, s in cells.items()
                      if k[2] == protocol)

        fam = family_table(rows, protocol)
        write_csv(fam, out_dir / f"cv_family_{protocol}.csv")

        pi = per_individual_rows(cells, protocol)
        write_csv(pi, out_dir / f"cv_per_individual_{protocol}.csv")

        diff = difficulty_rows(pi)
        write_csv(diff, out_dir / f"cv_individual_difficulty_{protocol}.csv")

        plot_family(fam, protocol, chance,
                    out_dir / f"cv_family_{protocol}.png")
        plot_individuals(diff, protocol, chance,
                         out_dir / f"cv_individuals_{protocol}.png")
        print(f"wrote {protocol}: family, per-individual, difficulty "
              f"(csv + png)")

        print(f"\n--- {PROTOCOL_LABEL.get(protocol, protocol)} "
              f"(chance {chance:.3f}) ---")
        print(f"{'family':14s} {'model':20s} {'O F1':>7s} {'B F1':>7s} "
              f"{'delta':>7s} {'O coh100':>9s} {'O agg':>7s} {'layer':>6s}")
        for r in fam:
            print(f"{r['family'][:13]:14s} {r['label'][:19]:20s} "
                  f"{_f(r['original_f1'])} {_f(r['bioda_f1'])} "
                  f"{_f(r['delta_original_minus_bioda'])} "
                  f"{_f(r['original_f1_cohort100']):>9s} "
                  f"{_f(r['original_f1_agg'])} "
                  f"{str(r['original_layer']):>6s}")

        leaky = [d for d in diff if d["leaky_split"] and d["audio"] == "original"]
        if leaky:
            print("  optimistic (train/test share a recording or session): "
                  + ", ".join(f"{d['individual']} F1 "
                              f"{d['f1_mean_over_models']:.3f}" for d in leaky))


def _f(v, width=7):
    return f"{v:{width}.4f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"


if __name__ == "__main__":
    main()
