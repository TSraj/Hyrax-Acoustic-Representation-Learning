#!/usr/bin/env python3
"""
Phase 3 - Step 34: the results table.

WHAT THIS ANSWERS
-----------------
Two questions, from the same pile of layer_probe_*.json files:

  1. ORIGINAL vs BIODA-DENOISED audio -- does denoising help or hurt?
  2. PER-BOUT vs AGGREGATED scoring  -- how much does pooling scores over a
     recording buy you?

Both are read WITHIN a split, never across one. session-holdout (8 animals,
chance 0.125) and by-file (10 animals, chance 0.100) are different tasks with
different chance levels; a number from one is not comparable to a number from
the other. Every output here is therefore keyed by split, and the audio
comparison is computed only between cells that agree on split, model and
condition.

WHAT IT WRITES
--------------
  summary_wide.csv        ONE ROW PER model x split. The table to read the
                          answer off: bioda F1, original F1, the delta, and the
                          same for the aggregated mode. Sorted worst-to-best
                          delta so a regression is the first thing you see.
  results_long.csv        one row per model x audio x split x mode x layer.
                          Everything, for any downstream plot.
  audio_comparison_<split>.png/.csv    original vs bioda, all models
  modes_<split>.png/.csv               per-bout vs aggregated, all models
  layers_<model>_<split>.png/.csv      the layer curves, one image per model

Figures are PNG at 300 DPI and every one has a CSV of the same base name
holding exactly the plotted values.

USAGE
-----
    python scripts/phase3_34_evaluation_report.py \
        --result-dir outputs/phase3/hyrax_probe_bout_session_holdout \
        --result-dir outputs/phase3/hyrax_probe_bout_by_file \
        --result-dir outputs/phase3/hyrax_probe_bout_original_session_holdout \
        --result-dir outputs/phase3/hyrax_probe_bout_original_by_file \
        --output-dir outputs/phase3/FINAL/10_audio_and_aggregation

Directories may be passed in any order and may be missing; whatever is present
is reported, and what is absent is listed so a half-finished run cannot be
mistaken for a complete one.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 300
# only frozen zero-shot cells belong in this report; adapted cells are the
# subject of a different experiment and mixing them in would compare a
# fine-tuned encoder against a frozen one under one column heading
CONDITION = "base"

AUDIO_LABEL = {"bioda_denoised": "BIODA denoised", "original": "Original"}
SPLIT_LABEL = {"session": "session-holdout", "file": "by-file"}


def parse_dir_arg(arg):
    """'path' or 'audio=path' -> (declared_audio_or_None, Path).

    Cells probed before `audio_version` was recorded carry no provenance, so
    the audio version has to be DECLARED for those directories. It is never
    guessed from the directory name: the whole comparison rests on not mixing
    the two audio versions, and a name is not evidence.
    """
    if "=" in arg:
        audio, path = arg.split("=", 1)
        audio = audio.strip()
        if audio not in AUDIO_LABEL:
            raise SystemExit(f"unknown audio version {audio!r}; "
                             f"expected one of {sorted(AUDIO_LABEL)}")
        return audio, Path(path)
    return None, Path(arg)


def load_cells(result_dirs, logger=print):
    """-> {(model, audio, split): summary}. Skips anything not frozen."""
    cells, skipped = {}, []
    for arg in result_dirs:
        declared, d = parse_dir_arg(arg)
        if not d.exists():
            skipped.append(f"{d} (missing)")
            continue
        for f in sorted(d.glob("layer_probe_*.json")):
            with open(f) as fh:
                s = json.load(fh)
            if s.get("condition") != CONDITION:
                continue
            recorded = s.get("audio_version")
            # A recorded version and a declared one that DISAGREE means the
            # directory holds something other than what was claimed. That is
            # exactly the mistake this report exists to prevent, so it stops.
            if recorded and declared and recorded != declared:
                raise SystemExit(
                    f"{f}\n  declared audio {declared!r} but the file records "
                    f"{recorded!r}. Refusing to build a table from this."
                )
            audio = recorded or declared
            if not audio:
                skipped.append(
                    f"{f.name} (no audio version recorded; pass the directory "
                    f"as e.g. bioda_denoised={d})")
                continue
            key = (s["model"], audio, s.get("split_by", "?"))
            if key in cells:
                skipped.append(f"{f} (duplicate of {key}, kept the first)")
                continue
            s["audio_version"] = audio
            cells[key] = s
    for s in skipped:
        logger(f"  skipped: {s}")
    check_distinct_encoders(cells, logger)
    return cells


def check_distinct_encoders(cells, logger=print):
    """Two models in one column must not be the same encoder.

    The 2026-09-07 run reported aves2_eat_bio and aves2_eat_all as two
    encoders across eight cells. They were one: avex had silently failed to
    apply either checkpoint (0/150 params matched, logged at INFO) so both ran
    the bare EAT backbone, and every metric came out byte-identical. Nothing in
    the table said so -- it took noticing that fifteen rows contained fourteen
    distinct numbers.

    Two independent checks, because either can fire alone. The fingerprint is
    definitive but only exists for cells probed after that fix; identical
    metrics catch the older files too, and would also catch a manifest or cache
    mix-up that duplicated a cell for some entirely different reason.
    """
    by_group = defaultdict(dict)
    for (model, audio, split), s in cells.items():
        by_group[(audio, split)][model] = s

    fp_dupes, metric_dupes = [], []
    for (audio, split), models in sorted(by_group.items()):
        seen_fp, seen_metrics = defaultdict(list), defaultdict(list)
        for model, s in sorted(models.items()):
            fp = s.get("extractor", {}).get("weights_sha1")
            if fp:
                seen_fp[fp].append(model)
            # the whole layer curve, not just the best number: two encoders
            # agreeing on one layer to 4 dp is luck, agreeing on all of them
            # is the same weights
            curve = tuple(sorted(
                (int(k), round(v["f1_macro_mean"], 6))
                for k, v in s["layers"].items()))
            seen_metrics[curve].append(model)

        for fp, ms in seen_fp.items():
            if len(ms) > 1:
                fp_dupes.append((audio, split, ms, fp[:16]))
        for _curve, ms in seen_metrics.items():
            if len(ms) > 1:
                metric_dupes.append((audio, split, ms))

    for audio, split, ms, fp in fp_dupes:
        logger(f"  FATAL: {audio}/{split}: {ms} share weight fingerprint {fp}")
    for audio, split, ms in metric_dupes:
        logger(f"  FATAL: {audio}/{split}: {ms} have identical layer curves")

    if fp_dupes or metric_dupes:
        raise SystemExit(
            "Refusing to build a table in which two models are the same "
            "encoder. Re-run the affected cells; if the models genuinely "
            "differ, the loader is not applying their checkpoints."
        )


def rows_long(cells):
    """One row per model x audio x split x mode x layer."""
    out = []
    for (model, audio, split), s in sorted(cells.items()):
        for layer, r in sorted(s["layers"].items(), key=lambda kv: int(kv[0])):
            base = dict(model=model, audio=audio, split=split,
                        n_classes=s["num_classes"], chance=round(s["chance"], 4),
                        layer=int(layer))
            out.append({**base, "mode": "per_bout",
                        "f1_macro": round(r["f1_macro_mean"], 4),
                        "f1_macro_std": round(r["f1_macro_std"], 4),
                        "accuracy": round(r["accuracy_mean"], 4),
                        "precision_macro": round(r["precision_macro_mean"], 4),
                        "recall_macro": round(r["recall_macro_mean"], 4),
                        "n_units": s["n_test"]})
            g = r.get("grouped")
            if g:
                out.append({**base, "mode": "aggregated",
                            "f1_macro": round(g["f1_macro_mean"], 4),
                            "f1_macro_std": round(g["f1_macro_std"], 4),
                            "accuracy": round(g["accuracy_mean"], 4),
                            "precision_macro": round(g["precision_macro_mean"], 4),
                            "recall_macro": round(g["recall_macro_mean"], 4),
                            "n_units": g["n_groups"]})
    return out


def best_of(s, mode):
    """Best layer and its metrics for one cell, in one mode."""
    if mode == "per_bout":
        layer = s["best_layer"]
        r = s["layers"][str(layer)]
        return dict(layer=layer, f1=r["f1_macro_mean"], acc=r["accuracy_mean"],
                    prec=r["precision_macro_mean"], rec=r["recall_macro_mean"],
                    n=s["n_test"])
    agg = s.get("aggregation")
    if not agg:
        return None
    return dict(layer=agg["best_layer"], f1=agg["best_f1_macro"],
                acc=agg["best_accuracy"], prec=agg["best_precision_macro"],
                rec=agg["best_recall_macro"], n=agg["n_groups"])


def rows_wide(cells):
    """ONE ROW PER model x split -- the table the answer is read off."""
    by_ms = defaultdict(dict)
    for (model, audio, split), s in cells.items():
        by_ms[(model, split)][audio] = s

    out = []
    for (model, split), per_audio in sorted(by_ms.items()):
        bio, org = per_audio.get("bioda_denoised"), per_audio.get("original")
        ref = bio or org
        row = {
            "model": model,
            "split": split,
            "n_classes": ref["num_classes"],
            "chance_f1": round(ref["chance"], 4),
        }
        for mode, tag in (("per_bout", "bout"), ("aggregated", "agg")):
            b = best_of(bio, mode) if bio else None
            o = best_of(org, mode) if org else None
            row[f"{tag}_bioda_f1"] = round(b["f1"], 4) if b else ""
            row[f"{tag}_bioda_acc"] = round(b["acc"], 4) if b else ""
            row[f"{tag}_bioda_layer"] = b["layer"] if b else ""
            row[f"{tag}_original_f1"] = round(o["f1"], 4) if o else ""
            row[f"{tag}_original_acc"] = round(o["acc"], 4) if o else ""
            row[f"{tag}_original_layer"] = o["layer"] if o else ""
            # POSITIVE delta = original is better than denoised
            row[f"{tag}_delta_f1"] = round(o["f1"] - b["f1"], 4) if (b and o) else ""
            row[f"{tag}_winner"] = ("original" if o["f1"] > b["f1"] else "bioda") \
                if (b and o) else ""
        # what aggregation buys, on whichever audio we have
        for tag, s in (("bioda", bio), ("original", org)):
            bb, ab = (best_of(s, "per_bout"), best_of(s, "aggregated")) if s else (None, None)
            row[f"agg_gain_{tag}"] = round(ab["f1"] - bb["f1"], 4) if (bb and ab) else ""
            row[f"n_groups_{tag}"] = ab["n"] if ab else ""
            row[f"min_groups_per_class_{tag}"] = (
                s["aggregation"]["min_groups_per_class"]
                if s and s.get("aggregation") else "")
        out.append(row)

    # worst delta first: a regression should be the first thing on the page
    out.sort(key=lambda r: (r["split"],
                            r["bout_delta_f1"] if r["bout_delta_f1"] != "" else 9))
    return out


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def bar_compare(rows, split, value_keys, labels, title, ylabel, out_png, chance):
    """Grouped bars, one pair per model. Also writes the matching CSV."""
    rows = [r for r in rows if r["split"] == split and
            all(r[k] != "" for k in value_keys)]
    if not rows:
        return False
    rows.sort(key=lambda r: -r[value_keys[0]])
    names = [r["model"] for r in rows]
    x = range(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.15), 5))
    for i, (k, lab) in enumerate(zip(value_keys, labels)):
        vals = [r[k] for r in rows]
        off = (i - 0.5) * width
        bars = ax.bar([xx + off for xx in x], vals, width, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(chance, ls="--", lw=1, color="grey")
    ax.text(len(names) - 0.4, chance + 0.005, f"chance {chance:.3f}",
            fontsize=7, color="grey", ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(max(r[k] for k in value_keys) for r in rows) * 1.28)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

    write_csv(out_png.with_suffix(".csv"),
              [{"model": r["model"], "split": r["split"],
                **{lab: r[k] for k, lab in zip(value_keys, labels)},
                "delta": round(r[value_keys[1]] - r[value_keys[0]], 4)}
               for r in rows])
    return True


def layer_figure(model, split, per_audio, out_png):
    """Layer curves for ONE model. One image per model on purpose: depths
    differ across encoders, so a shared x-axis would be a lie."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    csv_rows, plotted = [], False

    for audio, style in (("bioda_denoised", "-o"), ("original", "--s")):
        s = per_audio.get(audio)
        if not s:
            continue
        layers = sorted(s["layers"], key=int)
        for mode, alpha in (("per_bout", 1.0), ("aggregated", 0.45)):
            if mode == "aggregated" and "grouped" not in s["layers"][layers[0]]:
                continue
            vals = [(s["layers"][l]["f1_macro_mean"] if mode == "per_bout"
                     else s["layers"][l]["grouped"]["f1_macro_mean"])
                    for l in layers]
            ax.plot([int(l) for l in layers], vals, style, ms=3.5, lw=1.4,
                    alpha=alpha,
                    label=f"{AUDIO_LABEL.get(audio, audio)} - "
                          f"{'per bout' if mode == 'per_bout' else 'aggregated'}")
            plotted = True
            for l, v in zip(layers, vals):
                csv_rows.append({"model": model, "split": split, "audio": audio,
                                 "mode": mode, "layer": int(l),
                                 "f1_macro": round(v, 4)})

    if not plotted:
        plt.close(fig)
        return False

    ref = next(iter(per_audio.values()))
    ax.axhline(ref["chance"], ls=":", lw=1, color="grey")
    ax.text(0, ref["chance"] + 0.004, f"chance {ref['chance']:.3f}",
            fontsize=7, color="grey")
    ax.set_xlabel("layer  (0 = pre-transformer)")
    ax.set_ylabel("macro-F1")
    ax.set_title(f"{model} - {SPLIT_LABEL.get(split, split)}")
    ax.legend(frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    write_csv(out_png.with_suffix(".csv"), csv_rows)
    return True


def main():
    p = argparse.ArgumentParser(description="Audio + aggregation results table")
    p.add_argument("--result-dir", action="append", required=True,
                   help="repeatable; probe output directories to read")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("reading:")
    cells = load_cells(args.result_dir)
    if not cells:
        raise SystemExit("no frozen cells found in the given directories")

    have = defaultdict(set)
    for (model, audio, split) in cells:
        have[(model, split)].add(audio)
    print(f"\n{len(cells)} cells | "
          f"{len({m for m, _, _ in cells})} models | "
          f"splits {sorted({s for _, _, s in cells})}")

    incomplete = [f"{m} / {SPLIT_LABEL.get(s, s)}: only {sorted(a)}"
                  for (m, s), a in sorted(have.items()) if len(a) < 2]
    if incomplete:
        print("\nNOT YET COMPARABLE (need both audio versions):")
        for line in incomplete:
            print(f"  {line}")

    long_rows = rows_long(cells)
    wide_rows = rows_wide(cells)
    write_csv(out / "results_long.csv", long_rows)
    write_csv(out / "summary_wide.csv", wide_rows)
    print(f"\nwrote results_long.csv   ({len(long_rows)} rows)")
    print(f"wrote summary_wide.csv   ({len(wide_rows)} rows)")

    n_fig = 0
    for split in sorted({s for _, _, s in cells}):
        chance = next(c["chance"] for k, c in cells.items() if k[2] == split)
        if bar_compare(wide_rows, split,
                       ["bout_bioda_f1", "bout_original_f1"],
                       ["BIODA denoised", "Original"],
                       f"Audio version, per bout - {SPLIT_LABEL.get(split, split)}",
                       "macro-F1", out / f"audio_comparison_{split}.png", chance):
            n_fig += 1
        if bar_compare(wide_rows, split,
                       ["bout_bioda_f1", "agg_bioda_f1"],
                       ["per bout", "aggregated by recording"],
                       f"Evaluation mode - {SPLIT_LABEL.get(split, split)}",
                       "macro-F1", out / f"modes_{split}.png", chance):
            n_fig += 1

    for (model, split), audios in sorted(have.items()):
        per_audio = {a: cells[(model, a, split)] for a in audios}
        if layer_figure(model, split, per_audio,
                        out / f"layers_{model}_{split}.png"):
            n_fig += 1
    print(f"wrote {n_fig} figures (PNG 300 DPI, each with a matching CSV)")

    # the headline, printed so it is visible without opening anything
    done = [r for r in wide_rows if r["bout_delta_f1"] != ""]
    if done:
        print("\n" + "=" * 70)
        print("ORIGINAL vs BIODA  (positive delta = original is better)")
        print("=" * 70)
        for r in done:
            print(f"  {r['model']:<20} {SPLIT_LABEL.get(r['split'], r['split']):<16} "
                  f"bioda {r['bout_bioda_f1']:.4f}  original {r['bout_original_f1']:.4f}  "
                  f"delta {r['bout_delta_f1']:+.4f}  -> {r['bout_winner']}")
        wins = sum(r["bout_winner"] == "original" for r in done)
        print(f"\n  original wins {wins}/{len(done)} cells")
    else:
        print("\nno model has both audio versions yet -- delta columns are empty")


if __name__ == "__main__":
    main()
