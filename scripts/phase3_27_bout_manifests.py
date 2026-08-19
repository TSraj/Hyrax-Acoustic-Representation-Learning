#!/usr/bin/env python3
"""
Phase 3 - Step 27: bout-level hyrax manifests.

WHY THIS EXISTS
---------------
Every hyrax result so far was measured on the wrong unit. phase3_02 takes the
ground-truth bouts, CONCATENATES them into one wav per individual per split, and
the evaluation then cuts that wav into 5 s windows at 2.5 s stride. Bouts average
1.0-2.0 s, so each window holds 3-4 different bouts joined end to end, with
artificial splice points the encoder may well be keying on.

That was never a design decision. The concatenation happened first, and by the
time windowing was introduced the individual bouts were no longer addressable.

Nothing was lost, though: phase3_02 already parses every bout's source file,
start, end and session - it simply discards that structure at the last step.
This script keeps it.

WHAT IT PRODUCES
----------------
Two manifests, ONE ENTRY PER BOUT, each carrying `file` + `start` + `end`, so
the probe slices the exact GT segment at whatever length it happens to be. No
fixed window, no concatenation, no splices. Long bouts stay whole.

  hyrax_bout_session_holdout.json
      The strict split. Same 8 individuals and the same held-out session per
      individual as the existing manifest, so bout results are directly
      comparable to the window results they replace.

  hyrax_bout_by_file.json
      The 10 individuals with the most bouts, split BY RECORDING. This is the
      protocol the supervisor uses. It is looser than session holdout - two
      recordings from one session can land on opposite sides - so it should
      score higher, and the gap between the two manifests measures how much of
      the score is session leakage rather than individual identity.

Reads only Data/YearLocation/*/GTLabels/*.txt and the matching BIODA audio
paths. Writes no audio. phase3_02 is left untouched, so every existing manifest
and published number stays reproducible.

    python scripts/phase3_27_bout_manifests.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.utils.logging_utils import setup_logger  # noqa: E402

# the cohort and junk-session definitions live in phase3_02; importing them
# rather than retyping guarantees the session split matches the existing one
from phase3_02_create_manifests import (  # noqa: E402
    SESSION_TASK_INDIVIDUALS,
    SESSION_TASK_JUNK,
)

SR = 16000
# wav2vec2/HuBERT conv stacks need a few hundred samples before the receptive
# field is satisfied; shorter clips raise rather than degrade. Dropped bouts are
# counted and reported, never silently discarded.
MIN_BOUT_SECONDS = 0.4


def parse_bouts(data_dir, audio_subdir, logger):
    """One record per GT bout: source file, start, end, session, individual.

    Mirrors phase3_02.parse_bouts_from_gtlabels, but keeps every bout as its own
    addressable unit instead of folding it into a concatenation.
    """
    bouts_per_individual = defaultdict(list)
    session_profile = defaultdict(lambda: defaultdict(int))

    label_files = sorted(Path(data_dir).glob("*/GTLabels/*.txt"))
    logger.info(f"found {len(label_files)} label files")

    missing_audio = 0
    for label_file in label_files:
        bioda_dir = label_file.parent.parent / audio_subdir
        audio_name = label_file.stem.replace("_labels", "")
        audio_file = bioda_dir / f"{audio_name}.wav"

        if not audio_file.exists():
            matches = list(bioda_dir.glob(f"{audio_name}*.wav"))
            if not matches:
                missing_audio += 1
                continue
            audio_file = matches[0]

        parts = audio_name.split("_")
        if len(parts) < 2:
            continue
        individual = parts[0]
        session = parts[-1]

        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 3:
                    continue
                try:
                    start, end, label = float(cols[0]), float(cols[1]), cols[2]
                except (ValueError, IndexError):
                    continue
                if not label.lower().startswith("bout_"):
                    continue

                bouts_per_individual[individual].append({
                    "file": str(audio_file),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "session": session,
                    "individual": individual,
                    "label": label,
                })
                session_profile[individual][session] += 1

    if missing_audio:
        logger.warning(f"{missing_audio} label files had no matching audio")

    return dict(bouts_per_individual), dict(session_profile)


def valid_bouts(individual, bouts_per_individual):
    """Bouts with junk sessions removed -- same rule phase3_02 applies."""
    junk = SESSION_TASK_JUNK.get(individual, [])
    return [b for b in bouts_per_individual.get(individual, [])
            if b["session"] not in junk]


def apply_duration_floor(bouts, logger, tag):
    keep = [b for b in bouts if b["duration"] >= MIN_BOUT_SECONDS]
    dropped = len(bouts) - len(keep)
    if dropped:
        logger.info(f"    {tag}: dropped {dropped} bouts under {MIN_BOUT_SECONDS}s "
                    f"({dropped / max(len(bouts), 1) * 100:.1f}%)")
    return keep, dropped


def duration_stats(bouts):
    if not bouts:
        return {}
    d = sorted(b["duration"] for b in bouts)
    n = len(d)
    return {
        "n": n,
        "mean_s": round(sum(d) / n, 3),
        "median_s": round(d[n // 2], 3),
        "min_s": round(d[0], 3),
        "max_s": round(d[-1], 3),
        "p90_s": round(d[int(n * 0.9)], 3),
        "over_5s": sum(1 for x in d if x > 5.0),
        "total_minutes": round(sum(d) / 60, 2),
    }


def order_for_cache(items):
    """Group by source recording, then by start time.

    The probe caches the last few decoded wavs, so emitting bouts in this order
    turns ~3000 full-file decodes into roughly one per recording.
    """
    return sorted(items, key=lambda b: (b["file"], b["start"]))


def class_weights_from(split_items, classes):
    """Inverse frequency over BOUTS.

    Note this is now meaningful. In the concatenated manifests each individual
    contributed exactly one file per split, so the same formula produced uniform
    weights for everyone regardless of how much audio they actually had.
    """
    counts = defaultdict(int)
    for it in split_items:
        counts[it["individual"]] += 1
    total = len(split_items)
    return {c: (total / (len(classes) * counts[c]) if counts[c] else 0.0)
            for c in classes}


def build_session_holdout(bouts_per_individual, session_profile, logger):
    """8 individuals, hold out each one's LARGEST valid session. Matches phase3_02."""
    logger.info("\n" + "=" * 72)
    logger.info("SESSION HOLDOUT (bout level)")
    logger.info("=" * 72)

    target = [i for i in SESSION_TASK_INDIVIDUALS if i in bouts_per_individual]
    splits = {"train": [], "test": []}
    held_out, inventory, dropped_total = {}, {}, 0

    for ind in target:
        vb = valid_bouts(ind, bouts_per_individual)
        vb, dropped = apply_duration_floor(vb, logger, ind)
        dropped_total += dropped
        if not vb:
            continue

        # largest valid session becomes the test session -- phase3_02's rule
        sessions = defaultdict(int)
        for b in vb:
            sessions[b["session"]] += 1
        ho = max(sessions.items(), key=lambda kv: kv[1])[0]
        held_out[ind] = ho

        train = [b for b in vb if b["session"] != ho]
        test = [b for b in vb if b["session"] == ho]
        splits["train"].extend(train)
        splits["test"].extend(test)

        inventory[ind] = {
            "held_out_session": ho,
            "n_sessions": len(sessions),
            "train_bouts": len(train),
            "test_bouts": len(test),
        }
        logger.info(f"  {ind:<9} held-out {ho:<14} train {len(train):>5}  test {len(test):>5}")

    classes = sorted(inventory)
    return {
        "task": "hyrax_id_bout_session_holdout",
        "description": (
            "Bout-level hyrax individual ID. One entry per ground-truth bout, "
            "sliced at its true start/end -- NOT concatenated and NOT windowed. "
            "Train/test split by recording SESSION: each individual's largest "
            "valid session is held out, matching hyrax_id_session_holdout.json."
        ),
        "unit": "bout",
        "split_by": "session",
        "num_classes": len(classes),
        "individuals": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "class_weights": class_weights_from(splits["train"], classes),
        "held_out_sessions": held_out,
        "excluded_sessions": SESSION_TASK_JUNK,
        "min_bout_seconds": MIN_BOUT_SECONDS,
        "bouts_dropped_under_floor": dropped_total,
        "inventory": inventory,
        "duration_stats": {s: duration_stats(v) for s, v in splits.items()},
        "split_counts": {s: len(v) for s, v in splits.items()},
        "splits": {s: order_for_cache(v) for s, v in splits.items()},
        "comparability_note": (
            "Same 8 individuals and the same held-out session per individual as "
            "hyrax_id_session_holdout.json, so the ONLY change is the input unit: "
            "one real bout here versus a 5s window spanning 3-4 spliced bouts "
            "there. Any difference in score is attributable to that."
        ),
    }


def build_by_file(bouts_per_individual, logger, top_n=10, test_frac=0.2):
    """Top-N individuals by bout count, split by RECORDING.

    The supervisor's protocol. Recordings are assigned to test largest-first
    until test_frac of that individual's bouts is covered, with at least one
    recording held out and at least one kept for training.
    """
    logger.info("\n" + "=" * 72)
    logger.info(f"BY-FILE SPLIT (bout level, top {top_n} individuals)")
    logger.info("=" * 72)

    counts = {ind: len(b) for ind, b in bouts_per_individual.items()}
    target = [i for i, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]]
    logger.info(f"selected: {', '.join(target)}")

    splits = {"train": [], "test": []}
    inventory, dropped_total = {}, 0

    for ind in target:
        vb = valid_bouts(ind, bouts_per_individual)
        vb, dropped = apply_duration_floor(vb, logger, ind)
        dropped_total += dropped
        if not vb:
            continue

        by_file = defaultdict(list)
        for b in vb:
            by_file[b["file"]].append(b)

        if len(by_file) < 2:
            logger.warning(f"  {ind:<9} only {len(by_file)} recording -- cannot split by "
                           f"file, EXCLUDED")
            continue

        # deterministic: largest recordings first until the test quota is met
        ordered = sorted(by_file.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        quota = max(1, int(round(len(vb) * test_frac)))
        test_files, n_test = [], 0
        for path, bl in reversed(ordered):          # smallest first, finer control
            if n_test >= quota or len(test_files) >= len(ordered) - 1:
                break
            test_files.append(path)
            n_test += len(bl)

        train = [b for b in vb if b["file"] not in test_files]
        test = [b for b in vb if b["file"] in test_files]
        if not train or not test:
            logger.warning(f"  {ind:<9} degenerate split, EXCLUDED")
            continue

        splits["train"].extend(train)
        splits["test"].extend(test)
        inventory[ind] = {
            "n_recordings": len(by_file),
            "test_recordings": len(test_files),
            "train_bouts": len(train),
            "test_bouts": len(test),
        }
        logger.info(f"  {ind:<9} {len(by_file):>3} recordings ({len(test_files)} held out)  "
                    f"train {len(train):>5}  test {len(test):>5}")

    classes = sorted(inventory)
    return {
        "task": "hyrax_id_bout_by_file",
        "description": (
            f"Bout-level hyrax individual ID, {len(classes)} individuals with the "
            "most bouts, split by RECORDING rather than by session."
        ),
        "unit": "bout",
        "split_by": "file",
        "num_classes": len(classes),
        "individuals": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "class_weights": class_weights_from(splits["train"], classes),
        "test_fraction_target": test_frac,
        "min_bout_seconds": MIN_BOUT_SECONDS,
        "bouts_dropped_under_floor": dropped_total,
        "inventory": inventory,
        "duration_stats": {s: duration_stats(v) for s, v in splits.items()},
        "split_counts": {s: len(v) for s, v in splits.items()},
        "splits": {s: order_for_cache(v) for s, v in splits.items()},
        "leakage_note": (
            "A by-file split is LOOSER than session holdout: two recordings made "
            "in the same session can land on opposite sides, so recording-condition "
            "cues survive into test. Expect a higher score than the session-holdout "
            "manifest. The gap between the two is an estimate of that leakage, not "
            "of identity information."
        ),
    }


def main():
    p = argparse.ArgumentParser(description="Bout-level hyrax manifests")
    p.add_argument("--data-dir", default="Data/YearLocation")
    p.add_argument("--audio-subdir", default="BIODA/denoised",
                   help="denoiser version; BIODA is the one the audit supports")
    p.add_argument("--output-dir", default="outputs/phase3/manifests_bout")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("Phase3_BoutManifests", "INFO")
    logger.info("=" * 72)
    logger.info("PHASE 3 - STEP 27: BOUT-LEVEL MANIFESTS")
    logger.info("=" * 72)
    logger.info(f"source: {args.data_dir}/*/GTLabels + {args.audio_subdir}")

    if not Path(args.data_dir).exists():
        raise SystemExit(f"data dir not found: {args.data_dir}")

    bouts, profile = parse_bouts(args.data_dir, args.audio_subdir, logger)
    total = sum(len(v) for v in bouts.values())
    logger.info(f"parsed {total} bouts across {len(bouts)} individuals")

    allb = [b for v in bouts.values() for b in v]
    st = duration_stats(allb)
    logger.info(f"bout duration: mean {st['mean_s']}s  median {st['median_s']}s  "
                f"min {st['min_s']}s  max {st['max_s']}s  p90 {st['p90_s']}s")
    logger.info(f"bouts over 5s: {st['over_5s']} ({st['over_5s'] / st['n'] * 100:.1f}%) "
                f"-- these were previously split across windows")

    written = []
    for name, manifest in (
        ("hyrax_bout_session_holdout.json",
         build_session_holdout(bouts, profile, logger)),
        ("hyrax_bout_by_file.json",
         build_by_file(bouts, logger, args.top_n, args.test_frac)),
    ):
        path = out_dir / name
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        written.append((path, manifest))

    logger.info("\n" + "=" * 72)
    for path, m in written:
        logger.info(f"{path}")
        logger.info(f"   {m['num_classes']} individuals, split by {m['split_by']}, "
                    f"train {m['split_counts']['train']} / test {m['split_counts']['test']} bouts")
        ds = m["duration_stats"]["train"]
        logger.info(f"   train audio {ds['total_minutes']} min, "
                    f"mean bout {ds['mean_s']}s, {ds['over_5s']} bouts over 5s")

    logger.info("\nNEXT: probe these with")
    logger.info("  MANIFEST=outputs/phase3/manifests_bout/hyrax_bout_session_holdout.json \\")
    logger.info("  EXPERIMENT=adapt_species_id PROBE_TAG=bout_session_holdout \\")
    logger.info("  sbatch run_phase3_hyrax_layer_probe.sh")


if __name__ == "__main__":
    main()
