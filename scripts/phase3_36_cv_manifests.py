#!/usr/bin/env python3
"""
Phase 3 - Step 36: cross-validation manifests, all 18 individuals.

WHY THIS EXISTS
---------------
The published 15-encoder grid used ONE fixed partition per protocol and only
the 8 / 10 individuals with enough data to form it. Two consequences the
supervisor asked us to remove:

  * a single partition gives no error bar, and every score depends on which
    recordings happened to land in test
  * eight animals were never predicted at all

So: every recording is tested exactly once, and all 18 individuals are in.

WHAT IT DOES NOT TOUCH
----------------------
`manifests_bout/` and `manifests_bout_original/` are left exactly as they are.
The 15-encoder results already shown to the supervisor stay reproducible from
them; this writes a new tree.

THE TWO PROTOCOLS
-----------------
session_loso    Leave one SESSION out. Fold k holds out the k-th session of
                every individual that has one. An animal with 15 sessions is
                tested in 15 folds, one with 3 is tested in 3. Every session is
                held out exactly once, which is what leave-one-session-out
                means when the animals have wildly uneven session counts.

by_file_5fold   5 folds, grouped by RECORDING, assigned per individual
                largest-first so each fold carries a similar share of that
                animal's bouts. Repeated with 5 different partition seeds:
                one 5-fold partition is still a single partition, and repeating
                it costs nothing here because only the linear probe refits.

Both are recording-disjoint: no bout from a test recording is ever in train.

THE TWO ANIMALS THAT CANNOT BE SPLIT THAT WAY
---------------------------------------------
Counted from the annotations, not assumed:

    X0   16 bouts,  1 session,  1 recording
    O1   64 bouts,  1 session,  3 recordings

X0 has ONE recording in the entire dataset. Hold it out and X0 has no training
data at all, so it can never be recognised. O1 has three recordings but all
from one session, so leave-one-session-out removes all of it at once.

The supervisor's instruction is that all 18 are used, so these two fall back to
a WITHIN-recording (X0) or WITHIN-session (O1, session protocol only) split:
their bouts are divided between train and test inside the same recording.

That is leakage and it is not hidden. Train and test then share microphone,
distance, background and date, so those scores are optimistic and measure
partly "same recording" rather than "same animal". Every affected individual is
flagged in `split_exceptions`, carried into each item as `leaky_split`, and
must be reported per-individual rather than folded silently into the mean.

A RELATED CAVEAT, FOR THE WRITE-UP
----------------------------------
Session-disjoint folds stop bout leakage but cannot separate individual
identity from recording condition for an animal recorded on few dates. J9 (2
sessions), T1 (2), O1 (1) and X0 (1) are nearly confounded: "which animal" and
"which recording" are almost the same variable. `sessions` and `recordings` per
individual are therefore written into the inventory, so a high score on a
1-session animal is visibly weaker evidence than the same score on R3 (15
sessions). More folds do not fix this; it is a property of the data.

OUTPUT
------
  <output-dir>/cv_session_loso.json
  <output-dir>/cv_by_file_5fold.json

One file per protocol, holding every bout ONCE with its fold assignment per
repeat -- not one copy of the data per fold. Item schema is identical to
phase3_27's (file, start, end, duration, session, individual, label), so the
existing extraction path reads them unchanged.

USAGE
-----
    python scripts/phase3_36_cv_manifests.py \
        --audio-subdir BIODA/denoised \
        --output-dir outputs/phase3/manifests_cv

    python scripts/phase3_36_cv_manifests.py \
        --audio-subdir Audio \
        --output-dir outputs/phase3/manifests_cv_original

Run it once per audio version. The fold assignment is derived from session and
recording NAMES, which are identical in both trees, so the two versions differ
only in the path -- exactly as the original-vs-BIODA comparison requires.
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

from phase3_27_bout_manifests import (  # noqa: E402
    MIN_BOUT_SECONDS,
    apply_duration_floor,
    duration_stats,
    parse_bouts,
    valid_bouts,
)

N_FOLDS_BY_FILE = 5
N_REPEATS_BY_FILE = 5
# the repeat seeds are fixed so the partition is reproducible; they shuffle
# which recording lands in which fold, nothing else
REPEAT_SEEDS = [0, 1, 2, 3, 4]


def group_bouts(bouts, key):
    """-> {group value: [bout, ...]}, insertion ordered by first appearance."""
    out = defaultdict(list)
    for b in bouts:
        out[b[key]].append(b)
    return dict(out)


def assign_within_group(bouts, n_folds, seed=0):
    """Fold ids for bouts that CANNOT be split by group -- the leaky fallback.

    Used only for X0 (one recording) and, under the session protocol, O1 (one
    session). The bouts are dealt round-robin after a deterministic shuffle, so
    each fold gets a near-equal share and every bout is tested exactly once.
    """
    idx = list(range(len(bouts)))
    rng = _rng(seed)
    rng.shuffle(idx)
    folds = [0] * len(bouts)
    for position, i in enumerate(idx):
        folds[i] = position % n_folds
    return folds


def _rng(seed):
    import random
    return random.Random(seed)


def stable_seed(name):
    """A per-individual seed that does not move between processes.

    Python's builtin hash() is salted per interpreter unless PYTHONHASHSEED is
    set, so seeding from it gave X0 and O1 DIFFERENT within-recording splits in
    the BIODA and the original manifests. That silently breaks the pairing the
    whole original-vs-denoised comparison rests on: any delta for those two
    animals would then be partly the fold assignment, not the audio.
    """
    import hashlib
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def balanced_group_folds(groups, n_folds, seed):
    """Assign whole GROUPS to folds, largest first, into the emptiest fold.

    Largest-first keeps a 200-bout recording from landing beside another one
    and leaving a fold with almost nothing; the shuffle before sorting only
    breaks ties, so the partition is deterministic given the seed.

    -> {group value: fold index}
    """
    items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    rng = _rng(seed)
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    load = [0] * n_folds
    assignment = {}
    for name, bouts in items:
        f = min(range(n_folds), key=lambda k: (load[k], k))
        assignment[name] = f
        load[f] += len(bouts)
    return assignment


def build_session_loso(per_individual, logger):
    """Fold k holds out each individual's k-th session.

    Number of folds is the largest session count in the cohort, so an animal
    with fewer sessions simply contributes no test bouts to the later folds.
    Every session is still held out exactly once, which is the property that
    matters: pooled over folds, every bout has one out-of-fold prediction.
    """
    n_folds = max(len({b["session"] for b in v}) for v in per_individual.values())
    logger.info(f"session LOSO: {n_folds} folds "
                f"(= the largest session count in the cohort)")

    items, exceptions, inventory = [], {}, {}
    for ind in sorted(per_individual):
        bouts = per_individual[ind]
        sessions = group_bouts(bouts, "session")
        # sorted by name so the fold index is stable across audio versions
        order = sorted(sessions)

        if len(order) == 1:
            # cannot hold out the only session -- split inside it instead
            folds = assign_within_group(bouts, min(n_folds, len(bouts)),
                                        seed=stable_seed(ind))
            for b, f in zip(bouts, folds):
                items.append({**b, "folds": [f], "leaky_split": True})
            exceptions[ind] = {
                "reason": "single session",
                "n_sessions": 1,
                "n_recordings": len({b["file"] for b in bouts}),
                "fallback": "within-session split (train and test share a "
                            "session; score is optimistic)",
            }
            used = min(n_folds, len(bouts))
        else:
            for f, sess in enumerate(order):
                for b in sessions[sess]:
                    items.append({**b, "folds": [f], "leaky_split": False})
            used = len(order)

        inventory[ind] = {
            "bouts": len(bouts),
            "sessions": len(sessions),
            "recordings": len({b["file"] for b in bouts}),
            "folds_tested_in": used,
            "leaky_split": ind in exceptions,
        }
    return items, n_folds, 1, exceptions, inventory


def build_by_file(per_individual, logger):
    """5 folds grouped by recording, repeated with 5 partition seeds."""
    logger.info(f"by-file: {N_FOLDS_BY_FILE} folds x {N_REPEATS_BY_FILE} "
                f"repeats, grouped by recording")

    items, exceptions, inventory = [], {}, {}
    for ind in sorted(per_individual):
        bouts = per_individual[ind]
        recs = group_bouts(bouts, "file")

        if len(recs) == 1:
            # one recording in the whole dataset -- nothing to hold out
            per_bout_folds = [
                assign_within_group(bouts, min(N_FOLDS_BY_FILE, len(bouts)),
                                    seed=s)
                for s in REPEAT_SEEDS]
            for i, b in enumerate(bouts):
                items.append({**b,
                              "folds": [pf[i] for pf in per_bout_folds],
                              "leaky_split": True})
            exceptions[ind] = {
                "reason": "single recording",
                "n_sessions": len({b["session"] for b in bouts}),
                "n_recordings": 1,
                "fallback": "within-recording split (train and test share a "
                            "recording; score is optimistic)",
            }
        else:
            # a group cannot be in two folds at once, so an animal with fewer
            # recordings than folds is simply absent from some folds' test set
            k = min(N_FOLDS_BY_FILE, len(recs))
            maps = [balanced_group_folds(recs, k, seed=s) for s in REPEAT_SEEDS]
            for rec, rec_bouts in recs.items():
                for b in rec_bouts:
                    items.append({**b,
                                  "folds": [m[rec] for m in maps],
                                  "leaky_split": False})

        # count the folds actually used rather than deriving it from the
        # recording count: the leaky fallback splits BOUTS, so a one-recording
        # animal still spans several folds
        used = len({it["folds"][0] for it in items
                    if it["individual"] == ind})
        inventory[ind] = {
            "bouts": len(bouts),
            "sessions": len({b["session"] for b in bouts}),
            "recordings": len(recs),
            "folds_tested_in": used,
            "leaky_split": ind in exceptions,
        }
    return items, N_FOLDS_BY_FILE, N_REPEATS_BY_FILE, exceptions, inventory


def verify(items, n_folds, n_repeats, individuals, logger):
    """Refuse to write a manifest that does not do what it claims.

    Each check corresponds to a way the fold assignment could be silently
    wrong and still produce a plausible-looking number.
    """
    problems = []

    for r in range(n_repeats):
        # 1. every bout is tested exactly once per repeat -- true by
        #    construction (one fold id per bout), so what is checked is that
        #    the id is in range
        bad = [it for it in items if not 0 <= it["folds"][r] < n_folds]
        if bad:
            problems.append(f"repeat {r}: {len(bad)} bouts with a fold id "
                            f"outside 0..{n_folds - 1}")

        # 2. no recording spans two folds, except the declared leaky ones
        by_rec = defaultdict(set)
        for it in items:
            if not it["leaky_split"]:
                by_rec[it["file"]].add(it["folds"][r])
        split_recs = [f for f, fs in by_rec.items() if len(fs) > 1]
        if split_recs:
            problems.append(f"repeat {r}: {len(split_recs)} recordings appear "
                            f"in more than one fold, e.g. "
                            f"{Path(split_recs[0]).name}")

        # 3. every individual has training data in every fold it is tested in,
        #    otherwise its class is unlearnable there
        for ind in individuals:
            mine = [it for it in items if it["individual"] == ind]
            tested = {it["folds"][r] for it in mine}
            for f in tested:
                if not any(it["folds"][r] != f for it in mine):
                    problems.append(f"repeat {r}: {ind} has no training bouts "
                                    f"when fold {f} is held out")

    if problems:
        for p in problems:
            logger.error(f"  {p}")
        raise SystemExit("fold assignment failed verification; refusing to "
                         "write a manifest that misstates its own splits.")
    logger.info("  verified: fold ids in range, recordings not split across "
                "folds (except declared exceptions), every class trainable in "
                "every fold")


def build(protocol, per_individual, audio_subdir, logger):
    if protocol == "session_loso":
        items, n_folds, n_repeats, exc, inv = build_session_loso(
            per_individual, logger)
        desc = ("Leave-one-session-out over all 18 individuals. Fold k holds "
                "out each individual's k-th session; every session is tested "
                "exactly once.")
    else:
        items, n_folds, n_repeats, exc, inv = build_by_file(
            per_individual, logger)
        desc = (f"{N_FOLDS_BY_FILE}-fold cross-validation grouped by "
                f"recording, over all 18 individuals, repeated with "
                f"{N_REPEATS_BY_FILE} partition seeds.")

    classes = sorted(per_individual)
    verify(items, n_folds, n_repeats, classes, logger)

    return {
        "task": f"hyrax_id_cv_{protocol}",
        "description": desc,
        "unit": "bout",
        "protocol": protocol,
        "audio_subdir": audio_subdir,
        "num_classes": len(classes),
        "individuals": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "repeat_seeds": REPEAT_SEEDS[:n_repeats],
        "min_bout_seconds": MIN_BOUT_SECONDS,
        "split_exceptions": exc,
        "inventory": inv,
        "duration_stats": duration_stats(items),
        "leakage_note":
            "Folds are recording-disjoint EXCEPT for the individuals listed "
            "in split_exceptions, whose bouts are split inside a single "
            "recording or session because they have only one. Those scores "
            "are optimistic and must be read per-individual. Separately, an "
            "animal recorded in few sessions cannot be distinguished from its "
            "recording condition by any split; see `sessions` per individual "
            "in the inventory.",
        "items": items,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="Data/YearLocation")
    p.add_argument("--audio-subdir", default="BIODA/denoised",
                   help="'BIODA/denoised' or 'Audio'")
    p.add_argument("--output-dir", default="outputs/phase3/manifests_cv")
    args = p.parse_args()

    logger = setup_logger("Phase3_CVManifests", "INFO")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("PHASE 3 - STEP 36: CROSS-VALIDATION MANIFESTS (all 18)")
    logger.info("=" * 72)
    logger.info(f"source: {args.data_dir}/*/GTLabels + {args.audio_subdir}")

    bouts, _ = parse_bouts(args.data_dir, args.audio_subdir, logger)
    logger.info(f"parsed {sum(len(v) for v in bouts.values())} bouts across "
                f"{len(bouts)} individuals")

    per_individual, dropped_total = {}, 0
    for ind in sorted(bouts):
        vb = valid_bouts(ind, bouts)
        vb, dropped = apply_duration_floor(vb, logger, ind)
        dropped_total += dropped
        if vb:
            per_individual[ind] = vb

    logger.info(f"after junk-session removal and the {MIN_BOUT_SECONDS}s "
                f"floor: {sum(len(v) for v in per_individual.values())} bouts, "
                f"{len(per_individual)} individuals ({dropped_total} dropped)")

    for protocol, fname in [("session_loso", "cv_session_loso.json"),
                            ("by_file_5fold", "cv_by_file_5fold.json")]:
        logger.info("\n" + "=" * 72)
        logger.info(protocol.upper())
        logger.info("=" * 72)
        m = build(protocol, per_individual, args.audio_subdir, logger)
        path = out_dir / fname
        with open(path, "w") as f:
            json.dump(m, f, indent=2)
        logger.info(f"wrote {path}  ({len(m['items'])} bouts, "
                    f"{m['n_folds']} folds x {m['n_repeats']} repeats)")
        if m["split_exceptions"]:
            for ind, e in m["split_exceptions"].items():
                logger.warning(f"  EXCEPTION {ind}: {e['reason']} -> "
                               f"{e['fallback']}")


if __name__ == "__main__":
    main()
