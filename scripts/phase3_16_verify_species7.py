#!/usr/bin/env python3
"""
Phase A - Step 5: Verification gate for the 7-class species manifest + cache.

Run this BEFORE any adaptation run (Phase B). It is the go/no-go for the claim
that the encoder never sees hyrax during animal adaptation.

Checks, in order:

  MANIFEST
    1. num_classes == 7 and exactly the 7 expected non-hyrax species.
    2. 'hyrax' appears in NO class list, index map, or class-weight key.
    3. NO file path under outputs/phase3/hyrax_data/ appears in ANY split -
       this is the actual contamination test. The 8-class manifest sources its
       hyrax class from those concatenated per-individual wavs, which cover all
       18 individuals of the hyrax-ID task (including all 8 of the
       session-holdout cohort, 5 of them in the species TRAIN split).
    4. No item carries species == 'hyrax', and no item's individual matches a
       known hyrax individual ID (belt-and-braces against a mislabelled item).
    5. Removed-item count is exactly 18 (14 train / 2 val / 2 test), so the
       exclusion is provably complete rather than vacuously empty.
    6. Per-split totals match the 8-class manifest minus its hyrax items.
    7. Smallest class is above a viable training size.
    8. The comparability note is present.

  CACHE (only if --cache-dir is given and populated)
    9. The three new split cache hashes differ from the 8-class ones.
   10. Every 8-class cache file still exists, with unchanged size and mtime
       against a recorded baseline (or just still-present on first run).
   11. Cache row counts match the manifest split sizes, and the label arrays
       contain exactly 7 distinct classes with no index >= 7.

Exit code 0 = all pass. Non-zero = do not proceed to Phase B.

Usage:
    python scripts/phase3_16_verify_species7.py
    python scripts/phase3_16_verify_species7.py \
        --cache-dir outputs/phase3/window_cache_species7
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.5

EXPECTED_SPECIES = [
    'anuraset', 'bengalese_finch', 'macaque', 'marmoset',
    'picidae', 'wetlands_bird', 'zebra_finch',
]
HYRAX_DATA_PREFIX = 'outputs/phase3/hyrax_data'
EXPECTED_REMOVED = {'train': 14, 'val': 2, 'test': 2}
MIN_VIABLE_TRAIN_FILES = 50


class Gate:
    """Collects pass/fail results so every check runs before we exit."""

    def __init__(self):
        self.results = []

    def check(self, name, ok, detail="", always_show_detail=False):
        """detail explains a FAILURE, so it is only printed when the check fails
        (or when the caller explicitly wants the value echoed on success)."""
        self.results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"\n         {detail}" if detail and (not ok or always_show_detail)
                 else ""))
        return bool(ok)

    def section(self, title):
        print(f"\n{title}\n" + "-" * 78)

    @property
    def passed(self):
        return all(ok for _, ok, _ in self.results)

    def summary(self):
        n_fail = sum(1 for _, ok, _ in self.results if not ok)
        print("\n" + "=" * 78)
        if self.passed:
            print(f"VERIFICATION GATE: PASS ({len(self.results)} checks)")
            print("The 7-class manifest is clean. Phase B may proceed.")
        else:
            print(f"VERIFICATION GATE: FAIL ({n_fail}/{len(self.results)} checks failed)")
            print("DO NOT proceed to Phase B.")
            for name, ok, detail in self.results:
                if not ok:
                    print(f"  - {name}: {detail}")
        print("=" * 78)


def cache_key(items, label_key, max_windows_per_file):
    """Reproduce WindowedDataset._cache_key from phase3_10_lora_fine_tuning.py.

    Kept as an independent copy on purpose: this gate must be able to detect a
    key collision with the 8-class cache without importing the trainer.
    """
    h = hashlib.md5()
    h.update(f"{WINDOW_SECONDS}|{STRIDE_SECONDS}|{label_key}|{max_windows_per_file}".encode())
    for it in items:
        h.update(str(it['file']).encode())
    return h.hexdigest()[:12]


def all_items(manifest):
    for split in ('train', 'val', 'test'):
        for it in manifest['splits'].get(split, []):
            yield split, it


def verify_manifest(gate, m7, m8):
    gate.section("MANIFEST")

    # 1 - class count and identity
    gate.check("num_classes == 7", m7.get('num_classes') == 7,
               f"got {m7.get('num_classes')}")
    gate.check("class list is exactly the 7 expected non-hyrax species",
               list(m7.get('species', [])) == EXPECTED_SPECIES,
               f"got {m7.get('species')}")
    gate.check("species_to_idx is a contiguous 0..6 map",
               sorted(m7.get('species_to_idx', {}).values()) == list(range(7)),
               f"got {m7.get('species_to_idx')}")

    # 2 - hyrax absent from every class-level structure
    hyrax_in_meta = [
        field for field in ('species', 'species_to_idx', 'class_weights')
        if 'hyrax' in (m7.get(field) or {})
    ]
    gate.check("'hyrax' absent from species / species_to_idx / class_weights",
               not hyrax_in_meta, f"found in {hyrax_in_meta}")

    # 3 - THE contamination test: no hyrax_data path anywhere
    hyrax_paths = [(split, it['file']) for split, it in all_items(m7)
                   if str(it['file']).replace('\\', '/').startswith(HYRAX_DATA_PREFIX)]
    gate.check(f"no file path under {HYRAX_DATA_PREFIX}/ in any split",
               not hyrax_paths,
               f"{len(hyrax_paths)} contaminated items, e.g. {hyrax_paths[:3]}")

    # 4 - no hyrax-labelled item, no known hyrax individual ID
    hyrax_labelled = [(s, it['file']) for s, it in all_items(m7)
                      if it.get('species') == 'hyrax']
    gate.check("no item carries species == 'hyrax'", not hyrax_labelled,
               f"{len(hyrax_labelled)} items")

    hyrax_inds = set()
    for ids in (m8.get('hyrax_individuals') or {}).values():
        hyrax_inds.update(ids)
    leaked_inds = sorted({it['individual'] for _, it in all_items(m7)
                          if it.get('individual') in hyrax_inds})
    gate.check("no item's individual matches a known hyrax individual ID",
               not leaked_inds, f"leaked: {leaked_inds}")

    # 5 - exclusion is provably complete, not vacuous
    removed = m7.get('excluded_item_counts') or {}
    gate.check("excluded_species == ['hyrax']",
               m7.get('excluded_species') == ['hyrax'],
               f"got {m7.get('excluded_species')}")
    total_removed = sum(removed.values())
    print(f"         removed items: {total_removed} total "
          f"(train {removed.get('train', 0)}, val {removed.get('val', 0)}, "
          f"test {removed.get('test', 0)})")
    gate.check("removed-item count is exactly 18 (14 train / 2 val / 2 test)",
               {k: removed.get(k, 0) for k in EXPECTED_REMOVED} == EXPECTED_REMOVED
               and total_removed == 18,
               f"got {dict(removed)} (total {total_removed}); a zero here would "
               f"mean the exclusion never had anything to remove")

    # 6 - split totals equal the 8-class totals minus its hyrax items
    for split in ('train', 'val', 'test'):
        n7 = len(m7['splits'].get(split, []))
        n8 = len(m8['splits'].get(split, []))
        n8_hyrax = sum(1 for it in m8['splits'].get(split, [])
                       if it.get('species') == 'hyrax')
        gate.check(f"{split}: 7-class count == 8-class count - hyrax items "
                   f"({n8} - {n8_hyrax} = {n8 - n8_hyrax})",
                   n7 == n8 - n8_hyrax, f"got {n7}")

    # 6b - the non-hyrax file sets are identical (nothing else was dropped)
    files7 = {str(it['file']) for _, it in all_items(m7)}
    files8_nonhyrax = {str(it['file']) for _, it in all_items(m8)
                       if it.get('species') != 'hyrax'}
    gate.check("non-hyrax file set is byte-identical to the 8-class manifest",
               files7 == files8_nonhyrax,
               f"only in 7-class: {len(files7 - files8_nonhyrax)}, "
               f"only in 8-class: {len(files8_nonhyrax - files7)}")

    # 7 - viable class sizes
    counts = {}
    for it in m7['splits']['train']:
        counts[it['species']] = counts.get(it['species'], 0) + 1
    smallest = min(counts, key=counts.get)
    gate.check(f"smallest train class is viable "
               f"({smallest}={counts[smallest]} >= {MIN_VIABLE_TRAIN_FILES})",
               counts[smallest] >= MIN_VIABLE_TRAIN_FILES,
               f"per-class train counts: {counts}",
               always_show_detail=True)

    # 8 - comparability note carried in the manifest
    note = m7.get('comparability_note', '')
    gate.check("comparability_note present and mentions the f1_7 hazard",
               bool(note) and 'f1_7' in note and '7-WAY' in note,
               "missing or incomplete note")

    return counts


def verify_cache(gate, m7, m8, cache_dir, old_cache_dir, baseline_path):
    gate.section(f"CACHE ({cache_dir})")

    label_key, mwpf = 'species', 1
    keys7, keys8 = {}, {}
    for split in ('train', 'val', 'test'):
        keys7[split] = cache_key(m7['splits'][split], label_key, mwpf)
        keys8[split] = cache_key(m8['splits'][split], label_key, mwpf)

    # 9 - new hashes must differ from the 8-class ones
    collisions = [s for s in keys7 if keys7[s] == keys8[s]]
    gate.check("all three 7-class cache hashes differ from the 8-class hashes",
               not collisions,
               f"COLLISION on {collisions} - a shared key would silently feed "
               f"hyrax-contaminated windows into the 7-class run")
    for split in ('train', 'val', 'test'):
        print(f"         {split}: 7-class {keys7[split]}  |  8-class {keys8[split]}")

    # 10 - the 8-class SPECIES cache must be untouched.
    #
    # Scoped to the six filenames the 8-class species manifest hashes to. The
    # cache dir is shared with the hyrax tasks, whose files are irrelevant here
    # and must not be mistaken for the species baseline. The species cache was
    # built on HPC, so on a laptop none of the six will be present - that is a
    # SKIP, not a failure. A failure is a PARTIAL set: some present, some gone,
    # which is what actual damage looks like.
    expected_old = [f"{s}_{keys8[s]}_{kind}.npy"
                    for s in ('train', 'val', 'test')
                    for kind in ('windows', 'labels')]
    present = {}
    for name in expected_old:
        p = old_cache_dir / name
        if p.exists():
            st = p.stat()
            present[name] = {'size': st.st_size, 'mtime': int(st.st_mtime)}

    if not present:
        print(f"  [SKIP] 8-class species cache not on this filesystem "
              f"(built on HPC) - none of its {len(expected_old)} files in "
              f"{old_cache_dir}")
    else:
        missing_old = [n for n in expected_old if n not in present]
        gate.check("8-class species cache files all still present",
                   not missing_old,
                   f"{len(present)}/{len(expected_old)} present, missing: "
                   f"{missing_old} - a partial set means files were deleted")

        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            drifted = [n for n, meta in baseline.items()
                       if n not in present
                       or present[n]['size'] != meta['size']
                       or present[n]['mtime'] != meta['mtime']]
            gate.check("8-class species cache unchanged vs recorded baseline "
                       "(size + mtime)", not drifted, f"drifted: {drifted}")
        else:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_path, 'w') as f:
                json.dump(present, f, indent=2)
            print(f"  [INIT] recorded 8-class species cache baseline -> "
                  f"{baseline_path} ({len(present)} files); re-run to enforce "
                  f"immutability")

    # 11 - new cache contents match the manifest
    any_new = False
    for split in ('train', 'val', 'test'):
        lab_f = cache_dir / f"{split}_{keys7[split]}_labels.npy"
        win_f = cache_dir / f"{split}_{keys7[split]}_windows.npy"
        if not (lab_f.exists() and win_f.exists()):
            print(f"  [SKIP] {split}: cache not built yet ({lab_f.name})")
            continue
        any_new = True
        labels = np.load(lab_f)
        windows = np.load(win_f, mmap_mode='r')
        n_expected = len(m7['splits'][split])

        gate.check(f"{split}: cache rows == manifest files ({n_expected})",
                   len(labels) == n_expected and windows.shape[0] == n_expected,
                   f"labels {len(labels)}, windows {windows.shape[0]}")
        uniq = np.unique(labels)
        gate.check(f"{split}: labels span exactly 7 classes, max index 6",
                   len(uniq) == 7 and uniq.min() == 0 and uniq.max() == 6,
                   f"distinct={len(uniq)} min={uniq.min()} max={uniq.max()}; "
                   f"an index of 7 would be the old hyrax slot")

    if not any_new:
        print("\n  No 7-class cache built yet - run A3 (cache prep) then re-run "
              "this gate with --cache-dir to complete checks 9-11.")

    return keys7, keys8


def main():
    p = argparse.ArgumentParser(description="Phase A verification gate")
    p.add_argument("--manifest-7",
                   default="outputs/phase3/manifests_species7/species_id.json")
    p.add_argument("--manifest-8",
                   default="outputs/phase3/manifests/species_id.json")
    p.add_argument("--cache-dir", default=None,
                   help="7-class window cache dir (checks 9-11)")
    p.add_argument("--old-cache-dir", default="outputs/phase3/window_cache",
                   help="8-class window cache dir, asserted unchanged")
    p.add_argument("--baseline",
                   default="outputs/phase3/manifests_species7/_cache_baseline_8class.json",
                   help="Recorded size+mtime of the 8-class cache files")
    args = p.parse_args()

    print("=" * 78)
    print("PHASE A - VERIFICATION GATE: 7-class species manifest + cache")
    print("=" * 78)

    m7_path, m8_path = Path(args.manifest_7), Path(args.manifest_8)
    for label, path in (("7-class", m7_path), ("8-class", m8_path)):
        if not path.exists():
            print(f"\nERROR: {label} manifest not found: {path}")
            return 2

    with open(m7_path) as f:
        m7 = json.load(f)
    with open(m8_path) as f:
        m8 = json.load(f)

    print(f"\n7-class: {m7_path}")
    print(f"8-class: {m8_path}  (reference, must stay untouched)")

    gate = Gate()
    verify_manifest(gate, m7, m8)

    if args.cache_dir:
        verify_cache(gate, m7, m8, Path(args.cache_dir),
                     Path(args.old_cache_dir), Path(args.baseline))
    else:
        print("\n[SKIP] cache checks - pass --cache-dir to enable")

    gate.summary()

    if m7.get('comparability_note'):
        print("\nCOMPARABILITY NOTE (carried in the manifest):")
        for line in _wrap(m7['comparability_note'], 74):
            print(f"  {line}")

    return 0 if gate.passed else 1


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
