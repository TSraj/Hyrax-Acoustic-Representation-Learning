#!/bin/bash
# Submit the species_id multi-seed + low-fraction array (46 tasks).
#
# Run this from the login node (it is NOT itself a batch job):
#   ./run_phase3_lora_species_submit.sh
#
# Chains the cache prep job in front of the array. That job is idempotent: the
# species window cache already exists from the original sweep, so it will log
# "reusing window cache" and exit in seconds. It is kept in the chain only to
# cover the case where the cache was purged from scratch - without it, 46 array
# tasks would concurrently re-decode ~18k audio files.
#
# Mirrors run_phase3_lora_submit.sh, including the guard against the failure
# mode where the prep job fails to submit, $JID comes back empty, and the array
# then dies with "Job dependency problem".

set -e

cd "$(dirname "$0")"

if [ -x venv/bin/python ]; then
    PY="venv/bin/python"
else
    PY="python"
fi

# --- prerequisites -----------------------------------------------------------
SPECIES_MANIFEST="outputs/phase3/manifests/species_id.json"
HYRAX_MANIFEST="outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_session_holdout_ft.json"

missing=0
for f in "$SPECIES_MANIFEST" "$HYRAX_MANIFEST"; do
    if [ ! -f "$f" ]; then
        echo "MISSING manifest: $f"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    cat <<'EOM'

Manifests live under outputs/, which is gitignored, so they do not arrive with
a git pull. Regenerate them first:

  python scripts/phase3_02_create_manifests.py --tasks all

  python scripts/phase3_02_create_manifests.py \
      --audio-source bioda --tasks session_ft \
      --output-dir outputs/phase3/denoiser_screen/manifests/bioda

The species manifest must be the SAME one the seed-42 runs used - the window
cache key hashes its file list, so a regenerated manifest with a different file
order would miss the cache and rebuild ~18k files.
EOM
    exit 1
fi

if ! "$PY" -c "import peft" 2>/dev/null; then
    echo "WARNING: peft not importable via $PY."
    echo "  Batch jobs source venv/bin/activate themselves, so this may be fine."
    echo "  Re-run with STRICT=1 to make this fatal."
    [ "${STRICT:-0}" = "1" ] && exit 1
fi

# --- show what is about to run ------------------------------------------------
echo "Job table:"
bash run_phase3_lora_species_seeds.sh --list | head -5
echo "  ... (run 'bash run_phase3_lora_species_seeds.sh --list' for all 46)"
echo ""

# --- submit -------------------------------------------------------------------
echo "Submitting cache prep job (no-op if the cache is already built)..."
JID=$(sbatch --parsable run_phase3_lora_cache.sh)

if [ -z "$JID" ]; then
    echo "ERROR: cache job did not submit; not submitting the array."
    exit 1
fi
echo "  cache job: $JID"

echo "Submitting species array (depends on $JID)..."
AID=$(sbatch --parsable --dependency=afterok:"$JID" run_phase3_lora_species_seeds.sh)

if [ -z "$AID" ]; then
    echo "ERROR: array did not submit. Cancel the prep job with: scancel $JID"
    exit 1
fi
echo "  species array: $AID  (46 tasks, 4 concurrent)"

echo ""
echo "Queued. Watch with:  squeue -u \$USER"
echo "Results land in:     outputs/phase3/lora_sweep/species_id/<model>/frac<NN>/seed<S>/"
