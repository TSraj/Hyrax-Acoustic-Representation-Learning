#!/bin/bash
# Submit the LoRA sweep: cache prep job, then the 16-task array depending on it.
#
# Run this from the login node (it is NOT itself a batch job):
#   ./run_phase3_lora_submit.sh
#
# Guards against the failure mode where the prep job fails to submit, $JID comes
# back empty, and the array then dies with "Job dependency problem".

set -e

# Always operate from the repo root, whatever directory this was invoked from.
cd "$(dirname "$0")"

# Prefer the project venv. On the login node bare `python` is usually the system
# interpreter, which does not have peft even when the venv does.
if [ -x venv/bin/python ]; then
    PY="venv/bin/python"
else
    PY="python"
fi

# --- prerequisites -----------------------------------------------------------
HYRAX_MANIFEST="outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_session_holdout_ft.json"
SPECIES_MANIFEST="outputs/phase3/manifests/species_id.json"

missing=0
for f in "$HYRAX_MANIFEST" "$SPECIES_MANIFEST"; do
    if [ ! -f "$f" ]; then
        echo "MISSING manifest: $f"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    cat <<'EOM'

Manifests live under outputs/, which is gitignored, so they do not arrive with
a git pull. Regenerate them first:

  python scripts/phase3_02_create_manifests.py \
      --audio-source bioda --tasks session_ft \
      --output-dir outputs/phase3/denoiser_screen/manifests/bioda

  # only if species_id.json is absent:
  python scripts/phase3_02_create_manifests.py --tasks all
EOM
    exit 1
fi

if ! "$PY" -c "import peft" 2>/dev/null; then
    echo "WARNING: peft not importable via $PY."
    echo "  If you installed it inside the venv the batch jobs will still be fine,"
    echo "  since they run 'source venv/bin/activate' themselves."
    echo "  Continuing - re-run with STRICT=1 to make this fatal."
    [ "${STRICT:-0}" = "1" ] && exit 1
fi

# --- submit ------------------------------------------------------------------
echo "Submitting cache prep job..."
JID=$(sbatch --parsable run_phase3_lora_cache.sh)

if [ -z "$JID" ]; then
    echo "ERROR: cache job did not submit; not submitting the array."
    exit 1
fi
echo "  cache job: $JID"

echo "Submitting sweep array (depends on $JID)..."
AID=$(sbatch --parsable --dependency=afterok:"$JID" run_phase3_lora_sweep.sh)

if [ -z "$AID" ]; then
    echo "ERROR: array did not submit. Cancel the prep job with: scancel $JID"
    exit 1
fi
echo "  sweep array: $AID"

echo ""
echo "Queued. Watch with:  squeue -u \$USER"
echo "Results will land in outputs/phase3/lora_sweep/<task>/<model>/frac<NN>/"
