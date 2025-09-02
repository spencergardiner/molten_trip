#!/bin/bash
set -euo pipefail

# NOTE: run configuration values are exported variables


# Reconstruct arrays from space‑separated environment variables passed in.
# (Bash cannot export true arrays, so upstream scripts export space‑joined strings.)
IFS=' ' read -r -a TEMPERATURES_ARR <<< "${TEMPERATURES:-}"
IFS=' ' read -r -a GPU_INDEXES_ARR <<< "${GPU_INDEXES:-}"

if (( ${#GPU_INDEXES_ARR[@]} == 0 )); then
  echo "ERROR: GPU_INDEXES env var is empty or not set" >&2
  exit 1
fi
if (( ${#TEMPERATURES_ARR[@]} == 0 )); then
  echo "ERROR: TEMPERATURES env var is empty or not set" >&2
  exit 1
fi

# Make sure the base output directory exists
mkdir -p "$BASE_OUTPUT_DIR"

# Iterate over GPU indices and temperatures
for i in "${!GPU_INDEXES_ARR[@]}"; do
  GPU_INDEX=${GPU_INDEXES_ARR[$i]}
  TEMPERATURE=${TEMPERATURES_ARR[$(( i % ${#TEMPERATURES_ARR[@]} ))]}

  OUT_DIR="${BASE_OUTPUT_DIR}/eutectic_flinak_${TEMPERATURE}K"
  mkdir -p "$OUT_DIR"

  echo "Starting simulation on GPU ${GPU_INDEX}, output -> ${OUT_DIR}"

  python -m trip.tools.flinak_md_og \
  --temp $TEMPERATURE \
  --gpu $GPU_INDEX \
  --dt $DT \
  --t $SIM_TIME \
  --out $OUT_DIR \
  --model_file $TRIP_MODEL_FILE \
  --minimize $MINIMIZE \
  --npt $NPT \
  --friction $FRICTION \
    >> "${OUT_DIR}/md_run.log" 2>&1 &

done

# Wait for all background jobs to finish
wait
echo "All simulations complete."
