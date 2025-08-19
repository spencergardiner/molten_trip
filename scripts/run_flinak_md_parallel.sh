#!/bin/bash
set -euo pipefail

# --- Configuration ---
TEMPERATURES=(800 900 950 1000 1050 1100 1200 1300)
GPU_INDEXES=(0 1 2 3 4 5 6 7)  # Adjust based on available GPUs
DT=0.5              # in fs
TOTAL_TIME=100     # in ps
ENSEMBLE=NPT        # NPT or NVT
MINIMIZE=true
# check if in apptainer container
if [[ -n "${APPTAINER_CONTAINER:-}" ]]; then
  BASE_DIR="/mnt/trip"
else
  BASE_DIR="/lustre/orion/world-shared/gen006/sgardiner/molten_trip"
fi
BASE_OUTPUT_DIR="${BASE_DIR}/md_simulations"
TRIP_MODEL="${BASE_DIR}/models/200k_fw_0p01__r2_fw1_lr1e-3/200k_fw_0p01__r2_fw1_lr1e-3.pth"
CHECKPOINT_INTERVAL=100

# print current working directory
echo "Current working directory: $(pwd)"

# Derived flags
ENSEMBLE_FLAG=""
if [[ "$ENSEMBLE" == "NVT" ]]; then
  # --is_NPT disables NPT (store_false)
  ENSEMBLE_FLAG="--is_NPT"
fi

MIN_FLAG=""
if [[ "$MINIMIZE" == true ]]; then
  MIN_FLAG="--minimize"
fi

# Make sure the base output directory exists
mkdir -p "$BASE_OUTPUT_DIR"

# Iterate over 8 GPU indices
for i in "${!GPU_INDEXES[@]}"; do
  GPU_INDEX=${GPU_INDEXES[$i]}
  TEMPERATURE=${TEMPERATURES[$i % ${#TEMPERATURES[@]}]}


  OUT_DIR="${BASE_OUTPUT_DIR}/eutectic_flinak_2layers_${TEMPERATURE}K"
  mkdir -p "$OUT_DIR"

  echo "Starting simulation on GPU ${GPU_INDEX}, output -> ${OUT_DIR}"

  python -m trip.tools.flinak_md \
    --temperature "$TEMPERATURE" \
    --torch_gpu_index "$GPU_INDEX" \
    --dt "$DT" \
    --time "$TOTAL_TIME" \
    --output_dir "$OUT_DIR" \
    --trip_model_file "$TRIP_MODEL" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    $ENSEMBLE_FLAG \
    $MIN_FLAG \
    >> "${OUT_DIR}/run.log" 2>&1 &

done

# Wait for all background jobs to finish
wait
echo "All simulations complete."
