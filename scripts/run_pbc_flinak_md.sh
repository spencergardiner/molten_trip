#!/bin/bash
# run_flibenak_sim_pbc.sh - Uses proper PBC handling

# ----------- CONFIGURATION -------------
TEMPERATURE=1019          # in Kelvin
TORCH_GPU_INDEX=0
DT=0.1                       # time step in femtoseconds
SIM_TIME=1                  # total simulation time in nanoseconds
MINIMIZE=true
BASE_OUTPUT_DIR="./md_simulations"
TRIP_MODEL_FILE="./models/400k_3layers_fw_10/400k_3layers_fw_10.pth"
MINIMIZED_NAME="minimized"
TRAJECTORY_NAME="trajectory"
# ---------------------------------------

OUTPUT_DIR="${BASE_OUTPUT_DIR}/eutectic_flinak_${TEMPERATURE}K_pbc"
mkdir -p "$OUTPUT_DIR"
OUTPUT_LOG="${OUTPUT_DIR}/run.log"

# Construct Python command
CMD="python -m trip.tools.flinak_md_pbc \
  --temp $TEMPERATURE \
  --gpu $TORCH_GPU_INDEX \
  --dt $DT \
  --t $SIM_TIME \
  --out $OUTPUT_DIR \
  --model_file $TRIP_MODEL_FILE \
"

# Add minimization flag
if [ "$MINIMIZE" = true ]; then
  CMD+=" --minimize"
fi

# Print and run the command
echo "Running command:"
echo $CMD
eval $CMD | tee $OUTPUT_LOG
