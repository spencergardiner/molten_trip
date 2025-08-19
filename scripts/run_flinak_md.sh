#!/bin/bash
# run_flibenak_sim.sh

# ----------- CONFIGURATION -------------
TEMPERATURE=1018          # in Kelvin
TORCH_GPU_INDEX=0
DT=0.1                       # time step in femtoseconds
DT=0.1                       # time step in femtoseconds
SIM_TIME=20                  # total simulation time in picoseconds
ENSEMBLE_NPT=false            # set to false to run NVT
MINIMIZE=true
IS_NVE=false
BASE_OUTPUT_DIR="./md_simulations"
TRIP_MODEL_FILE="./models/400k_3layers_fw_10/400k_3layers_fw_10.pth"
MINIMIZED_NAME="minimized"
TRAJECTORY_NAME="trajectory"
# ---------------------------------------

OUTPUT_DIR="${BASE_OUTPUT_DIR}/eutectic_flinak_${TEMPERATURE}K"
mkdir -p "$OUTPUT_DIR"
OUTPUT_LOG="${OUTPUT_DIR}/run.log"

# Construct Python command
CMD="python -m trip.tools.flinak_md \
  --temperature $TEMPERATURE \
  --torch_gpu_index $TORCH_GPU_INDEX \
  --dt $DT \
  --time $SIM_TIME \
  --output_dir $OUTPUT_DIR \
  --trip_model_file $TRIP_MODEL_FILE \
  --minimized_name $MINIMIZED_NAME \
  --trajectory_name $TRAJECTORY_NAME"

# Add NPT/NVT flag
if [ "$ENSEMBLE_NPT" = true ]; then
  CMD+=" --is_NPT"
fi

# Add minimization flag
if [ "$MINIMIZE" = true ]; then
  CMD+=" --minimize"
fi

# add is_NVE flag
if [ "$IS_NVE" = true ]; then
  CMD+=" --is_NVE"
fi

# Print and run the command
echo "Running command:"
echo $CMD
eval $CMD | tee $OUTPUT_LOG
