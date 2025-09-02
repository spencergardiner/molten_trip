#!/bin/bash
# run_og_torch_flinak_md.sh
# TorchForce-integrated TrIP molten salt MD (FLiNaK eutectic)

############################################
# User configuration
############################################
TEMPERATURE=1026            # Kelvin
DT=0.1                      # fs timestep
SIM_TIME=0.02               # ns total time
FRICTION=1.0                # 1/ps Langevin friction
MINIMIZE=true               # Perform energy minimization
NPT=false                   # If true add MonteCarloBarostat (1 bar)
REPORT_INTERVAL=100         # Steps between trajectory frames & state data
CHECKPOINT_INTERVAL=100    # Steps between checkpoint writes
GPU_INDEX=3                 # Which GPU to use
TRIP_MODEL_FILE="./models/400k_3layers_fw_10/400k_3layers_fw_10.pth"  # TrIP checkpoint (kwargs-style OK)
BASE_OUTPUT_DIR="./md_simulations"  # Base output directory
############################################

set -euo pipefail

OUTPUT_DIR="${BASE_OUTPUT_DIR}/torchforce_flinak_${TEMPERATURE}K"
mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/run.log"

CMD=(python -m trip.tools.flinak_md_og_torch \
  --model_file "${TRIP_MODEL_FILE}" \
  --temp ${TEMPERATURE} \
  --dt ${DT} \
  --t ${SIM_TIME} \
  --friction ${FRICTION} \
  --gpu ${GPU_INDEX} \
  --minimize ${MINIMIZE} \
  --npt ${NPT} \
  --interval ${REPORT_INTERVAL} \
  --checkpoint_interval ${CHECKPOINT_INTERVAL} \
  --out "${OUTPUT_DIR}")

echo "Running TorchForce TrIP FLiNaK MD:"
echo "${CMD[@]}" | tee -a "${LOGFILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOGFILE}"
