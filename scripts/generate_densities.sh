#!/bin/bash
# Usage: ./run_density_vs_time_all.sh /path/to/simulations --skip-frames 100

set -e

# Input args
ROOT_DIR="$1"
shift  # remove the first arg (path)
EXTRA_ARGS="$@"  # optional arguments (like --skip-frames)

# Path to your python script
SCRIPT_PATH="./trip/tools/density_vs_time.py"

# Check script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: compute_density_vs_time.py not found at $SCRIPT_PATH"
    exit 1
fi

# Loop over each subdirectory
for sim_dir in "$ROOT_DIR"/*/; do
    echo "Processing directory: $sim_dir"

    # Find topology and trajectory files (assumes one of each)
    topo_file=$(find "$sim_dir" -maxdepth 1 -type f -name "*.pdb" | head -n 1)
    traj_file=$(find "$sim_dir" -maxdepth 1 -type f -name "*.dcd" | head -n 1)

    if [[ -z "$topo_file" || -z "$traj_file" ]]; then
        echo "  Skipping: Missing .pdb or .dcd file"
        continue
    fi

    # Run the Python script
    python "$SCRIPT_PATH" \
        --topo "$topo_file" \
        --traj "$traj_file" \
        --out-prefix "density_vs_time" \
        $EXTRA_ARGS
done
