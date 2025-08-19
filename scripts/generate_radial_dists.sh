#!/bin/bash
# Usage: ./run_rdfs_all.sh /path/to/simulations [extra args for RDF script]

set -e

# Input root directory
ROOT_DIR="$1"
shift  # shift off first argument (path)
EXTRA_ARGS="$@"  # pass remaining args to RDF script

# Path to the RDF script
SCRIPT_PATH="./trip/tools/calculate_radial_distributions.py"

# Check script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: RDF script not found at $SCRIPT_PATH"
    exit 1
fi

# Loop through subdirectories
for sim_dir in "$ROOT_DIR"/*/; do
    echo "Processing: $sim_dir"

    # Find topology and trajectory files
    topo_file=$(find "$sim_dir" -maxdepth 1 -type f -name "*.pdb" | head -n 1)
    traj_file=$(find "$sim_dir" -maxdepth 1 -type f -name "*.dcd" | head -n 1)

    if [[ -z "$topo_file" || -z "$traj_file" ]]; then
        echo "  Skipping: Missing .pdb or .dcd file"
        continue
    fi

    # Output directory
    output_dir="$sim_dir/rdf_output"
    mkdir -p "$output_dir"

    # Run RDF script
    python "$SCRIPT_PATH" \
        -t "$topo_file" \
        -d "$traj_file" \
        -o "$output_dir" \
        $EXTRA_ARGS
done
