#!/usr/bin/env python3
"""
compute_density_vs_time.py

Given a topology and trajectory:
  - load the MD trajectory & topology
  - for each frame (after skipping equilibration), compute instantaneous box volume
  - compute total mass once
  - calculate density (g/cm3) per frame
Finally, plot density vs. simulation time, saving the figure into
the same folder as the trajectory file.

Usage:
  python compute_density_vs_time.py \
      --topo path/to/my_system.pdb \
      --traj path/to/my_system.dcd \
      --skip-frames 100 \
      --out-prefix density_vs_time
"""

import argparse
import os
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

# atomic mass unit to grams
AMU_TO_G = 1.66053906660e-24
# 1 Å^3 to cm^3
ANG3_TO_CM3 = 1e-24

def compute_mass(universe):
    masses = universe.atoms.masses  # in amu
    total_mass_amu = masses.sum()
    return total_mass_amu * AMU_TO_G  # in grams

def compute_densities(universe, skip_frames=0):
    total_mass_g = compute_mass(universe)
    times = []
    densities = []

    for ts in universe.trajectory[skip_frames:]:
        # time in ps
        times.append(ts.time)

        # box volume in Å^3
        dims = ts.dimensions  # [a, b, c, alpha, beta, gamma]
        vol_ang3 = dims[0] * dims[1] * dims[2]
        vol_cm3 = vol_ang3 * ANG3_TO_CM3

        rho = total_mass_g / vol_cm3  # g/cm³
        densities.append(rho)

    return np.array(times), np.array(densities)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo', type=str, required=True,
                        help='Topology file (PDB, GRO, etc.)')
    parser.add_argument('--traj', type=str, required=True,
                        help='Trajectory file (DCD, XTC, etc.)')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Number of initial frames to skip (equilibration)')
    parser.add_argument('--out-prefix', type=str, default='density_vs_time',
                        help='Prefix for output CSV and plot')
    args = parser.parse_args()

    # Load universe
    print(f"Loading topology: {args.topo}")
    print(f"Loading trajectory: {args.traj}")
    u = mda.Universe(args.topo, args.traj)

    # Compute densities
    times, densities = compute_densities(u, skip_frames=args.skip_frames)
    print(f"Computed densities for {len(densities)} frames (skipped first {args.skip_frames}).")

    # Determine output directory (same as trajectory)
    traj_dir = os.path.dirname(os.path.abspath(args.traj))
    if not traj_dir:
        traj_dir = os.getcwd()

    # Save time–density data to CSV in traj dir
    out_csv = os.path.join(traj_dir, f"{args.out_prefix}.csv")
    header = "Time_ps,Density_g_per_cm3"
    data = np.column_stack((times, densities))
    np.savetxt(out_csv, data, header=header, delimiter=",", comments='')
    print(f"Saved time–density data to {out_csv}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(times, densities, marker='.', linestyle='-')
    plt.xlabel("Time (ps)")
    plt.ylabel("Density (g/cm³)")
    plt.title("Density vs. Time")
    plt.grid(True)
    plt.tight_layout()

    # Save plot PNG into the same directory as the trajectory
    out_png = os.path.join(traj_dir, f"{args.out_prefix}.png")
    plt.savefig(out_png, dpi=300)
    print(f"Saved density plot to {out_png}")
    plt.show()

if __name__ == "__main__":
    main()
