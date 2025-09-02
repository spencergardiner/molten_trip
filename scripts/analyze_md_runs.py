#!/usr/bin/env python3
"""
collect_and_plot_md_runs.py

Search recursively under a root directory for files named `md_run.log`.
For each found:
 - parse lines whose first field is an integer step and next four fields are numeric
 - build a pandas DataFrame indexed by step with columns:
   "Potential Energy (kJ/mole)","Total Energy (kJ/mole)","Temperature (K)","Density (g/mL)"
 - save DataFrame as md_run.csv in same directory as the log
 - plot Temperature vs Time (ps) where time_ps = step * step_size_fs / 1000.0
   and save plot as temp_vs_time.png in same directory.

Usage:
    python collect_and_plot_md_runs.py /path/to/root --step-fs 0.5
"""

import os
import argparse
import csv
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required. Install with: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib")

COLUMN_NAMES = [
    "Potential Energy (kJ/mole)",
    "Total Energy (kJ/mole)",
    "Temperature (K)",
    "Density (g/mL)",
]

def parse_md_run_log(path):
    steps = []
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # skip obvious non-csv lines
            if line.startswith("INFO:") or line.startswith("#") and "Step" not in line:
                # keep header starting with #"Step" will be skipped below
                pass
            # Try to parse CSV-style line
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            # first token should be integer step
            try:
                step = int(parts[0].strip().strip('"').strip("'"))
            except Exception:
                continue
            # next four tokens numeric
            try:
                vals = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            except Exception:
                continue
            steps.append(step)
            data.append(vals)
    if not steps:
        return None
    df = pd.DataFrame(data, index=steps, columns=COLUMN_NAMES)
    df.index.name = "Step"
    # sort by step index in case not ordered
    df = df.sort_index()
    return df

def make_temp_plot(df, out_png, step_fs, target_temp_str):
    # df indexed by step
    steps = df.index.to_numpy(dtype=float)
    time_ps = steps * (float(step_fs) / 1000.0)  # convert fs -> ps
    temps = df["Temperature (K)"].to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(time_ps, temps, lw=1.2, marker=None)
    plt.xlabel("Time (ps)")
    plt.ylabel("Temperature (K)")
    plt.title(f"Temperature vs Time, Target {target_temp_str}")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def process_root(root_path, step_fs):
    root = Path(root_path)
    if not root.exists():
        raise SystemExit(f"Input path {root} does not exist")
    found = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "md_run.log" in filenames:
            found += 1
            log_path = Path(dirpath) / "md_run.log"
            print(f"Processing: {log_path}")
            df = parse_md_run_log(log_path)
            if df is None or df.empty:
                print(f"  No numeric step/data lines found in {log_path}, skipping.")
                continue
            csv_out = Path(dirpath) / "md_run.csv"
            df.to_csv(csv_out, index=True)
            print(f"  Saved CSV: {csv_out}")
            # target temp contains the only number in the dir base name
            target_temp = os.path.basename(dirpath).split("_")[-1]
            png_out = Path(dirpath) / "temp_vs_time.png"
            try:
                make_temp_plot(df, png_out, step_fs, target_temp)
                print(f"  Saved plot: {png_out}")
            except Exception as e:
                print(f"  Failed to create plot for {log_path}: {e}")
    if found == 0:
        print("No md_run.log files found under", root)
    else:
        print(f"Finished. Processed {found} md_run.log file(s).")

def main():
    p = argparse.ArgumentParser(description="Collect MD run logs and plot temperature vs time.")
    p.add_argument("--root", help="Root path to search for md_run.log files")
    p.add_argument("--step-fs", type=float, default=0.5,
                   help="MD integrator step size in femtoseconds (default: 0.5 fs)")
    args = p.parse_args()
    process_root(args.root, args.step_fs)

if __name__ == "__main__":
    main()