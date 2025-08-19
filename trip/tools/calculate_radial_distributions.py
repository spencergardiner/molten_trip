import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from itertools import combinations, combinations_with_replacement


# look at / plot minimum distances between species
# re-write radial distribution function because this is very wrong
#


def compute_rdfs(topology, trajectory, output_dir, bins=200, range_max=10.0,
                 plot_overlaid=False, compare_self=False):
    # Load the simulation
    u = mda.Universe(topology, trajectory)
    # Identify unique elements
    elements = {atom.element for atom in u.atoms if atom.element}
    species = sorted(elements)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Choose pairing strategy
    if compare_self:
        pairs = combinations_with_replacement(species, 2)
    else:
        pairs = combinations(species, 2)

    # If plotting overlaid, initialize figure
    if plot_overlaid:
        plt.figure()

    for sp1, sp2 in pairs:
        sel1 = u.select_atoms(f"element {sp1}")
        sel2 = u.select_atoms(f"element {sp2}")
        if not sel1 or not sel2:
            continue

        # Reset trajectory before RDF computation
        u.trajectory.rewind()

        # Run RDF
        rdf = InterRDF(sel1, sel2, nbins=bins, range=(0.0, range_max))
        rdf.run()

        # Save data
        data = np.column_stack((rdf.bins, rdf.rdf))
        fname = os.path.join(output_dir, f"rdf_{sp1}_{sp2}.csv")
        np.savetxt(fname, data, header='r,g(r)', delimiter=',')
        print(f"Saved RDF data for {sp1}-{sp2} to {fname}")

        # Plotting
        if plot_overlaid:
            plt.plot(rdf.bins, rdf.rdf, label=f"{sp1}-{sp2}")
        else:
            plt.figure()
            plt.plot(rdf.bins, rdf.rdf)
            plt.xlabel(r"$r$ (Å)")
            plt.ylabel(r"$g(r)$")
            plt.title(f"RDF: {sp1}-{sp2}")
            plt.grid(True)
            out_png = os.path.join(output_dir, f"rdf_{sp1}_{sp2}.png")
            plt.savefig(out_png, dpi=300)
            # change y-lim to 0-2
            plt.xlim(0, 2)
            plt.close()
            print(f"Saved RDF plot for {sp1}-{sp2} to {out_png}")

    # Finalize overlaid plot
    if plot_overlaid:
        plt.xlabel(r"$r$ (Å)")
        plt.ylabel(r"$g(r)$")
        plt.title("Overlaid RDFs")
        plt.legend()
        plt.grid(True)
        out_png = os.path.join(output_dir, "rdf_overlaid.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved overlaid RDF plot to {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute radial distribution functions (RDF) for each species pair from a MD trajectory"
    )
    parser.add_argument("-t", "--topology", required=True,
                        help="Path to the PDB topology file")
    parser.add_argument("-d", "--dcd", required=True,
                        help="Path to the DCD trajectory file")
    parser.add_argument("-o", "--output", required=True,
                        help="Directory to store output CSVs and plots")
    parser.add_argument("--bins", type=int, default=200,
                        help="Number of histogram bins (default: 200)")
    parser.add_argument("--range", type=float, default=10.0,
                        help="Maximum distance to consider in Å (default: 10.0)")
    parser.add_argument("--plot_overlaid", action="store_true",
                        help="If set, overlay all RDF curves on a single plot")
    parser.add_argument("--compare_self", action="store_true",
                        help="If set, compute self RDFs (species-species); otherwise skip them")

    args = parser.parse_args()

    compute_rdfs(
        args.topology,
        args.dcd,
        args.output,
        bins=args.bins,
        range_max=args.range,
        plot_overlaid=True,
        compare_self=args.compare_self
    )


if __name__ == "__main__":
    main()
