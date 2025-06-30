#!/usr/bin/env python3
"""
Generate a cubic box of eutectic FLiNaK (LiF–NaF–KF) on a grid,
with random ion identities but global molar fractions maintained,
and write out a PDB suitable for OpenMM.
"""

import argparse
import numpy as np
from openmm.app import Topology, PDBFile
from openmm.app.element import Element
from openmm.unit import angstroms
from simtk import unit

def make_flinak_box(box_length, spacing=2.0,
                    x_frac=0.465, y_frac=0.115, z_frac=0.42,
                    output_pdb="flinak.pdb"):
    """
    box_length: float, Å
    spacing: float, grid spacing in Å
    x_frac: fraction of LiF pairs
    y_frac: fraction of NaF pairs
    z_frac: fraction of KF pairs
    output_pdb: filename to write PDB
    """
    # generate grid points
    pts = []
    coords = np.arange(0, box_length, spacing)
    for x in coords:
        for y in coords:
            for z in coords:
                pts.append((x, y, z))
    N = len(pts)
    if N % 2 != 0:
        raise ValueError("Total grid points must be even to form complete MF pairs.")
    # compute number of each pair (MF = metal + F)
    n_pairs = N // 2
    n_LiF = int(round(x_frac * n_pairs))
    n_NaF = int(round(y_frac * n_pairs))
    n_KF  = n_pairs - n_LiF - n_NaF
    # build list of ions
    ions = []
    # For each pair, we need one M⁺ and one F⁻
    ions += ["Li+"] * n_LiF + ["F-"] * n_LiF
    ions += ["Na+"] * n_NaF + ["F-"] * n_NaF
    ions += ["K+" ] * n_KF  + ["F-"] * n_KF
    if len(ions) != N:
        raise RuntimeError("Mismatch in total ion count.")
    # shuffle and assign to points
    np.random.shuffle(ions)
    # build Topology
    top = Topology()
    chain = top.addChain()
    # unit cell
    top.setUnitCellDimensions((box_length*angstroms,
                                box_length*angstroms,
                                box_length*angstroms))
    # add atoms
    positions = []
    for (x, y, z), ion in zip(pts, ions):
        if ion == "Li+":
            elem = Element.getBySymbol("Li")
            resname = "Li"
        elif ion == "Na+":
            elem = Element.getBySymbol("Na")
            resname = "Na"
        elif ion == "K+":
            elem = Element.getBySymbol("K")
            resname = "K"
        elif ion == "F-":
            elem = Element.getBySymbol("F")
            resname = "F"
        else:
            raise ValueError(f"Unknown ion {ion}")
        res = top.addResidue(resname, chain)
        top.addAtom(resname, elem, res)
        positions.append((x, y, z))
    # convert to OpenMM units
    positions = [(x*angstroms, y*angstroms, z*angstroms) for x, y, z in positions]
    # write PDB
    with open(output_pdb, "w") as f:
        PDBFile.writeFile(top, positions, f)
    print(f"Wrote {len(positions)} ions to {output_pdb}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FLiNaK grid PDB")
    parser.add_argument("--box", type=float, required=True,
                        help="Box edge length in Å")
    parser.add_argument("--spacing", type=float, default=2.0,
                        help="Grid spacing in Å")
    parser.add_argument("--out", type=str, default="flinak.pdb",
                        help="Output PDB filename")
    args = parser.parse_args()
    make_flinak_box(args.box, args.spacing, output_pdb=args.out)
