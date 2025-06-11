import argparse
import os

import pathlib
from typing import List


import warnings

import numpy as np
from tqdm import tqdm

import torch




from trip.data_loading import GraphConstructor
from trip.model import TrIP

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



def gen_pos_list(distance: float, device)-> List[float]:
    """Generate a list of positions for the interatomic distance plot."""
    return [
        torch.tensor([0.0, distance], device=device),
    ]

def run_stepped_interatomic_inference(
        model: TrIP,
        graph_constructor: GraphConstructor,
        species_1: int, 
        species_2: int,
        device,
        step_size: float = 0.01,
        min_distance: float = 0.5,
):
    """Run inference on a TrIP model to generate interatomic distance plots.

    Args:
        model (TrIP): The trained TrIP model.
        graph_constructor (GraphConstructor): The graph constructor for creating graphs.
        species_1 (int): The atomic number of the first species.
        species_2 (int): The atomic number of the second species.
        step_size (float): The step size for the distance increments.
        min_distance (float): The minimum interatomic distance to start from, in angstroms.

    Returns:
        dict: A dictionary containing the predicted energies and forces for each distance.
    """
    
    min_distance = 0.5

    species = torch.tensor([species_1, species_2], device=device, dtype=torch.long)
    boxsize = torch.tensor([[20, 20, 20]], device=device, dtype=torch.float32)

    outputs = {}
    for distance in np.arange(min_distance, model.cutoff + step_size, step_size):
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [distance, 0.0, 0.0]
            ],
            device=device,
            dtype=torch.float32
        )
        pos_list = [positions]

        graph = graph_constructor.create_graphs(pos_list, boxsize)
        graph.ndata['species'] = species
        graph = graph.to(device)

        with torch.amp.autocast('cuda', enabled=True):
            pred = model(graph, create_graph=False, standardized=True)
            pred_energy, pred_forces = pred

        outputs[distance] = {
            'energy': pred_energy.cpu().detach().numpy(),
            'forces': pred_forces.cpu().detach().numpy()
        }
    
    return outputs


def convert_atomic_number_to_symbol(atomic_number: int) -> str:
    """Convert atomic number to chemical symbol."""
    periodic_table = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
        6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
        15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
        19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
        27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        # Add more elements as needed
    }
    """Convert atomic number to chemical symbol."""

    if atomic_number not in periodic_table:
        warnings.warn(f"Atomic number {atomic_number} not found in periodic table. Returning 'Unknown(atomic_number)'.")
   
    return periodic_table.get(atomic_number, f'Unknown({atomic_number})')


def convert_symbol_to_atomic_number(symbol: str) -> int:
    """Convert chemical symbol to atomic number."""
    periodic_table = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
        'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14,
        'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        # Add more elements as needed
    }
    return periodic_table.get(symbol, -1)
    

def plot_interatomic_distance(
        outputs: dict,
        species_1: int,
        species_2: int,
        output_dir: pathlib.Path,
        step_size: float = 0.01,
        factor: float = 1.0,
):
    """Plot the interatomic distance and save the figure.

    Args:
        outputs (dict): The dictionary containing predicted energies and forces.
        species_1 (int): The atomic number of the first species.
        species_2 (int): The atomic number of the second species.
        output_dir (pathlib.Path): The directory to save the plot.
        step_size (float): The step size for the distance increments.
        factor (float): used to convert energies from Hartrees to a different unit (default is 1.0, no conversion).
    """
    energy_unit_label = "Ha"
    if round(factor) == 27:
        energy_unit_label = "eV"
    elif round(factor) in [627, 628]:
        energy_unit_label = "kcal/mol"
    elif factor == 1.0:
        energy_unit_label = "Ha"
    else:
        raise ValueError(f"Unsupported factor {factor}. Supported factors are 1.0 (Ha), 27.2 (eV), and 627.5 (kcal/mol).")

    distances = list(outputs.keys())
    energies = [outputs[distance]['energy'][0] * factor for distance in distances]

    print(energies)

    # Convert atomic numbers to chemical symbols
    species_1_symbol = convert_atomic_number_to_symbol(species_1)
    species_2_symbol = convert_atomic_number_to_symbol(species_2)

    # Create output directory if it doesn't exist
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Plot energy vs distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, marker=".", color='b', label='Predicted Energies')
    plt.legend()
    plt.title(f'Interatomic Distance Plot: {species_1_symbol}-{species_2_symbol}')
    plt.xlabel('Interatomic Distance (Å)')
    plt.ylabel(f"Energy ({energy_unit_label})")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_dir / f'interatomic_distance_{species_1_symbol}_{species_2_symbol}.png')




def main(args: argparse.Namespace):
    # get device
    device = torch.device(args.device)


    # Load the TrIP model
    model = TrIP.load(args.checkpoint_path)
    model.eval()
    model.to(device)

    # Create the graph constructor
    graph_constructor = GraphConstructor(cutoff=model.cutoff)

    species_list = [convert_symbol_to_atomic_number(symbol) for symbol in args.species]

    # create sets of all possible species pairs
    species_pairs = set()
    for i in range(len(species_list)):
        for j in range(i, len(species_list)):
            species_pairs.add((species_list[i], species_list[j]))

    # create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # iterate through all species pairs
    for species_1, species_2 in species_pairs:
        # Run inference for the current species pair
        outputs = run_stepped_interatomic_inference(
            model,
            graph_constructor,
            species_1,
            species_2,
            device=device,
            step_size=args.step_size,
            min_distance=args.min_distance

        )

        print(f"Processed species pair: {convert_atomic_number_to_symbol(species_1)}-{convert_atomic_number_to_symbol(species_2)}")

        # Plot and save the interatomic distance plot
        plot_interatomic_distance(outputs, species_1, species_2, output_dir, step_size=args.step_size)
        print(f"Saved plot for species pair: {convert_atomic_number_to_symbol(species_1)}-{convert_atomic_number_to_symbol(species_2)}")
    
    print(f"All plots saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interatomic distance plots for TrIP model.")
    parser.add_argument('--checkpoint_path', type=pathlib.Path, required=True, help="Path to the trained TrIP model checkpoint.")
    parser.add_argument('--species', type=str, nargs='+', required=True,
                        help="List of chemical symbols for the species to analyze (e.g., 'H', 'O', 'C').")
    parser.add_argument('--output_dir', type=pathlib.Path, default=pathlib.Path('interatomic_distance_plots'), help="Directory to save the plots.")
    parser.add_argument('--step_size', type=float, default=0.005, help="Step size for interatomic distance increments.")
    parser.add_argument('--min_distance', type=float, default=0.2, help="Minimum interatomic distance to start from (in angstroms).")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the inference on (e.g., 'cuda:0' or 'cpu').")

    args = parser.parse_args()
    
    main(args)