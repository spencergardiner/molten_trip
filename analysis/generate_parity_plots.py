"""Load a model, run inference, calculate metrics, save results, and generate parity plots."""

import argparse
import os
import logging
import pathlib
from typing import List, Callable


import warnings

import numpy as np
import torch.distributed
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.callbacks import BaseCallback, PerformanceCallback
from se3_transformer.runtime.loggers import LoggerCollection, DLLogger, WandbLogger, Logger
from se3_transformer.runtime.training import print_parameters_count
from se3_transformer.runtime.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity

from trip.data_loading import GraphConstructor, TrIPDataModule
from trip.model import TrIP
from trip.runtime.arguments import PARSER
from trip.runtime.callbacks import TrIPMetricCallback, TrIPLRSchedulerCallback
from trip.runtime.inference import evaluate

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def load_model(model_path: str, device: torch.device) -> TrIP:
    """Load a model from the specified path."""
    model = TrIP.load(model_path)
    return model.to(device)
    

def initialize_graph_constructor(model: TrIP) -> GraphConstructor:
    """Initialize the graph constructor for the model."""
    return GraphConstructor(model.cutoff)

def initialize_data_module(
    dataset_path: str,
    model: TrIP,

) -> TrIPDataModule:
    """Initialize the data module with the graph constructor."""
    return TrIPDataModule(
        batch_size=1,
        num_workers=4,
        data_file=dataset_path,
        si_tensor=model.si_tensor.to('cpu'),
        energy_std=model.energy_std,
    )

def run_inference(
        model: TrIP,
        dataloader: DataLoader,
        graph_constructor: GraphConstructor,
        device: torch.device,
) -> pd.DataFrame:
    """Run inference on the dataset and return results as a DataFrame.
    
    Args:
        model (TrIP): The trained model to use for inference.
        dataloader (DataLoader): The DataLoader containing the dataset.
        graph_constructor (GraphConstructor): The graph constructor for creating graphs.
        device (torch.device): The device to run the inference on.
    Returns:
        pd.DataFrame: A DataFrame containing the predicted and actual energies and forces.
            Cols: pred_energy, pred_force, actual_energy, actual_force, energy_diff, force_diff
            Each row is a single data point (conformation/ structure) in the dataset.
    """
    data_dict = {}
    index = 0
    for batch in tqdm(dataloader):
        species, pos_list, energy, forces, boxsize = to_cuda(batch, device)
        species = species.to(device)
        pos_list = [p.to(device) for p in pos_list]
        energy = energy.to(device)
        forces = forces.to(device)
        boxsize = boxsize.to(device)

        graph = graph_constructor.create_graphs(pos_list, boxsize)
        graph.ndata['species'] = species
        graph = graph.to(device)
        pred = model(graph, create_graph=False, standardized=True)

        print("\n\n\nPredictions:", pred)

        pred_energy, pred_forces = pred


        data_dict[index] = {
            'pred_energy': pred_energy.item(),
            'pred_force': pred_forces.cpu().numpy(),
            'actual_energy': energy.item(),
            'actual_force': forces.cpu().numpy(),
            'species': species.cpu().numpy(),
            'pos_list': [p.cpu().numpy() for p in pos_list],
            'boxsize': boxsize.cpu().numpy(),
        }
        
    print(f"Extracted {index} data points from the dataloader.")

    df = pd.DataFrame.from_dict(data_dict, orient='index')

    return df


def plot_actual_versus_predicted_energy(
    df: pd.DataFrame,
    title: str = "Actual vs Predicted Energy",
    save_dir: str = None
) -> None:
    """
    Plots actual vs predicted energy from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'actual_energy' and 'pred_energy' columns.
        title (str): Title of the plot.
    """
    # determine density of points relative to each other
    xy = np.vstack([ df['actual_energy'], df['pred_energy']])
    z = gaussian_kde(xy)(xy)
    z = z / z.max()  # Normalize the density values

    # create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df['actual_energy'], df['pred_energy'], alpha=0.5, c=z, cmap='viridis')
    plt.plot([df['actual_energy'].min(), df['actual_energy'].max()], 
                [df['actual_energy'].min(), df['actual_energy'].max()], 
                color='red', linestyle='--', label="Actual = Predicted Line")
    plt.legend()
    plt.xlabel('Actual Energy')
    plt.ylabel('Predicted Energy')
    plt.title(title)
    plt.grid()
    
    # save plot
    os.makedirs(save_dir, exist_ok=True)
    basename = f'{title.replace(" ", "_").lower()}.pdf'
    save_path = os.path.join(save_dir, basename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_energy_difference_histogram(
    df: pd.DataFrame,
    title: str = "Energy Difference Histogram",
    save_dir: str = None
) -> None:
    """
    Plots a histogram of the energy differences.

    Args:
        df (pd.DataFrame): DataFrame containing 'energy_diff' column.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(df['energy_diff'], bins=50, color='blue', alpha=0.7)
    plt.xlabel('Energy Difference')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid()

    # save plot
    os.makedirs(save_dir, exist_ok=True)
    basename = f'{title.replace(" ", "_").lower()}.pdf'
    save_path = os.path.join(save_dir, basename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_energy_difference_vs_actual_energy(
    df: pd.DataFrame,
    title: str = "Energy Difference vs Actual Energy",
    save_dir: str = None
) -> None:
    """
    Plots energy difference against actual energy.

    Args:
        df (pd.DataFrame): DataFrame containing 'actual_energy' and 'energy_diff' columns.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df['actual_energy'], df['energy_diff'], alpha=0.5, color='green')
    plt.xlabel('Actual Energy')
    plt.ylabel('Energy Difference')
    plt.title(title)
    plt.grid()

    # save plot
    os.makedirs(save_dir, exist_ok=True)
    basename = f'{title.replace(" ", "_").lower()}.pdf'
    save_path = os.path.join(save_dir, basename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def plot_force_difference_histogram(
    df: pd.DataFrame,
    title: str = "Force Difference Histogram",
    save_dir: str = None
) -> None:
    """
    Plots a histogram of the force differences.

    Args:
        df (pd.DataFrame): DataFrame containing 'force_diff' column.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(df['force_diff'], bins=50, color='orange', alpha=0.7)
    plt.xlabel('Force Difference')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid()

    # save plot
    os.makedirs(save_dir, exist_ok=True)
    basename = f'{title.replace(" ", "_").lower()}.pdf'
    save_path = os.path.join(save_dir, basename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def plot_force_difference_vs_actual_energy(
    df: pd.DataFrame,
    title: str = "Force Difference vs Actual Energy",
    save_dir: str = None
) -> None:
    """
    Plots force difference against actual energy.

    Args:
        df (pd.DataFrame): DataFrame containing 'actual_energy' and 'force_diff' columns.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df['actual_energy'], df['force_diff'], alpha=0.5, color='purple')
    plt.xlabel('Actual Energy')
    plt.ylabel('Force Difference')
    plt.title(title)
    plt.grid()

    # save plot
    os.makedirs(save_dir, exist_ok=True)
    basename = f'{title.replace(" ", "_").lower()}.pdf'
    save_path = os.path.join(save_dir, basename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def generate_parity_plots(
        df: pd.DataFrame,
        save_dir: str = "parity_plots"
) -> None:
    """
    Generate and save parity plots for energy and force predictions.

    Args:
        df (pd.DataFrame): DataFrame containing the results of the inference.
        save_dir (str): Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    plot_actual_versus_predicted_energy(df, save_dir=save_dir)
    plot_energy_difference_histogram(df, save_dir=save_dir)
    plot_energy_difference_vs_actual_energy(df, save_dir=save_dir)
    plot_force_difference_histogram(df, save_dir=save_dir)
    plot_force_difference_vs_actual_energy(df, save_dir=save_dir)



def main(args: argparse.Namespace) -> None:
    """Main function to run the inference and generate parity plots."""
    # Initialize distributed training
    init_distributed()

    # Set device rank
    device_rank = args.gpu_idx
    device = torch.device(f"cuda:{device_rank}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # Set random seed for reproducibility
    seed_everything(args.seed)
    # Load the model
    model = load_model(args.model_path, device)

    # Print model parameters count
    print_parameters_count(model)

    # Initialize graph constructor
    graph_constructor = initialize_graph_constructor(model)
    data_module = initialize_data_module(
        dataset_path=args.dataset_path,
        model=model,
    )

    dataloaders_dict = {
        "test": data_module.test_dataloader(),
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
    }

    updated_savedir_path = os.path.join(args.save_dir, os.path.basename(args.model_path).replace('.pth', ''))
    os.makedirs(updated_savedir_path, exist_ok=True)

    # run inference, generate plots, and save results
    for label, dataloader in dataloaders_dict.items():
        if not dataloader:
            logging.warning(f"No {label} dataloader found. Skipping inference for {label} dataset.")
            continue

        # Ensure the save directory exists
        output_dir = os.path.join(updated_savedir_path, label)
        os.makedirs(output_dir, exist_ok=True)


        # Run inference and save results
        df_save_path = os.path.join(output_dir, f"{label}_results.pkl")
        if os.path.exists(df_save_path):
            df = pd.read_pickle(df_save_path)
            logging.info(f"Results for {label} dataset already exist at {df_save_path}. Skipping inference.")
        else:
            logging.info(f"Running inference on {label} dataset...")
            df = run_inference(model, dataloader, graph_constructor, device)
            
            # Save the dataframe as a pickle file because I dont like worrying about csv formatting
            df.to_pickle(df_save_path)
            logging.info(f"Results saved to {df_save_path}")

        # Generate and save parity plots
        generate_parity_plots(df, save_dir=output_dir)
    logging.info(f"Parity plots saved to {updated_savedir_path}")
    
if __name__ == "__main__":
    # Parse command line arguments
    parser =argparse.ArgumentParser(description="Run inference and generate parity plots for TrIP model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained TrIP model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset for inference.")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results and plots.")
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU index to use for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run the main function
    main(args)
    # Handle warnings