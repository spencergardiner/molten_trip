from trip.data import AtomicData
import os
from openmm import *
from openmm.app import *
from openmm.unit import *
import torch


def get_species(symbols):
    symbol_list = AtomicData.get_symbols_list()
    symbol_dict = {symbol: i+1 for i, symbol in enumerate(symbol_list)}
    species = [symbol_dict[symbol] for symbol in symbols]
    return species

def save_pdb(pos, topo, name, out, **args):
    with open(os.path.join(out, name+'.pdb'), 'w') as f:
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu()
        pdbfile.PDBFile.writeFile(topo, pos, f)
        