# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import os
import h5py
import numpy as np
import torch

from trip.data_loading.container import Container

atomic_numbers_map = {
    b'Li': 3,
    b'Be': 4,
    b'F': 9,
    b'Na': 11,
    b'K': 19,
}


def iter_data_buckets(h5filename, keys=['energies', 'forces', 'cell']):
    """ Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    # irgnore atomic_numbers and coordinates when filtering by NaN values. IMPLIES they never have NaNs
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for dataset in f.values():
            # iterate through the next level in the hdf5 file
            for molecular_system in dataset.values():
                Nc = molecular_system['coordinates'].shape[0]  # number of configurations/ conformers of the system
                mask = np.ones(Nc, dtype=bool)  # [True, True, ...] of length Nc
                data = dict((k, molecular_system[k][()]) for k in keys)  # extract data from the molecule, dump into a dict
                for k in keys:
                    v = data[k].reshape(Nc, -1)  # flatten dimensions after the first. Each item in lsit is a row of features
                    mask = mask & ~np.isnan(v).any(axis=1)  # set rows with a NaN values to Falsw
                if not np.sum(mask):  # skip molecules with no valid data
                    continue
                return_data = dict((k, data[k][mask]) for k in keys)  # remove all conformers that have a NaN value in the target keys
                
                # convert species (strings) to atomic numbers (integers). Specific to FLiBeNaK
                atomic_numbers = []
                for atom in molecular_system['species']:
                    if atom in atomic_numbers_map:
                        atomic_numbers.append(atomic_numbers_map[atom])
                    else:
                        raise ValueError(f"Unknown atom type: {atom}")
                return_data['atomic_numbers'] = np.array(atomic_numbers, dtype=np.int32)

                # Extract cell / box size dimensions
                # FLiBeNaK 'cell' format is [[x,0,0], [0,y,0], [0,0,z]], but we only need [x,y,z] I think
                try:
                    extracted_cell_data = []
                    cell_data = molecular_system['cell'][()]
                    for conformer_idx in range(cell_data.shape[0]):
                        x_dim = cell_data[conformer_idx][0][0]
                        y_dim = cell_data[conformer_idx][1][1]
                        z_dim = cell_data[conformer_idx][2][2]
                        extracted_cell_data.append([x_dim, y_dim, z_dim])

                    return_data['boxsize'] = np.array(extracted_cell_data, dtype=np.float32)
                except IndexError:
                    # Handle the case where cell data is not available or has unexpected shape
                    raise ValueError("Cell data is not available or has unexpected shape. Cell data: {}".format(molecular_system['cell'][()]))

                return_data['coordinates'] = molecular_system['coordinates'][()][mask]  # discard conformers with NaN values
                yield return_data

            # Nc = grp['coordinates'].shape[0]
            # mask = np.ones(Nc, dtype=bool)
            # data = dict((k, grp[k][()]) for k in keys)
            # for k in keys:
            #     v = data[k].reshape(Nc, -1)
            #     mask = mask & ~np.isnan(v).any(axis=1)
            # if not np.sum(mask):
            #     continue
            # d = dict((k, data[k][mask]) for k in keys)
            # d['atomic_numbers'] = grp['atomic_numbers'][()]
            # d['coordinates'] = grp['coordinates'][()][mask]
            # yield d

# stack all of the extracted data into master lists. Each entry will be for a single molecule with 1+ conformers
species_data = []
pos_data = []
forces_data = []
energy_data = []
boxsize_data = []
num_list = []  # used to generate the train/val split

file_path = 'datasets/LiBeFKNa_it121.h5'
it = iter_data_buckets(file_path, keys=['energies', 'forces', 'cell'])

ev_to_hartree_divisor = 27.211386245981
for num, molecule in enumerate(it):
    # need to convert eV to Hartree for energies,
    # eV/A to Hartree/A for forces
    species_data.append(torch.tensor(molecule['atomic_numbers'], dtype=torch.long))
    pos_data.append(torch.tensor(molecule['coordinates'], dtype=torch.float32))
    energy_data.append(torch.tensor(molecule['energies'], dtype=torch.float32) / ev_to_hartree_divisor)  # convert eV to Hartree
    forces_data.append(torch.tensor(molecule['forces'], dtype=torch.float32) / ev_to_hartree_divisor) # convert eV/A to Hartree/A
    boxsize_data.append(torch.tensor(molecule['boxsize'], dtype=torch.float32))
    num_list.append(num)

test_idx = []
train_idx = []
val_idx = []
# Split the data into training, validation, and testing sets
# Here we will use a simple split: 80% training, 10% validation, and 10% testing
# This is a simple split, but you can modify it as needed
num_data = len(species_data)

test_size = int(0.1 * num_data)
val_size = int(0.1 * num_data)
train_size = num_data - test_size - val_size

np.random.seed(42)  # For reproducibility
shuffled_indices = np.random.permutation(num_data)

# Assign indices to train, val, and test sets
train_idx = shuffled_indices[:train_size].tolist()
val_idx = shuffled_indices[train_size:train_size + val_size].tolist()
test_idx = shuffled_indices[train_size + val_size:].tolist()

container = Container()

# print("Boxsize data shape: ", boxsize_data[0].shape)
# print("Length of boxsize data: ", len(boxsize_data))
# print("pos data shape: ", pos_data[0].shape)
# print("Length of pos data: ", len(pos_data))

def idx_lists(idx_list, *value_lists):
    """ Extract/ sort data in each list by the indices in idx_list and return a list of sorted lists.

    Args:
        idx_list (list): List of indices to extract/ sort by.
        *value_lists: Variable number of lists.
    Returns:
        new_value_lists (list[list]): List of sorted lists, with the same length as idx_list. 
        For value_lists l1, l2, l3, the output will be [sorted_l1, sorted_l2, sorted_l3].
    """
    new_value_lists = []
    for value_list in value_lists:
        new_value_lists.append([value_list[j] for j in idx_list])
    return new_value_lists

# *idx_lists(train_idx, species_data, pos_data, energy_data, forces_data, boxsize_data) --> extract data for training set
container.set_data('train', *idx_lists(train_idx, species_data, pos_data, energy_data, forces_data, boxsize_data))
container.set_data('val', *idx_lists(val_idx, species_data, pos_data, energy_data, forces_data, boxsize_data))
container.set_data('test', *idx_lists(test_idx, species_data, pos_data, energy_data, forces_data, boxsize_data))


# Do testing 
species_data = []
pos_data = []
energy_data = []
forces_data = []

# old code to process the comp6 test set (for use with Ani1x/2x)
####################################    ##############################
# data_dir = '/results'
# test_dir = os.path.join(data_dir, 'COMP6v1')
# species_dict = {b'H': 1, b'C': 6, b'N': 7, b'O': 8}
# for subdir in os.listdir(test_dir):
#     subpath = os.path.join(test_dir, subdir)
#     for file_name in os.listdir(subpath):
#         filepath = os.path.join(subpath, file_name)
#         with h5py.File(filepath, 'r') as f:
#             for main in f.values():
#                 for mol in main.values():
#                     species_data.append(torch.tensor([species_dict[atom] for atom in mol['species']], dtype=torch.long))
#                     pos_data.append(torch.tensor(np.array(mol['coordinates']), dtype=torch.float32))
#                     energy_data.append(torch.tensor(mol['energies'], dtype=torch.float32))
#                     forces_data.append(-torch.tensor(np.array(mol['forces']), dtype=torch.float32))  # COMP6's forces have wrong sign


# boxsize_data = [torch.full((pos_tensor.shape[0], 3), float('inf'), dtype=torch.float32) for pos_tensor in pos_data]

# container.set_data('test', species_data, pos_data, energy_data, forces_data, boxsize_data)
# save_path = os.path.join(data_dir, 'ani1x.h5')

save_path = 'datasets/processed_flibenak_it121.h5'
container.save_data(save_path)


# Again, don't need the old code for the comp6 test set
##########################################################
# # Now create a small subset for testing the code
# test_container = Container()

# subsets = ['train', 'val', 'test']

# for subset in subsets:
#     data = container.get_data(subset)
#     new_data = [data[0]]
#     for i, category in enumerate(data[1:]):
#         new_data.append([conf[0,None,...] for conf in category])
#     test_container.set_data(subset, *new_data)

# test_path = os.path.join(data_dir, 'test.h5')
# test_container.save_data(test_path)