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

# Questions for Bryce:
# - needed format of *data lists

import os
import h5py
import numpy as np
import torch
import time

from trip.data_loading.container import Container

def iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d

species_data = []
pos_data = []
forces_data = []
energy_data = []
num_list = []

print("Beginning extraction of ani1x")
# extracting data from ani1x dataset
ds_path = '/results/ani1xrelease.h5'
it = iter_data_buckets(ds_path, keys=['wb97x_dz.forces', 'wb97x_dz.energy'])

start_time = time.time()
mol_start_time = time.time()
for num, molecule in enumerate(it):
    species_data.append(torch.tensor(molecule['atomic_numbers'], dtype=torch.long))
    pos_data.append(torch.tensor(molecule['coordinates'], dtype=torch.float32))
    energy_data.append(torch.tensor(molecule['wb97x_dz.energy'], dtype=torch.float32))
    forces_data.append(torch.tensor(molecule['wb97x_dz.forces'], dtype=torch.float32))
    # num_list.append(num)
    
print("Time taken for ANI1x: ", time.time() - start_time)

# adding extracted data from ani2x dataset
ds_path = '/results/ANI-2x-wB97X-631Gd.h5'
with h5py.File(ds_path) as h5:
    
    start_time = time.time()
    for num_atoms, properties in h5.items(): #Iterate through like a dictionary
        atom_arrangements = np.array(properties['species'])
        unique_mol = np.unique(atom_arrangements, axis=0)
        print("Unique molecules to process: ", len(unique_mol))
        ctr = 0
        for type_mol in unique_mol:
            indices = np.where(np.all(np.array(atom_arrangements) == type_mol, axis=1))[0]
            mol_pos_data = []
            mol_energy_data = []
            mol_forces_data = []
            
            for idx in indices:
                mol_pos_data.append(torch.tensor(properties['coordinates'][idx], dtype=torch.float32))
                mol_energy_data.append(torch.tensor(properties['energies'][idx], dtype=torch.float32))
                mol_forces_data.append(torch.tensor(properties['forces'][idx], dtype=torch.float32))
            
            species_data.append(torch.tensor(type_mol, dtype=torch.long))
            pos_data.append(torch.stack(mol_pos_data))
            energy_data.append(torch.stack(mol_energy_data))
            forces_data.append(torch.stack(mol_forces_data))
        
            ctr += 1
            if ctr % 100 == 0:
                print(ctr, " molecules processed in ", time.time() - mol_start_time, " seconds")
                mol_start_time = time.time()
        
print("Time taken for ANI2x: ", time.time() - start_time)
            
num_array = np.arange(len(species_data))
train_idx = num_array[num_array % 20 != 0]
val_idx = num_array[num_array % 20 == 0]

container = Container()

def idx_lists(idx_list, *value_lists):
    new_value_lists = []
    for value_list in value_lists:
        new_value_lists.append([value_list[j] for j in idx_list])
    return new_value_lists

boxsize_data = [torch.full((pos_tensor.shape[0], 3), float('inf'), dtype=torch.float32) for pos_tensor in pos_data]

container.set_data('train', *idx_lists(train_idx, species_data, pos_data, energy_data, forces_data, boxsize_data))
container.set_data('val', *idx_lists(val_idx, species_data, pos_data, energy_data, forces_data, boxsize_data))


# Do testing. Here it looks like the goal is the same as with ani2x data extraction,
# and they've restructured the comp2v6 dataset to be the same as ani2x (?), so it shouldn't
# be too hard to modify the code to work with comp2v6
species_data = []
pos_data = []
energy_data = []
forces_data = []

print('Starting to extract COMP6v1 data')
data_dir = '/results'
test_dir = os.path.join(data_dir, 'COMP6v1')
species_dict = {b'H': 1, b'C': 6, b'N': 7, b'O': 8}
for subdir in os.listdir(test_dir):
    subpath = os.path.join(test_dir, subdir)
    for file_name in os.listdir(subpath):
        filepath = os.path.join(subpath, file_name)
        with h5py.File(filepath, 'r') as f:
            for main in f.values():
                for mol in main.values():
                    species_data.append(torch.tensor([species_dict[atom] for atom in mol['species']], dtype=torch.long))
                    pos_data.append(torch.tensor(np.array(mol['coordinates']), dtype=torch.float32))
                    energy_data.append(torch.tensor(mol['energies'], dtype=torch.float32))
                    forces_data.append(-torch.tensor(np.array(mol['forces']), dtype=torch.float32))  # COMP6's forces have wrong sign

print('Finished extracting COMP6v1 data and beginning v2')
ds_path = '/results/COMP6v2_wB97X-631Gd.h5'
with h5py.File(ds_path) as h5:
    
    for num_atoms, properties in h5.items(): #Iterate through like a dictionary
        atom_arrangements = np.array(properties['species'])
        unique_mol = np.unique(atom_arrangements, axis=0)
        for type_mol in unique_mol:
            indices = np.where(np.all(np.array(atom_arrangements) == type_mol, axis=1))[0]
            mol_pos_data = []
            mol_energy_data = []
            mol_forces_data = []
            
            for idx in indices:
                mol_pos_data.append(torch.tensor(properties['coordinates'][idx], dtype=torch.float32))
                mol_energy_data.append(torch.tensor(properties['energies'][idx], dtype=torch.float32))
                mol_forces_data.append(torch.tensor(properties['forces'][idx], dtype=torch.float32))
            
            species_data.append(torch.tensor(type_mol, dtype=torch.long))
            pos_data.append(torch.stack(mol_pos_data))
            energy_data.append(torch.stack(mol_energy_data))
            forces_data.append(torch.stack(mol_forces_data))
        

boxsize_data = [torch.full((pos_tensor.shape[0], 3), float('inf'), dtype=torch.float32) for pos_tensor in pos_data]

container.set_data('test', species_data, pos_data, energy_data, forces_data, boxsize_data)
save_path = '/results/ani2x.h5'

container.save_data(save_path)

# Now create a small subset for testing the code
test_container = Container()

subsets = ['train', 'val', 'test']

for subset in subsets:
    data = container.get_data(subset)
    new_data = [data[0]]
    for i, category in enumerate(data[1:]):
        new_data.append([conf[0,None,...] for conf in category])
    test_container.set_data(subset, *new_data)

test_path = '/results/test.h5'
test_container.save_data(test_path)
print('Data extraction complete')
