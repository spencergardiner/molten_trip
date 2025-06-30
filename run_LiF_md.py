import logging
import numpy as np
import torch
from simtk.openmm.app import Topology, Simulation, element
from simtk.openmm import System, NonbondedForce, Platform
from simtk import unit
from trip.tools.module import TrIPModule
from trip.tools.md import get_trip_force, get_system, get_simulation
from trip.tools.utils import save_pdb, get_species

# === Parameters ===
N = 100  # number of Li/F atoms each
L = 10.0 * unit.angstrom
total_atoms = 2 * N
dim = int(np.ceil(total_atoms ** (1/3)))
spacing = (L / dim).value_in_unit(unit.nanometer)
device = 'cuda:0'  # adjust for your setup

# === Build Topology & System ===
top = Topology()
chain = top.addChain()
res = top.addResidue('MIX', chain)
system = System()
elements = []

for i in range(N):
    top.addAtom('Li', element.lithium, res)
    system.addParticle(6.941 * unit.amu)
    elements.append('Li')
for i in range(N):
    top.addAtom('F', element.fluorine, res)
    system.addParticle(18.998 * unit.amu)
    elements.append('F')
# Initial positions: fill grid
coords = []
for x in range(dim):
    for y in range(dim):
        for z in range(dim):
            if len(coords) < total_atoms:
                coords.append([x * spacing, y * spacing, z * spacing])
positions = unit.Quantity(coords, unit=unit.nanometer)

# === Set up TrIP module ===
symbols = elements
species = get_species(symbols)
module = TrIPModule(species, 
                    model_file='models/model_fw1_80epochs/model_fw1_80epochs_e33.pth',
                    gpu=0).to(device)  # adapt init signature

# Convert to torch
pos_angstrom = positions.value_in_unit(unit.angstrom)
pos = torch.tensor(pos_angstrom, dtype=torch.float32, device=device)
boxsize = torch.tensor([L.value_in_unit(unit.angstrom)]*3, dtype=torch.float32, device=device)

# === Define Custom Force ===
trip_force = get_trip_force()
system = get_system(top, trip_force)

# === Simulation Setup ===
simulation = get_simulation(top, system, positions, temp=300.0, dt=1.0, out='test_md')  # adapt args

# === Energy Minimization ===
logging.info("Initial energy:")
e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
logging.info(f"  {e0}")

simulation.minimizeEnergy()  # local minimization per OpenMM API :contentReference[oaicite:5]{index=5}

state_min = simulation.context.getState(getEnergy=True, getPositions=True)
em = state_min.getPotentialEnergy()
pos_min = state_min.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
logging.info("Post-minimization energy:")
logging.info(f"  {em}")

# Save minimized structure
save_pdb(pos_min, top, 'minimized.pdb', 'test_md')

# === 10,000-Step MLIP-Driven MD ===
logging.info("Beginning 10,000-step MD simulation")
for step in range(1000):
    state = simulation.context.getState(getPositions=True)
    pos_nm = state.getPositions(asNumpy=True)
    pos_torch = torch.tensor(pos_nm * 10.0, dtype=torch.float32, device=device)  # nm→Å

    energy, forces = module(pos_torch, boxsize)
    coeff = 627.5 * (energy + torch.sum(pos_torch * forces)).item() / total_atoms
    forces_scaled = forces * 627.5 / unit.angstrom  # convert to kJ/mol·nm

    for idx in range(total_atoms):
        # setParticleParameters(index, charge, fx, fy, fz)
        trip_force.setParticleParameters(idx, idx, [coeff, *forces_scaled[idx].tolist()])
    trip_force.updateParametersInContext(simulation.context)

    simulation.step(1)

logging.info("Simulation complete. Saving final state.")
final_pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
save_pdb(final_pos, top, 'final.pdb', 'test_md')