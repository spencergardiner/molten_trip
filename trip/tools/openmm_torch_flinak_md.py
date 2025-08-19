from trip.tools.generate_flibenak_structures import create_pdb_box
from trip.tools.md import get_trip_force, get_system, get_simulation
from trip.tools.utils import get_species, save_pdb
from trip.tools.module import TrIPModule, TripOpenmmTorchForceModule
from openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter
from openmm.unit import femtosecond, kelvin, kilocalorie_per_mole, angstrom, nanosecond, picosecond
from openmm import System, LangevinIntegrator, Platform
from openmm.vec3 import Vec3
from openmmtorch import TorchForce

import torch
import os
from sys import stdout

import logging

logging.getLogger().setLevel(logging.INFO)

#csimulation configurations
time = 10 * picosecond  # in nanoseconds
dt = 0.5 * femtosecond  # in femtoseconds
temp = 1000 * kelvin  # in Kelvin

out_path = './eutectic_flinak_1000k'


rho_eutectic_flinak = lambda T: 2.5793 - 0.624e-3*T # [K]
pdb_start_structure_path = './flinak_1000K.pdb'

box_length = create_pdb_box(
    # atomic_system={'F':5000, 'Li': 2325, 'Na': 575 , 'K': 2100},
    atomic_system={'F': 1000, "Li": 465, "Na": 115, "K": 420}, # Full system
    # atomic_system={'F': 100, "Li": 47, "Na": 11, "K": 42}, # Reduced system for testing
    density=rho_eutectic_flinak(1000), # density at eutectic temperature ~1000K
    charges={'F': -1.0, 'Li': 1.0, 'Na': 1.0, 'K': 1.0},
    min_distance=2.0,
    decimals=6,
    double_box=False,
    axis_to_double='z',
    save_pdb_path=pdb_start_structure_path
)
print("Box length:", box_length)

pdb_file = PDBFile(pdb_start_structure_path)
topology = pdb_file.topology
positions = pdb_file.getPositions(asNumpy=True)



trip_model_file = './models/400k_3layers_fw_10/400k_3layers_fw_10.pth'
symbols = [atom.element.symbol for atom in topology.atoms()]
species = get_species(symbols)

logging.info("Initializing the TripOpenmmTorchForceModule")
model = TripOpenmmTorchForceModule(
    species=species,
    model_file=trip_model_file,
)
module = torch.jit.script(model)  # Convert to TorchScript for OpenMM-torch compatibility
torch_force = TorchForce(module)
torch_force.setUsePeriodicBoundaryConditions(True)


# setup the OpenMM system with the TorchForce
logging.info
system = System()
system.addForce(torch_force)

# get platform, properties
platform = Platform.getPlatformByName('HIP')  # Use OpenCL for GPU acceleration
properties = {"DeviceIndex": "0,1,2,3"}

# initialize integrator
logging.info("Creating Langevin integrator")
integrator = LangevinIntegrator(temp, 1/picosecond, dt)


logging.info("Creating simulation")
simulation = Simulation(topology, system, integrator, platform, properties)

simulation.context.setPositions(positions)  # update positions
simulation.context.setVelocitiesToTemperature(temp)
simulation.context.setPeriodicBoxVectors(
    Vec3(box_length, 0, 0),
    Vec3(0, box_length, 0),
    Vec3(0, 0, box_length)
)  # set periodic box vectors


# add reporters
simulation.reporters.append(DCDReporter(os.path.join(out_path, 'trajectory.dcd'), 1))
simulation.reporters.append(StateDataReporter(stdout, 1, step=True, temperature=True,
                                                potentialEnergy=True, totalEnergy=True))

# perform an energy minimization
logging.info("Minimizing energy")
simulation.minimizeEnergy()

# Run simulation
logging.info('Running simulation')
num_steps = int(time / dt)
simulation.step(num_steps)

# Save the final positions to a PDB file
final_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
final_pdb_file = os.path.join(out_path, 'final_positions.pdb')
with open(final_pdb_file, 'w') as f:
    PDBFile.writeFile(topology, final_positions, f)
logging.info(f"Final positions saved to {final_pdb_file}")



