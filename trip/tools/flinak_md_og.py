import argparse
import logging
import os
from sys import stdout
import time
import sys

import torch
import MDAnalysis as mda
from openmm import Platform, System, LangevinIntegrator, MonteCarloBarostat, CustomExternalForce, NonbondedForce, LocalEnergyMinimizer, VerletIntegrator
from openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter
from openmm.unit import femtosecond, kelvin, kilocalorie_per_mole, nanometer, angstrom, picosecond, bar, nanosecond

from trip.tools.generate_flibenak_structures import create_pdb_box
from trip.tools.utils import get_species, save_pdb
from trip.tools.module import TrIPModule
from openmm.unit import hartree, angstrom, kilocalorie_per_mole, nanometer, kilojoule_per_mole
from se3_transformer.runtime.utils import str2bool



def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--out', type=str, default='/results/',
                        help='The path to the output directory, default=/results/')
    parser.add_argument('--model_file', type=str, default='/results/trip_vanilla.pth',
                        help='Path to model file, default=/results/trip_vanilla.pth')
    parser.add_argument('--minimize', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to minimize the structure')
    parser.add_argument('--dt', type=float, default=0.5,
                        help='Step size in femtoseconds')
    parser.add_argument('--t', type=float, default=1.,
                        help='Simulation time in nanoseconds')
    parser.add_argument('--temp', type=float, default=298.,
                        help='Temperature in kelvin')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, default=0')
    args = parser.parse_args()
    return args

def get_trip_force():
    trip_force = CustomExternalForce('c-fx*x-fy*y-fz*z')
    trip_force.addPerParticleParameter('c')  # Correction term to get correct energy
    trip_force.addPerParticleParameter('fx')
    trip_force.addPerParticleParameter('fy')
    trip_force.addPerParticleParameter('fz')
    return trip_force

def get_system(topo, trip_force):
    system = System()
    for atom in topo.atoms():
        system.addParticle(atom.element.mass)
    system.addForce(trip_force)
    for index, atom in enumerate(topo.atoms()):
        trip_force.addParticle(index, (0, 0, 0, 0) * kilocalorie_per_mole/angstrom)

    return system
    
def get_simulation(topo, system, pos, temp, dt, out, gpu, **args):
    # integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
    integrator= VerletIntegrator(dt*femtosecond)
    platform = Platform.getPlatformByName('HIP')
    simulation = Simulation(topo, system, integrator, platform, {'DeviceIndex': str(gpu)})
    simulation.context.setPositions(pos.tolist() * angstrom)
    simulation.context.setVelocitiesToTemperature(temp * kelvin)
    simulation.reporters.append(DCDReporter(os.path.join(out, 'trajectory.dcd'), 1, enforcePeriodicBox=True))
    simulation.reporters.append(StateDataReporter(stdout, 1, step=True, temperature=True,
                                                  potentialEnergy=True, totalEnergy=True, density=True))
    return simulation


if __name__ == '__main__':
    # Setup
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    logging.info('============ TrIP =============')
    logging.info('|Molten Salt Molecular dynamics|')
    logging.info('===============================')

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    # set file paths
    out_path = args.out
    pdb_start = os.path.join(out_path, f"start_{args.temp:.0f}K.pdb")
    minimized_pdb = os.path.join(out_path, "minimized.pdb")
    traj_path = os.path.join(out_path, f"trajectory.dcd")

    # initialize system conditions
    temperature = args.temp * kelvin
    dt = args.dt * femtosecond
    total_time = args.t * nanosecond

    # Load / generate data
    if not os.path.exists(pdb_start):
        logging.info('PDB file not found, generating new box...')

        rho_flinak_eutectic = lambda T: 2.5793 - 0.624e-3 * T

        create_pdb_box(
            # atomic_system={'F': 100, "Li": 47, "Na": 11, "K": 42},
            atomic_system={"F": 200, "Li": 94, "Na": 22, "K": 84},
            # atomic_system={"F":1000, "Li": 470, "Na": 110, "K": 420},
            # atomic_system={"F":5000, "Li":2350, "Na": 550, "K": 2100},
            # atomic_system={'F': 400, "Li": 190, "Na": 45, "K": 170},
            density= rho_flinak_eutectic(args.temp),
            charges={'F': -1.0, 'Li': 1.0, 'Na': 1.0, 'K': 1.0},
            min_distance=2.0,
            decimals=6,
            double_box=False,
            axis_to_double='z',
            save_pdb_path=pdb_start
        )



    pdbf = PDBFile(pdb_start)
    topo = pdbf.topology
    symbols = [atom.element.symbol for atom in topo.atoms()]
    species = get_species(symbols)

    module = TrIPModule(species, **vars(args))
    for param in module.parameters():
        param.requires_grad_(False)

    pos = pdbf.getPositions(asNumpy=True) / angstrom
    pos = torch.tensor(pos, dtype=torch.float, device=device)
    boxsize = topo.getUnitCellDimensions() / angstrom
    boxsize = torch.tensor(boxsize, dtype=torch.float, device=device)

    # # Minimation procedure
    if args.minimize:
        module.log_energy(pos, boxsize)
        logging.info('Beginning minimization')
        pos = module.minimize(pos, boxsize)
        module.log_energy(pos, boxsize)
        save_pdb(pos, topo, 'minimized', **vars(args))
        logging.info('Finished minimization!')

    # Run simulation
    trip_force = get_trip_force()
    system = get_system(topo, trip_force)
    simulation = get_simulation(topo, system, pos, **vars(args))
    num_steps = int(1e6 * args.t / args.dt)  # 1e6 is the ratio of femtoseconds to nanoseconds



    logging.info('Beginning simulation')
    for i in range(num_steps):
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos = state.getPositions()
        pos = torch.tensor([[p.x, p.y, p.z] for p in pos],
                            dtype=torch.float, device=device, requires_grad=True)*10.0 # Nanometer to Angstrom conversion
        box_length = state.getPeriodicBoxVectors()[0][0].value_in_unit(angstrom)
        boxsize = torch.tensor([box_length, box_length, box_length], dtype=torch.float, device=device)
        energy, forces = module(pos, boxsize)
        c = 627.5 * kilocalorie_per_mole * (energy + torch.sum(pos * forces)).item() / len(pos)  # Energy correction per atom
        forces = forces * 627.5 * kilocalorie_per_mole / angstrom
        for index, atom in enumerate(topo.atoms()):
            trip_force.setParticleParameters(index, index, [c, *forces[index]])

        trip_force.updateParametersInContext(simulation.context)
        simulation.step(1)

    logging.info('Simulation finished successfully')

