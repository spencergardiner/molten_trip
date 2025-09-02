import argparse
import logging
import os
from sys import stdout

import torch
from openmm import Platform, System, LangevinIntegrator, MonteCarloBarostat, LocalEnergyMinimizer
from openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter
from openmm.unit import femtosecond, kelvin, picosecond, nanosecond, bar, nanometer
from openmmtorch import TorchForce

from se3_transformer.runtime.utils import str2bool

from trip.tools.generate_flibenak_structures import create_pdb_box
from trip.tools.utils import get_species, save_pdb
from trip.tools.module import TripOpenmmTorchForceModule


def parse_args():
    parser = argparse.ArgumentParser(description='Run molten salt MD using TrIP via TorchForce (fully integrated forces).')
    parser.add_argument('--out', type=str, default='./md_simulations/out', help='Output directory')
    parser.add_argument('--model_file', type=str, required=True, help='Path to TrIP model checkpoint')
    parser.add_argument('--temp', type=float, default=1000.0, help='Temperature in K')
    parser.add_argument('--dt', type=float, default=0.5, help='Timestep in fs')
    parser.add_argument('--t', type=float, default=0.05, help='Total simulation time (ns)')
    parser.add_argument('--friction', type=float, default=1.0, help='Langevin friction (1/ps)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--minimize', type=str2bool, nargs='?', const=True, default=True, help='Perform energy minimization before dynamics')
    parser.add_argument('--npt', type=str2bool, nargs='?', const=True, default=False, help='Use NPT ensemble (MonteCarloBarostat)')
    parser.add_argument('--interval', type=int, default=100, help='Trajectory/reporting stride (steps)')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Checkpoint interval (steps)')
    return parser.parse_args()


def generate_or_load_pdb(pdb_path, temp):
    if os.path.exists(pdb_path):
        return
    logging.info('PDB not found. Generating eutectic FLiNaK box...')
    rho_flinak_eutectic = lambda T: 2.5793 - 0.624e-3 * T
    create_pdb_box(
        atomic_system={"F": 200, "Li": 94, "Na": 22, "K": 84},
        density=rho_flinak_eutectic(temp),
        charges={'F': -1.0, 'Li': 1.0, 'Na': 1.0, 'K': 1.0},
        min_distance=2.0,
        decimals=6,
        double_box=False,
        axis_to_double='z',
        save_pdb_path=pdb_path
    )


def build_system(topo, torch_force, use_barostat, temperature):
    system = System()
    for atom in topo.atoms():
        system.addParticle(atom.element.mass)
    system.addForce(torch_force)
    if use_barostat:
        logging.info('Adding MonteCarloBarostat (1 bar)')
        system.addForce(MonteCarloBarostat(1 * bar, temperature))
    return system


def make_simulation(topo, system, positions, temperature, dt_fs, friction, gpu_index, out_dir, interval):
    integrator = LangevinIntegrator(temperature, friction / picosecond, dt_fs * femtosecond)
    platform = Platform.getPlatformByName('HIP')
    simulation = Simulation(topo, system, integrator, platform, {'DeviceIndex': str(gpu_index)})
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    traj_path = os.path.join(out_dir, 'trajectory.dcd')
    simulation.reporters.append(DCDReporter(traj_path, interval, enforcePeriodicBox=True, append=os.path.exists(traj_path)))
    simulation.reporters.append(StateDataReporter(stdout, interval, step=True, temperature=True, potentialEnergy=True, totalEnergy=True, density=True))
    return simulation


def load_checkpoint(simulation, out_dir):
    chk = os.path.join(out_dir, 'simulation.chk')
    step_file = os.path.join(out_dir, 'checkpoint.step')
    if os.path.exists(chk) and os.path.exists(step_file):
        try:
            simulation.loadCheckpoint(chk)
            with open(step_file) as f:
                step = int(f.read())
            logging.info(f'Resuming from checkpoint at step {step}')
            return step
        except Exception as e:
            logging.warning(f'Failed to load checkpoint: {e}. Starting fresh.')
    return 0


def save_checkpoint(simulation, out_dir, step):
    chk = os.path.join(out_dir, 'simulation.chk')
    step_file = os.path.join(out_dir, 'checkpoint.step')
    simulation.saveCheckpoint(chk)
    with open(step_file, 'w') as f:
        f.write(str(step))
    logging.info(f'Saved checkpoint at step {step}')


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    logging.info('=========== TrIP TorchForce MD ===========')

    os.makedirs(args.out, exist_ok=True)
    pdb_start = os.path.join(args.out, f'start_{int(args.temp)}K.pdb')
    minimized_pdb = os.path.join(args.out, 'minimized.pdb')

    # GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    logging.info(f'Using device {device}')

    # Prepare structure
    generate_or_load_pdb(pdb_start, args.temp)
    pdbf = PDBFile(pdb_start)
    topo = pdbf.topology
    init_positions = pdbf.getPositions(asNumpy=True)  # in nm (OpenMM units)

    # Species list for model
    symbols = [atom.element.symbol for atom in topo.atoms()]
    species = get_species(symbols)

    # Build Torch module
    trip_module = TripOpenmmTorchForceModule(species, args.model_file, map_location=device)
    trip_module.eval()

    # Prepare example tensors for potential tracing fallback
    # Positions already in OpenMM (nm); convert to torch
    example_positions = torch.tensor([[p.x, p.y, p.z] for p in pdbf.getPositions()], dtype=torch.float, device=device)
    # Construct boxvectors tensor (diagonal matrix) in nm
    uc = topo.getUnitCellDimensions()
    if uc is not None:
        boxvectors = torch.zeros((3, 3), dtype=torch.float, device=device)
        boxvectors[0, 0] = uc[0].value_in_unit(nanometer)
        boxvectors[1, 1] = uc[1].value_in_unit(nanometer)
        boxvectors[2, 2] = uc[2].value_in_unit(nanometer)
    else:
        # Fallback cubic box guess from max position
        max_coord = example_positions.max().item()
        boxvectors = torch.diag(torch.full((3,), max_coord * 1.2, dtype=torch.float, device=device))

    # Attempt scripting first; fall back to tracing if unsupported (e.g., due to contextmanager *args/**kwargs in deps)
    try:
        scripted = torch.jit.script(trip_module)
        logging.info('Successfully scripted TrIP module for TorchForce.')
    except Exception as e:
        logging.warning(f'script() failed ({e}); falling back to trace().')
        with torch.no_grad():
            scripted = torch.jit.trace(trip_module, (example_positions, boxvectors), strict=False)
        logging.info('Successfully traced TrIP module for TorchForce.')

    torch_force = TorchForce(scripted)  # Returns energy (kJ/mol) & forces (kJ/mol/nm)
    torch_force.setUsesPeriodicBoundaryConditions(True)
    torch_force.setOutputsForces(True)

    temperature = args.temp * kelvin
    system = build_system(topo, torch_force, args.npt, temperature)

    simulation = make_simulation(topo, system, init_positions, temperature, args.dt, args.friction, args.gpu, args.out, args.interval)

    # Minimization (OpenMM will call TorchForce for energies/forces)
    if args.minimize and not os.path.exists(minimized_pdb):
        logging.info('Minimizing structure with TorchForce potential...')
        LocalEnergyMinimizer.minimize(simulation)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos_nm = state.getPositions(asNumpy=True)  # nanometers
        pos_A = torch.tensor([[p.x, p.y, p.z] for p in state.getPositions()], dtype=torch.float) * 10.0
        save_pdb(pos_A, topo, 'minimized', out=args.out, temp=args.temp, model_file=args.model_file)
        logging.info('Minimization complete.')

    # Timestepping
    total_steps = int(1e6 * args.t / args.dt)
    start_step = load_checkpoint(simulation, args.out)
    if start_step >= total_steps:
        logging.info('Simulation already complete.')
        return

    logging.info(f'Starting dynamics from step {start_step} to {total_steps}')
    step = start_step
    while step < total_steps:
        chunk = min(args.checkpoint_interval, total_steps - step)
        simulation.step(chunk)
        step += chunk
        save_checkpoint(simulation, args.out, step)

    logging.info('Simulation finished successfully.')


if __name__ == '__main__':
    main()
