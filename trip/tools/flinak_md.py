#!/usr/bin/env python3
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
from openmm.unit import femtosecond, kelvin, kilocalorie_per_mole, nanometer, angstrom, picosecond, bar

from trip.tools.generate_flibenak_structures import create_pdb_box
from trip.tools.utils import get_species, save_pdb
from trip.tools.module import TrIPModule
from openmm.unit import hartree, angstrom, kilocalorie_per_mole, nanometer, kilojoule_per_mole

logging.getLogger().setLevel(logging.INFO)


def make_checkpoint(simulation, out_path, step):
    chk_path = os.path.join(out_path, "simulation.chk")
    step_path = os.path.join(out_path, "checkpoint.step")
    simulation.saveCheckpoint(chk_path)
    with open(step_path, "w") as f:
        f.write(str(step))
    logging.info(f"Saved checkpoint at step {step} to {chk_path}")


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
        trip_force.addParticle(index, (0, 0, 0, 0) * kilojoule_per_mole/nanometer)

    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * angstrom)
    charge = 0.0               # zero charge → no Coulomb
    sigma = 1.0 * angstrom    # arbitrary σ
    epsilon = 0.0    
        
    system.addForce(nb)
    for _ in range(system.getNumParticles()):
        nb.addParticle(charge, sigma, epsilon)

    # set periodic boundary conditions
    box_vectors = topo.getPeriodicBoxVectors()
    print("Topology periodic box vectors are:", box_vectors)
    system.setDefaultPeriodicBoxVectors(*box_vectors)

    return system
    

def load_checkpoint(simulation, out_path):
    chk_path = os.path.join(out_path, "simulation.chk")
    step_path = os.path.join(out_path, "checkpoint.step")
    if os.path.exists(chk_path) and os.path.exists(step_path):
        logging.info(f"Found checkpoint {chk_path}, resuming simulation")
        simulation.loadCheckpoint(chk_path)
        with open(step_path) as f:
            start_step = int(f.read())
        return start_step
    return 0


hartree_to_kJ_per_mol = 2625.5
nanometer_to_angstrom = 10.0
angstrom_to_nanometer = 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simulation of eutectic FLiNaK with restartable checkpointing."
    )
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in Kelvin.")
    parser.add_argument("--torch_gpu_index", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step (fs).")
    parser.add_argument("--time", type=float, default=20, help="Total sim time (ps).")
    parser.add_argument("--is_NPT", action="store_true", help="Use NPT if set; default is NVT.", default=True)
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--minimize", action="store_true", help="Minimize before running.")
    parser.add_argument("--minimized_name", type=str, default="minimized", help="Minimized PDB name.")
    parser.add_argument("--trajectory_name", type=str, default="trajectory", help="Name for trajectory file.")
    parser.add_argument("--trip_model_file", type=str,
                        default="./models/400k_3layers_fw_10/400k_3layers_fw_10.pth",
                        help="TrIP model .pth file.")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="How many steps between checkpoints.")
    parser.add_argument("--low_memory", action="store_true", 
                        help="Use low memory mode for TrIP model (slower but less memory).")
    parser.add_argument("--memory_efficient", action="store_true",
                        help="Use additional memory efficiency optimizations.")
    parser.add_argument("--is_NVE", action="store_true", help="Use NVE if set; default is NVT.")
    args = parser.parse_args()

    # log the entire python command
    logging.info(f"Running command: {' '.join(sys.argv)}")

    logging.info("\n\n\nStarting (or resuming) eutectic FLiNaK simulation with TrIP.")
    
    # Memory optimization settings
    # if args.memory_efficient:
    #     logging.info("Using memory efficient settings")
    #     # Increase checkpoint interval to reduce I/O and memory spikes
    #     args.checkpoint_interval = max(args.checkpoint_interval, 500)
    #     # Set smaller PyTorch memory allocation settings
    #     os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Apply additional memory optimizations for HIP/ROCm
    os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

    # convert units
    temperature = args.temperature * kelvin
    dt = args.dt * femtosecond
    total_time = args.time * picosecond
    is_NPT = args.is_NPT
    is_NVE = args.is_NVE

    # out_path = os.path.join(
    #     args.output_dir, f"eutectic_flinak_{args.temperature:.0f}K"
    # )
    out_path = args.output_dir
    os.makedirs(out_path, exist_ok=True)

    # file paths
    pdb_start = os.path.join(out_path, f"start_{args.temperature:.0f}K.pdb")
    minimized_pdb = os.path.join(out_path, f"{args.minimized_name}.pdb")
    traj_path = os.path.join(out_path, f"{args.trajectory_name}.dcd")

    rho_flinak_eutectic = lambda T: 2.5793 - 0.624e-3 * T

    # create starting box if needed
    if not os.path.exists(pdb_start):
        logging.info("Creating new box...")
        create_pdb_box(
            # atomic_system={'F': 100, "Li": 47, "Na": 11, "K": 42},
            atomic_system={"F": 200, "Li": 94, "Na": 22, "K": 84},
            # atomic_system={"F":1000, "Li": 470, "Na": 110, "K": 420},
            # atomic_system={"F":5000, "Li":2350, "Na": 550, "K": 2100},
            # atomic_system={'F': 400, "Li": 190, "Na": 45, "K": 170},
            density= rho_flinak_eutectic(temperature.value_in_unit(kelvin)),
            charges={'F': -1.0, 'Li': 1.0, 'Na': 1.0, 'K': 1.0},
            min_distance=2.0,
            decimals=6,
            double_box=False,
            axis_to_double='z',
            save_pdb_path=pdb_start
        )

    # read PDB, build system
    pdb = PDBFile(pdb_start)
    topo = pdb.topology
    positions = pdb.getPositions(asNumpy=True)
    trip_force = get_trip_force()
    system = get_system(topo, trip_force)

    # setup ML module
    torch.cuda.set_device(f"cuda:{args.torch_gpu_index}")
    symbols = [a.element.symbol for a in topo.atoms()]
    module = TrIPModule(species=get_species(symbols),
                        model_file=args.trip_model_file,
                        gpu=args.torch_gpu_index)

    # Disable gradients for model parameters to save memory during inference
    for param in module.parameters():
        param.requires_grad_(False)
    
    device = torch.device(f'cuda:{args.torch_gpu_index}')

    # prepare positions & box for ML
    pos_tensor = torch.tensor(positions.value_in_unit(angstrom),
                              dtype=torch.float, device=device)
    bv = system.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(angstrom)
    box_tensor = torch.tensor([bv, bv, bv], dtype=torch.float,
                              device=device)



    # build Simulation
    if is_NVE:
        print("NVE simulation")
        integrator = VerletIntegrator(dt)
    else:
        # NVT or NPT
        print("NVT/NPT simulation")
        integrator = LangevinIntegrator(temperature, 1/picosecond, dt)
        if is_NPT:
            print("NPT simulation")
            # system.addForce(MonteCarloBarostat(1*bar, temperature, 10))
    platform = Platform.getPlatformByName('HIP')
    sim = Simulation(topo, system, integrator, platform,
                     {'DeviceIndex': str(args.torch_gpu_index)})
    dcd_file_exists = os.path.exists(traj_path)
    sim.reporters.append(DCDReporter(traj_path, 1, enforcePeriodicBox=True, append=dcd_file_exists))
    sim.reporters.append(StateDataReporter(stdout, 1, step=True,
                                          temperature=True, potentialEnergy=True, totalEnergy=True, density=True))



    # resume from checkpoint if present
    start_step = load_checkpoint(sim, out_path)
    if start_step == 0:
        # fresh start: set positions & velocities
        # sim.context.setPositions(pos_tensor.cpu().numpy() * nanometer)
        sim.context.setPositions(pdb.getPositions())
        logging.info("Starting new simulation")


        # optional minimization
        if args.minimize and not os.path.exists(minimized_pdb):
            logging.info("Energy minimization …")
            # module.log_energy(pos_tensor, box_tensor)
            # pos_tensor = module.minimize(pos_tensor, box_tensor)
            # module.log_energy(pos_tensor, box_tensor)
            # # update state
            # sim.context.setPositions(pos_tensor.cpu().numpy() * angstrom)
        

            # minimize with OpenMM
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
            # save minimized structure with OpenMM PDBFile
            with open(minimized_pdb, 'w') as f:
                PDBFile.writeFile(topo, state.getPositions(), f)
            
            # clear GPU memory cache after minimization
            torch.cuda.empty_cache()

            logging.info(f"Minimized structure saved to {minimized_pdb}")
            print("Minimized potential energy (kJ/mol):", state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
            print("Minimized kinetic energy (kJ/mol):", state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
            print("Minimized total energy (kJ/mol):", state.getPotentialEnergy().value_in_unit(kilojoule_per_mole) + state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
            # pos_tensor = torch.tensor(state.getPositions().value_in_unit(angstrom), dtype=torch.float, device=device)
            pos = state.getPositions()
            pos_tensor = torch.tensor([[p.x, p.y, p.z] for p in pos],
                            dtype=torch.float, device=device)*10.0 # Nanometer to Angstrom conversion
            bv = state.getPeriodicBoxVectors()[0][0].value_in_unit(angstrom)
            box_tensor = torch.tensor([bv, bv, bv], dtype=torch.float, device=device)
            energy, _ = module(pos_tensor, box_tensor)
            print("TrIP minimized potential energy (kJ/mol):", energy * hartree_to_kJ_per_mol)

            sim.context.setVelocitiesToTemperature(temperature)

            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True, getEnergy=True)
            print("Post velocity potential energy (kJ/mol):", state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
            print("Post velocity kinetic energy (kJ/mol):", state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
            print("Post velocity total energy (kJ/mol):", state.getPotentialEnergy().value_in_unit(kilojoule_per_mole) + state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
            pos = state.getPositions()
            pos_tensor = torch.tensor([[p.x, p.y, p.z] for p in pos],
                            dtype=torch.float, device=device)*10.0 # Nanometer to Angstrom conversion
            bv = state.getPeriodicBoxVectors()[0][0].value_in_unit(angstrom)
            box_tensor = torch.tensor([bv, bv, bv], dtype=torch.float, device=device)
            energy, _ = module(pos_tensor, box_tensor)
            print("TrIP post velocity potential energy (kJ/mol):", energy * hartree_to_kJ_per_mol)

        elif args.minimize:
            logging.info("Using existing minimized structure")
            sim.context.setVelocitiesToTemperature(temperature)


    else:
        logging.info(f"Resuming at step {start_step}")
    


    # run the sim
    num_steps = int(total_time / dt)
    logging.info("Total steps: %d", num_steps)
    for step in range(start_step, num_steps):
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True, getEnergy=True)
        print("Potential Energy (kJ/mol):", state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        print("Kinetic Energy (kJ/mol):", state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
        print("Summed Energy (kJ/mol):", (state.getPotentialEnergy() + state.getKineticEnergy()).value_in_unit(kilojoule_per_mole))


        bv = state.getPeriodicBoxVectors()[0][0].value_in_unit(angstrom)
        box_tensor = torch.tensor([bv, bv, bv], device=device)

        # current_pos = torch.tensor(state.getPositions().value_in_unit(angstrom),
        #                            device=device, requires_grad=True)
        pos = state.getPositions()
        pos = torch.tensor([[p.x, p.y, p.z] for p in pos],
                            dtype=torch.float, device=device)*10.0 # Nanometer to Angstrom conversion

        energy, forces = module(pos, box_tensor)

        print("TrIP Energy (kJ/mol):", energy * hartree_to_kJ_per_mol)

        # Do calculations in torch tensors first - convert to OpenMM units later
        # virial_correction is Σ_i r_i·f_i (pos in Å, forces in Ha/Å) -> Hartree
        # unwrapped_state = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
        # unwrapped_pos = torch.tensor([[p.x, p.y, p.z] for p in unwrapped_state.getPositions()],
        #                               dtype=torch.float, device=device) * 10.0  # Nanometer to Angstrom conversion
        virial_correction = torch.sum(pos * forces)
        energy_per_particle = (energy + virial_correction) / len(pos)  # in Hartree

        # Convert from Hartree to OpenMM's kJ/mol manually for the CustomExternalForce 'c' parameter
        # c (per-particle) is an energy (kJ/mol)
        c = energy_per_particle * hartree_to_kJ_per_mol * kilojoule_per_mole
        print("C:", c.item(), "C*num_atoms:", c.item()*len(pos))

        # Convert forces from Ha/Å to kJ/mol/nm (to match c's units)
        forces_with_units = forces * hartree_to_kJ_per_mol * angstrom_to_nanometer * (kilojoule_per_mole / nanometer)

        # Diagnostic: reconstruct total ML energy (kJ/mol) from per-particle c and forces and compare
        try:
            # pos is in Å here; convert to nm for the dot product with forces_with_units
            pos_nm = pos * angstrom_to_nanometer  # Å -> nm (torch tensor)

            # numeric forces in kJ/mol/nm (torch tensor without units)
            numeric_forces_kj_per_nm = forces * hartree_to_kJ_per_mol * angstrom_to_nanometer

            # dot = Σ_i f_i(kJ/mol/nm) · r_i(nm) -> kJ/mol
            dot_sum_kj = torch.sum(numeric_forces_kj_per_nm * pos_nm)

            # c per particle in kJ/mol (tensor)
            c_per_particle_kj = energy_per_particle * hartree_to_kJ_per_mol

            # reconstructed total energy from the CustomExternalForce expression: Σ_i (c - f·r)
            reconstructed_total_kj = c_per_particle_kj * len(pos) - dot_sum_kj

            # module energy converted to kJ/mol
            ml_energy_kj = energy * hartree_to_kJ_per_mol

            diff_kj = (reconstructed_total_kj - ml_energy_kj).item()
            rel = abs(diff_kj) / (abs(ml_energy_kj) + 1e-12)
            print("Diagnostic: reconstructed_total_kJ - ML_energy_kJ (kJ/mol):", diff_kj, "rel:", rel)
            if rel > 1e-3:
                print("Warning: energy reconstruction mismatch exceeds 0.1% (possible unit/sign/precision issue)")
        except Exception as e:
            print("Diagnostic check failed:", e)

        for idx in range(topo.getNumAtoms()):
            trip_force.setParticleParameters(idx, idx, [c, *forces_with_units[idx]])
        trip_force.updateParametersInContext(sim.context)

        sim.step(1)
        print("\n")

        if (step + 1) % 10 == 0:  # Less frequent cleanup for normal mode
            torch.cuda.empty_cache()
        
        # checkpoint every N steps
        if (step + 1) % args.checkpoint_interval == 0:
            make_checkpoint(sim, out_path, step + 1)
            # Extra memory cleanup after checkpointing
            if args.memory_efficient:
                torch.cuda.empty_cache()

    # final checkpoint (optional)
    make_checkpoint(sim, out_path, num_steps)
    logging.info("Simulation complete.")
