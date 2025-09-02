import logging
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo 

from scipy.optimize import minimize
import torch
from torch.autograd.functional import hessian
import numpy as np
import time

import torchani

from trip.data_loading import GraphConstructor
# Import the TorchScript-friendly TrIP implementation explicitly for OpenMMTorch integration
from trip.model.model_openmmtorch import TrIP as TrIPTorchScript


class TrIPModule(torch.nn.Module):
    def __init__(self, species, model_file, gpu, constraints=[], **vars):
        super().__init__() 
        self.device = f'cuda:{gpu}'
        self.species_tensor = torch.tensor(species, dtype=torch.long, device=self.device)
        self.constraints = constraints
        if model_file == 'ani2x':
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
            self.forward = self.ani_forward
        else:
            self.model = TrIP.load(model_file, map_location=self.device)
            self.graph_constructor = GraphConstructor(cutoff=self.model.cutoff)
            self.forward = self.trip_forward

    def trip_forward(self, pos, boxsize, forces=True):
        graph = self.graph_constructor.create_graphs(pos, boxsize)  # Cutoff for 5-12 model is 3.0 A
        graph.ndata['species'] = self.species_tensor

        if forces:
            energy, forces = self.model(graph, forces=True)
            return energy.item(), forces
        else:
            energy = self.model(graph, forces=False)
            return energy.item()

    def ani_forward(self, pos, boxsize, forces=True):
        # Only supports boxsize == inf
        pos.requires_grad_(True)
        energy = self.model((self.species_tensor.unsqueeze(0), pos.unsqueeze(0))).energies.sum()
        if forces:
            forces = -torch.autograd.grad(energy.sum(), pos)[0]
            return energy.item(), forces
        else:
            return energy.item()

    def energy_np_(self, pos, boxsize):
        pos = pos.clone().detach().to(self.device).reshape(-1,3)
        with torch.no_grad():
            energy = self.forward(pos, boxsize=boxsize, forces=False)
        energy += self.calc_constraints(pos)
        if self.forward == self.ani_forward:
            try:    
                energy = energy.item()
            except AttributeError:
                pass

        return energy

    def energy_np(self, pos, boxsize):
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, dtype=torch.float, device=self.device)
        else:
            pos = pos.to(dtype=torch.float, device=self.device)

        pos = pos.reshape(-1, 3)
        with torch.no_grad():
            energy = self.forward(pos, boxsize=boxsize, forces=False)
        energy += self.calc_constraints(pos)
        return energy

    def jac_np(self, pos, boxsize):
        pos = torch.tensor(pos, dtype=torch.float, device=self.device).reshape(-1,3)
        _, forces = self.forward(pos, boxsize=boxsize)
        pos.requires_grad_(True)
        error = self.calc_constraints(pos)
        if error == 0:
            jac = torch.zeros_like(pos)
        else:
            jac = torch.autograd.grad(error, pos, create_graph=True)[0]
        jac -= forces
        pos.requires_grad_(False)
        return jac.detach().cpu().numpy().flatten()

    def hess(self, pos, boxsize):
        def energy(pos):
            graph = self.graph_constructor.create_graphs(pos.reshape(-1,3), boxsize)
            graph.ndata['species'] = self.species_tensor
            return self.model(graph, forces=False)
        return hessian(energy, pos.flatten())

    def timed_minimize(self, pos, boxsize, gpu, method='CG'):
        step_times = []  # Store step-wise timing
        gpu_memory_usage = []  # Store peak GPU memory usage (in MB) at each step

        # Initialize NVML for GPU monitoring
        torch.cuda.reset_peak_memory_stats()
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu)

        def energy_np_cpu(pos_flat, boxsize):
            pos_flat = torch.from_numpy(pos_flat.astype(np.float32)).to(self.device)
            energy = self.energy_np(pos_flat, boxsize)
            return energy.cpu().numpy() if isinstance(energy, torch.Tensor) else energy

        def callback(xk):
            """Callback function to record step times and GPU memory usage."""
            nonlocal step_start
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
            step_start = time.perf_counter()  # Reset timer for next step

            # Get GPU memory usage
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2) # gpu peak in MB
            gpu_memory_usage.append(peak_gpu)  # Track memory usage in MB
            torch.cuda.reset_peak_memory_stats()

        step_start = time.perf_counter()  # start time for first step
        sol = minimize(energy_np_cpu, pos.cpu().numpy().flatten(), args=boxsize,
                   method=method, jac=self.jac_np, callback=callback)
        
        # return only memory info
        return gpu_memory_usage

    def minimize(self, pos, boxsize, method='CG'):
        def energy_np_cpu(pos_flat, boxsize):
            pos_flat = torch.from_numpy(pos_flat.astype(np.float32)).to(self.device)
            energy = self.energy_np(pos_flat, boxsize)
            return energy.cpu().numpy() if isinstance(energy, torch.Tensor) else energy

        sol = minimize(energy_np_cpu, pos.cpu().numpy().flatten(), args=boxsize, method=method, jac=self.jac_np)
        return torch.tensor(sol.x, dtype=torch.float, device=self.device).reshape(-1,3)

    def calc_constraints(self, pos):
        return sum([constraint(pos) for constraint in self.constraints])

    def log_energy(self, pos, boxsize):
        with torch.no_grad():
            energy = self.forward(pos, boxsize, forces=False)
        logging.info(f'Energy: {energy*627.5:.2f}')






# import logging
# # from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo 

# from scipy.optimize import minimize
# import torch
# from torch.autograd.functional import hessian
# import numpy as np
# import time


from trip.data_loading import GraphConstructor
from trip.model import TrIP

class TripOpenmmTorchForceModule(torch.nn.Module):
    """Wrapper to use the TorchScript-friendly TrIP model inside OpenMMTorch.

    Notes
    -----
    - Positions passed in are in nanometers; TrIP was trained using Angstrom units.
      We therefore convert positions_nm -> positions_A by *10 before constructing
      the graph for the model.
    - Forces returned by the model are dE/d(Angstrom); physical forces w.r.t. nm
      require chain rule multiplication by 10.
    - Energy units: model returns Hartree; OpenMM expects kJ/mol.
    - Force units: model returns Hartree/Angstrom.
        Convert: Hartree/Angstrom * (2625.5 kJ/mol / Hartree) * (10 Angstrom / nm)
                = factor 26255.0 kJ/mol/nm.
    """
    def __init__(self, species, model_file, map_location: str = "cuda:0"):
        super().__init__()
        device = torch.device(map_location if torch.cuda.is_available() else "cpu")
        self.species_tensor = torch.tensor(species, dtype=torch.long, device=device)
        # Load the explicit-args TrIP implementation
        self.model = TrIPTorchScript.load(model_file, map_location=device)
        # Guard si_tensor (may be None depending on checkpoint)
        if getattr(self.model, 'si_tensor', None) is not None:
            self.model.si_tensor = self.model.si_tensor.to(device)
        
        self.graph_constructor = GraphConstructor(cutoff=self.model.cutoff)
        self.ha_to_kJmol = 2625.5  # Conversion factor from Hartree to kJ/mol
        self.force_unit_factor = self.ha_to_kJmol / 10.0  # Hartree/Å -> kJ/mol/nm

    def forward(self, positions: torch.Tensor, boxvectors: torch.Tensor):
        """Return potential energy (kJ/mol) and forces (kJ/mol/nm).

        Parameters
        ----------
        positions : (N,3) tensor in nanometers
        boxvectors : (3,3) tensor in nanometers
        """
        # Convert to Angstrom for model
        positions_angstrom = positions * 10.0
        positions_angstrom = positions_angstrom.requires_grad_(True)
        boxsize = boxvectors.diag() * 10.0  # (nm -> Å)
        graph = self.graph_constructor.create_graphs(positions_angstrom, boxsize)
        graph.ndata['species'] = self.species_tensor
        energy, forces = self.model(graph, forces=True)
        if isinstance(energy, torch.Tensor):
            energy_val = energy.item()
        else:
            energy_val = float(energy)
        # Convert energy Hartree -> kJ/mol
        energy_kJmol = energy_val * self.ha_to_kJmol
        # Convert forces Hartree/Å -> kJ/mol/nm (includes chain rule factor 10)
        forces_kJmol_per_nm = forces * self.force_unit_factor
        return energy_kJmol, forces_kJmol_per_nm
        


# class TrIPModule(torch.nn.Module):
#     def __init__(self, species, model_file, gpu, constraints=[]):
#         super().__init__() 
#         self.device = f'cuda:{gpu}'
#         self.species_tensor = torch.tensor(species, dtype=torch.long, device=self.device)

#         self.constraints = constraints
#         if model_file == 'ani2x':
#             self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
#             self.forward = self.ani_forward
#         else:
#             self.model = TrIP.load(model_file, map_location={"cuda:0":self.device})
#             self.model = self.model.to(self.device)
#             self.model.si_tensor = self.model.si_tensor.to(self.device)  # Ensure si_tensor is on the same device
#             self.graph_constructor = GraphConstructor(cutoff=self.model.cutoff)
#             self.forward = self.trip_forward

#     def trip_forward(self, pos, boxsize, forces=True):
#         graph = self.graph_constructor.create_graphs(pos, boxsize).to(self.device)  # Cutoff for 5-12 model is 3.0 A
#         graph.ndata['species'] = self.species_tensor

#         if forces:
#             energy, forces = self.model(graph, forces=True)
#             return energy.item(), forces
#         else:
#             energy = self.model(graph, forces=False)
#             return energy.item()

#     def ani_forward(self, pos, boxsize, forces=True):
#         # Only supports boxsize == inf
#         pos.requires_grad_(True)
#         energy = self.model((self.species_tensor.unsqueeze(0), pos.unsqueeze(0))).energies.sum()
#         if forces:
#             forces = -torch.autograd.grad(energy.sum(), pos)[0]
#             return energy.item(), forces
#         else:
#             return energy.item()

#     def energy_np_(self, pos, boxsize):
#         pos = pos.clone().detach().to(self.device).reshape(-1,3)
#         with torch.no_grad():
#             energy = self.forward(pos, boxsize=boxsize, forces=False)
#         energy += self.calc_constraints(pos)
#         if self.forward == self.ani_forward:
#             try:    
#                 energy = energy.item()
#             except AttributeError:
#                 pass

#         return energy

#     def energy_np(self, pos, boxsize):
#         if not torch.is_tensor(pos):
#             pos = torch.tensor(pos, dtype=torch.float, device=self.device)
#         else:
#             pos = pos.to(dtype=torch.float, device=self.device)

#         pos = pos.reshape(-1, 3)
#         with torch.no_grad():
#             energy = self.forward(pos, boxsize=boxsize, forces=False)
#         energy += self.calc_constraints(pos)
#         return energy

#     def jac_np(self, pos, boxsize):
#         pos = torch.tensor(pos, dtype=torch.float, device=self.device).reshape(-1,3)
#         _, forces = self.forward(pos, boxsize=boxsize)
#         pos.requires_grad_(True)
#         error = self.calc_constraints(pos)
#         if error == 0:
#             jac = torch.zeros_like(pos)
#         else:
#             jac = torch.autograd.grad(error, pos, create_graph=True)[0]
#         jac -= forces
#         pos.requires_grad_(False)
#         return jac.detach().cpu().numpy().flatten()

#     def hess(self, pos, boxsize):
#         def energy(pos):
#             graph = self.graph_constructor.create_graphs(pos.reshape(-1,3), boxsize)
#             graph.ndata['species'] = self.species_tensor
#             return self.model(graph, forces=False)
#         return hessian(energy, pos.flatten())

#     # def timed_minimize(self, pos, boxsize, gpu, method='CG'):
#     #     step_times = []  # Store step-wise timing
#     #     gpu_memory_usage = []  # Store peak GPU memory usage (in MB) at each step

#     #     # Initialize NVML for GPU monitoring
#     #     torch.cuda.reset_peak_memory_stats()
#     #     nvmlInit()
#     #     handle = nvmlDeviceGetHandleByIndex(gpu)

#     #     def energy_np_cpu(pos_flat, boxsize):
#     #         pos_flat = torch.from_numpy(pos_flat.astype(np.float32)).to(self.device)
#     #         energy = self.energy_np(pos_flat, boxsize)
#     #         return energy.cpu().numpy() if isinstance(energy, torch.Tensor) else energy

#     #     def callback(xk):
#     #         """Callback function to record step times and GPU memory usage."""
#     #         nonlocal step_start
#     #         step_end = time.perf_counter()
#     #         step_times.append(step_end - step_start)
#     #         step_start = time.perf_counter()  # Reset timer for next step

#     #         # Get GPU memory usage
#     #         memory_info = nvmlDeviceGetMemoryInfo(handle)
#     #         peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2) # gpu peak in MB
#     #         gpu_memory_usage.append(peak_gpu)  # Track memory usage in MB
#     #         torch.cuda.reset_peak_memory_stats()

#     #     step_start = time.perf_counter()  # start time for first step
#     #     sol = minimize(energy_np_cpu, pos.cpu().numpy().flatten(), args=boxsize,
#     #                method=method, jac=self.jac_np, callback=callback)
        
#     #     # return only memory info
#     #     return gpu_memory_usage

#     def minimize(self, pos, boxsize, method='CG'):
#         def energy_np_cpu(pos_flat, boxsize):
#             pos_flat = torch.from_numpy(pos_flat.astype(np.float32)).to(self.device)
#             energy = self.energy_np(pos_flat, boxsize)
#             return energy.cpu().numpy() if isinstance(energy, torch.Tensor) else energy

#         sol = minimize(energy_np_cpu, pos.cpu().numpy().flatten(), args=boxsize, method=method, jac=self.jac_np)
#         return torch.tensor(sol.x, dtype=torch.float, device=self.device).reshape(-1,3)

#     def calc_constraints(self, pos):
#         return sum([constraint(pos) for constraint in self.constraints])

#     def log_energy(self, pos, boxsize):
#         with torch.no_grad():
#             energy = self.forward(pos, boxsize, forces=False)
#         logging.info(f'Energy: {energy*627.5:.2f}')
