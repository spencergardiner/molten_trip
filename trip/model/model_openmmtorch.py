from copy import deepcopy
import pathlib
from typing import Dict, Optional, Any

import torch
from torch import Tensor
import torch.nn as nn

import dgl
from dgl import DGLGraph
from dgl.nn import SumPooling

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.layers.attention import AttentionBlockSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import get_populated_edge_features
from se3_transformer.runtime.utils import str2bool

from trip.model.layers import TrIPNorm
from trip.model.layers import WeightedEdgeSoftmax


class TrIPTransformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        graph_modules = []
        for _ in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory,
                                                   edge_softmax_fn=WeightedEdgeSoftmax()))
            if norm:
                graph_modules.append(TrIPNorm(fiber_hidden, nonlinearity=lambda x: x))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     pool=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree))
        self.graph_modules = nn.ModuleList(graph_modules)

    def forward(self,
                graph: DGLGraph,
                node_feats: Dict[str, Tensor],
                edge_feats: Dict[str, Tensor],
                scale: Tensor):
        basis = get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=True,
                          use_pad_trick=self.tensor_cores and not self.low_memory,
                          amp=torch.is_autocast_enabled())
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)
        basis = {key: value * scale[..., None, None, None] for key, value in basis.items()}
        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)
        feats = self.forward_graph_modules(node_feats, edge_feats, graph=graph, basis=basis, scale=scale)
        return feats['0'].squeeze(-1)

    def forward_graph_modules(self, node_feats, edge_feats, graph, basis, scale):
        for module in self.graph_modules:
            node_feats = module(node_feats, edge_feats, graph, basis, scale)
            if isinstance(module, AttentionBlockSE3):
                num_edge_channels = min(node_feats['0'].shape[1], edge_feats['0'].shape[1])
                edge_feats['0'] = edge_feats['0'][:, :num_edge_channels] + dgl.ops.copy_u(graph, node_feats['0'][:, :num_edge_channels])
        return node_feats

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_layers', type=int, default=7,
                            help='Number of stacked Transformer layers')
        parser.add_argument('--num_heads', type=int, default=8,
                            help='Number of heads in self-attention')
        parser.add_argument('--channels_div', type=int, default=2,
                            help='Channels division before feeding to attention layer')
        parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply a normalization layer after each attention block')
        parser.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply layer normalization between MLP layers')
        parser.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                            help='If true, will use fused ops that are slower but that use less memory')
        parser.add_argument('--tensor_cores', type=str2bool, nargs='?', const=True, default=False,
                            help='Whether to assume tensor core usage (affects fused bases).')
        return parser


class TrIPModel(TrIPTransformer):
    def __init__(self,
                 num_degrees: int,
                 num_channels: int,
                 energy_std: float,
                 cutoff: float,
                 coulomb_energy_unit: float,
                 si_tensor: Optional[Tensor] = None,
                 # Transformer arguments
                 num_layers: int = 7,
                 num_heads: int = 8,
                 channels_div: int = 2,
                 norm: bool = False,
                 use_layer_norm: bool = False,
                 tensor_cores: bool = False,
                 low_memory: bool = False):
        self.num_degrees = num_degrees
        self.num_channels = num_channels
        self.energy_std = energy_std
        self.cutoff = cutoff
        self.coulomb_energy_unit = coulomb_energy_unit
        self.si_tensor = si_tensor
        num_out_channels = num_channels * num_degrees
        super().__init__(fiber_in=Fiber.create(1, num_channels),
                         fiber_hidden=Fiber.create(num_degrees, num_channels),
                         fiber_out=Fiber.create(1, num_out_channels),
                         fiber_edge=Fiber.create(1, num_channels - 1),
                         num_layers=num_layers,
                         num_heads=num_heads,
                         channels_div=channels_div,
                         norm=norm,
                         use_layer_norm=use_layer_norm,
                         tensor_cores=tensor_cores,
                         low_memory=low_memory)
        self.embedding = nn.Embedding(100, num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_out_channels + num_channels, num_out_channels),
            nn.SiLU(),
            nn.Linear(num_out_channels, num_out_channels),
            nn.SiLU(),
            nn.Linear(num_out_channels, 1)
        )
        self.pool = SumPooling()

    def forward(self, graph, forces: bool = True, create_graph: bool = False, standardized: bool = False):
        atom_energies = self.forward_atom_energies(graph, standardized)
        energies = self.pool(graph, atom_energies)
        if not forces:
            return energies
        forces_tensor = -torch.autograd.grad(torch.sum(energies),
                                             graph.ndata['pos'],
                                             create_graph=create_graph)[0]
        return energies, forces_tensor

    def forward_atom_energies(self, graph, standardized: bool):
        dist = torch.norm(graph.edata['rel_pos'], p=2, dim=1)
        scale = self.scale_fn(dist, self.cutoff)
        species_embedding = self.embedding(graph.ndata['species'] - 1)
        node_feats = {'0': species_embedding.unsqueeze(-1)}
        radial_basis = self.get_radial_basis(dist)
        edge_feats = {'0': radial_basis.unsqueeze(-1)}
        feats = super().forward(graph, node_feats, edge_feats, scale)
        cat_feats = torch.cat([species_embedding, feats], dim=1)
        learned_energies = self.mlp(cat_feats).squeeze(-1)
        coulomb_energies = self.screened_coulomb(graph, dist, scale)
        atom_energies = learned_energies + coulomb_energies
        if standardized:
            return atom_energies
        atom_energies = atom_energies * self.energy_std
        species = graph.ndata['species']
        if self.si_tensor is not None:
            atom_energies = atom_energies + self.si_tensor[(species - 1).tolist()]
        return atom_energies

    @staticmethod
    def scale_fn(dist, cutoff):
        scale = torch.zeros_like(dist)
        mask = dist < cutoff
        if torch.any(mask):
            scale[mask] = TrIPModel.bump_fn(dist[mask] / cutoff, k=3)
        return scale

    @staticmethod
    def bump_fn(x, k=1):
        return torch.exp(1 - k / (1 - x ** 2))

    def screened_coulomb(self, graph, dist, scale):
        Z = graph.ndata['species']
        u, v = graph.edges()
        Z_u, Z_v = Z[u], Z[v]
        raw_energies = self.coulomb_energy_unit * Z_u * Z_v / (2 * dist)
        screen = TrIPModel.zbl_screening_fn(dist, Z_u, Z_v)
        screened_energies = raw_energies * screen * scale
        atom_coulomb_energies = dgl.ops.copy_e_sum(graph, screened_energies)
        return atom_coulomb_energies / self.energy_std

    @staticmethod
    def zbl_screening_fn(dist, Z_u, Z_v):
        au = 0.8854 * 0.529 / (Z_u ** 0.23 + Z_v ** 0.23)
        x = dist / au
        screening_factor = 0.1818 * torch.exp(-3.2 * x) \
                           + 0.5099 * torch.exp(-0.9423 * x) \
                           + 0.2802 * torch.exp(-0.4028 * x) \
                           + 0.02817 * torch.exp(-0.2016 * x)
        return screening_factor

    def get_radial_basis(self, dist) -> Tensor:
        return torch.zeros(len(dist), self.num_channels - 1, device=dist.device, dtype=dist.dtype)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_degrees', type=int, default=3,
                            help='Number of degrees to use. Hidden features types [0..num_degrees-1]')
        parser.add_argument('--num_channels', type=int, default=16,
                            help='Number of channels for hidden features')
        parser.add_argument('--cutoff', type=float, default=4.6,
                            help='Radius of graph neighborhood')
        parser.add_argument('--coulomb_energy_unit', type=float, default=0.529,
                            help='Value of e^2/(4*pi*e0) in preferred units (Ha*A)')
        TrIPTransformer.add_argparse_args(parser)
        return parser


class TrIP(TrIPModel):
    def __init__(self,
                 num_degrees: int = 3,
                 num_channels: int = 16,
                 energy_std: float = 1.0,
                 cutoff: float = 4.6,
                 coulomb_energy_unit: float = 0.529,
                 si_tensor: Optional[Tensor] = None,
                 num_layers: int = 7,
                 num_heads: int = 8,
                 channels_div: int = 2,
                 norm: bool = False,
                 use_layer_norm: bool = False,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 optimizer_type: str = 'adam',
                 learning_rate: float = 0.002,
                 momentum: float = 0.9,
                 weight_decay: float = 0.1,
                 force_weight: float = 0.1,
                 gamma: float = 1.0):
        super().__init__(num_degrees=num_degrees,
                         num_channels=num_channels,
                         energy_std=energy_std,
                         cutoff=cutoff,
                         coulomb_energy_unit=coulomb_energy_unit,
                         si_tensor=si_tensor,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         channels_div=channels_div,
                         norm=norm,
                         use_layer_norm=use_layer_norm,
                         tensor_cores=tensor_cores,
                         low_memory=low_memory)
        self.optimizer = TrIP.make_optimizer(self,
                                             optimizer_type=optimizer_type,
                                             learning_rate=learning_rate,
                                             momentum=momentum,
                                             weight_decay=weight_decay)
        # Store configuration explicitly (without **kwargs) for checkpoint compatibility
        self.kwargs: Dict[str, Any] = dict(num_degrees=num_degrees,
                                           num_channels=num_channels,
                                           energy_std=energy_std,
                                           cutoff=cutoff,
                                           coulomb_energy_unit=coulomb_energy_unit,
                                           si_tensor=si_tensor,
                                           num_layers=num_layers,
                                           num_heads=num_heads,
                                           channels_div=channels_div,
                                           norm=norm,
                                           use_layer_norm=use_layer_norm,
                                           tensor_cores=tensor_cores,
                                           low_memory=low_memory,
                                           optimizer_type=optimizer_type,
                                           learning_rate=learning_rate,
                                           momentum=momentum,
                                           weight_decay=weight_decay,
                                           force_weight=force_weight,
                                           gamma=gamma)

    def save(self, path: pathlib.Path, epoch: int):
        checkpoint = self.get_checkpoint(epoch)
        torch.save(checkpoint, str(path))

    def get_checkpoint(self, epoch: int):
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'kwargs': self.kwargs,
            'epoch': epoch
        }
        return checkpoint

    def load_state(self, checkpoint, map_location='cuda:0', weights_only: bool = False):
        torch.serialization.add_safe_globals([pathlib.PosixPath])
        if isinstance(checkpoint, (pathlib.Path, str)):
            checkpoint = torch.load(str(checkpoint), map_location=map_location, weights_only=weights_only)
        self.load_state_dict(checkpoint['state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict') is not None and not weights_only:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass
        # Preserve original kwargs for reference
        self.kwargs = deepcopy(checkpoint.get('kwargs', {}))
        return checkpoint

    @staticmethod
    def load(path: pathlib.Path, map_location='cuda:0', weights_only: bool = False):
        torch.serialization.add_safe_globals([pathlib.PosixPath])
        checkpoint = torch.load(str(path), map_location=map_location, weights_only=weights_only)
        old_kwargs = checkpoint.get('kwargs', {})
        # Map old kwargs (possibly from **kwargs version) to explicit arguments with defaults fallback
        def get(k, default):
            return old_kwargs[k] if k in old_kwargs else default

        model = TrIP(num_degrees=get('num_degrees', 3),
                     num_channels=get('num_channels', 16),
                     energy_std=get('energy_std', 1.0),
                     cutoff=get('cutoff', 4.6),
                     coulomb_energy_unit=get('coulomb_energy_unit', 0.529),
                     si_tensor=get('si_tensor', None),
                     num_layers=get('num_layers', 7),
                     num_heads=get('num_heads', 8),
                     channels_div=get('channels_div', 2),
                     norm=get('norm', False),
                     use_layer_norm=get('use_layer_norm', False),
                     tensor_cores=get('tensor_cores', False),
                     low_memory=get('low_memory', False),
                     optimizer_type=get('optimizer_type', 'adam'),
                     learning_rate=get('learning_rate', 0.002),
                     momentum=get('momentum', 0.9),
                     weight_decay=get('weight_decay', 0.1),
                     force_weight=get('force_weight', 0.1),
                     gamma=get('gamma', 1.0))
        model.to(device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        model.load_state(checkpoint, map_location, weights_only=weights_only)
        return model

    @staticmethod
    def make_optimizer(model, optimizer_type, learning_rate, momentum, weight_decay, **_ignored):
        parameters = TrIP.add_weight_decay(model, weight_decay, skip_list=['embedding.weight', 'mlp.4.weight'])
        if optimizer_type == 'adam':
            return torch.optim.Adam(parameters, lr=learning_rate, betas=(momentum, 0.999),
                                    weight_decay=weight_decay)
        elif optimizer_type == 'lamb':
            return torch.optim.AdamW(parameters, lr=learning_rate, betas=(momentum, 0.999),
                                     weight_decay=weight_decay)
        else:
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum,
                                   weight_decay=weight_decay)

    @staticmethod
    def add_weight_decay(model, weight_decay, skip_list=None):
        if skip_list is None:
            skip_list = []
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list or 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    @staticmethod
    def loss_fn(pred, target, beta=2e-1):
        calc_loss = lambda x: beta * (torch.mean(torch.sqrt(x + beta ** 2)) - beta)
        energy_loss = calc_loss((pred[0] - target[0]) ** 2)
        forces_loss = calc_loss(torch.sum((pred[1] - target[1]) ** 2, dim=1))
        return energy_loss, forces_loss

    @staticmethod
    def error_fn(pred, target, num_atoms=0):
        end = len(pred[0]) - num_atoms
        energy_error = torch.sum((pred[0][:end] - target[0][:end]) ** 2)
        end = len(pred[1]) - num_atoms
        forces_error = torch.sum((pred[1][:end] - target[1][:end]) ** 2)
        return energy_error, forces_error

    @staticmethod
    def add_argparse_args(parent_parser):
        opt_parser = parent_parser.add_argument_group('Optimizer')
        opt_parser.add_argument('--optimizer_type', choices=['adam', 'sgd', 'lamb'], default='adam')
        opt_parser.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.002)
        opt_parser.add_argument('--gamma', type=float, default=1.0)
        opt_parser.add_argument('--momentum', type=float, default=0.9)
        opt_parser.add_argument('--weight_decay', type=float, default=0.1)

        model_parser = parent_parser.add_argument_group('Model architecture')
        model_parser.add_argument('--force_weight', type=float, default=0.1,
                                  help='Weight force losses relative to energy losses')
        TrIPModel.add_argparse_args(model_parser)
        return parent_parser
