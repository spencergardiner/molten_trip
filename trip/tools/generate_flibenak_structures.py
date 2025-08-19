import numpy as np
from ase.data import atomic_masses, atomic_numbers
from itertools import product
from openmm.app import Topology, PDBFile
from openmm.app.element import Element
from openmm.unit import angstroms
from simtk import unit
from openmm.vec3 import Vec3

def calculate_mass(atomic_system, units='g'):
    """
        Calculates the mass of the atomic system in grams. Receives a dict containing the atomic species or type as
        the key and how many atoms of this type the system has as the value. For instance, {'Bi': 50, 'Ca': 20} means
        that there is 50 Bi atoms and 20 Ca atoms in our system.
    """
    N_a = 6.02214e23
    mass = sum([n_atoms/N_a * atomic_masses[atomic_numbers[atom_type]] for atom_type, n_atoms in atomic_system.items()])

    return mass if units == 'g' else mass*1.0e-3 # Returns mass in kg if units is not 'g'


def box_dimension_estimator(atomic_system, density, density_unit='g/cm3', decimal_digits=6):
    """First approximation to the cubic box/cell size given a density and a system.

    The idea is very simple, since ρ = m_system / V_box, then V_box = m_system / ρ.

    Args:
        atomic_system (dict): A dictionary containing the atom type and the number of the corresponding atoms that we
                              wish to simulate, e.g. atoms={'Na': 32, 'Cl': 32}
        density (float): The density of the molecule at the temperature that we wish to run MD
        density_unit (str): The units we are using. Can be either 'g/cm3' or 'kg/m3'
        decimal_digits (int): Number of decimal digits of the box length.

    Returns:
        (float): estimated box length [Angstrom]
    """
    if density_unit == 'kg/m3':
        density /= 10**3 # Turns density to [g/cm^3]

    box_volume = calculate_mass(atomic_system) / density # [cm^3]
    L = box_volume**(1/3) * 1e8 # [Angstrom]

    return np.around(L, decimal_digits)


def to_mic(r_ij, box_length):
    """Impose minimum image convention on the vector r_ij = r_i - r containing the distances between atom 'i' and all
       other atoms in the system.

    The idea behind PBC is that there are infinite identical copies of the simulation box to emulate a bulk material.
    In that case, if we have 100 atoms in our system and we wish to calculate the distance between a reference atom
    and the other 99 atoms in the box, then we must consider the closest "copy" of each of the 99 atoms to this
    reference atom. This is the minimum image convention (MIC), i.e. we consider the smallest distance between the
    reference atom and the "infinite" copies of the other atoms.

    Args:
        r_ij (np.array): A (n_atoms, 3) matrix r_ij = r_i - r or a (n_atoms, n_atoms, 3) 3-D array containing all r_i-r.
                       To better understand, for the 2-D array case, each row stores (x_i - x_j, y_i - y_j, z_i - z_j)
                       or (dx, dy, dz).
        box_length (float): Length of cubic cell

    Returns:
      np.array: r_ij under MIC

    OBS1: If we have a non-cubic box then we must first transfor the non-cubic box into a cubic box
    OBS2: We can use abs(r_ij) as well because r_ij stores the coordinate differences between atoms 'i' and 'j
          (dx, dy, dz) and the Euclidean distance doesn't depend on the sign of (dx, dy, dz) but only its
          norm as we use their squared values d = sqrt(dx^2 + dy^2 + dz^2)
    """
    # If r_ij/L is < 0.5 then it is already the minimum distance, but if it is > 0.5 we subtract a box length from it.
    # This works because np.round() will round x > 0.5 to 1 and x <= 0.5 to 0. The idea is that the maximum allowed
    # distance along each dimension has to be at most L/2.
    return r_ij - box_length * np.round(r_ij/box_length) # abs(r_ij) - box_length * np.round(abs(r_ij)/box_length)


def generate_coordinates(n_atoms, box_length, min_distance=2.0, max_iter=30, max_tries=25, decimal_digits=6,
                         scale_coords=True, save_txt=False):
    """
    Generate random atomic coordinates inside a cubic box respecting the min_distance constraint.

    Args:
        n_atoms (int) => Number of atom in the system and thus the number of coordinates that will be generated.
        box_length (float) => The parameter of the cubic box where the simulation takes place, where the volume of the
                              box is equal to (box_length)**3
        min_distance (float): Minimum absolute distance between any two atoms in Angstroms.
        max_iter (int): Maximum number of tries to place an atom in a specific grid.
        max_tries (int): Maximum number of tries to rerun the algorithm and try to fit the atoms that are missing to
                         reach the desired number of atoms.
        decimal_digits (int) => Controls the number of decimal digits of the coordinates
        scale_coords (bool): Scales distances by the box length such that all coordinates are now between [0,1].
        save_txt (bool) => If set to 'True', a text file will be created with the coordinates

    Returns:
        (np.array): A (N,3) np.array where 'N' is the total number of atoms in the system.
    """

    N = int(box_length // min_distance)
    grid_spacing = box_length / N
    grid = np.arange(0, box_length, grid_spacing)

    if N ** 3 < n_atoms:
        raise ValueError(
            f'Density is too high, there is no way or it will take a very long time to place {n_atoms} atoms '
            f'in a box length of {box_length:.2f} given that each atom must be {min_distance} apart')

    # Creating a grid that when indexed by (i,j,k) returns the base coordinate at that point
    mesh_grid = np.array(np.meshgrid(grid, grid, grid)).T
    # Swapping axes such that meshgrid[a,b,c] corresponds to gridspacing * [a,b,c]
    mesh_grid = np.swapaxes(mesh_grid, 0, 2)
    mesh_grid = np.swapaxes(mesh_grid, 0, 1)
    occupied_mesh_grid = np.zeros_like(mesh_grid)  # Will store the absolute position of each atom in the grid

    all_indices = np.indices((N, N, N)).T.reshape(-1, 3)
    np.random.shuffle(all_indices)
    available_indices = list(map(tuple, all_indices))
    occupied_indices = []

    aux_neighbors_idx = list(product((0, -1, 1), (0, -1, 1), (0, -1, 1)))
    del aux_neighbors_idx[0]  # Removing (0,0,0) since it is useless
    aux_neighbors_idx = np.array(aux_neighbors_idx)

    rng = np.random.default_rng()
    total_tries = 0

    while len(occupied_indices) < n_atoms:
        if not available_indices:
            # Refill available_indices with the indices that are not occupied after a full sweep over all indices.
            available_indices = set(map(tuple, all_indices.tolist())).difference(occupied_indices)
            total_tries += 1
            # This increase of max_iter speeds the code because we are now trying harder to fit the remaining atoms
            # in the box for a given index, so we don't need to calculate the neighbors and the whole cycle again.
            max_iter += 2
            if total_tries > max_tries:
                raise Exception(f'Exceeded maximum iterations, only {len(occupied_indices)} were placed')

        idx = available_indices.pop()
        neighbor_list = np.array(idx) + aux_neighbors_idx

        # Applying PBC to the neighbors
        neighbor_list[neighbor_list == -1] = N - 1
        neighbor_list[neighbor_list == N] = 0

        # Getting only the coordinates of the occupied neighbors from the neighbor list since we only need to check
        # distance for them (there is no reason to check distance of an empty grid).
        i_neighbors, j_neighbors, k_neighbors = neighbor_list[:, 0], neighbor_list[:, 1], neighbor_list[:, 2]
        neighbors_coords = occupied_mesh_grid[tuple(i_neighbors), tuple(j_neighbors), tuple(k_neighbors)]
        occupied_neighbors_coords = neighbors_coords[np.any(neighbors_coords, axis=1)] # Empty grid have coords (0,0,0)

        for _ in range(max_iter):
            candidate_coord = mesh_grid[idx] + rng.uniform(0, grid_spacing, size=3)
            distances = np.linalg.norm(to_mic(candidate_coord - occupied_neighbors_coords, box_length), axis=1)
            if np.all(distances > min_distance):
                occupied_mesh_grid[idx] = candidate_coord
                occupied_indices.append(tuple(idx))
                break

    occupied_mesh_grid = occupied_mesh_grid.reshape(-1, 3)
    coords = occupied_mesh_grid[np.any(occupied_mesh_grid, axis=1)]
    np.random.shuffle(coords)

    if scale_coords:
        coords /= box_length

    if save_txt:
        np.savetxt(f'{n_atoms}_positions.txt', coords, fmt=f'%.{decimal_digits}f')

    return coords


def create_lammps_data_file(atomic_system, style='full', charges=None, min_distance=2.0, box_length=None, density=None,
                             decimals=6, double_box=False, axis_to_double='z', filename='lammps_data'):
    """Generates a data file for LAMMPS

    Args:
        atomic_system (dict): A dictionary containing the atom type as key and the number of this type of atom as value.
        style (str): Style of lammps data file, can be either "full", "atomic", or "charge".
        charges (dict): A dictionary containing atom type as keys and its charge as values.
        min_distance (float): Minimum distance between any two atoms under pbc.
        box_length (float): The length of the box that the simulation will take place.
        density (float): The density of the system. This parameter is specified when we do not know the box_size.
        decimals (int): Decimal places to write the atomic coordinates.

    """

    if box_length is None and density is None:
        raise Exception('Please specify a box size or a density so the box size can be estimated')
    elif box_length is None:
        density = density/1000 if density > 100 else density # convert to g/cm^3 if it is in kg/m^3
        box_length = box_dimension_estimator(atomic_system, density)

    atom_types, atom_nums = zip(*atomic_system.items())
    n_atoms = sum(atom_nums)
    lammps_type = range(1, len(atom_types)+1)
    type_array = np.hstack([[t] * n for t, n in zip(lammps_type, atom_nums)])
    charge_array = np.zeros(n_atoms) if charges is None else np.hstack([[charges[at]] * atomic_system[at] for at in atom_types])
    box_vec = np.ones(3) * box_length

    if double_box: # Used if we want to create a rectangular box
        target_idx = ['x', 'y', 'z'].index(axis_to_double)
        coord1 = box_length * generate_coordinates(n_atoms, box_length, decimal_digits=decimals, min_distance=min_distance)
        coord2 = box_length * generate_coordinates(n_atoms, box_length, decimal_digits=decimals, min_distance=min_distance)
        coord2[:,target_idx] += box_length
        positions = np.vstack((coord1, coord2))
        n_atoms *= 2
        type_array = np.hstack((type_array, type_array))
        charge_array = np.hstack((charge_array, charge_array))
        box_vec[target_idx] *= 2

    else:
        positions = box_length * generate_coordinates(n_atoms, box_length, decimal_digits=decimals,
                                                min_distance=min_distance)

    coordinate_matrix = np.hstack((np.arange(1, n_atoms+1).reshape(-1,1),
                                   np.zeros((n_atoms,1), dtype=int),
                                   type_array.reshape(-1,1),
                                   charge_array.reshape(-1,1),
                                   positions
                                   ))

    with open(f'{filename}.txt', 'w') as f:
        f.write('Start File for LAMMPS' + 2*'\n')

        f.write(f'{n_atoms} atoms' + '\n')
        f.write(f'{len(atom_types)} atom types' + 2*'\n')

        f.write(f'0.00000000 {box_vec[0]} xlo xhi' + '\n')
        f.write(f'0.00000000 {box_vec[1]} ylo yhi' + '\n')
        f.write(f'0.00000000 {box_vec[2]} zlo zhi' + 2*'\n')

        f.write(f'Masses' + 2*'\n')

        for i, atom in enumerate(atom_types):
            f.write(f'{i+1} {atomic_masses[atomic_numbers[atom]]}\n')
        f.write('\n' + 'Atoms'  + 2*'\n')

        for c in coordinate_matrix:
            if style == 'atomic':
                f.write(f'{int(c[0])} {int(c[2])} '
                        f'{c[4]:.{decimals}f} {c[5]:.{decimals}f} {c[6]:.{decimals}f}\n')
            elif style == 'charge':
                f.write(f'{int(c[0])} {int(c[2])} {c[3]:.1f} '
                        f'{c[4]:.{decimals}f} {c[5]:.{decimals}f} {c[6]:.{decimals}f}\n')
            elif style == 'full':
                f.write(f'{int(c[0])} {int(c[1])} {int(c[2])} {c[3]:.1f} '
                        f'{c[4]:.{decimals}f} {c[5]:.{decimals}f} {c[6]:.{decimals}f}\n')


def create_pdb_box(atomic_system,
                      charges=None,
                      min_distance=2.0,
                      box_length=None,
                      density=None,
                      decimals=6,
                      double_box=False,
                      axis_to_double='z',
                      save_pdb_path='./pdb_output.pdb'
                      ):
    """
    Build a PDB file with a box of atoms, respecting the minimum distance between them.
    
    Args:
        atomic_system (dict):  {atom_type_str: count, ...}
        charges (dict):        {atom_type_str: partial_charge, ...} or None
        min_distance (float):  minimum inter‐ionic distance under PBC (Å)
        box_length (float):    box edge in Å; or if None, inferred from density
        density (float):       g/cm^3 (if box_length is None)
        decimals (int):        grid‐packing precision passed to generate_coordinates
        double_box (bool):     if True, duplicate along `axis_to_double`
        axis_to_double (str):  one of 'x','y','z'
    
    Returns:
        topology (openmm.app.Topology)
        positions (list of Vec3 Quantity in Å)
        charge_array (np.ndarray of floats)
    """
    # --- determine box_length
    if box_length is None and density is None:
        raise ValueError("Specify either box_length or density")
    if box_length is None:
        # convert kg/m^3 → g/cm^3 if needed
        d = density/1000 if density > 100 else density
        box_length = box_dimension_estimator(atomic_system, d)
    
    # flatten types → arrays
    atom_types, atom_counts = zip(*atomic_system.items())
    n_atoms = sum(atom_counts)
    # LAMMPS‐style integer types used only to build charge array
    # here we just make one big list of per-atom charges
    if charges is None:
        charge_array = np.zeros(n_atoms)
    else:
        charge_array = np.hstack([
            np.full(count, charges[atype], dtype=float)
            for atype, count in zip(atom_types, atom_counts)
        ])
    
    # generate positions (fractional), then scale by box_length
    coords = box_length * generate_coordinates(
        n_atoms, box_length,
        decimal_digits=decimals,
        min_distance=min_distance
    )
    
    # handle optional double‐box
    box_vec = np.ones(3) * box_length
    if double_box:  # used to create a rectangular box
        idx = {'x':0,'y':1,'z':2}[axis_to_double]
        coords2 = box_length * generate_coordinates(
            n_atoms, box_length,
            decimal_digits=decimals,
            min_distance=min_distance
        )
        coords2[:,idx] += box_length
        coords = np.vstack([coords, coords2])
        charge_array = np.concatenate([charge_array, charge_array])
        box_vec[idx] *= 2
    
    # --- build the OpenMM Topology
    top = Topology()
    chain = top.addChain()
    # set orthorhombic cell vectors
    # top.setUnitCellDimensions((box_vec[0]*angstroms,
    #                            box_vec[1]*angstroms,
    #                            box_vec[2]*angstroms))

    v1 = Vec3(box_vec[0], 0, 0) / 10 # convert to nm
    v2 = Vec3(0, box_vec[1], 0) / 10
    v3 = Vec3(0, 0, box_vec[2]) / 10

    # top.setPeriodicBoxVectors(v1, v2, v3)

    
    positions = [Vec3(*coord) * angstroms for coord in coords]


    # create one residue per atom (named by element)
    # we need to re‐explode atomic_system to match coords order
    # so we reconstruct the per‐atom type list in the same order as charges/coords
    type_list = []
    for atype, count in zip(atom_types, atom_counts):
        type_list.extend([atype]*count)
    if double_box:
        type_list = type_list + type_list
    
    for atype in type_list:
        # choose element by symbol
        elem = Element.getBySymbol(atype)
        res = top.addResidue(atype, chain)
        top.addAtom(atype, elem, res)
    

    positions = np.array(coords)
    positions = unit.Quantity(positions, unit.angstroms)  # ensure positions are in angstroms
    # print("Positions", positions)

    # set periodic box vectors
    top.setPeriodicBoxVectors((v1, v2, v3))

    PDBFile.writeFile(top, positions, open(save_pdb_path, 'w'))

    return box_length
    



# rho_eutectic_flinak = lambda T: 2.5793 - 0.624e-3*T # [K]
# # create_lammps_data_file(atomic_system={'F': 5000, 'Li': 2325, 'Na': 575 , 'K': 2100}, density=rho_eutectic_flinak(1000), style='atomic')
# flinath_calc = lambda T: 3.6437 - 5.686e-4 * T

# create_pdb_box(
#     atomic_system={'F': 5000, 'Li': 2325, 'Na': 575 , 'K': 2100},
#     density=rho_eutectic_flinak(1000), # density at eutectic temperature ~1000K
#     charges={'F': -1.0, 'Li': 1.0, 'Na': 1.0, 'K': 1.0},
#     min_distance=2.0,
#     decimals=6,
#     double_box=False,
#     axis_to_double='z',
#     save_pdb_path='./flinak_1000K.pdb'
# )

# FLiNaK (46.5-11.5-42 mol % of LiF-NaF-KF), ~720K melting point
# LiF => 2325
# NaF => 575
# KF  => 2100
# Total atoms = 2325*2 + 575*2 + 2100*2 = 10000
# F: 5000, Li: 2325, Na: 575, K: 2100