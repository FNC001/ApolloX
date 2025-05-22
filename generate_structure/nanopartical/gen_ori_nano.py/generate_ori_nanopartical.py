from ase import Atoms
from ase.build import bulk
from ase.io import write
import numpy as np

# Create a copper single crystal with face-centered cubic (fcc) structure
cu_bulk = bulk('Cu', 'fcc', a=3.6)  # 'a' is the lattice constant for copper

# Generate a nanoparticle by cutting from a large crystal
def create_nanoparticle(atom, size, vacuum=10.0):
    # Expand the structure by replicating the unit cell
    supercell = atom * (size, size, size)
    
    # Calculate the radius of the nanoparticle (assumed spherical)
    radius = size * atom.cell.lengths()[0] / 2
    
    # Create a mask to select atoms within a spherical region
    center = np.sum(supercell.positions, axis=0) / len(supercell)
    mask = np.linalg.norm(supercell.positions - center, axis=1) < radius
    
    # Filter atoms using the mask
    nanoparticle_atoms = supercell[mask]

    # Calculate the bounding box and add a vacuum layer
    cell_size = np.ptp(nanoparticle_atoms.positions, axis=0) + 2 * vacuum
    nanoparticle = Atoms(
        nanoparticle_atoms.symbols,
        positions=nanoparticle_atoms.positions - nanoparticle_atoms.positions.min(axis=0) + vacuum,
        cell=cell_size,
        pbc=True
    )

    return nanoparticle

# Set the size of the nanoparticle (in number of unit cells along each direction)
nanoparticle_size = 8  # For example, an 8x8x8 unit cell nanoparticle
vacuum_layer = 15.0    # Add 15 Å of vacuum around the nanoparticle

try:
    nanoparticle = create_nanoparticle(cu_bulk, nanoparticle_size, vacuum_layer)
    # Save the nanoparticle structure in POSCAR format
    write('POSCAR', nanoparticle, format='vasp')
    print("POSCAR file has been saved with the nanoparticle including a vacuum layer.")
except Exception as e:
    print("An error occurred while saving the POSCAR file:", e)

from ase.visualize import view
view(nanoparticle)
