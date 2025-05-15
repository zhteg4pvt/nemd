# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module builds crystals.
"""
import functools

import crystals
import numpy as np
from rdkit import Chem

from nemd import structure


class Crystal(crystals.Crystal):
    """
    Crystal from the database with unit cell lattice vectors scaled.
    """

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': command-line options
        """
        super().__init__(*args, **kwargs)
        self.options = options

    @classmethod
    def from_database(cls, options=None, **kwargs):
        crystal = super().from_database(options.name)
        vecs = crystal.lattice_vectors
        if not np.allclose(options.scale_factor or 0, 1):
            vecs = [x * y for x, y in zip(vecs, options.scale_factor)]
        return cls(crystal, vecs, crystal.source, options=options, **kwargs)

    @property
    @functools.cache
    def super_cell(self):
        """
        Stretched (or compressed) the unit cell, and duplicate in dimensions.

        :return 'crystals.crystal.Supercell': the super cell.
        """
        return self.supercell(*self.options.dimension)

    @property
    @functools.cache
    def mol(self):
        """
        Return the crystal as a molecule.

        :return 'rdkit.Chem.rdchem.Mol': the molecule in the supercell.
        """
        atoms = sorted(self.super_cell.atoms,
                       key=lambda x: tuple(x.coords_fractional))
        # Build molecule from atoms
        emol = Chem.EditableMol(Chem.Mol())
        for atom in atoms:
            emol.AddAtom(Chem.rdchem.Atom(atom.atomic_number))
        mol = emol.GetMol()
        # Set lattice parameters for the molecule
        lattice_params = self.super_cell.lattice_parameters
        vecs = tuple(np.array(lattice_params[:3]) * self.super_cell.dimensions)
        vecs += lattice_params[3:]
        mol = structure.Mol(mol, vecs=vecs)
        # Add conformer
        conf = structure.Conf(mol.GetNumAtoms())
        xyz = np.array([x.coords_cartesian for x in atoms])
        conf.setPositions(xyz)
        mol.AddConformer(conf)
        return mol
