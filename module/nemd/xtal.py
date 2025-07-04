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
    def fromDatabase(cls, options, **kwargs):
        """
        Get crystal from th database with unit cell stretched or compressed.

        :param options namedtuple: the command line options.
        :return `Crystal`: the crystal.
        """
        crystal = cls.from_database(options.name)
        vecs = crystal.lattice_vectors
        if not np.allclose(options.scale_factor or 0, 1):
            vecs = [x * y for x, y in zip(vecs, options.scale_factor)]
        return cls(crystal, vecs, crystal.source, options=options, **kwargs)

    @functools.cached_property
    def supercell(self):
        """
        See parent.
        """
        return super().supercell(*self.options.dimension)

    @functools.cached_property
    def mol(self):
        """
        Return the crystal as a molecule.

        :return 'Chem.Mol': the supercell molecule.
        """
        atoms = sorted(self.supercell.atoms,
                       key=lambda x: tuple(x.coords_fractional))
        # Build molecule from atoms
        emol = Chem.EditableMol(Chem.Mol())
        for atom in atoms:
            emol.AddAtom(Chem.rdchem.Atom(atom.atomic_number))
        mol = emol.GetMol()
        # Set lattice parameters for the molecule
        lattice_params = self.supercell.lattice_parameters
        vecs = tuple(np.array(lattice_params[:3]) * self.supercell.dimensions)
        vecs += lattice_params[3:]
        mol = structure.Mol(mol, vecs=vecs)
        # Add conformer
        conf = structure.Conf(mol.GetNumAtoms())
        conf.SetPositions(np.array([x.coords_cartesian for x in atoms]))
        mol.AddConformer(conf)
        return mol
