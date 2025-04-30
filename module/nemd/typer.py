# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module types the atoms based on the opls-ua force field.
"""
import collections
import functools

import numpy as np
import pandas as pd
from rdkit import Chem

from nemd import envutils
from nemd import logutils
from nemd import rdkitutils
from nemd import symbols

logger = logutils.Logger.get(__file__)


class Typer:
    """
    Type the atoms and map SMILES fragments.
    """
    RES_NUM = symbols.RES_NUM
    TYPE_ID = symbols.TYPE_ID

    def __init__(self, wmodel=symbols.TIP3P):
        """
        :param wmodel str: the model type for water
        """
        self.wmodel = wmodel
        self.mol = None
        self.res_num = None
        self.mx = None

    def type(self, mol):
        """
        Assign atom types for force field assignment.

        :param mol 'rdkit.Chem.rdchem.Mol': molecule to assign FF types
        """
        self.setUp(mol)
        self.doTyping()
        self.setResNum()

    def setUp(self, mol):
        """
        Set up the typer for the new molecule typing.

        :param mol 'rdkit.Chem.rdchem.Mol': molecule to assign FF types
        """
        self.mol = mol
        self.res_num = 0
        self.mx = min([symbols.MAX_INT32, self.mol.GetNumAtoms()])

    def doTyping(self):
        """
        Match the substructure with SMILES and assign atom type.

        :raise KeyError: if any atoms are not marked.
        """
        unmarked = self.mol.GetNumAtoms()
        for idx, sml in self.smiles.iterrows():
            if unmarked <= 0:
                break
            matches = self.mol.GetSubstructMatches(sml.mol, maxMatches=self.mx)
            if not matches:
                continue
            logger.debug(f"assignAtomType {sml.sml}, {matches}")
            marks = []
            for match in matches:
                marked = list(self.mark(match, sml))
                if not marked:
                    continue
                unmarked -= len(marked)
                self.res_num += 1
                marks.append(marked)
            cnt = collections.Counter([len(x) for x in marks])
            counted = ','.join([f'{x}*{y}' for x, y in cnt.items()])
            logger.debug(f"{sml.sml}: {len(marks)} matches ({counted})")
        logger.debug(f"{unmarked} / {self.mol.GetNumAtoms()} unmarked")
        if not unmarked:
            return
        atoms = self.mol.GetAtoms()
        atom = next(x for x in atoms if not x.HasProp(self.RES_NUM))
        raise KeyError(f'Typing missed {atom.GetSymbol()} {atom.GetIdx()}')

    @property
    @functools.cache
    def smiles(self):
        """
        Return the smiles-based typing table.

        :return `pd.DataFrame`: the smiles-based typing table
        """
        namepath = envutils.get_data('ff', 'oplsua', 'typer.parquet')
        sml = pd.read_parquet(namepath)
        sml['mp'] = sml['mp'].map(lambda x: [x - 1 for x in x])
        water_range = range(*eval(sml.index.name))
        water_dsc = symbols.WATER_DSC.format(model=self.wmodel)
        to_drop = [x for x in water_range if sml.iloc[x].dsc != water_dsc]
        sml.drop(index=to_drop, inplace=True)
        sml.hs = sml.hs.apply(eval)
        sml['hs'] = sml['hs'].map(lambda x: x if x is None else {
            x - 1: y - 1
            for x, y in x.items()
        })
        sml['mol'] = [rdkitutils.MolFromSmiles(x) for x in sml.sml]
        sml['deg'] = [
            np.array(list(map(self.getDeg, x.GetAtoms()))) for x in sml.mol
        ]
        return sml.iloc[::-1]

    @staticmethod
    def getDeg(atom):
        """
        Get the degree of the atom. (the hydrogen atoms on carbons are not
        counted towards in the degree in the united atom model)

        :param atom `rdkit.Chem.rdchem.Atom`: the atom to get degree of
        :return list: the degree of the atom
        """
        degree = atom.GetDegree()
        if atom.GetSymbol() != symbols.CARBON:
            degree += atom.GetNumImplicitHs()
        return degree

    def mark(self, match, sml):
        """
        Marker atoms with type id, res_num, and bonded_atom id for vdw/charge
        table lookup, charge balance, and bond searching.

        :param match tuple: atom ids of one match
        :param sml `'pandas.Series'`: the smiles to search and mark matches
        :return generator int: one of marked atom id
        """
        # Filter substructure matches based on connectivity. The connecting
        # atoms usually have different connectivities. For example, first C
        # in 'CC(=O)O' fragment terminates while the second 'C' in 'CCC(=O)O'
        # molecule is connected to two carbons. Mark the first C in 'CC(=O)O'
        # fragment as None so that molecule won't type this terminating atom.
        deg = [self.mol.GetAtomWithIdx(x).GetDegree() for x in match]
        ids = [[x, y] for x, y, z in zip(match, sml.mp, deg == sml.deg) if z]
        for atom_id, type_id in ids:
            atom = self.mol.GetAtomWithIdx(atom_id)
            if atom.HasProp(self.TYPE_ID):
                # This atom has been marked, skip
                continue
            # Mark the atom and it's hydrogen neighbors
            h_nbrs = [
                x for x in atom.GetNeighbors()
                if x.GetSymbol() == symbols.HYDROGEN
            ]
            for idx, atom in enumerate([atom] + h_nbrs):
                # Neighboring hydrogen when idx != 0
                aid = atom.GetIdx() if idx else atom_id
                tid = sml.hs[type_id] if idx else type_id.item()
                # TYPE_ID defines vdw and charge
                atom.SetIntProp(self.TYPE_ID, tid)
                atom.SetIntProp(self.RES_NUM, self.res_num)
                yield aid
                logger.debug(
                    f"{atom.GetSymbol()}{atom.GetDegree()} {aid} {tid}")

    def setResNum(self):
        """
        Set the residue number based on the fragments (SMILES match results).
        """
        logger.debug(f"{self.res_num} residues found.")
        bonds = self.mol.GetBonds()
        bonded = [[x.GetBeginAtom(), x.GetEndAtom()] for x in bonds]
        res_num = [[y.GetIntProp(self.RES_NUM) for y in x] for x in bonded]
        cbonds = [x for x, y in zip(bonded, res_num) if y[0] != y[1]]
        emol = Chem.EditableMol(Chem.Mol(self.mol))
        for batom, eatom in cbonds:
            emol.RemoveBond(batom.GetIdx(), eatom.GetIdx())
        frags = Chem.GetMolFrags(emol.GetMol())
        for idx, aids in enumerate(Chem.GetMolFrags(emol.GetMol())):
            for aid in aids:
                self.mol.GetAtomWithIdx(aid).SetIntProp(self.RES_NUM, idx)
        logger.debug(f"{len(frags)} residues reassigned.")
