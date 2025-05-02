# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module reads, parses and assigns opls-ua parameters.
"""
import collections
import functools
import os

import methodtools
import numpy as np
import pandas as pd
import scipy
from rdkit import Chem

from nemd import builtinsutils
from nemd import envutils
from nemd import logutils
from nemd import rdkitutils
from nemd import symbols

logger = logutils.Logger.get(__file__)

IDX = 'idx'
TYPE_ID = symbols.TYPE_ID
OPLSUA = symbols.OPLSUA.lower()
DIRNAME = envutils.get_data('ff', OPLSUA)


class Base(pd.DataFrame, builtinsutils.Object):
    """
    Base class of oplsua assignment.
    """

    @classmethod
    @property
    def parquet(cls):
        """
        Return the path of the parquet file.

        :return str: the path of the parquet file.
        """
        return os.path.join(DIRNAME, f"{cls.name}.parquet")

    @classmethod
    @property
    def npy(cls):
        """
        Return the pathname of the npy file.

        :return str: the npy pathname for mapping.
        """
        return os.path.join(DIRNAME, f"{cls.name}.npy")


class Charge(Base):
    """
    The class to hold charge information.
    """

    def __init__(self, **kwargs):
        """
        See parent.
        """
        super().__init__(pd.read_parquet(self.parquet), **kwargs)


class Smiles(Charge):
    """
    The class to hold smiles.
    """

    def __init__(self, **kwargs):
        """
        See parent.
        """
        super().__init__(**kwargs)
        self.hs = self.hs.apply(eval)


class Vdw(Charge):
    """
    The class to hold VDW information.
    """


class Atom(Charge):
    """
    The class to hold atom information.
    """
    HYDROGEN = symbols.HYDROGEN

    @methodtools.lru_cache()
    @property
    def atomic_number(self):
        """
        Return the atomic number of each atom.

        :return `np.ndarray`: the atomic number of each atom.
        """
        return self.Z.values

    @methodtools.lru_cache()
    @property
    def connectivity(self):
        """
        Return the connectivity of each atom.

        :return `np.ndarray`: the connectivity of each atom.
        """
        return self.conn.values


class BondIndex(np.ndarray):
    """
    This class maps the row values (value1, value2) to the corresponding index
    of an array (index). The mapping is done by sparse matrix, which is fast and
    memory-efficient.

    1) locate the row index by value1, value2
    2) return the row indices for rows containing any of the given values.

    for example:

    data = [[index_1, value1_1, value2_1],
            [index_2, value1_2, value2_2],
            [index_3, value1_3, value2_2],
            ...]

    Map(data).index(value1_2, value2_2) --> index_2
    Map(data).indexes(value1_2, value2_2) --> [index_2, index_3]
    """
    START = 0
    END = -1

    def __new__(cls, data):
        """
        :param data `np.ndarray`: [index, value1, value2, value3, value4] as row

        :return `Dihedral`: the map from values to the row index.
        """
        obj = np.array(data).transpose().view(cls)
        return obj

    def index(self, row):
        """
        Return the matched index given the row values.

        :return int: the matched index.
        """
        if row[self.START] > row[self.END]:
            row = row[::-1]
        block = self.getBlock(row[1:-1])
        try:
            index = block[row[0], row[-1]]
        except IndexError:
            # when index exceeds the range
            return
        if index != 0 or row == self.zero:
            return index

    @methodtools.lru_cache()
    def getBlock(self, *args):
        """
        Get the sparse block which maps the given values to the row index.

        :return 'csr_matrix': the sparse matrix from values to the row index.
        """
        return self.getCsr(self)

    def getCsr(self, dat, dtype=np.int16):
        """
        Convert the data to a sparse matrix.

        :param dat 3xN `np.ndarray`: [index, value1, value2] as each row.
        :return 'csr_matrix': the sparse matrix from values to the row index.
        """
        return scipy.sparse.coo_matrix((dat[0], dat[1:]), dtype=dtype).tocsr()

    @methodtools.lru_cache()
    @property
    def zero(self):
        """
        Return the zero row values.

        :return tuple: the row values that correspond to the zero index.
        """
        index = np.where(self[0] == 0)[0]
        return tuple(self[1:, index].transpose().flatten())

    @methodtools.lru_cache()
    def getFlipped(self, row):
        indexes, head_tail = self.getPartial(row)
        return self.flipped(indexes, head_tail)

    def getPartial(self, row):
        """
        Get the indexes of rows whose values match any of the given values.

        :param row tuple: match rows containing any of the values.
        :return 'np.ndarray': the indexes of the matching rows.
        """
        block = self.getBlock()
        indexes = (block[row, :].data, block[:, row].data)
        indexes = np.sort(np.concatenate(indexes))
        return indexes, self.head_tail[indexes]

    @methodtools.lru_cache()
    @property
    def head_tail(self):
        return self[[1, -1]].transpose()

    def flipped(self, indexes, head_tail):
        indexes = np.repeat(indexes, repeats=2, axis=0)
        head_tail = np.repeat(head_tail, repeats=2, axis=0)
        head_tail[1::2, :] = np.fliplr(head_tail[1::2, :])
        return indexes, head_tail


class AngleIndex(BondIndex):
    """
    This class maps the row values (value1, value2, value3) to the corresponding
    index of an array (index). The mapping is done by sparse matrix and
    dictionary lookup, which is fast and memory-efficient.

    1) locate the row range by value2
    2) match the value1 and value3 within the range

    for example:

    data = [[index_1, value1_1, value2_1, value3_1],
            [index_2, value1_2, value2_2, value3_2],
            [index_3, value1_3, value2_2, value3_3],
            ...]

    Map(data).index(value1_2, value2_2, value3_2) --> index_2
    Map(data).indexes(value1_3, value2_2, value3_4) --> [index_2, index_3]
    """

    @methodtools.lru_cache()
    def getBlock(self, key):
        """
        Get the sparse block which maps the given values to the row index.

        :param key tuple of int: the value in the middle of the row.
        :return 'csr_array': a matrix mapping atom 1d1 and 1d4 to dihedral id.
        """
        matches = np.where(self[2] == key[0])[0]
        return self.getCsr(self[[0, 1, 3], matches[0]:matches[-1] + 1])

    @methodtools.lru_cache()
    def getPartial(self, row):
        """
        Get the indexes of rows with the given two middle values matched.

        :param row list of ints: the row values whose middle value to match.
        :return 'ndarray': the indexes of the matching rows.
        :raise IndexError: when block is not found.
        """
        blk = self.getBlock(row[1:-1])
        return blk.data, self.head_tail[blk.data]


class DihedralIndex(AngleIndex):
    """
    This class maps the row values (value1, value2, value3, value4) to the
    corresponding index of an array (index). The mapping is done by sparse
    matrix and dictionary lookup, which is fast and memory-efficient.

    1) locate the row range by value2, value3
    2) match the value1 and value4 within the range

    for example:

    data = [[index_1, value1_1, value2_1, value3_1, value4_1],
            [index_2, value1_2, value2_2, value3_2, value4_2],
            [index_3, value1_3, value2_2, value3_3, value4_2],
            ...]

    Map(data).index(value1_2, value2_2, value3_2, value4_2) --> index_2
    Map(data).indexes(value1_2, value2_2) --> [index_2, index_3]
    """
    START = 1
    END = 2

    @methodtools.lru_cache()
    def getBlock(self, key):
        """
        Get the sparse block.

        :param key tuple of two ints: the two values in the middle of a row.
        :return 'csr_array': a matrix mapping atom 1d1 and 1d4 to the indexes.
        """
        start = self.getMap()[key]
        end = self.getMap(shift=-1)[key] + 1
        return self.getCsr(self[[0, 1, 4], start:end])

    @methodtools.lru_cache()
    def getMap(self, shift=1):
        """
        Return the mapping from values to the row start and end indices.

        :param shift int: shift the array by the given value.
        :return 'csr_array': a sparse matrix mapping ids to the limits.
        """
        coords = self[[2, 3]]  # value2, value3
        selected = [np.roll(x, shift=shift) != x for x in coords]
        sel = np.array(selected).any(axis=0)
        return self.getCsr(self[[0, 2, 3]][:, sel])

    @methodtools.lru_cache()
    def getFlipped(self, row):
        """
        Get the indexes of rows with the given two middle values matched.

        :param row list of ints: the row values whose middle value to match.
        :return 'ndarray': the indexes of the matching rows.
        :raise IndexError: when block is not found.
        """
        flipped = row[self.START] > row[self.END]
        indexes, head_tail = self.getPartial(row[::-1] if flipped else row)
        if flipped:
            return indexes, np.fliplr(head_tail)
        if row[self.START] == row[self.END]:
            return self.flipped(indexes, head_tail)
        return indexes, head_tail


class Bond(Charge):
    """
    The class to hold Bond information.
    """
    ID_COLS = ['id1', 'id2']
    INDEX_CLASS = BondIndex
    # https://pandas.pydata.org/docs/development/extending.html
    _metadata = ['atoms']

    def __init__(self, *args, atoms=None, **kwargs):
        """
        :param atoms `Atom`: the atom information.
        """
        super().__init__(*args, **kwargs)
        self.atoms = atoms

    def getMatched(self, atoms):
        """
        Get force field matched bonds. The searching and approximation follows:

        1) Forced type mapping for the connections.
        2) Exact match.
        3) End atom matching based on symbol and connectivity

        :param atoms list: bonded atoms.
        :return int: the index of the first match
        """
        tids = self.getTypes(atoms)
        idx = self.row.index(tids)
        if idx is not None:
            return idx
        return self.getPartial(tids, atoms)

    def getTypes(self, atoms):
        """
        Get the type ids of the atoms.

        :param atoms `list` of `rdkit.Chem.rdchem.Atom`: the atoms to type
        :return tuple of int: the atom type ids.
        """
        # FIXME: more standard and general algorithm to match and appropriate
        #   the connecting bonds, angles, and dihedrals
        # TYPE_ID takes care of specific charge and vdw parameters (specific).
        tids = (x.GetIntProp(TYPE_ID) for x in atoms)
        # Forced type mapping for each atom.
        # Per-atom mapping generates general types (force field moieties).
        tids = tuple(self.maps[0].get(x, x) for x in tids)
        # Forced type mapping for connecting atoms.
        # e.g., CH2-COOH --> alpha-COOH; HO-C=O --> C-OH (Tyr)
        # Manually connects atoms between force field moieties.
        return self.getCtype(tids)

    @methodtools.lru_cache()
    @property
    def maps(self):
        """
        Return the per-atom type mapping and connection-atom mapping.

        :return dict, dict: the mapping for atom types.
        """
        with open(self.npy, 'rb') as fh:
            tmap = {x: y for x, y in np.load(fh)}
            cmap = {tuple(x): tuple(y) for x, y in zip(*np.load(fh))}
            cmap.update({x[::-1]: y[::-1] for x, y in cmap.items()})
        return tmap, cmap

    def getCtype(self, tids):
        """
        Get the type ids of the connecting atoms.

        :param tids tuple: the atom type ids to map from.
        :return tuple: the mapped type ids.
        """
        try:
            return self.maps[1][tids]
        except KeyError:
            return tids

    @methodtools.lru_cache()
    @property
    def row(self):
        """
        The mapping from a row of atom type ids to the index of the exact match,
        or the indexes and head-tail of all partial matches.

        :return `nemd.numpyutils.Bond` (sub-)class: the mapping object.
        """
        return self.INDEX_CLASS(self[self.ID_COLS].reset_index().values)

    def getPartial(self, tids, atoms):
        """
        Return the partial matches based on the symbols and connectivities.

        :param tids list of int: the type ids of the atoms
        :param atoms 'Chem.rdchem.Atom' list: bond, angle, or dihedral atoms

        :return int: the index of the first match
        :raise IndexError: failed to find any matches.
        """
        logger.debug(f"No exact match for {self.name} between atom {tids}.")
        try:
            indexes, head_tail = self.row.getFlipped(tids)
        except IndexError:
            raise IndexError(f"No partial match for {self.name} ({tids}).")
        # Check atomic number
        atomic = self.atoms.atomic_number[head_tail]
        atom_zs = [atoms[0].GetAtomicNum(), atoms[-1].GetAtomicNum()]
        z_matches = (atomic == atom_zs).all(axis=1)
        # Check connectivity
        conns = self.atoms.connectivity[head_tail]
        atom_conns = [self.getConn(atoms[0]), self.getConn(atoms[-1])]
        con_matches = (conns == atom_conns).all(axis=1)
        found = z_matches & con_matches
        if not found.any():
            found = z_matches
        try:
            index = np.nonzero(found)[0][0]
        except IndexError:
            raise IndexError(f"No params for {self.name} between atom {tids}.")
        asked = [f"{x}_{y}" for x, y in zip(atom_zs, atom_conns)]
        found = [f"{x}_{y}" for x, y in zip(atomic[index], conns[index])]
        logger.debug(f"{asked} replaced by {found}")
        return indexes[index]

    @staticmethod
    def getConn(atom):
        """
        Get the atomic connectivity information.

        :param atom 'Chem.rdchem.Atom': the connectivity of this atom.
        :return int: the number of bonds including the implicit hydrogen.
        """
        try:
            implicit_h_num = atom.GetIntProp(symbols.IMPLICIT_H)
        except KeyError:
            implicit_h_num = atom.GetNumImplicitHs()
        return implicit_h_num + atom.GetDegree()

    @methodtools.lru_cache()
    @property
    def has_h(self):
        """
        Return an array indicating the hydrogen presence for each row.

        :return `np.ndarray`: whether each bond has hydrogen atoms.
        """
        return (self.atoms.atomic_number[self.ids] == 1).any(axis=1)

    @methodtools.lru_cache()
    @property
    def ids(self):
        """
        Return the atom ids

        :return 'np.ndarray': the atom ids.
        """
        return self[self.ID_COLS].values


class Angle(Bond):
    """
    The class to hold Angle information.
    """
    ID_COLS = ['id1', 'id2', 'id3']
    INDEX_CLASS = AngleIndex


class Dihedral(Angle):
    """
    The class to hold Dihedral information.
    """
    ID_COLS = ['id1', 'id2', 'id3', 'id4']
    INDEX_CLASS = DihedralIndex

    def getCtype(self, tids):
        """
        Get the type ids of the connecting atoms.

        :param tids tuple: the atom type ids to map from.
        :return tuple: the mapped type ids.
        """
        mids = super().getCtype(tids[1:-1])
        return tuple((tids[0], *mids, tids[-1]))


class Improper(Bond):
    """
    The class to hold improper information.
    """
    ID_COLS = ['id1', 'id2', 'id3', 'id4']
    ATOMIC_NUMBERS = [6, 1, 7, 8]

    def getMatched(self, atoms):
        """
        Get force field matched improper by counting the symbols.

        :param atoms list: bonded atoms.
        :return int: the index of the match.
        """
        conn_atomic = [self.getConn(atoms[2])]
        conn_atomic += [x.GetAtomicNum() for x in atoms]
        hashed = self.hash(conn_atomic)
        return self.index_map[hashed]

    @classmethod
    def hash(cls, conn_atomic):
        """
        Hash improper cluster information.

        :param conn_atomic list: The first is the center atom connectivity
            (implicit hydrogen included) following by the atomic numbers of
            four atoms. (the third one is the center)
        """
        counted = collections.Counter(conn_atomic[1:])
        count = [counted.get(x, 0) for x in cls.ATOMIC_NUMBERS]
        # the center atom's atomic number and connectivity followed by the count
        # e.g., [3, 6, 7, 6, 8] --> 6 3 2 0 1 1
        return tuple([conn_atomic[3], conn_atomic[0], *count])

    @methodtools.lru_cache()
    @property
    def index_map(self):
        """
        Return the mapping from a hashed improper to type id.

        NOTE: mapping may be only good for this specific force field (even this
            specific file only).

        :return dict: the mapping from a hashed improper to type id.
        """
        # neighbors of CC(=O)C and CC(O)C have the same symbols
        # The third one is the center ('Improper Torsional Parameters' in prm)
        conns = self.atoms.connectivity[self.id3]
        atomic = self.atoms.atomic_number[self.ids]
        conn_atomic = np.concatenate((conns.reshape(-1, 1), atomic), axis=1)
        unique, indexes = np.unique(conn_atomic, axis=0, return_index=True)
        counted = [self.hash(x) for x in unique]
        return {x: y for x, y in zip(counted, indexes)}


class Parser:
    """
    Parsed force field information, mapping details, and atomic typer.
    """

    def __init__(self, wmodel=symbols.TIP3P):
        """
        :param wmodel str: the model type for water
        """
        self.wmodel = wmodel
        self.typer = Typer(wmodel=self.wmodel)

    def type(self, mol):
        """
        Type the molecule by the force field typer.

        :param mol 'Chem.rdchem.Mol': the molecule to type.
        """
        self.typer.type(mol)

    @property
    @functools.cache
    def atoms(self):
        """
        Set atom types based on the 'Atom Type Definitions' block.

        :return `Atom`: the atom information.
        """
        return Atom()

    @property
    @functools.cache
    def vdws(self):
        """
        Set vdw parameters based on 'Van der Waals Parameters' block.

        :return `Vdw`: the vdw information.
        """
        return Vdw()

    @property
    @functools.cache
    def charges(self):
        """
        Set charges based on 'Atomic Partial Charge Parameters' block.

        :return `Charge`: the charge information.
        """
        return Charge()

    @property
    @functools.cache
    def bonds(self):
        """
        Set bond parameters based on 'Bond Stretching Parameters' block.

        :return `Bond`: the bond information.
        """
        return Bond(atoms=self.atoms)

    @property
    @functools.cache
    def angles(self):
        """
        Set angle parameters based on 'Angle Bending Parameters' block.

        :return `Angle`: the angle information.
        """
        return Angle(atoms=self.atoms)

    @property
    @functools.cache
    def impropers(self):
        """
        Set improper parameters based on 'Improper Torsional Parameters' block.

        :return `Improper`: the improper information.
        """
        return Improper(atoms=self.atoms)

    @property
    @functools.cache
    def dihedrals(self):
        """
        Set dihedral parameters based on 'Torsional Parameters' block.

        :return `Dihedral`: the dihedral information.
        """
        return Dihedral(atoms=self.atoms)

    def molecular_weight(self, mol):
        """
        The molecular weight of one molecule.

        :param mol Chem.rdchem.Mol: the input molecule.
        :return float: the total molecular weight.
        """
        tids = [x.GetIntProp(TYPE_ID) for x in mol.GetAtoms()]
        return round(sum(self.atoms.mass.values[tids]), 4)

    @classmethod
    @functools.cache
    def get(cls, wmodel=symbols.TIP3P):
        """
        Read and parser opls force field file.

        :param wmodel str: the model type for water
        :return 'OplsParser': the parser with force field information
        """
        return cls(wmodel=wmodel)


class Typer:
    """
    Type the atoms and map SMILES fragments.
    """
    RES_NUM = symbols.RES_NUM

    def __init__(self, wmodel=symbols.TIP3P):
        """
        :param wmodel str: the model type for water
        """
        self.wmodel = wmodel
        self.mol = None
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
        self.mx = min([symbols.MAX_INT32, self.mol.GetNumAtoms()])

    def doTyping(self):
        """
        Match the substructure with SMILES and assign atom type.

        :raise KeyError: if any atoms are not marked.
        """
        unmarked, res_num = self.mol.GetNumAtoms(), 0
        for idx, sml in self.smiles.iterrows():
            if unmarked <= 0:
                break
            matches = self.mol.GetSubstructMatches(sml.mol, maxMatches=self.mx)
            if not matches:
                continue
            logger.debug(f"assignAtomType {sml.sml}, {matches}")
            marks = []
            for match in matches:
                marked = list(self.mark(match, sml, res_num))
                if not marked:
                    continue
                unmarked -= len(marked)
                res_num += 1
                marks.append(marked)
            cnt = collections.Counter([len(x) for x in marks])
            counted = ','.join([f'{x}*{y}' for x, y in cnt.items()])
            logger.debug(f"{sml.sml}: {len(marks)} matches ({counted})")
        logger.debug(f"{unmarked} / {self.mol.GetNumAtoms()} unmarked")
        logger.debug(f"{res_num} residues found.")
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
        sml = Smiles()
        water = sml[sml.sml == 'O']
        to_drop = water.dsc != symbols.WATER_DSC.format(model=self.wmodel)
        sml.drop(index=water[to_drop].index, inplace=True)
        sml['mol'] = [rdkitutils.MolFromSmiles(x) for x in sml.sml]
        return sml

    def mark(self, match, sml, res_num):
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
        deg = [
            self.mol.GetAtomWithIdx(x).GetDegree() == y
            for x, y in zip(match, sml.deg)
        ]
        ids = [[x, y] for x, y, z in zip(match, sml.mp, deg) if z]
        for atom_id, type_id in ids:
            atom = self.mol.GetAtomWithIdx(atom_id)
            if atom.HasProp(TYPE_ID):
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
                tid = sml.hs[type_id] if idx else type_id
                # TYPE_ID defines vdw and charge
                atom.SetIntProp(TYPE_ID, tid)
                atom.SetIntProp(self.RES_NUM, res_num)
                yield aid
                logger.debug(
                    f"{atom.GetSymbol()}{atom.GetDegree()} {aid} {tid}")

    def setResNum(self):
        """
        Set the residue number based on the fragments (SMILES match results).
        """
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
