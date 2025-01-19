# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module parses, analyzes, and validates constitutional repeat units.
"""
import collections
import functools

from rdkit import Chem

from nemd import structure
from nemd import symbols

WILD_CARD = symbols.WILD_CARD


class MoietyError(ValueError):
    """
    Error raised when a polymer cannot be built out of these moieties.
    """
    pass


class Moiety(structure.Mol):
    """
    Class to hold and validate a moiety.
    """

    HEAD_ID = 0
    TAIL_ID = 1

    @property
    @functools.cache
    def wild_card(self):
        """
        Get the wild card atoms of the molecule.

        :return rdkit.Chem.rdchem.Atom: the wild_card atoms.
        :raises MoietyError: the wildcard caps > 1 atoms.
        """
        atoms = [x for x in self.GetAtoms() if x.GetSymbol() == WILD_CARD]
        return atoms

    def capping(self, role_id=TAIL_ID):
        """
        Get the capping atoms of the molecule according to the capping type.

        :param role_id int: the role id of the capping atoms.
        :return rdkit.Chem.rdchem.Atom: the capping atoms.
        """
        if not all([len(x.GetNeighbors()) == 1 for x in self.wild_card]):
            raise MoietyError(f"{WILD_CARD} caps > 1 atoms. ({self.smiles})")
        return [x for x in self.wild_card if x.GetAtomMapNum() == role_id]

    @property
    @functools.cache
    def smiles(self):
        """
        Get the SMILES string of the molecule.

        :return str: the SMILES string of the molecule
        """
        return Chem.MolToSmiles(self)


class Moieties(collections.UserDict):
    """
    Class to hold moieties and validate roles.
    """

    REGULAR = 'regular'
    TERMINATOR = 'terminator'
    INITIATOR = 'initiator'
    MONOMER = 'monomer'
    ROLES = {0: REGULAR, 1: TERMINATOR, 2: MONOMER}

    def getRole(self, moiety):
        """
        Get the role of the molecule based on the capping count and map num.

        :raise MoietyError: unsupported role.
        :return str: the role of the molecule.
        """
        num = len(moiety.wild_card)
        if num and num == len(moiety.capping()):
            # Marked initiator
            return self.INITIATOR

        try:
            return self.ROLES[num]
        except IndexError:
            # FIXME: branching units
            raise MoietyError(f"{moiety.smiles} contains > 2 {WILD_CARD}.")


class Mol(Moieties):
    """
    Class to hold moieties and validate the capability of building a polymer.
    """

    def __init__(self, mol, allow_reg=True):
        """
        :param cru str: the cru string.
        :param allow_reg bool: allow regular molecule beyond repeat units.
        """
        self.mol = mol
        self.allow_reg = allow_reg
        self.frags = collections.defaultdict(list)

    def run(self):
        """
        Main method to run.
        """
        self.setMoietys()
        self.checkReg()
        self.checkInitiator()
        self.checkTerminator()
        self.checkMonomer()

    def setMoietys(self):
        """
        Set the molecule fragments in the cru with their roles.
        """
        for mol_frag in Chem.GetMolFrags(self.mol, asMols=True):
            moiety = Moiety(mol_frag)
            self.frags[self.getRole(moiety)].append(moiety)

    def checkReg(self):
        """
        Check the regular molecules.

        :raise MoietyError: regular molecules are not allowed or mixed with
            constitutional repeat units.
        """
        regulars = self.frags[self.REGULAR]
        if regulars and len(self.frags) > 1:
            raise MoietyError(
                f"{Chem.MolToSmiles(self.mol)} mixes regular molecules with "
                f"constitutional repeat units.")
        if self.allow_reg:
            return
        raise MoietyError(f"{regulars[0].smiles} doesn't contain {WILD_CARD}")

    def checkInitiator(self):
        """
        Check the initiator. If there is no initiator, try to convert one
        terminator into an initiator.

        :raise MoietyError: multiple initiators are found.
        """
        initiators = self.frags[self.INITIATOR]
        match len(initiators):
            case 0:
                if len(self.frags[self.TERMINATOR]) < 2:
                    return
                # Initiator: single wildcard moiety with the largest AtomMapNum
                func = lambda x: -x.wild_card[0].GetAtomMapNum()
                initiator = sorted(self.frags[self.TERMINATOR], key=func)[0]
                initiator.wild_card[0].SetAtomMapNum(Moiety.TAIL_ID)
                self.frags[self.INITIATOR] = [initiator]
                self.frags[self.TERMINATOR].remove(initiator)
            case 1:
                return
            case _:
                smiles = ' '.join([x.smiles for x in initiators])
                raise ValueError(f"Multiple initiators found in {smiles}.")

    def checkTerminator(self):
        """
        Check the terminator.

        :raise MoietyError: Multiple terminators are found.
        """
        terminators = self.frags[self.TERMINATOR]
        match len(terminators):
            case 0:
                return
            case 1:
                terminators[0].wild_card[0].SetAtomMapNum(Moiety.HEAD_ID)
            case _:
                miles = ' '.join([x.smiles for x in terminators])
                raise MoietyError(f"Multiple terminators found in {miles}.")

    def checkMonomer(self):
        """
        Check the monomer.

        :raise MoietyError: The monomer does not have a head marked.
        """
        for mol in self.frags[self.MONOMER]:
            map_nums = [x.GetAtomMapNum() for x in mol.wild_card]
            if Moiety.HEAD_ID not in map_nums:
                raise MoietyError(f"Head atom not found in {mol.smiles}.")
            if Moiety.TAIL_ID not in map_nums:
                index = map_nums.index(Moiety.HEAD_ID) ^ 1
                mol.wild_card[index].SetAtomMapNum(Moiety.TAIL_ID)

    def getSmiles(self, canonize=True):
        """
        Get the SMILES string out of the fragments. (map num updated)

        :param canonize bool: whether to canonize the SMILES string.
        :return str: the SMILES string of the fragments.
        """
        mols = [y for x in self.frags.values() for y in x]
        smiles = Chem.MolToSmiles(functools.reduce(Chem.CombineMols, mols))
        return Chem.CanonSmiles(smiles) if canonize else smiles
