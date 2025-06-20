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

STAR = symbols.STAR
REGULAR = 'regular'
TERMINATOR = 'terminator'
INITIATOR = 'initiator'
MONOMER = 'monomer'


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

    def getCapping(self, role_id=TAIL_ID):
        """
        Get the capping atoms.

        :param role_id int: the role id of the capping atoms.
        :return list of rdkit.Chem.rdchem.Atom: the capping atoms.
        :raises MoietyError: the wildcard caps > 1 atoms.
        """
        if not all([len(x.GetNeighbors()) == 1 for x in self.stars]):
            raise MoietyError(f"{STAR} of {self.smiles} doesn't cap one atom.")
        return [x for x in self.stars if x.GetAtomMapNum() == role_id]

    @functools.cached_property
    def stars(self):
        """
        Get the atoms of the star symbol.

        :return list of rdkit.Chem.rdchem.Atom: the star atoms.
        """
        return [x for x in self.GetAtoms() if x.GetSymbol() == STAR]

    @functools.cached_property
    def role(self, roles=(REGULAR, TERMINATOR, MONOMER)):
        """
        Get the role.

        :param roles tuple: the moiety role.
        :raise MoietyError: unsupported roles.
        :return str: the role.
        """
        num = len(self.stars)
        if num and num == len(self.getCapping()):
            # The map num of '*' is explicitly set.
            return INITIATOR
        try:
            return roles[num]
        except IndexError:
            # FIXME: branching units
            raise MoietyError(f"{self.smiles} contains > 2 {STAR}.")


class Mol(structure.Mol):
    """
    Class to validate the moieties.
    """

    def __init__(self, *args, allow_reg=True, **kwargs):
        """
        :param allow_reg bool: allow regular molecule beyond repeat units.
        """
        super().__init__(*args, **kwargs)
        self.allow_reg = allow_reg
        self.moieties = collections.defaultdict(list)

    def run(self):
        """
        Main method to run.
        """
        self.setMoieties()
        self.checkReg()
        self.setInitiator()
        self.setTerminator()
        self.setMonomer()

    def setMoieties(self):
        """
        Set the molecule fragments in the cru with their roles.
        """
        for frag in self.GetMolFrags(asMols=True):
            moiety = Moiety(frag)
            self.moieties[moiety.role].append(moiety)

    def checkReg(self):
        """
        Check the regular molecules.

        :raise MoietyError: regular molecules are not allowed or mixed with cru.
        """
        regulars = self.moieties[REGULAR]
        if not regulars:
            return
        if not self.allow_reg:
            raise MoietyError(f"{regulars[0].smiles} doesn't contain {STAR}.")
        if len(self.moieties) > 1:
            raise MoietyError(f"{self.smiles} mixes regular with cru.")

    def setInitiator(self):
        """
        Set the initiator.

        :raise MoietyError: multiple initiators are found.
        """
        initiators = self.moieties[INITIATOR]
        match len(initiators):
            case 0:
                if len(self.moieties[TERMINATOR]) < 2:
                    return
                # Marked terminators use HEAD ID (0) as the map num
                initiator = sorted(
                    self.moieties[TERMINATOR],
                    key=lambda x: x.stars[0].GetAtomMapNum())[-1]
                self.moieties[TERMINATOR].remove(initiator)
                initiator.stars[0].SetAtomMapNum(Moiety.TAIL_ID)
                self.moieties[INITIATOR] = [initiator]
            case 1:
                return
            case _:
                smiles = ' '.join([x.smiles for x in initiators])
                raise MoietyError(f"Multiple initiators found: {smiles}.")

    def setTerminator(self):
        """
        Set the terminator.

        :raise MoietyError: Multiple terminators are found.
        """
        terminators = self.moieties[TERMINATOR]
        match len(terminators):
            case 0:
                return
            case 1:
                terminators[0].stars[0].SetAtomMapNum(Moiety.HEAD_ID)
            case _:
                smiles = ' '.join([x.smiles for x in terminators])
                raise MoietyError(f"Multiple terminators found: {smiles}.")

    def setMonomer(self):
        """
        Set the monomer.

        :raise MoietyError: The monomer does not have a head marked.
        """
        if not self.moieties:
            raise MoietyError(f'No moieties found.')
        for mol in self.moieties[MONOMER]:
            map_nums = [x.GetAtomMapNum() for x in mol.stars]
            if Moiety.HEAD_ID not in map_nums:
                raise MoietyError(f"Head atom not found in {mol.smiles}.")
            if Moiety.TAIL_ID not in map_nums:
                head = map_nums.index(Moiety.HEAD_ID)
                tail = next(x for i, x in enumerate(mol.stars) if i != head)
                tail.SetAtomMapNum(Moiety.TAIL_ID)

    def getSmiles(self, canonize=True):
        """
        Get the SMILES string out of the moieties. (map num updated)

        :param canonize bool: whether to canonize the SMILES string.
        :return str: the SMILES string of the moieties.
        """
        mols = [y for x in self.moieties.values() for y in x]
        smiles = Chem.MolToSmiles(functools.reduce(Chem.CombineMols, mols))
        return Chem.CanonSmiles(smiles) if canonize else smiles
