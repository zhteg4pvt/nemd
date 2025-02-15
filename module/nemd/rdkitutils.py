# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
rdkit utilities.
"""
import os
from contextlib import contextmanager

import rdkit
from rdkit import Chem
from rdkit import rdBase

from nemd import logutils


def MolFromSmiles(*args, **kwargs):
    """
    Construct molecule from a SMILES, and raise errors instead of printing to
    the script on failure. In addition, this module can be lazily imported.

    :return `rdkit.Chem.rdchem.Mol`: the constructed molecule.
    :raise ValueError: if no molecules can be constructed.
    """
    with capture_logging() as logs:
        mol = Chem.MolFromSmiles(*args, **kwargs)
        if mol is None:
            raise ValueError("\n".join(f"{x}: {y}" for x, y in logs.items()))
        return mol


@contextmanager
def capture_logging(logger=None):
    """
    Capture the rdkit logging messages. The captured messages are either printed
    to the logger or return as a dictionary.

    :param logger 'logging.Logger': the logger to print the messages.
    :return dict: keys are log levels (e.g. 'WARNING', 'ERROR'), and values are
        logging messages.
    """
    with open(os.devnull, 'w') as devnull:
        stream = rdkit.log_handler.setStream(devnull)
        has_handler = logger and logger.hasHandlers()
        hdlr = logger.handlers[0] if has_handler else logutils.Handler()
        rdkit.logger.addHandler(hdlr)
        rdBase.LogToPythonLogger()
        try:
            yield None if has_handler else hdlr.logs
        finally:
            rdkit.log_handler.setStream(stream)
            rdkit.logger.removeHandler(hdlr)
            rdBase.LogToCppStreams()


# def get_mol_from_smiles(smiles_str, embeded=True, mol_id=1):
#     with rdkit_preserve_hs() as ps:
#         mol = Chem.MolFromSmiles(smiles_str, ps)
#     if not embeded:
#         return mol
#     with ignore_warnings():
#         AllChem.EmbedMolecule(mol, useRandomCoords=True)
#     mol.GetConformer().SetIntProp(pnames.MOL_ID, mol_id)
#     return mol
#
# @contextmanager
# def rdkit_preserve_hs():
#     ps = Chem.SmilesParserParams()
#     ps.removeHs = False
#     try:
#         yield ps
#     finally:
#         ps.removeHs = True
