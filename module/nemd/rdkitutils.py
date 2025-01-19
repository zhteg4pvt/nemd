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
from rdkit import RDLogger
from rdkit import rdBase

from nemd import logutils


@contextmanager
def rdkit_preserve_hs():
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    try:
        yield ps
    finally:
        ps.removeHs = True


@contextmanager
def ignore_warnings():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)
    try:
        yield lg
    finally:
        lg.setLevel(RDLogger.WARNING)


def MolFromSmiles(*args, **kwargs):
    with capture_logging() as logs:
        mol = Chem.MolFromSmiles(*args, **kwargs)
        if mol is None:
            msg = "\n".join(f"{x}: {y}" for x, y in logs.items())
            raise ValueError(msg)
        return mol


@contextmanager
def capture_logging():
    """
    Capture logging messages from rdkit and return them as a dict.

    :return dict: keys are log levels (e.g. 'WARNING', 'ERROR'), and values are
        logging messages.
    """
    with open(os.devnull, 'w') as devnull:
        stream = rdkit.log_handler.setStream(devnull)
        hdlr = logutils.Handler()
        rdkit.logger.addHandler(hdlr)
        rdBase.LogToPythonLogger()
        try:
            yield hdlr.logs
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
