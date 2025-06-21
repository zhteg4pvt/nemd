# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Test configuration, such as global testing fixtures...
"""
import collections
import contextlib
import os
import random
import shutil
from unittest import mock

import flow
import pytest

from nemd import envutils
from nemd import frame
from nemd import jobutils
from nemd import np
from nemd import oplsua
from nemd import osutils
from nemd import pytestutils
from nemd import structure


@contextlib.contextmanager
def lines(filename='filename'):
    """
    Get the file handler of a temporary file and read the read lines on existing.

    :param filename str: temporary filename
    :return '_io.TextIOWrapper', list: the file handler, the written lines.
    """
    lines = []
    try:
        with open(filename, 'w') as wfh:
            yield wfh, lines
    finally:
        with open(filename, 'r') as rfh:
            lines += [x.rstrip('\n') for x in rfh.readlines()]


@pytest.fixture
def tmp_dir(tmpdir):
    """
    Create a temporary directory and change to it for the duration of the test.

    :param tmpdir '_pytest._py.path.LocalPath': The temporary directory factory.
    :return tmpdir '_pytest._py.path.LocalPath': The temporary directory.
    """
    with osutils.chdir(tmpdir, rmtree=True):
        yield tmpdir


@pytest.fixture
def tmp_line(tmp_dir):
    """
    Get the handler to write and written lines.

    :return 'function': The function to get handler to write and written lines.
    """
    return lines


@pytest.fixture
def env(ekey, evalue):
    """
    Temporarily set environment.

    :param ekey str: The environmental keyword.
    :param evalue str: the environmental value.
    :return environ dict: the environment.
    """
    environ = {} if evalue is None else {ekey: evalue}
    with mock.patch.dict('os.environ', environ, clear=True):
        yield environ


@pytest.fixture
def raises(expected):
    """
    Return function to open context management for the exception assertion.

    :param request '_pytest.fixtures.SubRequest': The requested information.
    :param expected 'type' or any: The raised exception class (e.g. ValueError),
        or the expected return instance
    :return 'ContextManager': context manager to assert the raise
    """
    return pytestutils.Raises.ctxmgr(expected)


@pytest.fixture
def flow_opr():
    """
    Yield FlowProject with restorable _OPERATION_*.

    :return `flow.FlowProject`: the patched flow.FlowProject
    """
    project = flow.FlowProject
    functions = project._OPERATION_FUNCTIONS
    preconditions = project._OPERATION_PRECONDITIONS
    postconditions = project._OPERATION_POSTCONDITIONS
    project._OPERATION_FUNCTIONS = []
    project._OPERATION_PRECONDITIONS = collections.defaultdict(list)
    project._OPERATION_POSTCONDITIONS = collections.defaultdict(list)
    yield project
    project._OPERATION_FUNCTIONS = functions
    project._OPERATION_PRECONDITIONS = preconditions
    project._OPERATION_POSTCONDITIONS = postconditions


@pytest.fixture
def copied(dirname, tmp_dir):
    """
    Locate and copy an itest.

    :return `str`: the dirname to the original itest
    """
    if dirname is None:
        return
    test_dir = envutils.test_data(dirname)
    shutil.copytree(test_dir, os.curdir, dirs_exist_ok=True)
    return test_dir


@pytest.fixture
def jobs(copied):
    """
    Return the signac jobs of a copied itest.

    :return `list`: signac jobs
    """
    try:
        project = flow.project.FlowProject.get_project(os.curdir)
    except LookupError:
        return []
    else:
        return list(project.find_jobs())


@pytest.fixture
def job(jobname, copied):
    """
    Return the job of a copied itest.

    :return `jobutils.Job`: Job loaded from a job json file
    """
    return jobutils.Job(jobname)


@pytest.fixture
def frm(file):
    """
    Return a trajectory frame.

    :param file str: the trajectory file
    :return `frame.Frame`: loaded frame
    """
    with open(file, 'r') as fh:
        return frame.Frame.read(fh)


@pytest.fixture
def smol(mol, cnum, seed):
    """
    Return a molecule of conformers from random seed.

    :param mol `Mol`: the input molecule.
    :param cnum `int`: the number of conformers.
    :param seed int: the random seed for the embedding.
    :return `structure.Mol`: the molecule
    """
    mol.EmbedMultipleConfs(numConfs=cnum, randomSeed=seed)
    return mol


@pytest.fixture
def emol(mol, cnum):
    """
    Return a molecule with conformers.

    :param mol `Mol`: the input molecule.
    :param cnum `int`: the number of conformers.
    :return `structure.Mol`: the molecule
    """
    mol.EmbedMultipleConfs(numConfs=cnum)
    return mol


@pytest.fixture
def mol(smiles):
    """
    Return a molecule.

    :param smiles str: the input smiles
    :return `structure.Mol`: the molecule
    """
    if smiles is not None:
        return structure.Mol.MolFromSmiles(smiles)


@pytest.fixture
def tmol(mol, ff=oplsua.Parser.get()):
    """
    Return a molecule.

    :param mol `Mol`: the input molecule.
    :param ff `oplsua.Parser`: the force field parser.
    :return `structure.Mol`: the molecule
    """
    ff.type(mol)
    return mol


@pytest.fixture
def mols(smiles, cnum, seed):
    """
    Return molecules of conformers from random seed.

    :param smiles str: the input smiles
    :param cnum `int`: the conformer number of each molecule.
    :return list: the molecules
    """
    mols = [structure.Mol.MolFromSmiles(x) for x in smiles]
    for mol in mols:
        mol.EmbedMultipleConfs(cnum, randomSeed=seed)
    return mols


@pytest.fixture
def random_seed(seed):
    """
    Set the random state using the seed.

    :param seed `int`: the random seed.
    """
    np.random.seed(seed)
    random.seed(seed)


@pytest.fixture
def called(expected):
    """
    Assert function called with the expected.

    :param expected str: expected called with.
    :return `mock.Mock`: the mocked function.
    """
    mocked = mock.Mock()
    yield mocked
    if expected:
        mocked.assert_called_with(expected)
    else:
        mocked.assert_not_called()