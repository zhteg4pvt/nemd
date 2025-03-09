# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Customization before importing the nemd module.

1) Set the DEBUG state from the environment variable
2) lazy_import so that modules are materialized only on accessing attributes.
3) Import modules that are frequently used.
"""
import os

DEBUG = bool(os.environ.get('DEBUG'))

if not DEBUG:
    # lazy_import creates faked module objects which are expected to be
    # materialized by accessing their attributes.
    # FIXME: StreamHandler added by basicConfig() in lazy_import/__init__.py
    import logging
    logger = logging.getLogger()
    handlers = logger.handlers[:]
    import lazy_import
    for handler in set(logger.handlers).difference(handlers):
        logger.removeHandler(handler)
        handler.close()
    # FIXME: rdkit.rdBase are not lazy-imported due to `ImportError: xx requires
    #  module rdkit, but it couldn't be loaded.` on "from rdkit import Chem;
    #  Chem.MolFromSmiles"
    # FIXME: rdkitutils are not lazy-imported due to `TypeError: cannot pickle
    #  'module' object` on cloudpickle.dumps(self) in flow/project.py
    # FIXME: sh are not lazy-imported due to `AttributeError: module 'sh' has no
    #  attribute 'grep'` on sh.grep
    modules = [
        'numpy', 'pandas', 'numba', 'networkx', 'rdkit', 'rdkit.Chem', 'scipy',
        'signac', 'flow', 'wurlitzer', 'pkgutil', 'chemparse', 'psutil'
    ]
    names = [
        'constants', 'table', 'box', 'frame', 'dist', 'traj', 'structure',
        'lmpatomic', 'lmpfull', 'structutils', 'oplsua', 'stillinger', 'cru',
        'polymutils', 'xtal', 'lmplog', 'psutils', 'alamode', 'molview'
    ]
    for name in modules + [f"nemd.{x}" for x in names]:
        lazy_import.lazy_module(name)
import numpy as np
import pandas as pd
