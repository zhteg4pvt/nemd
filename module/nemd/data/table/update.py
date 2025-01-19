# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This script generates periodic table.
"""
import mendeleev
import pandas as pd
from rdkit import Chem
from sqlalchemy.orm.session import close_all_sessions

MAX_ATOMIC_NUM = 118
table = Chem.GetPeriodicTable()
symbols = [table.GetElementSymbol(x) for x in range(MAX_ATOMIC_NUM + 1)]
atomic_numbers = [x for x in range(MAX_ATOMIC_NUM + 1)]
atomic_weights = [table.GetAtomicWeight(x) for x in range(MAX_ATOMIC_NUM + 1)]
elements = [mendeleev.element(x) for x in symbols[1:]]
cpk_colors = [None] + [x.cpk_color for x in elements]
pd.DataFrame(
    {
        'atomic_number': atomic_numbers,
        'atomic_weight': atomic_weights,
        'cpk_color': cpk_colors
    },
    index=symbols).to_parquet('table.parquet')
# mendeleev established sqlite3 connections, which throw out ProgrammingError
# SQLite objects created in a thread can only be used in that same thread.
close_all_sessions()
