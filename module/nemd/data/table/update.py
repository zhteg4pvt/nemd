# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This script generates periodic table.
"""
import mendeleev
import pandas as pd
from rdkit import Chem

table = Chem.GetPeriodicTable()
MAX_ATOMIC_NUM = table.GetMaxAtomicNumber()
symbols = [table.GetElementSymbol(x) for x in range(MAX_ATOMIC_NUM + 1)]
atomic_numbers = [x for x in range(MAX_ATOMIC_NUM + 1)]
atomic_weights = [table.GetAtomicWeight(x) for x in range(MAX_ATOMIC_NUM + 1)]
cpk_colors = [None] + [mendeleev.element(x).cpk_color for x in symbols[1:]]
data = dict(atomic_number=atomic_numbers,
            atomic_weight=atomic_weights,
            cpk_color=cpk_colors)
data = pd.DataFrame(data, index=pd.Index(symbols, name='symbol'))
data.to_parquet('table.parquet')
