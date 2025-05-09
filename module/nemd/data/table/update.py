# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This script generates periodic table.
"""
import mendeleev
import pandas as pd
from rdkit import Chem


class Table:
    """
    Periodic Table.
    """

    def __init__(self):
        self.data = None
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        table = Chem.GetPeriodicTable()
        atomic_numbers = [x for x in range(table.GetMaxAtomicNumber() + 1)]
        symbols = [table.GetElementSymbol(x) for x in atomic_numbers]
        cpk_colors = [mendeleev.element(x).cpk_color for x in symbols[1:]]
        atomic_weights = [table.GetAtomicWeight(x) for x in atomic_numbers]
        data = dict(atomic_number=atomic_numbers,
                    atomic_weight=atomic_weights,
                    cpk_color=[None] + cpk_colors)
        self.data = pd.DataFrame(data, index=pd.Index(symbols, name='symbol'))

    def write(self):
        """
        Write data.
        """
        self.data.to_parquet('table.parquet')


if __name__ == "__main__":
    Table().write()
