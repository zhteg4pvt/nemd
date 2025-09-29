# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Periodic Table.
"""
import pandas as pd

from nemd import envutils

pathname = envutils.Src().get('table', 'table.parquet')
TABLE = pd.read_parquet(pathname, engine="fastparquet")
