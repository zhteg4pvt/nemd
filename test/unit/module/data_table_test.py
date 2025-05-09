import os.path

import pytest

from nemd.data.table import update


@pytest.mark.slow
class TestSmiles:

    @pytest.fixture
    def table(self):
        return update.Table()

    def testWrite(self, table, tmp_dir):
        assert (119, 3) == table.data.shape
        table.write()
        assert os.path.exists('table.parquet')