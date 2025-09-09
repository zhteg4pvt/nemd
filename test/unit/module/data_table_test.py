import os.path

import pytest

try:
    from nemd.data.table import update
except ImportError:
    pytest.skip('table not available', allow_module_level=True)


@pytest.mark.slow
class TestSmiles:

    @pytest.fixture
    def table(self):
        return update.Table()

    def testWrite(self, table, tmp_dir):
        assert (119, 3) == table.data.shape
        table.write()
        assert os.path.exists('table.parquet')
