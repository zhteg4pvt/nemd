from nemd import table


class TestTable:

    def test_shape(self):
        assert (119, 3) == table.TABLE.shape
