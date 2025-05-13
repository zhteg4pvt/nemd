import pytest

from nemd import sw


class TestFunc:

    @pytest.mark.parametrize('elements,expected', [(('Si', ), 'Si.sw'),
                                                   (('Ar', ), None)])
    def testGetFile(self, elements, expected):
        file = sw.get_file(*elements)
        assert file is None if expected is None else file.endswith('.sw')
