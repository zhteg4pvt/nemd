import pytest

from nemd import sw


class TestFunc:

    @pytest.mark.parametrize('elements,isabs,expected',
                             [(('Si', ), True, 'Si.sw'),
                              (('Ar', ), False, None),
                              (('Si', ), False, 'Si.sw')])
    def testGetFile(self, elements, isabs, expected, tmp_dir):
        if not isabs and expected:
            with open(expected, 'w'):
                pass
        file = sw.get_file(*elements)
        assert file == expected