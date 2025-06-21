import pytest

from nemd import pytestutils


class TestFunc:

    @pytest.mark.parametrize('expected', [3.14, ValueError])
    def testCtxmgr(self, expected):
        with pytestutils.Raises.ctxmgr(expected):
            if isinstance(expected, float):
                return
            raise expected


@pytestutils.Raises
class TestRaises:

    @pytest.mark.parametrize('expected', [3.14, ValueError])
    def testNew(self, expected):
        if isinstance(expected, float):
            return
        raise expected