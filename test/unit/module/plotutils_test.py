from unittest import mock

import pandas as pd
import pytest

from nemd import plotutils


class TestFunc:

    @pytest.mark.parametrize('inav', [(False), (True)])
    def testGetPyplot(self, inav):
        import matplotlib
        obackend = matplotlib.get_backend()
        with mock.patch('nemd.plotutils.print'):
            with plotutils.get_pyplot(inav=inav) as plt:
                plt.show = mock.Mock()
        assert inav == plt.show.called
        assert obackend == matplotlib.get_backend()
