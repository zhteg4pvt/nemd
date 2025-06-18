import pytest

from nemd import lmplog


class TestThermo:
    MINIMIZE = [
        ['Step', 'Temp', 'E_pair', 'E_mol', 'TotEng', 'Press'],
        [0, 0, 951591.43, 3135.108, 954726.54, 2988313.7],
        [1000, 0.10138438, -3306.2155, 1032.3684, -2272.9408, 1862.3769],
        [2000, 0.33174263, -3500.6115, 1021.026, -2476.6199, 710.07054]
    ]

    @pytest.fixture
    def thermo(self, data):
        return lmplog.Thermo(data[1:], columns=data[0])

    # @pytest.mark.parametrize('data', [(MINIMIZE)])
    # def testSetup(self, thermo):
    #     breakpoint()
