import time

import numpy as np
import pytest

from nemd import psutils
from nemd import symbols


class TestProcess:

    @pytest.fixture
    def process(self):
        return psutils.Process()

    def testGetUsed(self, process):
        assert process.getUsed()


class TestMemory:

    @pytest.fixture
    def mem(self, intvl):
        return psutils.Memory(intvl=intvl)

    @pytest.mark.parametrize('intvl', [(0.01)])
    def testSetPeak(self, mem):
        mem.stop.set()
        mem.setPeak(mem.peak, mem.stop, mem.intvl)
        assert 0.0 == mem.peak.get()

    @pytest.mark.parametrize('intvl', [(0.01)])
    def testResult(self, mem):
        mem.start()
        time.sleep(0.02)
        assert mem.result is not None
