import os
import subprocess

import pytest

from nemd import constants


class TestFunc:

    def testLogger(self):
        cmd = "nemd_run -c 'import nemd, logging; assert not logging.getLogger().handlers'"
        env = os.environ | {'DEBUG': ''}
        proc = subprocess.run(cmd,
                              text=True,
                              stdout=subprocess.PIPE,
                              shell=True,
                              env=env)
        assert not proc.returncode

    @pytest.mark.parametrize('ekey', ['DEBUG'])
    @pytest.mark.parametrize('evalue', [(''), ('1')])
    def testLazyImport(self, evalue, env):
        cmd = "nemd_run -c 'from nemd import constants; print(repr(constants))'"
        proc = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              text=True,
                              shell=True)
        assert bool(evalue) ^ ('Lazily-loaded' in proc.stdout)
        assert 1e-08 == constants.ANG_TO_CM
        assert not repr(constants).startswith('Lazily-loaded')
