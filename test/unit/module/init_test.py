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

    @pytest.mark.parametrize('debug', [(''), ('1')])
    def testLazyImport(self, debug):
        cmd = "nemd_run -c 'from nemd import constants; print(repr(constants))'"
        env = os.environ | {'DEBUG': debug}
        proc = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              text=True,
                              shell=True,
                              env=env)
        if debug:
            assert not proc.stdout.startswith('Lazily-loaded')
            return
        assert proc.stdout.startswith('Lazily-loaded')
        assert 1e-08 == constants.ANG_TO_CM
        assert not repr(constants).startswith('Lazily-loaded')
