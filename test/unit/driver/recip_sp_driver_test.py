from unittest import mock

import pytest
import recip_sp_driver as driver


class TestArgumentParser:

    @pytest.mark.parametrize("miller_indices,valid", [(['1', '2'], True),
                                                      (['1'], False),
                                                      (['1', '1', '1'], False),
                                                      (['0', '0'], False)])
    def testParseArgs(self, miller_indices, valid):
        parser = driver.ArgumentParser()
        argv = [driver.FLAG_MILLER_INDICES] + miller_indices
        with mock.patch.object(parser, 'error'):
            parser.parse_args(argv)
            assert not valid == parser.error.called
