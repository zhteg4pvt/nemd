from unittest import mock

import pytest
import recip_sp_driver as driver


class TestFunction:

    @mock.patch('nemd.parserutils.ArgumentParser.error')
    @pytest.mark.parametrize("argv,called",
                             [(['1', '2'], False), (['1'], True),
                              (['0', '0'], True)])
    def testValidateOptions(self, error_mock, argv, called):
        driver.validate_options([driver.FLAG_MILLER_INDICES] + argv)
        assert error_mock.called == called
