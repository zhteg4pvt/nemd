import pandas as pd
import pytest

from nemd import analyzer
from nemd import envutils
from nemd import parserutils

TEST0045 = envutils.test_data('itest', '0045_test')
TEST0037 = envutils.test_data('itest', '0037_test')


class TestBase:

    @pytest.fixture
    def base(self, NAME, UNIT, LABEL, PROP, DATA_EXT, args, parm):
        Base = type(
            'Base', (analyzer.Base, ),
            dict(NAME=NAME,
                 UNIT=UNIT,
                 LABEL=LABEL,
                 PROP=PROP,
                 DATA_EXT=DATA_EXT))
        options = parserutils.Workflow().parse_args(args)
        if parm is not None:
            parm = pd.Series(index=pd.Index([], name=parm))
        return Base(options=options, parm=parm)

    @pytest.mark.parametrize('UNIT,LABEL,PROP', [(None, ) * 3])
    @pytest.mark.parametrize(
        'NAME,DATA_EXT,args,parm,expected',
        [('base', '.csv', ['-JOBNAME', 'name'], None, 'name_base.csv'),
         ('anal', '.npy', ['-JOBNAME', 'job'], 0, 'workspace/job_anal_0.npy')])
    def testOutfile(self, base, expected):
        assert expected == base.outfile

    @pytest.mark.parametrize('UNIT,PROP', [(None, None)])
    @pytest.mark.parametrize('dirname', [TEST0037])
    @pytest.mark.parametrize(
        'NAME,DATA_EXT,LABEL,args,parm,expected',
        [('xyz', '.xyz', ['-JOBNAME', 'name'], None, 'name_base.csv')])
    def testReadData(self, jobs):
        breakpoint()
