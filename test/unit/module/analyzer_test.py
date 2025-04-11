import pandas as pd
import pytest

from nemd import analyzer
from nemd import parserutils


class TestBase:

    @pytest.fixture
    def base(self, NAME, PROP, ERR_LB, args, parm):
        attrs = dict(NAME=NAME, PROP=PROP, ERR_LB=ERR_LB)
        Base = type('Base', (analyzer.Base, ), attrs)
        options = parserutils.Workflow().parse_args(args)
        if parm is not None:
            parm = pd.Series(index=pd.Index([], name=parm))
        return Base(options=options, parm=parm)

    @pytest.mark.parametrize('NAME,PROP,ERR_LB,args,parm,expected',
                             [('base', 'prop', 'St Dev', ['-JOBNAME', 'name'],
                               None, 'name_base.csv'),
                              ('base', 'prop', 'St Dev', ['-JOBNAME', 'job'],
                               0, 'workspace/job_base_0.csv')])
    def testOutfile(self, base, expected):
        assert expected == base.outfile
