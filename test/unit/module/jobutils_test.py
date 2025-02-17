import json

import pytest

from nemd import jobutils


class TestFunc:

    @pytest.mark.parametrize(
        'cmd,first,all_args',
        [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'], '1', ['1', '2']),
         ([], None, None)])
    def testGetArg(self, cmd, first, all_args):
        assert first == jobutils.get_arg(cmd, '-cru_num')
        val = first if first else 'val'
        assert val == jobutils.get_arg(cmd, '-cru_num', default='val')
        assert all_args == jobutils.get_arg(cmd, '-cru_num', first=False)
        vals = all_args if all_args else ['val']
        assert vals == jobutils.get_arg(cmd,
                                        '-cru_num',
                                        default=['val'],
                                        first=False)

    @pytest.mark.parametrize(
        'cmd,popped,num',
        [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'], ['1', '2'], 2),
         ([], None, 0), (['[Ar]', '-cru_num', '2', '-DEBUG'], ['2'], 2)])
    def testPopArg(self, cmd, popped, num):
        assert popped == jobutils.pop_arg(cmd, '-cru_num')
        assert num == len(cmd)

    @pytest.mark.parametrize('cmd,expected',
                             [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'
                                ], ['[Ar]', '-cru_num', '5', '2', '-DEBUG']),
                              ([], ['-cru_num', '5']),
                              (['[Ar]', '-cru_num', '2', '-DEBUG'
                                ], ['[Ar]', '-cru_num', '5', '-DEBUG'])])
    def testSetArg(self, cmd, expected):
        assert expected == jobutils.set_arg(cmd, '-cru_num', '5')

    @pytest.mark.parametrize('file,expected',
                             [('mol_bldr_driver.py', 'mol_bldr'),
                              ('cb_lmp_log_workflow.py', 'cb_lmp_log'),
                              ('cb_test.py', 'cb_test')])
    def testGetName(self, file, expected):
        assert expected == jobutils.get_name(file)

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'name')])
    @pytest.mark.parametrize('jobname', [None, 'jobname'])
    @pytest.mark.parametrize('file', [False, True])
    @pytest.mark.parametrize('log', [False, True])
    def testAddOutfile(self, jobname, file, log, tmp_dir, env):
        jobutils.add_outfile('file', jobname=jobname, file=file, log=log)
        with open('signac_job_document.json') as fh:
            data = json.load(fh)
            name = 'name' if jobname is None else 'jobname'
            assert {name: ['file']} == data['outfiles']
            assert ({name: 'file'} if file else None) == data.get('outfile')
            assert ({name: 'file'} if log else None) == data.get('logfile')
