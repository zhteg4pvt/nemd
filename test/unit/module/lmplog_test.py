import conftest
import pytest

from nemd import envutils
from nemd import lmplog
from nemd import parserutils

HEX = envutils.test_data('hexane_liquid', 'lammps_runner_lammps.log')
SI = envutils.test_data('0044', 'lammps1', 'lmp.log')


@pytest.fixture
def log(args):
    options = parserutils.LmpLog().parse_args(args)
    return lmplog.Log(infile=options.log, options=options, delay=True)


@conftest.require_src
class TestLog:

    @pytest.mark.parametrize('args,expected', [([HEX], (151, 6, 1.0, 'real')),
                                               ([SI], (1, 5, 0.001, 'metal'))])
    def testSetup(self, log, expected):
        log.setUp()
        assert expected == (*log.thermo.shape, log.timestep, log.unit)

    @pytest.mark.parametrize('args,expected',
                             [([HEX], (126, 7, 26, 1.0, 'real')),
                              ([SI], (1, 6, 0, 0.001, 'metal'))])
    def testRead(self, log, expected):
        log.read()
        assert expected == (*log.thermo.shape, len(log), log.timestep,
                            log.unit)

    @pytest.mark.parametrize('args,expected', [([HEX], (126, 26, 151, 0))])
    def testConcat(self, log, expected):
        log.read()
        assert expected[:2] == (log.thermo.shape[0], len(log))
        log.concat()
        assert expected[2:] == (log.thermo.shape[0], len(log))

    @pytest.mark.parametrize('args,expected', [([HEX], (151, 6, 0, 0.001)),
                                               ([SI], (1, 5, 0, 0.001)),
                                               ([HEX, '-slice', '1', '5', '2'],
                                                (2, 6, 0, 0.001))])
    def testFinalize(self, log, expected):
        log.read()
        log.concat()
        log.finalize()
        assert expected == (*log.thermo.shape, len(log), log.thermo.timestep)


@conftest.require_src
class TestThermo:

    EMPTY = envutils.test_data('ar', 'empty.log')

    @pytest.fixture
    def thermo(self, log):
        log.read()
        timstep = log.getTimestep(log.timestep, backend=True)
        return lmplog.Thermo(log.thermo,
                             timestep=timstep,
                             unit=log.unit,
                             options=log.options)

    @pytest.mark.parametrize(
        'args,expected',  # yapf: disable
        [([HEX, '-last_pct', '0.1'], [
            'Time (ps) (113)', 'Temp (K)', 'E_pair (kcal/mol)',
            'E_mol (kcal/mol)', 'TotEng (kcal/mol)', 'Press (atm)',
            'Volume (Å^3)'
        ]),
         ([SI], [
             'Time (ps) (0)', 'Temp (K)', 'E_pair (eV)', 'E_mol (eV)',
             'TotEng (eV)', 'Press (bar)'
         ]), ([EMPTY], [None])])
    def testSetUp(self, thermo, expected):
        assert expected == [thermo.index.name, *thermo.columns]

    @pytest.mark.parametrize('args,expected',
                             [([HEX, '-last_pct', '0.8'], [19.0, 82.027]),
                              ([SI, '-last_pct', '0.9'], [0.0, 0.0])])
    def testRange(self, thermo, expected):
        assert expected == thermo.range
