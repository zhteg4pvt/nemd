import pytest

from nemd import envutils
from nemd import lmplog
from nemd import parserutils

HEX = envutils.test_data('hexane_liquid', 'lammps_runner_lammps.log')
SI = envutils.test_data('0044', 'lammps1', 'lmp.log')


@pytest.fixture
def log(file):
    return lmplog.Log(infile=file, delay=True)


class TestLog:

    @pytest.mark.parametrize('file,expected', [(HEX, (151, 6, 1.0, 'real')),
                                               (SI, (1, 5, 0.001, 'metal'))])
    def testSetup(self, log, expected):
        log.setUp()
        assert expected == (*log.thermo.shape, log.timestep, log.unit)

    @pytest.mark.parametrize('file,expected',
                             [(HEX, (126, 7, 26, 1.0, 'real')),
                              (SI, (1, 6, 0, 0.001, 'metal'))])
    def testRead(self, log, expected):
        log.read()
        assert expected == (*log.thermo.shape, len(log), log.timestep,
                            log.unit)

    @pytest.mark.parametrize('file,expected', [(HEX, (126, 26, 151, 0))])
    def testSetThermo(self, log, expected):
        log.read()
        assert expected[:2] == (log.thermo.shape[0], len(log))
        log.setThermo()
        assert expected[2:] == (log.thermo.shape[0], len(log))

    @pytest.mark.parametrize('file,expected', [(HEX, (151, 6, 0, 0.001)),
                                               (SI, (1, 5, 0, 0.001))])
    def testFinalize(self, log, expected):
        log.read()
        log.finalize()
        assert expected == (*log.thermo.shape, len(log), log.thermo.timestep)


class TestThermo:

    @pytest.fixture
    def thermo(self, log, args):
        log.read()
        options = parserutils.LmpLog().parse_args([log.infile] + args)
        timstep = log.getTimestep(log.timestep, backend=True)
        return lmplog.Thermo(log.thermo,
                             timestep=timstep,
                             unit=log.unit,
                             options=options)

    @pytest.mark.parametrize(
        'file,args,expected',  # yapf: disable
        [(HEX, ['-last_pct', '0.1'], [
            'Time (ps) (113)', 'Temp (K)', 'E_pair (kcal/mol)',
            'E_mol (kcal/mol)', 'TotEng (kcal/mol)', 'Press (atm)',
            'Volume (Å^3)'
        ]),
         (SI, [], [
             'Time (ps) (0)', 'Temp (K)', 'E_pair (eV)', 'E_mol (eV)',
             'TotEng (eV)', 'Press (bar)'
         ])])
    def testSetUp(self, thermo, expected):
        assert expected == [thermo.index.name, *thermo.columns]

    @pytest.mark.parametrize('file,args,expected',
                             [(HEX, ['-last_pct', '0.8'], [19.0, 82.027]),
                              (SI, ['-last_pct', '0.9'], [0.0, 0.0])])
    def testRange(self, thermo, expected):
        assert expected == thermo.range
