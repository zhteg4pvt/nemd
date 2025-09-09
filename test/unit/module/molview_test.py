from unittest import mock

import numpy as np
import pytest

from nemd import envutils
from nemd import lmpfull
from nemd import molview
from nemd import traj

FRM = envutils.test_data('water', 'three.custom')
RDR = lmpfull.Reader.fromTest('water', 'polymer_builder.data')


@pytest.mark.skipif(FRM is None, reason='Trajectory not available')
@pytest.mark.parametrize('file', [FRM])
class TestFrame:

    @pytest.fixture()
    def frm(self, file, rdr):
        return molview.Frame(traj.Traj(file), rdr=rdr)

    @pytest.mark.parametrize('rdr,expected', [(None, [9, 9, 9, 1, 1, 1]),
                                              (RDR, [9, 9, 9, 2, 2, 2])])
    def testSetUp(self, frm, expected):
        np.testing.assert_almost_equal(frm.nunique(), expected)

    @pytest.mark.parametrize('rdr,expected', [(None, (1000, 56.409))])
    def testUpdate(self, frm, expected):
        copied = frm.copy()
        frm.update(frm.trj[-1])
        assert not (copied == frm)[['xu', 'yu', 'zu']].any().any()
        assert expected == (frm.step, frm.box.max().max())

    @pytest.mark.parametrize('rdr,expected',
                             [(None, ['X', 5, '#FF1493', 9, 3]),
                              (RDR, ['O', 3.1507, '#f00000', 3, 3])])
    def testCoords(self, frm, expected):
        info, coords = next(frm.coords)
        assert expected == [*info, *coords.shape]

    @pytest.mark.parametrize('rdr,expected', [(None, ['X']),
                                              (RDR, ['O', 'H'])])
    def testElements(self, frm, expected):
        assert (expected == frm.ele).all()

    @pytest.mark.parametrize('rdr,expected', [(None, 0), (RDR, 12)])
    def testBonds(self, frm, expected):
        assert expected == len(list(frm.bonds))

    @pytest.mark.parametrize(
        'rdr,expected', [(None, [[-2.77391, 4.24531], [-0.797575, 6.221645],
                                 [-2.3512099, 4.6680098]])])
    def testLims(self, frm, expected):
        np.testing.assert_almost_equal(frm.lims, expected, decimal=6)

    @pytest.mark.parametrize('rdr,expected', [(None, [0, 834, 1000])])
    def testIter(self, frm, expected):
        steps = [x.step for x in frm.iter()]
        np.testing.assert_almost_equal(steps, expected)


@pytest.mark.skipif(FRM is None, reason='Trajectory not available')
@pytest.mark.parametrize('file', [FRM])
class TestFigure:

    @pytest.fixture
    def fig(self, file, rdr):
        return molview.Figure(traj.Traj(file), rdr=rdr, delay=True)

    @pytest.mark.parametrize('ekey', ['INTERAC'])
    @pytest.mark.parametrize('rdr,evalue,expected', [(None, None, 13),
                                                     (RDR, '1', 26)])
    def testSetUp(self, fig, expected, evalue, env):
        with mock.patch.object(fig, 'show') as mocked:
            fig.setUp()
            assert bool(evalue) == mocked.called
        assert expected == len(fig.data)

    @pytest.mark.parametrize('rdr,expected', [(None, 13), (RDR, 26)])
    def testTraces(self, fig, expected):
        assert expected == len(fig.traces)

    @pytest.mark.parametrize('rdr,expected', [(None, 1), (RDR, 2)])
    def testScatters(self, fig, expected):
        assert expected == len(list(fig.scatters))

    @pytest.mark.parametrize('rdr,expected', [(None, 0), (RDR, 12)])
    def testLines(self, fig, expected):
        assert expected == len(list(fig.lines))

    @pytest.mark.parametrize('rdr', [(None), (RDR)])
    def testEdges(self, fig):
        assert 12 == len(list(fig.edges))

    @pytest.mark.parametrize('rdr', [(None), (RDR)])
    def testUpdateFrame(self, fig):
        fig.updateFrame()
        assert 3 == len(fig.frames)

    @pytest.mark.parametrize('rdr', [(None), (RDR)])
    def testUpdateLayout(self, fig):
        fig.updateFrame()
        fig.updateLayout()
        assert 3 == len(fig.layout['sliders'][0]['steps'])

    @pytest.mark.parametrize('rdr', [(None), (RDR)])
    def testScene(self, fig):
        assert ['xaxis', 'yaxis', 'zaxis',
                'aspectmode'] == list(fig.scene.keys())

    @pytest.mark.parametrize('rdr,expected', [(None, 3)])
    def testSlider(self, fig, expected):
        fig.updateFrame()
        assert expected == len(fig.slider['steps'])

    @pytest.mark.parametrize('rdr,expected', [(None, 2)])
    def testButtons(self, fig, expected):
        assert expected == len(fig.buttons['buttons'])
