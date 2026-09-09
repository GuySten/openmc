"""Tests for CellSetFilter, which bins crossings by region boundary."""

import numpy as np
import pytest

import openmc


def four_cell_model(ux):
    """Four cells in a row along x, with a monodirectional source.

    Cells 1..4 span x in [-10, -2], [-2, 0], [0, 2] and [2, 10]. The source
    fires every particle along ``ux``, so it crosses all three interior
    boundaries in turn before leaking. Cells are void, so nothing scatters.
    """
    openmc.reset_auto_ids()

    planes = [openmc.XPlane(x) for x in (-2.0, 0.0, 2.0)]
    lo = openmc.XPlane(-10.0, boundary_type='vacuum')
    hi = openmc.XPlane(10.0, boundary_type='vacuum')
    ymin = openmc.YPlane(-5.0, boundary_type='vacuum')
    ymax = openmc.YPlane(5.0, boundary_type='vacuum')
    zmin = openmc.ZPlane(-5.0, boundary_type='vacuum')
    zmax = openmc.ZPlane(5.0, boundary_type='vacuum')
    box = +ymin & -ymax & +zmin & -zmax

    cells = [
        openmc.Cell(cell_id=1, region=+lo & -planes[0] & box),
        openmc.Cell(cell_id=2, region=+planes[0] & -planes[1] & box),
        openmc.Cell(cell_id=3, region=+planes[1] & -planes[2] & box),
        openmc.Cell(cell_id=4, region=+planes[2] & -hi & box),
    ]

    model = openmc.Model()
    model.geometry = openmc.Geometry(cells)

    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((-5.0 * np.sign(ux), 0.0, 0.0))
    src.angle = openmc.stats.Monodirectional((ux, 0.0, 0.0))

    model.settings.run_mode = 'fixed source'
    model.settings.batches = 1
    model.settings.particles = 100
    model.settings.source = src

    # left half and right half
    return model, cells[:2], cells[2:]


def test_cellset_filter_validation():
    """The Python API rejects malformed regions and senses."""
    with pytest.raises(ValueError):
        # a bare cell is not a region; it has to be wrapped in an iterable
        openmc.CellSetFilter([openmc.Cell(cell_id=1)])

    with pytest.raises(ValueError):
        openmc.CellSetFilter([])

    with pytest.raises(ValueError):
        openmc.CellSetFilter([[]])

    with pytest.raises(ValueError):
        openmc.CellSetFilter([[1, 2]], sense='sideways')

    with pytest.raises(ValueError):
        # a repeated sense would silently duplicate bins
        openmc.CellSetFilter([[1, 2]], sense=['net', 'net'])


def test_cellset_filter_bins():
    """Bins are region-major and count regions times senses."""
    f = openmc.CellSetFilter([[1, 2, 3], [4, 5]], sense=['net', 'out', 'in'])
    assert f.num_bins == 6
    assert f.bins[0] == ((1, 2, 3), 'net')
    assert f.bins[1] == ((1, 2, 3), 'out')
    assert f.bins[3] == ((4, 5), 'net')


def test_cellset_filter_xml_roundtrip():
    f = openmc.CellSetFilter([[1, 2, 3], [4, 5]], sense=['out', 'in'])
    g = openmc.CellSetFilter.from_xml_element(f.to_xml_element())
    assert g == f
    assert g.regions == f.regions
    assert g.sense == f.sense


def test_cellset_filter_equality_ignores_cell_order():
    """A region is a set, so the order cells are listed in is not meaningful."""
    assert (openmc.CellSetFilter([[1, 2, 3]])
            == openmc.CellSetFilter([[3, 1, 2]]))


@pytest.mark.parametrize('ux', [1.0, -1.0])
def test_cellset_current_senses(ux, run_in_tmpdir):
    """Net, outgoing and incoming currents across a region boundary.

    The source cell and the direction of travel are mirrored between the two
    parameters, so the outgoing current must come out positive either way. That
    is the property a surface-normal sign convention cannot provide: it would
    make one of the two negative.
    """
    model, left, right = four_cell_model(ux)

    tally = openmc.Tally()
    tally.filters = [openmc.CellSetFilter([left, right],
                                          sense=['net', 'out', 'in'])]
    tally.scores = ['current']
    model.tallies = [tally]

    model.run(apply_tally_results=True)
    j = tally.mean.reshape(2, 3)

    # Particles cross the left/right boundary exactly once, in the direction of
    # travel. The two interior crossings (cell 1 to 2, and cell 3 to 4) are
    # internal to a region and must not be counted at all.
    source_side, other_side = (0, 1) if ux > 0 else (1, 0)

    assert j[source_side, 0] == pytest.approx(1.0)   # net, leaving
    assert j[source_side, 1] == pytest.approx(1.0)   # out
    assert j[source_side, 2] == pytest.approx(0.0)   # in

    assert j[other_side, 0] == pytest.approx(-1.0)   # net, entering
    assert j[other_side, 1] == pytest.approx(0.0)    # out
    assert j[other_side, 2] == pytest.approx(1.0)    # in


def test_cellset_region_to_region(run_in_tmpdir):
    """Pinning both ends of a crossing gives a directed region-to-region current."""
    model, left, right = four_cell_model(1.0)

    forward = openmc.Tally(name='left to right')
    forward.filters = [openmc.CellSetFilter([left], sense='out'),
                       openmc.CellSetFilter([right], sense='in')]
    forward.scores = ['current']

    backward = openmc.Tally(name='right to left')
    backward.filters = [openmc.CellSetFilter([right], sense='out'),
                        openmc.CellSetFilter([left], sense='in')]
    backward.scores = ['current']

    model.tallies = [forward, backward]
    model.run(apply_tally_results=True)

    assert forward.mean.flat[0] == pytest.approx(1.0)
    assert backward.mean.flat[0] == pytest.approx(0.0)


def test_cellset_internal_crossings_excluded(run_in_tmpdir):
    """A region spanning the whole model has no boundary to cross."""
    model, left, right = four_cell_model(1.0)

    tally = openmc.Tally()
    tally.filters = [openmc.CellSetFilter([left + right], sense=['out', 'in'])]
    tally.scores = ['current']
    model.tallies = [tally]

    model.run(apply_tally_results=True)

    # Every crossing is between two cells of the one region, so nothing scores
    # even though the particles cross three surfaces each.
    assert tally.mean.flat[0] == pytest.approx(0.0)
    assert tally.mean.flat[1] == pytest.approx(0.0)


def test_cellset_statepoint_roundtrip(run_in_tmpdir):
    """The filter and its results survive a statepoint round trip."""
    model, left, right = four_cell_model(1.0)

    tally = openmc.Tally()
    tally.filters = [openmc.CellSetFilter([left, right],
                                          sense=['net', 'out', 'in'])]
    tally.scores = ['current']
    model.tallies = [tally]

    sp_file = model.run()
    with openmc.StatePoint(sp_file) as sp:
        out = sp.tallies[tally.id]
        filt = out.filters[0]

        assert isinstance(filt, openmc.CellSetFilter)
        assert filt.regions == ((1, 2), (3, 4))
        assert filt.sense == ('net', 'out', 'in')

        j = out.mean.reshape(2, 3)
        assert j[0, 0] == pytest.approx(1.0)
        assert j[1, 2] == pytest.approx(1.0)

        # Region and sense are separately selectable in a dataframe
        df = out.get_pandas_dataframe()
        assert list(df['cellset sense'][:3]) == ['net', 'out', 'in']
        assert df['cellset'][0] == '1, 2'
