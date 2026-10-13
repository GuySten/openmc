import pytest

from openmc.data.talys import (
    adjustment_factor, read_best_adjustments, read_omp_file)


OMP = """\
  26  54   2                                                              Fe054
   1 -11.34   0.000
   1.186   0.663  58.2    0.0071 0.000019  13.2   78.0
   1.278   0.536  15.4    0.0223  10.90
   1.000   0.580   6.1    0.0040  -3.1  160.0
   2 -11.34   0.000
   1.215   0.670  54.2    0.0077 0.000022   9.5   88.0
   1.278   0.536  15.4    0.0223  10.90
   1.000   0.580   6.1    0.0040  -3.1  160.0
  26  56   1   C
   1  -9.42   0.000
   1.186   0.663  56.8    0.0071 0.000019  13.0   80.0
   1.282   0.532  15.3    0.0211  10.90
   1.000   0.580   6.1    0.0040  -3.1  160.0
"""


def test_read_omp_file(tmp_path):
    path = tmp_path / 'n-Fe.omp'
    path.write_text(OMP)
    sets = read_omp_file(path)
    assert sorted(sets) == [54, 56]
    # Only the first (non-dispersive) set of Fe54 is used
    fe54 = sets[54]
    assert fe54['ef'] == -11.34
    assert (fe54['rv'], fe54['av'], fe54['v1']) == (1.186, 0.663, 58.2)
    assert (fe54['v2'], fe54['v3']) == (0.0071, 0.000019)
    assert (fe54['w1'], fe54['w2']) == (13.2, 78.0)
    assert (fe54['rd'], fe54['ad'], fe54['d1']) == (1.278, 0.536, 15.4)
    assert (fe54['d2'], fe54['d3']) == (0.0223, 10.90)
    assert (fe54['rso'], fe54['aso'], fe54['vso1']) == (1.0, 0.58, 6.1)
    assert (fe54['vso2'], fe54['wso1'], fe54['wso2']) == (0.004, -3.1, 160.)
    assert fe54['v4'] == 7.0e-9
    assert not fe54['coupled']
    assert sets[56]['coupled']
    assert sets[56]['ef'] == -9.42


BEST = """\
#avadjustF n 2. 10. 1.
rvadjust n 1. 1. 6. 3.5 1.02
rvadjust n 0.99 6. 20. 13. 0.98
d1adjust n 1.10 5. 25. 1.20
awdadjust n 1. awd.table
v1adjust p 1.05
gnadjust 26 56 1.1
"""


def test_read_best_adjustments(tmp_path):
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'tables' / 'awd.table').write_text(
        '4.0 1.0\n4.0 0.5\n5.0 0.8\n6.0 1.0\n')
    path = tmp_path / 'n-Fe056.talys'
    path.write_text(BEST)
    adj = read_best_adjustments(path)
    assert sorted(adj) == ['awd', 'd1', 'rv']
    # The constant factor is the one of the last line
    assert adj['rv']['value'] == 0.99
    assert adj['rv']['entries'] == [{'range': [1., 6., 3.5, 1.02]},
                                    {'range': [6., 20., 13., 0.98]}]
    # A line with fewer than four energy parameters only sets the constant
    assert adj['d1'] == {'value': 1.10, 'entries': []}
    # Repeated energies are skipped
    assert adj['awd']['entries'] == [
        {'table': [[4.0, 1.0], [5.0, 0.8], [6.0, 1.0]]}]


def test_unsupported_keyword(tmp_path):
    path = tmp_path / 'n-Fe056.talys'
    path.write_text('rvadjustF n 1. 10. 2.\n')
    with pytest.raises(ValueError):
        read_best_adjustments(path)


def test_adjustment_factor():
    table = {'value': 1.0, 'entries': [{'table': [[4., 1.], [5., 0.8]]}]}
    assert adjustment_factor(table, 4.5) == pytest.approx(0.9)
    # Outside the table only the constant factor remains
    assert adjustment_factor(table, 20.) == 1.0

    bump = {'value': 0.98, 'entries': [{'range': [1., 6., 3.5, 1.02]}]}
    # At the middle energy the factor equals the depth D
    assert adjustment_factor(bump, 3.5) == pytest.approx(0.98*1.02)
    assert adjustment_factor(bump, 1.0) == pytest.approx(0.98)
    assert adjustment_factor(bump, 6.0) == pytest.approx(0.98)
    assert 0.98 < adjustment_factor(bump, 5.0) < 0.98*1.02
