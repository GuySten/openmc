"""Tests for the k-effective columns written during an eigenvalue simulation."""

import re

import pytest

import openmc


# Matches MIN_REALIZATIONS_TO_COMBINE in include/openmc/math_functions.h
MIN_REALIZATIONS_TO_COMBINE = 4

# A generation row looks like "        7/1    1.01234    1.00987 +/- 0.00321",
# with the average k columns absent until a combined estimator is available
_GENERATION_ROW = re.compile(
    r"^\s+(\d+)/(\d+)\s+(\d+\.\d+)"
    r"(?:\s+(\d+\.\d+)\s*\+/-\s*(\d+\.\d+))?\s*$"
)


def _generation_rows(output):
    """Parse the batch/generation table out of captured OpenMC output.

    Returns one (batch, generation, k, average) tuple per row, where average is
    either None or a (mean, std_dev) pair.

    """
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if 'Bat./Gen.' in line)

    rows = []
    for line in lines[start:]:
        match = _GENERATION_ROW.match(line)
        if match is None:
            continue
        batch, generation, k, average, std_dev = match.groups()
        rows.append((
            int(batch),
            int(generation),
            float(k),
            None if average is None else (float(average), float(std_dev))
        ))
    return rows


def test_average_k_column_is_combined_estimator(run_in_tmpdir, capsys):
    inactive = 2
    batches = 12

    model = openmc.examples.pwr_pin_cell()
    model.settings.particles = 200
    model.settings.inactive = inactive
    model.settings.batches = batches
    model.settings.statepoint = {'batches': [batches - 1, batches]}

    model.run()
    rows = _generation_rows(capsys.readouterr().out)

    # One row per batch, since there is a single generation per batch
    assert [batch for batch, *_ in rows] == list(range(1, batches + 1))

    # The combined estimator needs MIN_REALIZATIONS_TO_COMBINE active batches
    # to have been accumulated, and a batch is only accumulated after its
    # generations have been written, hence the extra batch
    first_with_average = inactive + MIN_REALIZATIONS_TO_COMBINE + 1
    for batch, _, _, average in rows:
        if batch < first_with_average:
            assert average is None, f'batch {batch} reported an average k'
        else:
            assert average is not None, f'batch {batch} reported no average k'

    # By the same token, what is written for the final batch is the combined
    # estimator as it stood at the end of the preceding batch, which is the
    # value that batch's statepoint holds
    keff = openmc.StatePoint(f'statepoint.{batches - 1}.h5').keff
    average, std_dev = rows[-1][3]

    # Both are compared at the precision they were written with
    assert average == pytest.approx(keff.nominal_value, abs=1e-5)
    assert std_dev == pytest.approx(keff.std_dev, abs=1e-5)
