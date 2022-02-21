# Standard library
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
import pytest

# Neurencoding repo imports
import neurencoding.design_matrix as dm
import neurencoding.utils as mut

BINWIDTH = 0.02
VARTYPES = {
    'trial_start': 'timing',
    'trial_end': 'timing',
    'stim_onset': 'timing',
    'feedback': 'timing',
    'wheel_traces': 'continuous',
    'deltas': 'value',
}


@pytest.fixture(scope='module')
def trialsdf():
    # Generate fake trial start, stimon, feedback, and end times
    starts = np.array([0, 1.48, 2.93, 4.67, 6.01, 7.31, 8.68, 9.99, 11.43, 12.86])
    ends = np.array([1.35, 2.09, 3.53, 5.23, 6.58, 7.95, 9.37, 11.31, 12.14, 13.26])
    stons = starts + 0.1
    fdbks = np.array([0.24, 1.64, 3.15, 4.81, 6.23, 7.50, 8.91, 10.16, 11.64, 13.05])
    deltavals = np.arange(1, 11)

    # Figure out how many bins each trial is and generate non-monotonic trace of fake wheel
    whlpath = Path(__file__).parent.joinpath('fixtures', 'design_wheel_traces_test.p')
    if whlpath.exists():
        fakewheels = np.load(whlpath, allow_pickle=True)

    # Store trialsdf for later use
    trialsdf = pd.DataFrame({
        'trial_start': starts,
        'trial_end': ends,
        'stim_onset': stons,
        'feedback': fdbks,
        'wheel_traces': fakewheels,
        'deltas': deltavals,
    })
    return trialsdf


def binf(x):
    return np.ceil(x / BINWIDTH).astype(int)


def test_init(trialsdf):
    """
    test whether the generation of a DesignMatrix object works as intended, subtracting trial
    start times from timing columns and storing the appropriate internal attributes
    """
    with pytest.raises(KeyError):
        design = dm.DesignMatrix(trialsdf, vartypes={
            'trial_start': 'timing',
        })
    with pytest.raises(ValueError):
        design = dm.DesignMatrix(trialsdf,
                                 vartypes={
                                     'trial_start': 'notatype',
                                     'trial_end': 'notatype',
                                     'stim_onset': 'notatype',
                                     'feedback': 'notatype',
                                     'wheel_traces': 'notatype',
                                     'deltas': 'notatype',
                                 })
    design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)
    assert hasattr(design, 'base_df')
    assert hasattr(design, 'trialsdf')
    assert np.all(
        np.isclose(design.trialsdf['duration'],
                   trialsdf['trial_end'] - trialsdf['trial_start'],
                   atol=BINWIDTH / 2))
    assert design.binwidth == BINWIDTH
    assert hasattr(design, 'covar') and isinstance(design.covar, dict)
    assert hasattr(design, 'vartypes') and design.vartypes is VARTYPES
    assert hasattr(design, 'compiled') and not design.compiled
    for var in VARTYPES:
        if VARTYPES[var] == 'timing':
            assert np.all(
                np.isclose(design.trialsdf[var],
                           trialsdf[var] - trialsdf['trial_start'],
                           atol=1e-5))

    return


def test_addcov(trialsdf):
    """
    test the generic .add_covariate() method in DesignMatrix, making sure it stores the passed args
    to the internal covar dict properly, throwing errors along the way for timing mismatches and
    finally incrementing the currcol value for the current last DM column.
    """
    pass


def test_timingcov(trialsdf):
    """
    Test that adding a timing covariate exposes the correct values in design['covar'], and that
    the internal regressor structures are as expected.
    """
    design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)
    tbases = mut.raised_cosine(0.2, 3, binf)
    with pytest.raises(ValueError):
        design.add_covariate_timing('timingtest', 'asdf', tbases)
    with pytest.raises(TypeError):
        design.add_covariate_timing('timingtest', 'wheel_traces', tbases)
    design.add_covariate_timing(
        'timingtest',
        'stim_onset',
        tbases,
        deltaval='deltas',
        offset=-BINWIDTH,
        cond=lambda r: r.deltas < 6,  # End time of 6th trial
        desc='test timing regressor')
    rkey = 'timingtest'
    assert rkey in design.covar
    assert design.covar[rkey]['description'] == 'test timing regressor'
    assert np.all(design.covar[rkey]['bases'] == tbases)
    assert np.all(design.covar[rkey]['valid_trials'] == np.array(range(5)))
    assert design.covar[rkey]['offset'] == -BINWIDTH
    assert np.sum(design.covar[rkey]['regressor'][2]) == pytest.approx(3)
    assert all(
        np.flatnonzero(design.covar[rkey]['regressor'][i]).shape[0] == 1
        for i in design.trialsdf.index)
    nzinds = [np.flatnonzero(design.covar[rkey]['regressor'][i]) for i in design.trialsdf.index]
    assert np.all(np.array(nzinds) == pytest.approx(0.1 / BINWIDTH))

    return


def test_construct(trialsdf):
    """
    Check whether or not design matrix construction works as intended
    """

    # Design matrix instance
    design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)

    # Separate bases for wheel and timing
    tbases = mut.raised_cosine(0.2, 3, binf)
    wbases = mut.raised_cosine(0.1, 2, binf)
    # Add covariates one by one. Add different offsets on timings to test
    design.add_covariate_timing('start', 'trial_start', tbases, offset=0.02)
    design.add_covariate_timing('stim_on', 'stim_onset', tbases)
    design.add_covariate_timing('feedback', 'feedback', tbases, offset=-0.02)
    design.add_covariate('wheelpos', trialsdf.wheel_traces, wbases, offset=-0.1)
    design.compile_design_matrix()  # Finally compile
    # Load target DM
    npy_file = Path(__file__).parent.joinpath('fixtures', 'design_matrix_test.npy')
    if npy_file.exists():
        ref_dm = np.load(npy_file)
        assert np.allclose(design.dm, ref_dm), "Design matrix produced didn't match reference"
