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


def test_addboxcar(trialsdf):
    """
    Test the internal method for adding a boxcar covariate with a defined start and end time taken
    from the columnd of the trials dataframe passed to the constructor.
    """
    # Make sure type enforcement and data checks work
    with pytest.raises(AttributeError):  # duplicate cov checking works?
        design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)
        design.add_covariate_boxcar('testcov', 'stim_onset', 'feedback')
        design.add_covariate_boxcar('testcov', 'stim_onset', 'feedback')
    with pytest.raises(KeyError):  # non-existing column checks work?
        design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)
        design.add_covariate_boxcar('testcov', 'stim_onset', 'notacolumn')
    with pytest.raises(KeyError):  # Column check for height works?
        design.add_covariate_boxcar('testcov', 'stim_onset', 'feedback', height='notacolumn')
    with pytest.raises(TypeError):  # Only timing bounds on boxcar end?
        design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)
        design.add_covariate_boxcar('testcov', 'stim_onset', 'wheel_traces')
    with pytest.raises(TypeError):  # Only timing bounds on boxcar start?
        design.add_covariate_boxcar('testcov', 'wheel_traces', 'stim_onset')
    with pytest.raises(IndexError):  # Correctly reject mismatched manual height series?
        tmpheights = pd.Series(np.arange(100))
        design.add_covariate_boxcar('testcov', 'stim_onset', 'feedback', height=tmpheights)
    with pytest.raises(IndexError):  # Correctly reject mismatched height indices?
        tmpheights = pd.Series(np.arange(10), index=np.arange(0, 20, 2, dtype=int))
        design.add_covariate_boxcar('testcov', 'stim_onset', 'feedback', height=tmpheights)

    # Check whether the method is generating a reasonable regressor
    design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)
    design.add_covariate_boxcar('testboxcar',
                                'stim_onset',
                                'feedback',
                                cond=lambda tr: tr.deltas <= 5,
                                height='deltas',
                                desc='Test boxcar covariate')
    assert 'testboxcar' in design.covar
    assert np.all(
        design.covar['testboxcar']['valid_trials'] == trialsdf.index[trialsdf.deltas <= 5])
    assert design.covar['testboxcar']['bases'] is None
    assert design.covar['testboxcar']['offset'] == 0
    startinds = np.array(
        [np.flatnonzero(x > 0)[0] for x in design.covar['testboxcar']['regressor']])
    endinds = np.array(
        [np.flatnonzero(x > 0)[-1] for x in design.covar['testboxcar']['regressor']])
    for i, idx in enumerate(trialsdf.index):
        curr_reg = design.covar['testboxcar']['regressor'].loc[idx]
        assert startinds[i] == pytest.approx(0.1 / BINWIDTH)
        assert endinds[i] == pytest.approx(design.binf(design.trialsdf.loc[idx, 'feedback']))
        assert np.all(curr_reg[np.flatnonzero(curr_reg)] == trialsdf.loc[idx, 'deltas'])
    return


def test_addraw(trialsdf):
    # TODO: Write this
    pass


def test_addcov(trialsdf):
    """
    test the generic .add_covariate() method in DesignMatrix, making sure it stores the passed args
    to the internal covar dict properly, throwing errors along the way for timing mismatches and
    finally incrementing the currcol value for the current last DM column.
    """
    design = dm.DesignMatrix(trialsdf, vartypes=VARTYPES, binwidth=BINWIDTH)
    tbases = mut.raised_cosine(0.2, 3, binf)
    with pytest.raises(ValueError):
        # Check that if the length of reg. doesn't match duration the method will throw an error.
        reglist = [np.ones(binf(dur)) for dur in design.trialsdf.duration]
        dummyregressor = pd.Series(reglist, index=design.trialsdf.index)
        dummyregressor.loc[0] = np.pad(dummyregressor.loc[0], (0, 1))  # Regressor too long on tr 1
        design.add_covariate('testreg', dummyregressor, None)
    with pytest.raises(ValueError):
        # Check that if trials not present in the design DF are in the regressor an error occurs
        dummyregressor.loc[0] = dummyregressor.loc[0][:-1]  # Fix our earlier issue
        dummyregressor = pd.concat([dummyregressor, pd.Series([10], index=[1000])])
        design.add_covariate('testreg', dummyregressor, None)
    design.add_covariate('wheel',
                         trialsdf.wheel_traces,
                         tbases,
                         offset=-0.2,
                         cond=lambda tr: tr.deltas <= 5,
                         desc='testdesc')
    assert np.all(design.covar['wheel']['valid_trials'] == trialsdf.index[trialsdf.deltas <= 5])
    assert np.all(design.covar['wheel']['bases'] == tbases)
    assert design.covar['wheel']['offset'] == -0.2
    assert np.all(design.covar['wheel']['regressor'] == trialsdf.wheel_traces)
    assert design.currcol == 3
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
    # TODO: Add a boxcar covariate and a raw covariate to test those helper methods
    design.compile_design_matrix()  # Finally compile
    # Load target DM
    npy_file = Path(__file__).parent.joinpath('fixtures', 'design_matrix_test.npy')
    if npy_file.exists():
        ref_dm = np.load(npy_file)
        assert np.allclose(design.dm, ref_dm), "Design matrix produced didn't match reference"
