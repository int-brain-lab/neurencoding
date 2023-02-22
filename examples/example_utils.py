import numpy as np
import logging
from scipy import interpolate
try:
    from one.api import ONE
    from brainbox.io.one import SessionLoader
except ImportError:
    raise ImportError("Could not import ONE. 'ibllib' and 'one-api' packages are necessary for the"
                      " example scripts to run. Please install them using pip")

class TimeSeries(dict):
    """A subclass of dict with dot syntax, enforcement of time stamping"""

    def __init__(self, times, values, columns=None, *args, **kwargs):
        """TimeSeries objects are explicity for storing time series data in which entry (row) has
        a time stamp associated. TS objects have obligatory 'times' and 'values' entries which
        must be passed at construction, the length of both of which must match. TimeSeries takes an
        optional 'columns' argument, which defaults to None, that is a set of labels for the
        columns in 'values'. These are also exposed via the dot syntax as pointers to the specific
        columns which they reference.

        :param times: an ordered object containing a list of timestamps for the time series data
        :param values: an ordered object containing the associated measurements for each time stamp
        :param columns: a tuple or list of column labels, defaults to none. Each column name will
            be exposed as ts.colname in the TimeSeries object unless colnames are not strings.

        Also can take any additional kwargs beyond times, values, and columns for additional data
        storage like session date, experimenter notes, etc.

        Example:
        timestamps, mousepos = load_my_data()  # in which mouspos is T x 2 array of x,y coordinates
        positions = TimeSeries(times=timestamps, values=mousepos, columns=('x', 'y'),
                               analyst='John Cleese', petshop=True,
                               notes=("Look, matey, I know a dead mouse when I see one, "
                                      'and I'm looking at one right now."))
        """
        super(TimeSeries, self).__init__(
            times=np.array(times), values=np.array(values), columns=columns, *args, **kwargs
        )
        self.__dict__ = self
        self.columns = columns
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)

        # Enforce times dict key which contains a list or array of timestamps
        if len(self.times) != len(values):
            raise ValueError("Time and values must be of the same length")

        # If column labels are passed ensure same number of labels as columns, then expose
        # each column label using the dot syntax of a Bunch
        if isinstance(self.values, np.ndarray) and columns is not None:
            if self.values.shape[1] != len(columns):
                raise ValueError("Number of column labels must equal number of columns in values")
            self.update({col: self.values[:, i] for i, col in enumerate(columns)})

    def copy(self):
        """Return a new TimeSeries instance which is a copy of the current TimeSeries instance."""
        return TimeSeries(super(TimeSeries, self).copy())


def sync(
    dt, times=None, values=None, timeseries=None, offsets=None, interp="zero", fillval=np.nan
):
    """
    Function for resampling a single or multiple time series to a single, evenly-spaced, delta t
    between observations. Uses interpolation to find values.

    Can be used on raw numpy arrays of timestamps and values using the 'times' and 'values' kwargs
    and/or on brainbox.TimeSeries objects passed to the 'timeseries' kwarg. If passing both
    TimeSeries objects and numpy arrays, the offsets passed should be for the TS objects first and
    then the numpy arrays.

    Uses scipy's interpolation library to perform interpolation.
    See scipy.interp1d for more information regarding interp and fillval parameters.

    :param dt: Separation of points which the output timeseries will be sampled at
    :type dt: float
    :param timeseries: A group of time series to perform alignment or a single time series.
        Must have time stamps.
    :type timeseries: tuple of TimeSeries objects, or a single TimeSeries object.
    :param times: time stamps for the observations in 'values']
    :type times: np.ndarray or list of np.ndarrays
    :param values: observations corresponding to the timestamps in 'times'
    :type values: np.ndarray or list of np.ndarrays
    :param offsets: tuple of offsets for time stamps of each time series. Offsets for passed
        TimeSeries objects first, then offsets for passed numpy arrays. defaults to None
    :type offsets: tuple of floats, optional
    :param interp: Type of interpolation to use. Refer to scipy.interpolate.interp1d for possible
        values, defaults to np.nan
    :type interp: str
    :param fillval: Fill values to use when interpolating outside of range of data. See interp1d
        for possible values, defaults to np.nan
    :return: TimeSeries object with each row representing synchronized values of all
        input TimeSeries. Will carry column names from input time series if all of them have column
        names.
    """
    #########################################
    # Checks on inputs and input processing #
    #########################################

    # Initialize a list to contain times/values pairs if no TS objs are passed
    if timeseries is None:
        timeseries = []
    # If a single time series is passed for resampling, wrap it in an iterable
    elif isinstance(timeseries, TimeSeries):
        timeseries = [timeseries]
    # Yell at the user if they try to pass stuff to timeseries that isn't a TimeSeries object
    elif not all([isinstance(ts, TimeSeries) for ts in timeseries]):
        raise TypeError(
            "All elements of 'timeseries' argument must be brainbox.TimeSeries "
            "objects. Please uses 'times' and 'values' for np.ndarray args."
        )
    # Check that if something is passed to times or values, there is a corresponding equal-length
    # argument for the other element.
    if (times is not None) or (values is not None):
        if len(times) != len(values):
            raise ValueError("'times' and 'values' must have the same number of elements.")
        if type(times[0]) is np.ndarray:
            if not all([t.shape == v.shape for t, v in zip(times, values)]):
                raise ValueError(
                    "All arrays in 'times' must match the shape of the"
                    " corresponding entry in 'values'."
                )
            # If all checks are passed, convert all times and values args into TimeSeries objects
            timeseries.extend([TimeSeries(t, v) for t, v in zip(times, values)])
        else:
            # If times and values are only numpy arrays and lists of arrays, pair them and add
            timeseries.append(TimeSeries(times, values))

    # Adjust each timeseries by the associated offset if necessary then load into a list
    if offsets is not None:
        tstamps = [ts.times + os for ts, os in zip(timeseries, offsets)]
    else:
        tstamps = [ts.times for ts in timeseries]
    # If all input timeseries have column names, put them together for the output TS
    if all([ts.columns is not None for ts in timeseries]):
        colnames = []
        for ts in timeseries:
            colnames.extend(ts.columns)
    else:
        colnames = None

    #################
    # Main function #
    #################

    # Get the min and max values for all timeseries combined after offsetting
    tbounds = np.array([(np.amin(ts), np.amax(ts)) for ts in tstamps])
    if not np.all(np.isfinite(tbounds)):
        # If there is a np.inf or np.nan in the time stamps for any of the timeseries this will
        # break any further code so we check for all finite values and throw an informative error.
        raise ValueError(
            "NaN or inf encountered in passed timeseries.\
                          Please either drop or fill these values."
        )
    tmin, tmax = np.amin(tbounds[:, 0]), np.amax(tbounds[:, 1])
    if fillval == "extrapolate":
        # If extrapolation is enabled we can ensure we have a full coverage of the data by
        # extending the t max to be an whole integer multiple of dt above tmin.
        # The 0.01% fudge factor is to account for floating point arithmetic errors.
        newt = np.arange(tmin, tmax + 1.0001 * (dt - (tmax - tmin) % dt), dt)
    else:
        newt = np.arange(tmin, tmax, dt)
    tsinterps = [
        interpolate.interp1d(ts.times, ts.values, kind=interp, fill_value=fillval, axis=0)
        for ts in timeseries
    ]
    syncd = TimeSeries(newt, np.hstack([tsi(newt) for tsi in tsinterps]), columns=colnames)
    return syncd

def load_trials_df(
    eid,
    one=None,
    t_before=0.0,
    t_after=0.2,
    ret_wheel=False,
    ret_abswheel=False,
    wheel_binsize=0.02,
    addtl_types=[],
    trials_mask=None,
):
    """
    Generate a pandas dataframe of per-trial timing information about a given session.
    Each row in the frame will correspond to a single trial, with timing values indicating timing
    session-wide (i.e. time in seconds since session start). Can optionally return a resampled
    wheel velocity trace of either the signed or absolute wheel velocity.
    The resulting dataframe will have a new set of columns, trial_start and trial_end, which define
    via t_before and t_after the span of time assigned to a given trial.
    (useful for bb.modeling.glm)
    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : one.api.OneAlyx, optional
        one object to use for loading. Will generate internal one if not used, by default None
    t_before : float, optional
        Time before stimulus onset to include for a given trial, as defined by the trial_start
        column of the dataframe. If zero, trial_start will be identical to stimOn, by default 0.
    t_after : float, optional
        Time after feedback to include in the trail, as defined by the trial_end
        column of the dataframe. If zero, trial_end will be identical to feedback, by default 0.
    ret_wheel : bool, optional
        Whether to return the time-resampled wheel velocity trace, by default False
    ret_abswheel : bool, optional
        Whether to return the time-resampled absolute wheel velocity trace, by default False
    wheel_binsize : float, optional
        Time bins to resample wheel velocity to, by default 0.02
    addtl_types : list, optional
        List of additional types from an ONE trials object to include in the dataframe. Must be
        valid keys to the dict produced by one.load_object(eid, 'trials'), by default empty.
    trials_mask : list, optional
        List of trial indices to include in the dataframe. If None, all trials will be included.
    Returns
    -------
    pandas.DataFrame
        Dataframe with trial-wise information. Indices are the actual trial order in the original
        data, preserved even if some trials do not meet the maxlen criterion. As a result will not
        have a monotonic index. Has special columns trial_start and trial_end which define start
        and end times via t_before and t_after
    """
    if not one:
        one = ONE()

    if ret_wheel and ret_abswheel:
        raise ValueError("ret_wheel and ret_abswheel cannot both be true.")

    # Define which datatypes we want to pull out
    trialstypes = [
        "choice",
        "probabilityLeft",
        "feedbackType",
        "feedback_times",
        "contrastLeft",
        "contrastRight",
        "goCue_times",
        "stimOn_times",
    ]
    trialstypes.extend(addtl_types)

    loader = SessionLoader(one=one, eid=eid)
    loader.load_session_data(pose=False, motion_energy=False, pupil=False, wheel=False)
    if ret_wheel or ret_abswheel:
        loader.load_wheel(smooth_size=0.001)

    trials = loader.trials
    starttimes = trials.stimOn_times
    endtimes = trials.feedback_times
    trialsdf = trials[trialstypes]
    if trials_mask is not None:
        trialsdf = trialsdf.loc[trials_mask]
    trialsdf["trial_start"] = trialsdf["stimOn_times"] - t_before
    trialsdf["trial_end"] = trialsdf["feedback_times"] + t_after
    tdiffs = trialsdf["trial_end"] - np.roll(trialsdf["trial_start"], -1)
    if np.any(tdiffs[:-1] > 0):
        logging.warning(
            f"{sum(tdiffs[:-1] > 0)} trials overlapping due to t_before and t_after "
            "values. Try reducing one or both!"
        )
    if not ret_wheel and not ret_abswheel:
        return trialsdf

    wheel = loader.wheel
    whlt, whlv = wheel.times, wheel.velocity
    starttimes = trialsdf["trial_start"]
    endtimes = trialsdf["trial_end"]
    wh_endlast = 0
    trials = []
    for (start, end) in np.vstack((starttimes, endtimes)).T:
        wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
        wh_endind = np.searchsorted(whlt[wh_endlast:], end, side="right") + wh_endlast + 4
        wh_endlast = wh_endind
        tr_whlvel = whlv[wh_startind:wh_endind]
        tr_whlt = whlt[wh_startind:wh_endind] - start
        whlseries = TimeSeries(tr_whlt, tr_whlvel, columns=["whlvel"])
        whlsync = sync(wheel_binsize, timeseries=whlseries, interp="previous")
        trialstartind = np.searchsorted(whlsync.times, 0)
        trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
        trvel = whlsync.values[trialstartind : trialendind + trialstartind]
        if np.abs((trialendind - len(trvel))) > 0:
            raise IndexError("Mismatch between expected length of wheel data and actual.")
        if ret_wheel:
            trials.append(trvel)
        elif ret_abswheel:
            trials.append(np.abs(trvel))
    trialsdf["wheel_velocity"] = trials
    return trialsdf
