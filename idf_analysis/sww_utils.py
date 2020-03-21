__author__ = "David Camhy, Markus Pichler"
__copyright__ = "Copyright 2018, University of Technology Graz"
__credits__ = ["David Camhy", "Markus Pichler"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Camhy, Markus Pichler"

import pytz
from datetime import tzinfo
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np

from .definitions import COL


class TimezoneError(Exception):
    """Some Error With a Timezone"""


class IdfError(Exception):
    """Some Error Within this Package"""


def check_tz(timezone):
    """
    check the give timezone

    :param timezone:
    :type timezone: tzinfo | str
    :return:
    :rtype: tzinfo
    """
    if isinstance(timezone, str):
        try:
            return pytz.timezone(timezone)
        except:
            raise TimezoneError('"{}" is not a timezone or unknown'.format(timezone))
    # else:
    #     return timezone
    elif isinstance(timezone, tzinfo):
        return timezone
    else:
        raise TimezoneError('unknown timezone format: "{}"'.format(type(timezone)))


def remove_timezone(data):
    """
    convert the timezone to wintertime and then remove the timezone from the dataframe index, to not get in conflict
    with the plot functions

    Args:
        data (pandas.DataFrame | pandas.Series): data with datetimeindex 

    Returns:
        pandas.DataFrame | pandas.Series: data with datetimeindex without timezone
    """
    no_tz = data.copy()
    index = no_tz.index
    native_timezone = pytz.timezone('Etc/GMT-1')
    if index.tz is None:
        raise TimezoneError("Dataframe must have a timezone information")
    timezone = check_tz(native_timezone)
    idx = index.tz_convert(check_tz(timezone))
    no_tz.index = idx.tz_localize(None)
    return no_tz


########################################################################################################################
def guess_freq(date_time_index, default=pd.Timedelta(minutes=1)):
    """
    guess the frequency by evaluating the most often frequency

    Args:
        date_time_index (pandas.DatetimeIndex): index of a time-series
        default (pandas.Timedelta):

    Returns:
        pandas.DateOffset: frequency of the date-time-index
    """
    freq = date_time_index.freq
    if pd.notnull(freq):
        return to_offset(freq)

    if not len(date_time_index) <= 3:
        freq = pd.infer_freq(date_time_index)  # 'T'

        if pd.notnull(freq):
            return to_offset(freq)

        delta_series = date_time_index.to_series().diff(periods=1).fillna(method='backfill')
        counts = delta_series.value_counts()
        counts.drop(pd.Timedelta(minutes=0), errors='ignore')

        if counts.empty:
            delta = default
        else:
            delta = counts.index[0]
            if delta == pd.Timedelta(minutes=0):
                delta = default
    else:
        delta = default

    return to_offset(delta)


########################################################################################################################
def year_delta(years):
    return pd.Timedelta(days=365.25 * years)


########################################################################################################################
def rain_events(series, ignore_rain_below=0, min_gap=pd.Timedelta(hours=4)):
    """
    get rain events as a table with start and end times

    Args:
        series (pandas.Series): rain series
        ignore_rain_below (float): where it is considered as rain
        min_gap (pandas.Timedelta): 4 hours of no rain between events

    Returns:
        pandas.DataFrame: table of the rain events
    """
    # best OKOSTRA adjustment with 0.0
    # by ignoring 0.1 mm the results are getting bigger

    # remove values below a from the database
    temp = series[series > ignore_rain_below].index.to_series()

    # 4 hours of no rain between events

    event_end = temp[temp.diff(periods=-1) < -min_gap]
    event_end = event_end.append(temp.tail(1), ignore_index=True)

    event_start = temp[temp.diff() > min_gap]
    event_start = event_start.append(temp.head(1), ignore_index=True)
    event_start = event_start.sort_values().reset_index(drop=True)

    events = pd.concat([event_start, event_end], axis=1, ignore_index=True)
    events.columns = [COL.START, COL.END]
    return events


def event_number_to_series(events, index):
    """
    make a time-series where the value of the event number is paste to the <index>

    Args:
        events (pandas.DataFrame):
        index (pandas.DatetimeIndex):

    Returns:
        pandas.Series:
    """
    ts = pd.Series(index=index)

    events_dict = events.to_dict(orient='index')
    for event_no, event in events_dict.items():
        ts[event[COL.START]: event[COL.END]] = event_no

    return ts


########################################################################################################################
def agg_events(events, series, agg='sum'):
    """

    Args:
        events (pandas.DataFrame): table of events
        series (pandas.Series): time-series data
        agg (str | function): aggregation of time-series

    Returns:
        numpy.ndarray: result of function of every event
    """
    if events.empty:
        return np.array([])

    if events.index.size > 3500:
        res = series.groupby(event_number_to_series(events, series.index)).agg(agg).values
    else:
        res = events.apply(lambda event: series[event[COL.START]:event[COL.END]].agg(agg), axis=1).values
    return res


########################################################################################################################
def event_duration(events):
    """
    calculate the event duration

    Args:
        events (pandas.DataFrame): table of events with COL.START and COL.END times

    Returns:
        pandas.Series: duration of each event
    """
    return events[COL.END] - events[COL.START]


########################################################################################################################
def rain_bar_plot(rain, ax=None, color=None, reverse=False):
    """
    make a standard precipitation/rain plot

    Args:
        rain (pandas.Series):
        ax (matplotlib.axes.Axes):
        ylabel (str):
        color (str):
        reverse (bool):

    Returns:
        matplotlib.axes.Axes: rain plot
    """
    if color is None:
        color = '#1E88E5'

    ax = rain.plot(ax=ax, drawstyle='steps-mid', color=color, solid_capstyle='butt', solid_joinstyle='miter')
    ax.fill_between(rain.index, rain.values, 0, step='mid', zorder=1000, color=color)

    if reverse:
        # ax.set_ylim(top=0, bottom=rain.max() * 1.1)
        ax.set_ylim(bottom=0)
        ax.invert_yaxis()
    else:
        ax.set_ylim(bottom=0)

    return ax


########################################################################################################################
def resample_rain_series(series):
    """

    Args:
        series (pandas.Series):

    Returns:
        tuple[pandas.Series, int]: the resampled series AND the final frequency in minutes
    """
    resample_minutes = (
        (pd.Timedelta(hours=5), 1),
        (pd.Timedelta(hours=12), 2),
        (pd.Timedelta(days=1), 5),
        (pd.Timedelta(days=2), 10),
        (pd.Timedelta(days=3), 15),
        (pd.Timedelta(days=4), 20)
    )

    dur = series.index[-1] - series.index[0]
    freq = guess_freq(series.index)

    minutes = 1
    for duration_limit, minutes in resample_minutes:
        if dur < duration_limit:
            break

    if freq.delta > pd.Timedelta(minutes=minutes):
        return series, int(freq / pd.Timedelta(minutes=1))

    # print('resample_rain_series: ', dur, duration_limit, minutes)
    return series.resample('{}T'.format(minutes)).sum(), minutes
