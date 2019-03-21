__author__ = "David Camhy, Markus Pichler"
__copyright__ = "Copyright 2018, University of Technology Graz"
__credits__ = ["David Camhy", "Markus Pichler"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Camhy, Markus Pichler"

import pytz
from tzlocal import get_localzone
from datetime import tzinfo
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import _delta_to_tick as delta_to_freq
from pandas import DatetimeIndex, DateOffset, Timedelta, DataFrame


class TimezoneError(Exception):
    """Some Error With a Timezone"""


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


def remove_timezone(df):
    """
    convert the timezone to wintertime and then remove the timezone from the dataframe index, to not get in conflict
    with the plot functions

    :param df: data
    :type df: DataFrame

    :return: converted data
    :rtype: DataFrame
    """
    no_tz = df.copy()
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
    return most often frequency in the format [minutes]T eg: "1T" when the frequency is one minute

    :param DatetimeIndex date_time_index:
    :param Timedelta default:
    :rtype: DateOffset
    """

    # ---------------------------------
    def _get_freq(freq):
        if isinstance(freq, str):
            freq = to_offset(freq)

        return freq

    # ---------------------------------
    freq = date_time_index.freq
    if pd.notnull(freq):
        return _get_freq(freq)

    if not len(date_time_index) <= 3:
        freq = pd.infer_freq(date_time_index)  # 'T'

        if pd.notnull(freq):
            return _get_freq(freq)

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

    freq = delta_to_freq(delta)
    return _get_freq(freq)


########################################################################################################################
def year_delta(years):
    return pd.Timedelta(days=365.25 * years)


########################################################################################################################
def rain_events(series, ignore_rain_below=0, min_gap=pd.Timedelta(hours=4)):
    """
    get rain events as a table with start and end times

    :param series: rain series
    :type series: pd.Series

    :param ignore_rain_below: where it is considered as rain
    :type ignore_rain_below: float

    :param min_gap: 4 hours of no rain between events
    :type min_gap: pd.Timedelta

    :return: table of the rain events
    :rtype: pd.DataFrame
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
    events.columns = ['start', 'end']
    return events


########################################################################################################################
def agg_events(events, series, agg='sum'):
    """

    :param events: table of events
    :type events: pd.DataFrame

    :param series: timeseries data
    :type series: pd.Series

    :param agg: aggregation of timeseries
    :type agg: str | function

    :return: result of function of every event
    :rtype: pd.Series
    """

    def _agg_event(event):
        return series[event['start']:event['end']].agg(agg)

    return events.apply(_agg_event, axis=1)
