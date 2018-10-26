__author__ = "David Camhy, Markus Pichler"
__copyright__ = "Copyright 2018, University of Technology Graz"
__credits__ = ["David Camhy", "Markus Pichler"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Camhy, Markus Pichler"

import pytz
from tzlocal import get_localzone
from pandas import DataFrame, DatetimeIndex, Series
from datetime import tzinfo
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import _delta_to_tick as delta_to_freq
from pandas import DatetimeIndex, Series, DateOffset, Timedelta


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

def convert_index_timezone(index, timezone=get_localzone()):
    """
    convert timezone of the DatetimeIndex into a given timezone

    :param index: index
    :type index: DatetimeIndex

    :param timezone: timezone in which the index will be converted - default: timezone of the computer
    :type timezone: tzinfo | str

    :return: converted index
    :rtype: DatetimeIndex
    """
    if index.tz is None:
        raise TimezoneError("Dataframe must have a timezone information")
    return index.tz_convert(check_tz(timezone))


def remove_index_timezone(index, native_timezone=pytz.timezone('Etc/GMT-1')):
    idx = convert_index_timezone(index, timezone=check_tz(native_timezone))
    idx = idx.tz_localize(None)
    return idx


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
    no_tz.index = remove_index_timezone(no_tz.index)
    return no_tz


########################################################################################################################
########################################################################################################################
########################################################################################################################
def tick_printable(tick):
    """
    converts DateOffset format "<x * freq>" to "x freq" for better readability

    :type  tick: DateOffset
    :rtype: str
    """
    return str(tick).replace('<', '').replace('>', '').replace('* ', '')


########################################################################################################################
def time_steps(date_time_index):
    """
    get all time steps existing within the DataFrame

    :type date_time_index: DatetimeIndex
    :rtype: Series[Timedelta]
    """
    return date_time_index.to_series().diff(periods=1).fillna(method='backfill')


def guess_freq(date_time_index, to_str=False, default=pd.Timedelta(minutes=1)):
    """
    return most often frequency in the format [minutes]T eg: "1T" when the frequency is one minute

    :param DatetimeIndex date_time_index:
    :param bool to_str:
    :param Timedelta default:
    :rtype: DateOffset
    """
    # ---------------------------------
    def _get_freq(freq, to_str):
        if isinstance(freq, str):
            freq = to_offset(freq)
        if to_str:
            return tick_printable(freq)
        else:
            return freq

    # ---------------------------------
    freq = date_time_index.freq
    if pd.notnull(freq):
        return _get_freq(freq, to_str)

    if not len(date_time_index) <= 3:
        freq = pd.infer_freq(date_time_index)  # 'T'

        if pd.notnull(freq):
            return _get_freq(freq, to_str)

        delta_series = time_steps(date_time_index)
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
    return _get_freq(freq, to_str)


########################################################################################################################
def year_delta(years):
    return pd.Timedelta(days=365.2425 * years)


########################################################################################################################
def rain_events(series, ignore_rain_below=0.001, min_gap=pd.Timedelta(hours=4)):
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
