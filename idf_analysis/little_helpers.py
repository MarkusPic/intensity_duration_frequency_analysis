__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
from pandas import Timedelta
from tqdm.auto import tqdm

from .definitions import COL


def delta2min(time_delta):
    """
    convert timedelta to float in minutes

    Args:
        time_delta (pandas.Timedelta, pandas.DateOffset):

    Returns:
        float: the timedelta in minutes
    """
    if isinstance(time_delta, pd.DateOffset):
        time_delta = time_delta.delta
    return int(time_delta.total_seconds() / 60)


def minutes_readable(minutes):
    """
    convert the duration in minutes to a more readable form

    Args:
        minutes (float | int): duration in minutes

    Returns:
        str: duration as a string
    """
    if minutes <= 60:
        return '{:0.0f} min'.format(minutes)
    elif 60 < minutes < 60 * 24:
        minutes /= 60
        if minutes % 1:
            fmt = '{:0.1f} h'
        else:
            fmt = '{:0.0f} h'
        return fmt.format(minutes)
    elif 60 * 24 <= minutes:
        minutes /= 60 * 24
        if minutes % 1:
            fmt = '{:0.1f} d'
        else:
            fmt = '{:0.0f} d'
        return fmt.format(minutes)
    else:
        return str(minutes)


def duration_steps_readable(durations):
    """
    convert the durations to a more readable form

    Args:
        durations (list[int | float]): in minutes

    Returns:
        list[str]: of the readable duration list
    """
    return [minutes_readable(i) for i in durations]


def height2rate(height_of_rainfall, duration):
    """
    calculate the specific rain flow rate in [l/(s*ha)]
    if 2 array-like parameters are give, a element-wise calculation will be made.
    So the length of the array must be the same.

    Args:
        height_of_rainfall (float | np.ndarray | pd.Series): height_of_rainfall: in [mm]
        duration (float | np.ndarray | pd.Series): in minutes

    Returns:
        float | np.ndarray | pd.Series: specific rain flow rate in [l/(s*ha)]
    """
    return height_of_rainfall / duration * (1000 / 6)


def rate2height(rain_flow_rate, duration):
    """
    convert the rain flow rate to the height of rainfall in [mm]
    if 2 array-like parameters are give, a element-wise calculation will be made.
    So the length of the array must be the same.

    Args:
        rain_flow_rate (float | np.ndarray | pd.Series): in [l/(s*ha)]
        duration (float | np.ndarray | pd.Series): in minutes

    Returns:
        float | np.ndarray | pd.Series: height of rainfall in [mm]
    """
    return rain_flow_rate * duration / (1000 / 6)


def frame_looper(size, columns, label='return periods'):
    if size > 30000:  # if > 3 weeks, use a progressbar
        return tqdm(columns, desc=f'calculating {label} data-frame')
    else:
        return columns


def event_caption(event, unit='mm'):
    caption = 'rain event\n'
    if (COL.START in event) and (COL.END in event):
        caption += f'between {event[COL.START]:%Y-%m-%d %H:%M} and '
        if f'{event[COL.START]:%Y-%m-%d}' == f'{event[COL.END]:%Y-%m-%d}':
            caption += f'{event[COL.END]:%H:%M}\n'
        elif f'{event[COL.START]:%Y-%m-}' == f'{event[COL.END]:%Y-%m-}':
            caption += f'{event[COL.END]:%d %H:%M}\n'
        else:
            caption += f'{event[COL.END]:%Y-%m-%d %H:%M}\n'

    if COL.LP in event:
        caption += f'with a total sum of {event[COL.LP]:0.1f} {unit}\n'

    if COL.DUR in event:
        caption += f' and a duration of {timedelta_readable(event[COL.DUR])}'

    caption += '.\n'

    if COL.MAX_PERIOD in event:
        caption += f' The maximum return period was {return_period_formatter(event[COL.MAX_PERIOD])} a\n'

        if COL.MAX_PERIOD_DURATION in event:
            caption += f' at a duration of {minutes_readable(event[COL.MAX_PERIOD_DURATION])}.'

    return caption


def return_period_formatter(t):
    if t < 1:
        return '< 1'
    elif t > 200:
        return '$\\gg$ 100'
    elif t > 100:
        return '> 100'
    else:
        return f'{t:0.1f}'


def timedelta_components_plus(td, min_freq='T'):
    """Schaltjahre nicht miteinbezogen"""
    l = []

    # years, weeks
    days_year = 365
    days_week = 7

    for component, value in td.round(min_freq).components._asdict().items():
        if component == 'days':
            years, value = value // days_year, value % days_year
            l.append([int(years), 'years'])

            value -= years // 4

            weeks, value = value // days_week, value % days_week
            l.append([int(weeks), 'weeks'])

        l.append([value, component])
    return l


def timedelta_components_readable(l, short=False, sep=', '):
    s = sep.join(
        ['{}{}{}'.format(v, '' if short else ' ', c[0] if short else (c if v > 1 else c[:-1])) for v, c in l if v > 0])
    if not short:
        # replace last "," with "and"
        s = ' and '.join(s.rsplit(sep, 1))
    return s


def timedelta_readable(td, min_freq='T', short=False, sep=', '):
    """Schaltjahre nicht miteinbezogen"""
    return timedelta_components_readable(timedelta_components_plus(td, min_freq), short=short, sep=sep)


def timedelta_readable2(d1, d2, min_freq='T', short=False, sep=', '):
    td = d2 - d1

    years = None
    if td > Timedelta(days=365):
        d2_new = d2.replace(year=d1.year)

        if d2_new < d1:
            d2_new = d2_new.replace(year=d1.year + 1)

        years = d2.year - d2_new.year

        td = d2_new - d1

    l = timedelta_components_plus(td, min_freq)

    if years is not None:
        l[0][0] = years

    return timedelta_components_readable(l, short=short, sep=sep)
