import datetime

import pandas as pd

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
        try:
            time_delta = pd.Timedelta(time_delta)
        except ValueError:
            time_delta = time_delta._pd_timedelta
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
        return f'{minutes:0.0f} min'
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

    if 2 array-like parameters are give, an element-wise calculation will be made.
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

    if 2 array-like parameters are give, an element-wise calculation will be made.
    So the length of the array must be the same.

    Args:
        rain_flow_rate (float | np.ndarray | pd.Series): in [l/(s*ha)]
        duration (float | np.ndarray | pd.Series): in minutes

    Returns:
        float | np.ndarray | pd.Series: height of rainfall in [mm]
    """
    return rain_flow_rate * duration / (1000 / 6)


def get_progress_bar(iterator, desc=None):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterator, desc=desc)
    except ModuleNotFoundError:
        return iterator


def frame_looper(size, columns, label='return periods'):
    if size > 30000:  # if > 3 weeks, use a progressbar
        return get_progress_bar(columns, desc=f'calculating {label} data-frame')
    else:
        return columns


def event_caption_eng(event, unit='mm'):
    """
    Generates a human-readable caption for a rain event.

    The caption includes details such as the event's start and end times, total rainfall, duration,
    and maximum return period (if available). The output is formatted for readability.

    Args:
        event (dict or pandas.Series): A dictionary or Series containing event details. Expected keys include:
            - COL.START: Start time of the event.
            - COL.END: End time of the event.
            - COL.LP: Total rainfall sum (optional).
            - COL.DUR: Duration of the event (optional).
            - COL.MAX_PERIOD: Maximum return period (optional).
            - COL.MAX_PERIOD_DURATION: Duration of the maximum return period (optional).
        unit (str, optional): Unit for rainfall (default: 'mm').

    Returns:
        str: A formatted string describing the rain event.

    Example:
        Given an event with:
            - COL.START = pd.Timestamp('2023-01-01 12:00')
            - COL.END = pd.Timestamp('2023-01-01 14:00')
            - COL.LP = 15.5
            - COL.DUR = pd.Timedelta(hours=2)
            - COL.MAX_PERIOD = 10
            - COL.MAX_PERIOD_DURATION = pd.Timedelta(minutes=30)

        The output might look like:
            "rain event
             between 2023-01-01 12:00 and 14:00
             with a total sum of 15.5 mm
             and a duration of 2 hours.
             The maximum return period was 10 years
             at a duration of 30 minutes."
    """
    return event_caption(event, unit=unit, lang='en')


def event_caption_ger(event, unit='mm'):
    """
    Generates a human-readable caption for a rain event in german.

    The caption includes details such as the event's start and end times, total rainfall, duration,
    and maximum return period (if available). The output is formatted for readability.

    Args:
        event (dict or pandas.Series): A dictionary or Series containing event details. Expected keys include:
            - COL.START: Start time of the event.
            - COL.END: End time of the event.
            - COL.LP: Total rainfall sum (optional).
            - COL.DUR: Duration of the event (optional).
            - COL.MAX_PERIOD: Maximum return period (optional).
            - COL.MAX_PERIOD_DURATION: Duration of the maximum return period (optional).
        unit (str, optional): Unit for rainfall (default: 'mm').

    Returns:
        str: A formatted string describing the rain event.

    Example:
        Given an event with:
            - COL.START = pd.Timestamp('2023-01-01 12:00')
            - COL.END = pd.Timestamp('2023-01-01 14:00')
            - COL.LP = 15.5
            - COL.DUR = pd.Timedelta(hours=2)
            - COL.MAX_PERIOD = 10
            - COL.MAX_PERIOD_DURATION = pd.Timedelta(minutes=30)

        The output might look like:
            "Regenereignis
             von 01.01.2023 12:00 bis 14:00
             mit einer Regensumme von 15.5 mm
             und einer Dauer von 2 Stunden.
             Die maximale Wiederkehrperiode war 10 a
             bei einer Dauerstufe von 30 minutes."
    """
    return event_caption(event, unit=unit, lang='de')


def event_caption(event, unit='mm', lang='en'):
    """
    Generates a human-readable caption for a rain event in English or German.

    Args:
        event (dict or pandas.Series): A dictionary or Series containing event details. Expected keys include:
            - COL.START: Start time of the event.
            - COL.END: End time of the event.
            - COL.LP: Total rainfall sum (optional).
            - COL.DUR: Duration of the event (optional).
            - COL.MAX_PERIOD: Maximum return period (optional).
            - COL.MAX_PERIOD_DURATION: Duration of the maximum return period in minutes (optional).
        unit (str, optional): Unit for rainfall (default: 'mm').
        lang (str, optional): Language ('en' for English, 'de' for German, default: 'en').

    Returns:
        str: A formatted string describing the rain event.

    Example:
        Given an event with:
            - COL.START = pd.Timestamp('2023-01-01 12:00')
            - COL.END = pd.Timestamp('2023-01-01 14:00')
            - COL.LP = 15.5
            - COL.DUR = pd.Timedelta(hours=2)
            - COL.MAX_PERIOD = 10
            - COL.MAX_PERIOD_DURATION = 30

        The german output might look like:
            "Regenereignis
             von 01.01.2023 12:00 bis 14:00
             mit einer Regensumme von 15.5 mm
             und einer Dauer von 2 Stunden.
             Die maximale Wiederkehrperiode war 10 a
             bei einer Dauerstufe von 30 minutes."

        The english output might look like:
            "rain event
             between 2023-01-01 12:00 and 14:00
             with a total sum of 15.5 mm
             and a duration of 2 hours.
             The maximum return period was 10 years
             at a duration of 30 minutes."
    """
    captions = {
        'en': {
            'event': 'rain event',
            'with_total': 'with a total sum of',
            'and_duration': 'and a duration of',
            'max_period': 'The maximum return period was',
            'at_duration': 'at a duration of',
            'years': 'years'
        },
        'de': {
            'event': 'Regenereignis',
            'with_total': 'mit einer Regensumme von',
            'and_duration': 'und einer Dauer von',
            'max_period': 'Die maximale Wiederkehrperiode war',
            'at_duration': 'bei einer Dauerstufe von',
            'years': 'a'
        }
    }

    cap = captions.get(lang, captions['en'])
    caption = f"{cap['event']}\n"

    if (COL.START in event) and (COL.END in event):
        start = event[COL.START]
        end = event[COL.END]

        if lang == 'de':
            if start.date() == end.date():
                # Beide Zeitpunkte sind am selben Tag
                caption += f"am {start.strftime('%d.%m.%Y von %H:%M')} bis {end.strftime('%H:%M')}\n"
            elif start.year == end.year:
                # Beide Zeitpunkte im selben Jahr
                if start.month == end.month:
                    # Beide Zeitpunkte im selben Jahr und Monat
                    caption += f"von {start.strftime('%d.%m.%Y %H:%M')} bis {end.strftime('%d. %H:%M')}\n"
                else:
                    caption += f"von {start.strftime('%d.%m.%Y %H:%M')} bis {end.strftime('%d.%m. %H:%M')}\n"
            else:
                # Unterschiedliche Jahre
                caption += f"von {start.strftime('%d.%m.%Y %H:%M')} bis {end.strftime('%d.%m.%Y %H:%M')}\n"
        else:
            caption += f'between {start:%Y-%m-%d %H:%M} and '
            if f'{start:%Y-%m-%d}' == f'{end:%Y-%m-%d}':
                # same day
                caption += f'{end:%H:%M}\n'
            elif f'{start:%Y-%m-}' == f'{end:%Y-%m-}':
                # same month
                caption += f'{end:%d %H:%M}\n'
            else:
                # different month
                caption += f'{end:%Y-%m-%d %H:%M}\n'

    if COL.LP in event:
        caption += f"{cap['with_total']} {event[COL.LP]:0.1f} {unit}\n"

    if COL.DUR in event:
        caption += f"{cap['and_duration']} {timedelta_readable(event[COL.DUR], lang=lang)}.\n"

    if COL.MAX_PERIOD in event:
        caption += f"{cap['max_period']} {return_period_formatter(event[COL.MAX_PERIOD])} {cap['years']}\n"

        if COL.MAX_PERIOD_DURATION in event:
            caption += f"{cap['at_duration']} {minutes_readable(event[COL.MAX_PERIOD_DURATION])}."

    return caption


def return_period_formatter(t):
    """
    Formats a return period value into a human-readable string.

    The function categorizes the return period value into specific ranges and returns a formatted string
    based on the range. This is useful for displaying return periods in a clear and concise manner.

    Args:
        t (float): The return period value to format.

    Returns:
        str: A formatted string representing the return period. The formatting rules are:
             - If t < 1: Returns "< 1".
             - If t > 200: Returns "$\\gg$ 100" (indicating much greater than 100).
             - If t > 100: Returns "> 100".
             - Otherwise: Returns the value formatted to one decimal place (e.g., "50.0").

    Example:
        >>> return_period_formatter(0.5)
        '< 1'
        >>> return_period_formatter(150)
        '> 100'
        >>> return_period_formatter(250)
        '$\\gg$ 100'
        >>> return_period_formatter(50.123)
        '50.1'
    """
    if t < 1:
        return '< 1'
    elif t > 200:
        return '$\\gg$ 100'
    elif t > 100:
        return '> 100'
    else:
        return f'{t:0.1f}'


def timedelta_components_plus(td, min_freq='min'):
    """
    Decomposes a timedelta into its components, approximating years and weeks.

    Args:
        td (datetime.timedelta or pandas.Timedelta): The time difference to decompose.
        min_freq (str, optional): The minimum frequency for rounding (e.g., 'min', 's'). Defaults to 'min'.

    Returns:
        list: A list of lists, where each sublist contains a numerical value and its corresponding time unit.

    Note:
        Leap years are not considered in year calculations.
        Possible components: [years, weeks, days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds]
    """
    list_of_components = []

    if isinstance(td, datetime.timedelta):
        td = pd.to_timedelta(td)

    # years, weeks
    days_year = 365
    days_week = 7

    for label_component, value in td.round(min_freq).components._asdict().items():
        if label_component == 'days':
            years, value = value // days_year, value % days_year
            list_of_components.append([int(years), 'years'])

            value -= years // 4

            weeks, value = value // days_week, value % days_week
            list_of_components.append([int(weeks), 'weeks'])

        list_of_components.append([value, label_component])
    return list_of_components


# Dictionary with English time units as keys and their German translations as values
time_units = {
    "years": "Jahre",
    "year": "Jahr",
    "weeks": "Wochen",
    "week": "Woche",
    "days": "Tage",
    "day": "Tag",
    "hours": "Stunden",
    "hour": "Stunde",
    "minutes": "Minuten",
    "minute": "Minute",
    "seconds": "Sekunden",
    "second": "Sekunde",
    "milliseconds": "Millisekunden",
    "millisecond": "Millisekunde",
    "microseconds": "Mikrosekunden",
    "microsecond": "Mikrosekunde",
    "nanoseconds": "Nanosekunden",
    "nanosecond": "Nanosekunde"
}


def timedelta_components_readable(list_of_components, short=False, sep=', ', lang='en'):
    """
    Converts a list of time components into a human-readable string.

    Args:
        list_of_components (list): A list of [value, unit] pairs representing time components.
        short (bool, optional): If True, uses abbreviated unit names (e.g., 'y' for years). Defaults to False.
        sep (str, optional): Separator between components. Defaults to ', '.
        lang (str, optional): Language ('en' for English, 'de' for German, default: 'en').

    Returns:
        str: A formatted string representing the time components, with the last component joined by "and".

    Note:
        - Singular units (e.g., "1 year" instead of "1 years") are handled automatically.
        - The last separator is replaced with "and" for better readability unless short mode is enabled.

    Example:
        timedelta_components_readable([(2, 'days'), (3, 'hours')]) -> '2 days and 3 hours'
    """
    result = []

    for value, label_component in list_of_components:
        if value > 0:
            if short:
                unit_sep = ''
                unit = label_component[0]
            else:
                unit_sep = ' '
                unit = label_component
                if value == 1:
                    unit = label_component[:-1]
            if lang == 'de':
                unit = time_units.get(unit, unit)
            result.append(f'{value}{unit_sep}{unit}')

    s = sep.join(result)

    if not short:
        # replace last "," with "and"
        s = ' and '.join(s.rsplit(sep, 1))
    return s


def timedelta_readable(td, min_freq='min', short=False, sep=', ', lang='en'):
    """
    Converts a timedelta into a human-readable string.

    Args:
        td (datetime.timedelta or pandas.Timedelta): The time difference to format.
        min_freq (str, optional): The minimum frequency for rounding (e.g., 'min', 's'). Defaults to 'min'.
        short (bool, optional): Whether to use abbreviated unit names (e.g., 'h' for hours). Defaults to False.
        sep (str, optional): Separator used between components in the output string. Defaults to ', '.
        lang (str, optional): Language ('en' for English, 'de' for German, default: 'en').

    Returns:
        str: A formatted string representing the time duration.

    Note:
        Leap years are not considered in year calculations.

    Example:
        timedelta_readable(pd.Timedelta(days=400, hours=5)) -> '1 year, 5 weeks and 5 hours'
    """
    return timedelta_components_readable(timedelta_components_plus(td, min_freq), short=short, sep=sep, lang=lang)


def timedelta_readable2(d1, d2, min_freq='min', short=False, sep=', ', lang='en'):
    """
    Computes the difference between two dates and returns a human-readable string.

    Args:
        d1 (datetime-like): The start date.
        d2 (datetime-like): The end date.
        min_freq (str, optional): The minimum frequency for rounding (e.g., 'min', 's'). Defaults to 'min'.
        short (bool, optional): Whether to use abbreviated unit names (e.g., 'h' for hours). Defaults to False.
        sep (str, optional): Separator used between components in the output string. Defaults to ', '.
        lang (str, optional): Language ('en' for English, 'de' for German, default: 'en').

    Returns:
        str: A formatted string representing the time difference.

    Note:
        - Approximates the number of years by adjusting for the closest full year.
        - Leap years are not considered in year calculations.

    Example:
        timedelta_readable2(pd.Timestamp('2020-01-01'), pd.Timestamp('2023-06-15'))
        -> '3 years, 5 months and 14 days'
    """
    td = d2 - d1

    years = None
    if td > pd.Timedelta(days=365):
        d2_new = d2.replace(year=d1.year)

        if d2_new < d1:
            d2_new = d2_new.replace(year=d1.year + 1)

        years = d2.year - d2_new.year

        td = d2_new - d1

    l = timedelta_components_plus(td, min_freq)

    if years is not None:
        l[0][0] = years

    return timedelta_components_readable(l, short=short, sep=sep, lang=lang)
