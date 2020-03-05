import pandas as pd


def delta2min(time_delta):
    """
    convert timedelta to float in minutes

    Args:
        time_delta (pandas.Timedelta):

    Returns:
        float: the timedelta in minutes
    """
    # time_delta.total_seconds() / 60
    return time_delta / pd.Timedelta(minutes=1)


def minutes_readable(minutes):
    """
    convert the duration in minutes to a more readable form

    Args:
        minutes (float | int): duration in minutes

    Returns:
        str: duration as a string
    """
    if minutes <= 60:
        return '{:0.0f}min'.format(minutes)
    elif 60 < minutes < 60 * 24:
        minutes /= 60
        if minutes % 1:
            fmt = '{:0.1f}h'
        else:
            fmt = '{:0.0f}h'
        return fmt.format(minutes)
    elif 60 * 24 <= minutes:
        minutes /= 60 * 24
        if minutes % 1:
            fmt = '{:0.1f}d'
        else:
            fmt = '{:0.0f}d'
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
    duration_strings = list()
    for i, minutes in enumerate(durations):
        duration_strings.append(minutes_readable(minutes))
    return duration_strings