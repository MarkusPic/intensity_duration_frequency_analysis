import warnings

from math import floor, e

import numpy as np
import pandas as pd
from tqdm import tqdm

from .definitions import *
from .sww_utils import year_delta, guess_freq, rain_events, agg_events


def annual_series(rolling_sum_values, year_index):
    """
    create an annual series of the maximum overlapping sum per year and calculate the "u" and "w" parameters
    acc. to DWA-A 531 chap. 5.1.5

    Args:
        rolling_sum_values (numpy.ndarray): array with maximum rolling sum per event per year.
        year_index (numpy.ndarray): array with year of the event.

    Returns:
        tuple[float, float]: parameter u and w from the annual series for a specific duration step as a tuple
    """
    annually_series = pd.Series(rolling_sum_values).groupby(year_index).max().values
    # annually_series = pd.Series(data=rolling_sum_values,
    #                             index=events[COL.START].values).resample('AS').max().index
    annually_series = np.sort(annually_series)[::-1]

    mean_sample_rainfall = annually_series.mean()
    sample_size = annually_series.size

    index = np.arange(sample_size) + 1
    x = -np.log(np.log((sample_size + 0.2) / (sample_size - index + 0.6)))
    x_mean = x.mean()

    w = ((x * annually_series).sum() - sample_size * mean_sample_rainfall * x_mean) / \
        ((x ** 2).sum() - sample_size * x_mean ** 2)
    u = mean_sample_rainfall - w * x_mean

    return {PARAM.U: u, PARAM.W: w}


def _plotting_formula(k, l, m):
    """
    plotting function acc. to DWA-A 531 chap. 5.1.3 for the partial series

    Args:
        k (float): running index
        l (float): sample size
        m (float): measurement period

    Returns:
        float: estimated empirical return period
    """
    return (l + 0.2) * m / ((k - 0.4) * l)


def partial_series(rolling_sum_values, measurement_period):
    """
    create an partial series of the largest overlapping sums and calculate the "u" and "w" parameters
    acc. to DWA-A 531 chap. 5.1.4

    Args:
        rolling_sum_values (numpy.ndarray): array with maximum rolling sum per event
        measurement_period (float): in years

    Returns:
        tuple[float, float]: parameter u and w from the partial series for a specific duration step as a tuple
    """
    partially_series = rolling_sum_values
    partially_series = np.sort(partially_series)[::-1]

    # use only the (2-3 multiplied with the number of measuring years) of the biggest
    # values in the database (-> acc. to ATV-A 121 chap. 4.3; DWA-A 531 chap. 4.4)
    # as an requirement for the extreme value distribution
    threshold_sample_size = int(floor(measurement_period * e))
    partially_series = partially_series[:threshold_sample_size]

    mean_sample_rainfall = partially_series.mean()
    sample_size = threshold_sample_size
    index = np.arange(sample_size) + 1
    log_return_periods = np.log(_plotting_formula(index, sample_size, measurement_period))
    ln_t_n_mean = log_return_periods.mean()

    w = ((log_return_periods * partially_series).sum() - sample_size * mean_sample_rainfall * ln_t_n_mean) / \
        ((log_return_periods ** 2).sum() - sample_size * ln_t_n_mean ** 2)

    u = mean_sample_rainfall - w * ln_t_n_mean

    return {PARAM.U: u, PARAM.W: w}


def _improve_factor(interval):
    """
    correction factor acc. to DWA-A 531 chap. 4.3

    Args:
        interval (float): length of the interval: number of observations per duration

    Returns:
        float: correction factor
    """
    improve_factor = {1: 1.14,
                      2: 1.07,
                      3: 1.04,
                      4: 1.03,
                      5: 1.00,
                      6: 1.00}

    return np.interp(interval,
                     list(improve_factor.keys()),
                     list(improve_factor.values()))


def calculate_u_w(file_input, duration_steps, series_kind):
    """
    statistical analysis for each duration step acc. to DWA-A 531 chap. 5.1
    save the parameters of the distribution function as interim results
    acc. to DWA-A 531 chap. 4.4: use the annual series only for measurement periods over 20 years


    Args:
        file_input (pandas.Series): precipitation data
        duration_steps (list[int] | numpy.ndarray): in minutes
        series_kind (str): 'annual' or 'partial'

    Returns:
        pandas.DataFrame: with key=durations and values=dict(u, w)
    """
    ts = file_input.copy()
    # -------------------------------
    # measuring time in years
    measurement_start, measurement_end = ts.index[[0, -1]]
    measurement_period = (measurement_end - measurement_start) / year_delta(years=1)
    if round(measurement_period, 1) < 10:
        warnings.warn("The measurement period is too short. The results may be inaccurate! "
                      "It is recommended to use at least ten years. "
                      "(-> Currently {}a used)".format(measurement_period))

    # -------------------------------
    base_frequency = guess_freq(ts.index)  # DateOffset/Timedelta

    # ------------------------------------------------------------------------------------------------------------------
    interim_results = dict()

    # -------------------------------
    # acc. to DWA-A 531 chap. 4.2:
    # The values must be independent of each other for the statistical evaluations.
    # estimated four hours acc. (SCHILLING 1984)
    # for larger durations - use the duration as minimal gap
    #
    # use only duration for splitting events
    # may increase design-rain-height of smaller durations
    #
    # -------------------------------
    pbar = tqdm(duration_steps, desc='Calculating Parameters u and w')
    for duration_integer in pbar:
        pbar.set_description('Calculating Parameters u and w for duration {:0.0f}'.format(duration_integer))

        duration = pd.Timedelta(minutes=duration_integer)

        if duration < pd.Timedelta(base_frequency):
            continue

        events = rain_events(ts, min_gap=duration)

        # correction factor acc. to DWA-A 531 chap. 4.3
        improve = _improve_factor(duration / base_frequency)

        roll_sum = ts.rolling(duration).sum()

        # events[COL.rolling_sum_valuesAX_OVERLAPPING_SUM] = agg_events(events, roll_sum, 'max') * improve
        rolling_sum_values = agg_events(events, roll_sum, 'max') * improve

        if series_kind == ANNUAL:
            interim_results[duration_integer] = annual_series(rolling_sum_values, events[COL.START].year.values)
        elif series_kind == PARTIAL:
            interim_results[duration_integer] = partial_series(rolling_sum_values, measurement_period)
        else:
            raise NotImplementedError

    # -------------------------------
    interim_results = pd.DataFrame.from_dict(interim_results, orient='index')
    interim_results.index.name = COL.DUR
    return interim_results
