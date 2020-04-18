__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import warnings

import pandas as pd
import numpy as np
from math import e, floor
from collections import OrderedDict
from tqdm import tqdm

from .sww_utils import guess_freq, rain_events, agg_events, year_delta
from .definitions import DWA, ATV, DWA_adv, PARTIAL, ANNUAL, LOG1, LOG2, HYP, LIN, COL, PARAM, PARAM_COL


########################################################################################################################
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


########################################################################################################################
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


########################################################################################################################
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


########################################################################################################################
def calculate_u_w(file_input, duration_steps, series_kind):
    """
    statistical analysis for each duration step acc. to DWA-A 531 chap. 5.1
    save the parameters of the distribution function as interim results
    acc. to DWA-A 531 chap. 4.4: use the annual series only for measurement periods over 20 years


    Args:
        file_input (pandas.Series): precipitation data
        duration_steps (list[int] | numpy.ndarray): in minutes
        series_kind (str): annual or partial

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

        # events[COL.Mrolling_sum_valuesAX_OVERLAPPING_SUM] = agg_events(events, roll_sum, 'max') * improve
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


########################################################################################################################
def folded_log_formulation(duration, param, case, param_mean=None, duration_mean=None):
    """

    Args:
        duration:
        param:
        case:
        param_mean:
        duration_mean:

    Returns:

    """
    if param_mean and duration_mean:
        mean_ln_duration = np.log(duration_mean)
        mean_ln_param = np.log(param_mean)
    else:
        param_mean = param.mean()
        mean_ln_param = np.log(param).mean()
        mean_ln_duration = np.log(duration).mean()

    divisor = ((np.log(duration) - mean_ln_duration) ** 2).sum()

    if case == LOG2:
        # for the twofold formulation
        b = ((np.log(param) - mean_ln_param) * (np.log(duration) - mean_ln_duration)).sum() / divisor
        a = mean_ln_param - b * mean_ln_duration

    elif case == LOG1:
        # for the onefold formulation
        b = ((param - param_mean) * (np.log(duration) - mean_ln_duration)).sum() / divisor
        a = param_mean - b * mean_ln_duration

    else:
        raise NotImplementedError

    return float(a), float(b)


########################################################################################################################
def hyperbolic_formulation(duration, param, a_start=20.0, b_start=15.0, param_mean=None, duration_mean=None):
    """

    Args:
        duration:
        param:
        a_start:
        b_start:
        param_mean:
        duration_mean:

    Returns:

    """

    # ------------------------------------------------------------------------------------------------------------------
    def get_param(dur, par, a_, b_):
        i = -a_ / (dur + b_)
        if param_mean:
            i_mean = - param_mean / duration_mean
            param_mean_ = param_mean
        else:
            i_mean = i.mean()
            param_mean_ = par.mean()

        b_ = ((par - param_mean_) * (i - i_mean)).sum() / ((i - i_mean) ** 2).sum()
        a_ = param_mean_ - b_ * i_mean
        return a_, b_

    # ------------------------------------------------------------------------------------------------------------------
    iteration_steps = 0

    a = a_start
    b = b_start

    conditions = True
    while conditions:
        conditions = False

        a_s = a
        b_s = b
        a, b = get_param(duration, param, a, b)
        conditions = (abs(a - a_s) > 0.005) or (abs(b - b_s) > 0.005) or conditions
        a = (a + a_s) / 2
        b = (b + b_s) / 2

        iteration_steps += 1
    return float(a), float(b)


########################################################################################################################
def get_duration_steps(worksheet):
    """
    duration step boundary for the various distribution functions in minutes

    Args:
        worksheet (str):

    Returns:
        tuple[int, int]: duration steps in minutes
    """
    return {
        # acc. to ATV-A 121 chap. 5.2 (till 2012)
        ATV: (60 * 3, 60 * 48),
        # acc. to DWA-A 531 chap. 5.2.1
        DWA_adv: (60 * 3, 60 * 24),
        # acc. to DWA-A 531 chap. 5.2.1
        DWA: (60, 60 * 12)
    }[worksheet]


########################################################################################################################
def get_approaches(worksheet):
    """
    approaches depending on the duration and the parameter

    Args:
        worksheet (str): worksheet name for the analysis:
            - 'DWA-A_531'
            - 'ATV-A_121'
            - 'DWA-A_531_advektiv' (yet not implemented)

    Returns:
        list[dict]: table of approaches depending on the duration and the parameter
    """
    approach_list = list()

    # acc. to ATV-A 121 chap. 5.2.1
    if worksheet == ATV:
        approach_list.append(OrderedDict({PARAM_COL.FROM: None,
                                          PARAM_COL.TO: None,
                                          PARAM_COL.U: LOG2,
                                          PARAM_COL.W: LOG1}))

    elif worksheet == DWA:
        duration_bound_1, duration_bound_2 = get_duration_steps(worksheet)

        approach_list.append(OrderedDict({PARAM_COL.FROM: 0,
                                          PARAM_COL.TO: duration_bound_1,
                                          PARAM_COL.U: HYP,
                                          PARAM_COL.W: LOG2}))
        approach_list.append(OrderedDict({PARAM_COL.FROM: duration_bound_1,
                                          PARAM_COL.TO: duration_bound_2,
                                          PARAM_COL.U: LOG2,
                                          PARAM_COL.W: LOG2}))
        approach_list.append(OrderedDict({PARAM_COL.FROM: duration_bound_2,
                                          PARAM_COL.TO: np.inf,
                                          PARAM_COL.U: LIN,
                                          PARAM_COL.W: LIN}))

    else:
        raise NotImplementedError

    return approach_list


def split_interim_results(parameter, interim_results):
    new_parameters = list()
    for i, row in enumerate(parameter):
        it_res = interim_results.loc[row[PARAM_COL.FROM]: row[PARAM_COL.TO]]
        if it_res.empty or it_res.index.size == 1:
            continue
        row[COL.DUR] = list(map(int, it_res.index.values))
        for p in PARAM.U_AND_W:  # u or w
            row[PARAM_COL.VALUES(p)] = list(map(float, it_res[p].values))
        new_parameters.append(row)

    return new_parameters


########################################################################################################################
def _calc_params(parameter, params_mean=None, duration_mean=None):
    """
    calculate parameters a_u, a_w, b_u and b_w and add it to the dict

    Args:
        parameter (list[dict]):
        params_mean (dict[float]):
        duration_mean (float):

    Returns:
        list[dict]: parameters
    """
    for row in parameter:
        if not COL.DUR in row:
            continue

        dur = np.array(row[COL.DUR])

        for p in PARAM.U_AND_W:  # u or w
            a_label = PARAM_COL.A(p)  # a_u or a_w
            b_label = PARAM_COL.B(p)  # b_u or b_w

            param = np.array(row[PARAM_COL.VALUES(p)])  # values for u or w of the annual/partial series

            approach = row[p]
            if params_mean:
                param_mean = params_mean[p]
            else:
                param_mean = None

            # ----------------------------
            if approach in [LOG1, LOG2]:
                row[a_label], row[b_label] = folded_log_formulation(dur, param, case=approach,
                                                                    param_mean=param_mean, duration_mean=duration_mean)

            # ----------------------------
            elif approach == HYP:
                a_start = 20.0
                if a_label in row and not np.isnan(row[a_label]):
                    a_start = row[a_label]

                b_start = 15.0
                if b_label in row and not np.isnan(row[b_label]):
                    b_start = row[b_label]

                row[a_label], row[b_label] = hyperbolic_formulation(dur, param, a_start=a_start, b_start=b_start,
                                                                    param_mean=param_mean, duration_mean=duration_mean)

            # ----------------------------
            elif approach == LIN:
                pass

            # ----------------------------
            else:
                raise NotImplementedError

            # ----------------------------

    return parameter


########################################################################################################################
def get_parameters(interim_results, worksheet=DWA):
    """
    get calculation parameters

    Args:
        interim_results (pandas.DataFrame):
        worksheet (str):

    Returns:
        list[dict]: parameters
    """
    parameter = get_approaches(worksheet)
    parameter = split_interim_results(parameter, interim_results)
    parameter = _calc_params(parameter)

    # -------------------------------------------------------------
    # the balance between the different duration ranges acc. to DWA-A 531 chap. 5.2.4
    duration_step = parameter[0][PARAM_COL.TO]
    durations = np.array([duration_step - 0.001, duration_step + 0.001])

    if any(durations < interim_results.index.values.min()) or any(durations > interim_results.index.values.max()):
        return parameter

    u, w = get_u_w(durations, parameter)
    parameter = _calc_params(parameter, params_mean=dict(u=np.mean(u), w=np.mean(w)), duration_mean=duration_step)
    # -------------------------------------------------------------
    return parameter


########################################################################################################################
def get_row(duration, parameter):
    for row in parameter:
        if row[PARAM_COL.FROM] <= duration <= row[PARAM_COL.TO]:
            return row


def get_scalar_param(p, duration, parameter):
    """

    Args:
        duration (float | int): in minutes
        parameter (list[dict]):

    Returns:
        (float, float): u, w
    """
    row = get_row(duration, parameter)

    if row is None:
        return np.NaN

    approach = row[p]

    if approach == LOG1:
        a = row[PARAM_COL.A(p)]
        b = row[PARAM_COL.B(p)]
        return a + b * np.log(duration)

    elif approach == LOG2:
        a = row[PARAM_COL.A(p)]
        b = row[PARAM_COL.B(p)]
        return np.exp(a) * np.power(duration, b)

    elif approach == HYP:
        a = row[PARAM_COL.A(p)]
        b = row[PARAM_COL.B(p)]
        return a * duration / (duration + b)

    elif approach == LIN:
        return np.interp(duration, row[COL.DUR], row[PARAM_COL.VALUES(p)])


def get_array_param(p, duration, parameter):
    """

    Args:
        duration (numpy.ndarray): in minutes
        parameter (list[dict]):

    Returns:
        (numpy.ndarray, numpy.ndarray): u, w
    """
    return np.vectorize(lambda d: get_scalar_param(p, d, parameter))(duration)


########################################################################################################################
def get_u_w(duration, parameter):
    """

    Args:
        duration (numpy.ndarray| list | float | int): in minutes
        parameter (list[dict]):

    Returns:
        (float, float): u, w
    """
    if isinstance(duration, (list, np.ndarray)):
        func = get_array_param
    else:
        func = get_scalar_param

    return (func(p, duration, parameter) for p in PARAM.U_AND_W)


########################################################################################################################
def depth_of_rainfall(u, w, return_period, series_kind):
    """
    calculate the height of the rainfall h in L/m² = mm

    Args:
        u (float): parameter depending on duration
        w (float): parameter depending on duration
        return_period (float): in years
        series_kind (str): ['partial', 'annual']

    Returns:
        float: height of the rainfall h in L/m² = mm
    """
    if series_kind == ANNUAL:
        if return_period <= 10:
            return_period = np.exp(1.0 / return_period) / (np.exp(1.0 / return_period) - 1.0)

        log_tn = -np.log(np.log(return_period / (return_period - 1.0)))

    else:
        log_tn = np.log(return_period)

    return u + w * log_tn
