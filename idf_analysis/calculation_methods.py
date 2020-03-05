__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
import numpy as np
from numpy import NaN
from math import e, floor

from tqdm import tqdm

from .sww_utils import guess_freq, rain_events, agg_events
from .definitions import DWA, ATV, DWA_adv, PARTIAL, ANNUAL, LOG1, LOG2, HYP, LIN, COL, PARAM, PARAM_COL


########################################################################################################################
def annual_series(events):
    """
    create an annual series of the maximum overlapping sum per year and calculate the "u" and "w" parameters
    acc. to DWA-A 531 chap. 5.1.5

    Args:
        events (pandas.DataFrame): with columns=[start, end, max_overlapping_sum]

    Returns:
        tuple[float, float]: parameter u and w from the annual series for a specific duration step as a tuple
    """
    annually_series = pd.Series(data=events[COL.MAX_OVERLAPPING_SUM].values,
                                index=events[COL.START].values,
                                name=COL.MAX_OVERLAPPING_SUM).resample('AS').max()
    annually_series = annually_series.sort_values(ascending=False).reset_index(drop=True)

    mean_sample_rainfall = annually_series.mean()
    sample_size = annually_series.count()

    x = -np.log(np.log((sample_size + 0.2) / (sample_size - (annually_series.index.values + 1.0) + 0.6)))
    x_mean = x.mean()

    w = ((x * annually_series).sum() - sample_size * mean_sample_rainfall * x_mean) / \
        ((x ** 2).sum() - sample_size * x_mean ** 2)
    u = mean_sample_rainfall - w * x_mean

    return {PARAM.U: u, PARAM.W: w}


########################################################################################################################
def partial_series(events, measurement_period):
    """
    create an partial series of the largest overlapping sums and calculate the "u" and "w" parameters
    acc. to DWA-A 531 chap. 5.1.4

    Args:
        events (pandas.DataFrame): with columns=[start, end, max_overlapping_sum]
        measurement_period (float): in years

    Returns:
        tuple[float, float]: parameter u and w from the partial series for a specific duration step as a tuple
    """
    partially_series = pd.Series(data=events[COL.MAX_OVERLAPPING_SUM].values, name=COL.MAX_OVERLAPPING_SUM)
    partially_series = partially_series.sort_values(ascending=False).reset_index(drop=True)

    # use only the (2-3 multiplied with the number of measuring years) of the biggest
    # values in the database (-> acc. to ATV-A 121 chap. 4.3; DWA-A 531 chap. 4.4)
    # as an requirement for the extreme value distribution

    size_threshold_value = int(floor(measurement_period * e))
    partially_series = partially_series.iloc[:size_threshold_value].copy()  # 1

    mean_sample_rainfall = partially_series.mean()
    sample_size = partially_series.count()

    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    log_return_periods = np.log(_plotting_formula(partially_series.index.values + 1, sample_size, measurement_period))
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
def calculate_u_w(file_input, duration_steps, measurement_period, series_kind):
    """
    statistical analysis for each duration step acc. to DWA-A 531 chap. 5.1
    save the parameters of the distribution function as interim results
    acc. to DWA-A 531 chap. 4.4: use the annual series only for measurement periods over 20 years


    Args:
        file_input (pandas.Series): precipitation data
        duration_steps (list[int] | numpy.ndarray): in minutes
        measurement_period (float): duration of the series in years
        series_kind (str): annual or partial

    Returns:
        pandas.DataFrame: with index=durations and columns=[u, w]
    """
    ts = file_input.copy()
    base_frequency = guess_freq(file_input.index)  # DateOffset/Timedelta

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
        pbar.set_description(f'Calculating Parameters u and w for duration {duration_integer}')
        duration = pd.Timedelta(minutes=duration_integer)

        if duration < pd.Timedelta(base_frequency):
            continue

        events = rain_events(file_input, min_gap=duration)

        # correction factor acc. to DWA-A 531 chap. 4.3
        improve = _improve_factor(duration / base_frequency)

        events[COL.MAX_OVERLAPPING_SUM] = agg_events(events, ts.rolling(duration).sum(), 'max') * improve

        if series_kind == ANNUAL:
            interim_results[duration_integer] = annual_series(events)
        elif series_kind == PARTIAL:
            interim_results[duration_integer] = partial_series(events, measurement_period)
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

    return a, b


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
    return a, b


########################################################################################################################
def formulation(approach, duration, param, a_start=20.0, b_start=15.0, param_mean=None, duration_mean=None):
    """

    Args:
        approach:
        duration:
        param:
        a_start:
        b_start:
        param_mean:
        duration_mean:

    Returns:

    """
    if approach in [LOG1, LOG2]:
        return folded_log_formulation(duration, param, case=approach, param_mean=param_mean,
                                      duration_mean=duration_mean)

    elif approach == HYP:
        return hyperbolic_formulation(duration, param, a_start=a_start, b_start=b_start, param_mean=param_mean,
                                      duration_mean=duration_mean)

    elif approach == LIN:
        return NaN, NaN

    else:
        raise NotImplementedError


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
def get_approach_table(worksheet):
    """
    table of approaches depending on the duration and the parameter

    Args:
        worksheet (str): worksheet name for the analysis:
            - 'DWA-A_531'
            - 'ATV-A_121'
            - 'DWA-A_531_advektiv' (yet not implemented)

    Returns:
        pandas.DataFrame: table of approaches depending on the duration and the parameter
    """
    approach_list = list()

    # acc. to ATV-A 121 chap. 5.2.1
    if worksheet == ATV:
        approach_list.append({PARAM_COL.FROM: None,
                              PARAM_COL.TO: None,
                              PARAM_COL.U: LOG2,
                              PARAM_COL.W: LOG1})

    elif worksheet == DWA:
        duration_bound_1, duration_bound_2 = get_duration_steps(worksheet)

        approach_list.append({PARAM_COL.FROM: 0,
                              PARAM_COL.TO: duration_bound_1,
                              PARAM_COL.U: HYP,
                              PARAM_COL.W: LOG2})
        approach_list.append({PARAM_COL.FROM: duration_bound_1,
                              PARAM_COL.TO: duration_bound_2,
                              PARAM_COL.U: LOG2,
                              PARAM_COL.W: LOG2})
        approach_list.append({PARAM_COL.FROM: duration_bound_2,
                              PARAM_COL.TO: np.inf,
                              PARAM_COL.U: LIN,
                              PARAM_COL.W: LIN})

    else:
        raise NotImplementedError

    return approach_list


########################################################################################################################
def _calc_params(parameter, interim_results, params_mean=None, duration_mean=None):
    """
    calculate parameters a_u, a_w, b_u and b_w and add it to the dict

    Args:
        parameter (list[dict]):
        interim_results (pandas.DataFrame):
        params_mean (dict[float]):
        duration_mean (float):

    Returns:
        list[dict]: parameters
    """
    for row in parameter:
        it_res = interim_results.loc[row[PARAM_COL.FROM]: row[PARAM_COL.TO]]
        if it_res.empty:
            continue

        dur = it_res.index.values

        for p in PARAM.U_AND_W:
            a_label = PARAM_COL.A(p)
            b_label = PARAM_COL.B(p)

            param = it_res[p].values

            approach = row[p]
            if params_mean:
                param_mean = params_mean[p]
            else:
                param_mean = None

            a_start = 20.0
            if a_label in row and not np.isnan(row[a_label]):
                a_start = row[a_label]

            b_start = 15.0
            if b_label in row and not np.isnan(row[b_label]):
                b_start = row[b_label]

            a, b = formulation(approach, dur, param, a_start=a_start, b_start=b_start,
                               param_mean=param_mean, duration_mean=duration_mean)
            row[a_label] = a
            row[b_label] = b
    return parameter


########################################################################################################################
def get_parameter(interim_results, worksheet=DWA):
    """
    get calculation parameters

    Args:
        interim_results (pandas.DataFrame):
        worksheet (str):

    Returns:
        list[dict]: parameters
    """
    parameter = get_approach_table(worksheet)

    parameter = _calc_params(parameter, interim_results)

    # the balance between the different duration ranges acc. to DWA-A 531 chap. 5.2.4
    duration_step = parameter[0][PARAM_COL.TO]
    durations = np.array([duration_step - 0.001, duration_step + 0.001])

    if any(durations < interim_results.index.values.min()):
        return parameter

    u, w = get_u_w(durations, parameter, interim_results)
    parameter = _calc_params(parameter, interim_results, params_mean=dict(u=np.mean(u), w=np.mean(w)),
                             duration_mean=duration_step)
    return parameter


########################################################################################################################
def get_u_w(duration, parameter, interim_results):
    """

    Args:
        duration (numpy.ndarray | float | int): in minutes
        parameter (list[dict]):
        interim_results:

    Returns:
        (float, float): u, w
    """
    if isinstance(duration, list):
        duration = np.array(duration)

    # ------------------------------------------------------------------------------------------------------------------
    def _calc_param(a, b, duration, approach, interim_results):
        """
        calc u(D) or w(D)

        Args:
            a (float):
            b (float):
            duration (float | numpy.ndarray):
            approach (str):
            interim_results (pandas.Series):

        Returns:
            float: parameter
        """
        if approach == LOG1:
            return a + b * np.log(duration)
        elif approach == LOG2:
            return np.exp(a) * np.power(duration, b)
        elif approach == HYP:
            return a * duration / (duration + b)
        elif approach == LIN:
            return np.interp(duration, interim_results.index.values, interim_results.values)
        else:
            raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    res = {}
    for row in parameter:

        if isinstance(duration, (int, float)):
            if not (duration > row[PARAM_COL.FROM]) & (duration <= row[PARAM_COL.TO]):
                continue
            else:
                dur = duration
        else:
            dur = duration[(duration > row[PARAM_COL.FROM]) & (duration <= row[PARAM_COL.TO])]
            if not dur.size:
                continue

        for p in PARAM.U_AND_W:
            approach = row[p]
            a = row[PARAM_COL.A(p)]
            b = row[PARAM_COL.B(p)]

            new = _calc_param(a, b, dur, approach, interim_results[p])
            if p in res:
                res[p] = np.array(list(res[p]) + list(new))
            else:
                res[p] = new
    return [res[i] for i in PARAM.U_AND_W]


########################################################################################################################
def depth_of_rainfall(u, w, series_kind, return_period):
    """
    calculate the height of the rainfall h in L/m² = mm

    Args:
        u (float): parameter
        w (float): parameter
        series_kind (str): ['partial', 'annual']
        return_period (float): in years

    Returns:
        float: height of the rainfall h in L/m² = mm
    """
    if series_kind == ANNUAL and return_period <= 10:
        return_period_asteriks = np.exp(1.0 / return_period) / (np.exp(1.0 / return_period) - 1.0)
        return u + w * (-np.log(np.log(return_period_asteriks / (return_period_asteriks - 1.0))))

    elif series_kind == ANNUAL and return_period > 10:
        return u + w * (-np.log(np.log(return_period / (return_period - 1.0))))

    else:
        return u + w * np.log(return_period)
