__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
import numpy as np
from numpy import NaN
from math import e, floor, ceil
from .sww_utils import guess_freq, rain_events, agg_events
from .definitions import DWA, ATV, DWA_adv, PARTIAL, ANNUAL, LOG1, LOG2, HYP, LIN

# from my_helpers import check

MAX_OSUM = 'max_overlapping_sum'


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
    annually_series = pd.Series(data=events[MAX_OSUM].values, index=events['start'].values, name=MAX_OSUM).resample(
        'AS').max()
    annually_series = annually_series.sort_values(ascending=False).reset_index(drop=True)

    mean_sample_rainfall = annually_series.mean()
    sample_size = annually_series.count()

    x = -np.log(np.log((sample_size + 0.2) / (sample_size - (annually_series.index.values + 1.0) + 0.6)))
    x_mean = x.mean()

    w = ((x * annually_series).sum() - sample_size * mean_sample_rainfall * x_mean) / \
        ((x ** 2).sum() - sample_size * x_mean ** 2)
    u = mean_sample_rainfall - w * x_mean

    return {'u': u, 'w': w}


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
    partially_series = pd.Series(data=events[MAX_OSUM].values, name=MAX_OSUM)
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

    return {'u': u, 'w': w}


########################################################################################################################
def _calculate_u_w_OLD(file_input, duration_steps, measurement_period, series_kind):
    """
    statistical analysis for each duration step acc. to DWA-A 531 chap. 5.1
    save the parameters of the distribution function as interim results
    acc. to DWA-A 531 chap. 4.4: use the annual series only for measurement periods over 20 years


    Args:
        file_input (pandas.Series): data
        duration_steps (list[int] | numpy.ndarray): in minutes
        measurement_period (float): duration of the series in years
        series_kind (str): annual or partial series

    Returns:
        pandas.DataFrame: with index = duration and columns = [u, w]
    """
    # check('start')
    ts = file_input.copy()
    # ts = ts.dropna()
    base_frequency = guess_freq(file_input.index)  # DateOffset/Timedelta
    ts = ts.resample(base_frequency).sum()

    # ts = ts.asfreq(base_frequency)
    # ts = ts.fillna(0)

    # ------------------------------------------------------------------------------------------------------------------
    def _calc_overlapping_sum_max(event, duration):
        """
        calculation of the maximum of the overlapping sum of the series
        acc. to DWA-A 531 chap. 4.2

        Args:
            event (pandas.Series): event with index=[start, end]
            duration (pandas.Timedelta): of the calculation step

        Returns:
            float: maximum of the overlapping sum
        """
        data = ts.loc[event['start']:event['end']].copy()
        interval = int(round(duration / base_frequency))

        # correction factor acc. to DWA-A 531 chap. 4.3
        improve = [1.140, 1.070, 1.040, 1.030]

        if interval == 1:
            return data.max() * improve[0]

        data = data.rolling(window=interval).sum()

        if interval > 4:
            return data.max()
        else:
            return data.max() * improve[interval - 1]

    # ------------------------------------------------------------------------------------------------------------------
    interim_results = pd.DataFrame(index=duration_steps, columns=['u', 'w'], dtype=float)
    interim_results.index.name = 'duration'

    # acc. to DWA-A 531 chap. 4.2:
    # The values must be independent of each other for the statistical evaluations.
    # estimated four hours acc. (SCHILLING 1984)
    # for larger durations - use the duration as minimal gap
    minimal_gap = pd.Timedelta(hours=4)
    # check('Events')
    events = rain_events(file_input, ignore_rain_below=0.0, min_gap=minimal_gap)
    # check(' - done')
    for duration_index in duration_steps:
        print(duration_index)
        duration = pd.Timedelta(minutes=duration_index)
        if duration > minimal_gap:
            events = rain_events(file_input, ignore_rain_below=0.0, min_gap=duration)

        # check('osum')
        events[MAX_OSUM] = events.apply(_calc_overlapping_sum_max, axis=1, duration=duration)

        # check('series calc')
        if series_kind == ANNUAL:
            interim_results.loc[duration_index] = annual_series(events)
        elif series_kind == PARTIAL:
            interim_results.loc[duration_index] = partial_series(events, measurement_period)
        else:
            raise NotImplementedError
        # check(' - done')
    return interim_results


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

    for duration_integer in duration_steps:
        duration = pd.Timedelta(minutes=duration_integer)

        if duration < pd.Timedelta(base_frequency):
            continue

        events = rain_events(file_input, min_gap=duration)

        # correction factor acc. to DWA-A 531 chap. 4.3
        improve = _improve_factor(duration / base_frequency)

        events[MAX_OSUM] = agg_events(events, ts.rolling(duration).sum(), 'max') * improve

        if series_kind == ANNUAL:
            interim_results[duration_integer] = annual_series(events)
        elif series_kind == PARTIAL:
            interim_results[duration_integer] = partial_series(events, measurement_period)
        else:
            raise NotImplementedError

    # -------------------------------
    interim_results = pd.DataFrame.from_dict(interim_results)
    interim_results.index.name = 'duration'
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
        tuple[int, int]:
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
    approach_table = pd.DataFrame(columns=['von', 'bis', 'u', 'w', 'a_u', 'b_u', 'a_w', 'b_w'])

    # acc. to ATV-A 121 chap. 5.2.1
    if worksheet == ATV:
        approach_table = approach_table.append(dict(von=None, bis=None, u=LOG2, w=LOG1), ignore_index=True)

    elif worksheet == DWA:
        duration_bound_1, duration_bound_2 = get_duration_steps(worksheet)

        approach_table = approach_table.append(dict(von=0, bis=duration_bound_1,
                                                    u=HYP, w=LOG2), ignore_index=True)
        approach_table = approach_table.append(dict(von=duration_bound_1, bis=duration_bound_2,
                                                    u=LOG2, w=LOG2), ignore_index=True)
        approach_table = approach_table.append(dict(von=duration_bound_2, bis=np.inf,
                                                    u=LIN, w=LIN), ignore_index=True)

    else:
        raise NotImplementedError

    return approach_table


########################################################################################################################
def get_parameter(interim_results, worksheet=DWA):
    """
    get calculation parameters

    Args:
        interim_results (pandas.DataFrame):
        worksheet (str):

    Returns:
        pandas.DataFrame: parameters
    """
    parameter = get_approach_table(worksheet)

    # ------------------------------------------------------------------------------------------------------------------
    def _calc_params(parameter_, params_mean=None, duration_mean=None):
        parameters = parameter_.copy()
        for index, row in parameters.iterrows():
            it_res = interim_results.loc[row['von']: row['bis']]
            if it_res.empty:
                continue
            for p in ['u', 'w']:
                param = it_res[p].values
                dur = it_res.index.values
                approach = row[p]
                if params_mean:
                    param_mean = params_mean[p]
                else:
                    param_mean = None

                a_start = row.loc['a_{}'.format(p)]
                b_start = row.loc['b_{}'.format(p)]
                if np.isnan(a_start):
                    a_start = 20.0
                if np.isnan(b_start):
                    b_start = 15.0

                a, b = formulation(approach, dur, param, a_start=a_start, b_start=b_start,
                                   param_mean=param_mean, duration_mean=duration_mean)
                parameters.loc[index, 'a_{}'.format(p)] = a
                parameters.loc[index, 'b_{}'.format(p)] = b
        return parameters

    # ------------------------------------------------------------------------------------------------------------------
    parameter = _calc_params(parameter)

    # the balance between the different duration ranges acc. to DWA-A 531 chap. 5.2.4
    duration_step = parameter['bis'][0]
    durations = np.array([duration_step - 0.001, duration_step + 0.001])

    if any(durations < interim_results.index.values.min()):
        return parameter

    u, w = get_u_w(durations, parameter, interim_results)
    parameter = _calc_params(parameter, params_mean=dict(u=np.mean(u), w=np.mean(w)), duration_mean=duration_step)
    return parameter


########################################################################################################################
def get_u_w(duration, parameter, interim_results):
    """

    Args:
        duration (numpy.ndarray | float | int): in minutes
        parameter:
        interim_results:

    Returns:

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
    for index, row in parameter.iterrows():

        if isinstance(duration, (int, float)):
            if not (duration > row['von']) & (duration <= row['bis']):
                continue
            else:
                dur = duration
        else:
            dur = duration[(duration > row['von']) & (duration <= row['bis'])]
            if not dur.size:
                continue

        for p in ['u', 'w']:
            approach = row[p]
            a, b = row.loc[['{}_{}'.format(i, p) for i in ['a', 'b']]]
            new = _calc_param(a, b, dur, approach, interim_results[p])
            if p in res:
                res[p] = np.array(list(res[p]) + list(new))
            else:
                res[p] = new
    return [res[i] for i in ['u', 'w']]


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


########################################################################################################################
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
