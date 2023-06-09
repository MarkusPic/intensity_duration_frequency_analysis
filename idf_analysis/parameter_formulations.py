import numpy as np
from scipy.optimize import curve_fit

from .definitions import APPROACH


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

    if case == APPROACH.LOG2:
        # for the twofold formulation
        b = ((np.log(param) - mean_ln_param) * (np.log(duration) - mean_ln_duration)).sum() / divisor
        a = mean_ln_param - b * mean_ln_duration

    elif case == APPROACH.LOG1:
        # for the onefold formulation
        b = ((param - param_mean) * (np.log(duration) - mean_ln_duration)).sum() / divisor
        a = param_mean - b * mean_ln_duration

    else:
        raise NotImplementedError

    return float(a), float(b)


def hyperbolic_formulation(duration: np.array, param: np.array, a_start=20.0, b_start=15.0, param_mean=None, duration_mean=None):
    """
    Computes the hyperbolic formulation using the given parameters.

    Args:
        duration (np.array): Array of durations.
        param (np.array): Array of parameters.
        a_start (float): Initial value for 'a' parameter.
        b_start (float): Initial value for 'b' parameter.
        param_mean (float, optional): Mean value of parameters. Defaults to None.
        duration_mean (float, optional): Mean value of durations. Defaults to None.

    Returns:
        tuple[float, float]: Computed value of 'a' and 'b'.
    """

    def get_param(dur, par, a_, b_):
        """
        Computes 'a' and 'b' parameters based on given durations and parameters.

        Args:
            dur (np.array): Array of durations.
            par (np.array): Array of parameters.
            a_ (float): Current value of 'a' parameter.
            b_ (float): Current value of 'b' parameter.

        Returns:
            tuple[float, float]: Computed value of 'a' and 'b'.
        """
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


def hyperbolic_function(D, a, b):
    return a * D / (D + b)


def hyperbolic_formulation_chatgpt_opt(duration: np.array, param: np.array, a_start=20.0, b_start=15.0, param_mean=None, duration_mean=None):
    # TODO: param_mean and duration_mean for parameter - balance

    # Perform the curve fitting
    initial_guess = [a_start, b_start]  # initial guess for a and b
    fit_params, _ = curve_fit(hyperbolic_function, duration, param, p0=initial_guess)

    # Extract the optimized parameter values
    a_fit, b_fit = fit_params
    return float(a_fit), float(b_fit)
