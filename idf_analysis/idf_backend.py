__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

from collections import OrderedDict

import pandas as pd
import numpy as np
import os

from .definitions import *
from .event_series_analysis import calculate_u_w
from .in_out import write_yaml, read_yaml
from .parameter_formulations import folded_log_formulation, hyperbolic_formulation


class IdfParameters:
    def __init__(self, series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=False):
        self.series_kind = series_kind

        self._durations = None
        self._set_default_durations(extended_durations)

        self.parameters_series = {}  # parameters u and w (distribution function) from the event series analysis
        self.parameters_final = {}  # parameters of the distribution function after the regression

        self.get_approaches(worksheet)

    # -------------------------------------------------------------
    def __bool__(self):
        return bool(self.parameters_series)

    # -------------------------------------------------------------
    def calc_from_series(self, series):
        u_w = calculate_u_w(series, self.durations, self.series_kind)
        for p in PARAM.U_AND_W:
            self.parameters_series[p] = np.array([d[p] for d in u_w.values()])
        self._calc_params()
        self._balance_parameter_change()

    def reverse_engineering(self, idf_table):
        durations = idf_table.index.values
        u = idf_table[1].values
        w_dat = idf_table.sub(u, axis=0)
        # --
        w = []
        for dur in durations:
            dat = w_dat.loc[dur]
            log_tn_i = np.log(dat.index.values)
            w.append(sum(dat.values * log_tn_i) / sum(log_tn_i ** 2))

        # -------
        # ax = dat.plot(logx=True)
        # fig = ax.get_figure()
        # fig.show()
        # print(results.summary())

        self.durations = durations
        self.parameters_series[PARAM.U] = np.array(u)
        self.parameters_series[PARAM.W] = np.array(w)
        self._calc_params()
        self._balance_parameter_change()

    # -------------------------------------------------------------
    @property
    def durations(self):
        return np.array(self._durations)

    @durations.setter
    def durations(self, durations):
        self._durations = list(durations)

    def _set_default_durations(self, extended_durations=False):
        # Suggestion from ATV-A 121 (ATV-DVWK-Regelwerk 2/2001)

        # sampling points of the duration steps in minutes
        duration_steps = [5, 10, 15, 20, 30, 45, 60, 90]
        # self._duration_steps += [i * 60 for i in [3, 4.5, 6, 7.5, 10, 12, 18]]  # duration steps in hours
        duration_steps += [i * 60 for i in [2, 3, 4, 6, 9, 12, 18]]  # duration steps in hours
        if extended_durations:
            duration_steps += [i * 60 * 24 for i in [1, 2, 3, 4, 5, 6]]  # duration steps in days

        self.durations = duration_steps

    def filter_durations(self, freq_minutes):
        self.limit_duration(lowest=freq_minutes)

    def limit_duration(self, lowest=None, highest=None):
        bool_array = self.durations == False
        if lowest is not None:
            bool_array |= self.durations >= lowest
        if highest is not None:
            bool_array |= self.durations <= highest

        self.durations = self.durations[bool_array]
        for param in PARAM.U_AND_W:
            if param in self.parameters_series:
                self.parameters_series[param] = self.parameters_series[param][bool_array]
        # self.parameters_series[PARAM.W] = self.parameters_series[PARAM.W][bool_array]

    # -------------------------------------------------------------
    def get_approaches(self, worksheet):
        """
        Approaches depending on the duration and the parameter.

        Args:
            worksheet (str): worksheet name for the analysis:
                - 'DWA-A_531'
                - 'ATV-A_121'
                - 'DWA-A_531_advektiv' (yet not implemented)

        Returns:
            list[dict]: table of approaches depending on the duration and the parameter
        """
        self.parameters_final = read_yaml(os.path.join(os.path.dirname(__file__), 'approaches', worksheet + '.yaml'))

    # def _dur_upper_bound(self, dur_min):
    #     lower_bounds = list(self.parameters_final.keys())
    #     return lower_bounds[lower_bounds.index(dur_min) + 1]

    def _iter_params(self):
        lower_bounds = list(self.parameters_final.keys())
        return zip(lower_bounds, lower_bounds[1:] + [np.inf])

    def _calc_params(self, params_mean=None, duration_mean=None):
        """
        Calculate parameters a_u, a_w, b_u and b_w and add it to the dict

        Args:
            params_mean (dict[float]):
            duration_mean (float):

        Returns:
            list[dict]: parameters
        """
        for dur_lower_bound, dur_upper_bound in self._iter_params():
            param_part = (self.durations >= dur_lower_bound) & (self.durations <= dur_upper_bound)

            if param_part.sum() == 1:
                del self.parameters_final[dur_lower_bound]
                # Only one duration step in this duration range.
                # Only one value available in the series for this regression.
                continue

            params_dur = self.parameters_final[dur_lower_bound]

            dur = self.durations[param_part]

            for p in PARAM.U_AND_W:  # u or w
                params = params_dur[p]
                values_series = self.parameters_series[p][param_part]

                approach = params[PARAM.FUNCTION]

                if params_mean:
                    param_mean = params_mean[p]
                else:
                    param_mean = None

                # ----------------------------
                if approach in [APPROACH.LOG1, APPROACH.LOG2]:
                    params[PARAM.A], params[PARAM.B] = folded_log_formulation(dur, values_series, case=approach,
                                                                              param_mean=param_mean,
                                                                              duration_mean=duration_mean)

                # ----------------------------
                elif approach == APPROACH.HYP:
                    a_start = 20.0
                    if PARAM.A in params and not np.isnan(params[PARAM.A]):
                        a_start = params[PARAM.A]

                    b_start = 15.0
                    if PARAM.B in params and not np.isnan(params[PARAM.B]):
                        b_start = params[PARAM.B]

                    params[PARAM.A], params[PARAM.B] = hyperbolic_formulation(dur, values_series, a_start=a_start,
                                                                              b_start=b_start,
                                                                              param_mean=param_mean,
                                                                              duration_mean=duration_mean)

                # ----------------------------
                elif approach == APPROACH.LIN:
                    pass

                # ----------------------------
                else:
                    raise NotImplementedError

                # ----------------------------

    def _balance_parameter_change(self):
        # the balance between the different duration ranges acc. to DWA-A 531 chap. 5.2.4
        duration_step = list(self.parameters_final.keys())[1]
        durations = np.array([duration_step - 0.001, duration_step + 0.001])

        # if the interim results end here

        if any(durations < self.durations.min()) or \
                any(durations > self.durations.max()):
            return

        u, w = self.get_u_w(durations)
        self._calc_params(params_mean=dict(u=np.mean(u), w=np.mean(w)), duration_mean=duration_step)

    def measured_points(self, return_periods, max_duration=None):
        """
        get the calculation results of the rainfall with u and w without the estimation of the formulation

        Args:
            return_periods (float | np.array | list | pd.Series): return period in [a]
            max_duration (float): max duration in [min]

        Returns:
            pd.Series: series with duration as index and the height of the rainfall as data
        """
        interim_results = pd.DataFrame(index=self.durations)
        interim_results.index.name = PARAM.DUR
        interim_results[PARAM.U] = self.parameters_series[PARAM.U]
        interim_results[PARAM.W] = self.parameters_series[PARAM.W]

        if max_duration is not None:
            interim_results = interim_results.loc[:max_duration].copy()

        return pd.Series(index=interim_results.index,
                         data=interim_results['u'] + interim_results['w'] * np.log(return_periods))

    def get_duration_section(self, duration, param):
        for lower, upper in self._iter_params():
            if lower <= duration <= upper:
                return self.parameters_final[lower][param]

    def get_scalar_param(self, p, duration):
        """

        Args:
            p (str): name of the parameter 'u' or 'w'
            duration (float | int): in minutes

        Returns:
            (float, float): u, w
        """
        param = self.get_duration_section(duration, p)

        if param is None:
            return np.NaN

        elif param[PARAM.FUNCTION] == APPROACH.LOG1:
            a = param[PARAM.A]
            b = param[PARAM.B]
            return a + b * np.log(duration)

        elif param[PARAM.FUNCTION] == APPROACH.LOG2:
            a = param[PARAM.A]
            b = param[PARAM.B]
            return np.exp(a) * np.power(duration, b)

        elif param[PARAM.FUNCTION] == APPROACH.HYP:
            a = param[PARAM.A]
            b = param[PARAM.B]
            return a * duration / (duration + b)

        elif param[PARAM.FUNCTION] == APPROACH.LIN:
            return np.interp(duration, self.durations, self.parameters_series[p])

    def get_array_param(self, p, duration):
        """

        Args:
            p (str): name of the parameter 'u' or 'w'
            duration (numpy.ndarray): in minutes

        Returns:
            (numpy.ndarray, numpy.ndarray): u, w
        """
        return np.vectorize(lambda d: self.get_scalar_param(p, d))(duration)

    def get_u_w(self, duration):
        """
        calculate the u and w parameters depending on the durations

        Args:
            duration (numpy.ndarray| list | float | int): in minutes

        Returns:
            (float, float): u and w
        """
        if isinstance(duration, (list, np.ndarray)):
            func = self.get_array_param
        else:
            func = self.get_scalar_param

        return (func(p, duration) for p in PARAM.U_AND_W)

    @classmethod
    def from_interim_results_file(cls, interim_results_fn, worksheet=METHOD.KOSTRA):
        """DEPRECIATED: for compatibility reasons. To use the old file and convert it to the new parameters file"""
        interim_results = pd.read_csv(interim_results_fn, index_col=0, header=0)
        p = cls(worksheet, interim_results)
        return p

    def to_yaml(self, filename):
        def to_basic(a):
            return list([round(float(i), 4) for i in a])
        data = OrderedDict({
            PARAM.SERIES: self.series_kind,
            PARAM.DUR: to_basic(self.durations),
            PARAM.PARAMS_SERIES: {p: to_basic(l) for p, l in self.parameters_series.items()},
            PARAM.PARAMS_FINAL: self.parameters_final
        })
        write_yaml(data, filename)

    @classmethod
    def from_yaml(cls, filename):
        data = read_yaml(filename)
        if isinstance(data, list):
            return cls.from_yaml_depreciated(filename)
        p = cls(series_kind=data[PARAM.SERIES])
        p.durations = data[PARAM.DUR]
        # list to numpy.array
        p.parameters_series = {p: np.array(l) for p, l in data[PARAM.PARAMS_SERIES].items()}
        p.parameters_final = data[PARAM.PARAMS_FINAL]
        return p

    @classmethod
    def from_yaml_depreciated(cls, filename, series_kind=SERIES.PARTIAL):
        data = read_yaml(filename)
        p = cls(series_kind=series_kind)
        p.durations = []
        p.parameters_series = {PARAM.U: [], PARAM.W: []}
        p.parameters_final = {}
        for part in data:
            start_dur = float(part['von'])
            start_idx = 0
            if start_dur in part[COL.DUR]:
                start_idx = 1
            p._durations += part[COL.DUR][start_idx:]
            part_params = {}

            for u_or_w in PARAM.U_AND_W:
                part_params[u_or_w] = {PARAM.FUNCTION: part[u_or_w]}
                if part[u_or_w] != APPROACH.LIN:
                    part_params[u_or_w][PARAM.A] = part[f'{PARAM.A}_{u_or_w}']
                    part_params[u_or_w][PARAM.B] = part[f'{PARAM.B}_{u_or_w}']

                p.parameters_series[u_or_w] += part[PARAM.VALUES(u_or_w)][start_idx:]

            p.parameters_final[start_dur] = part_params

        p.parameters_series = {param: np.array(l) for param, l in p.parameters_series.items()}
        if '.yaml' in filename:
            os.rename(filename, filename.replace('.yaml', '_OLD.yaml'))
            p.to_yaml(filename)
        return p
