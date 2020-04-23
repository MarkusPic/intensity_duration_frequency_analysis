__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
import numpy as np
from collections import OrderedDict

from .definitions import *
from .event_series_analysis import calculate_u_w
from .in_out import write_yaml, read_yaml
from .parameter_formulations import folded_log_formulation, hyperbolic_formulation


class IdfParameters:
    def __init__(self):
        self.worksheet = None
        self.interim_results = None  # parameters u and w from the event series analysis
        self._data = list()

    def to_yaml(self, filename):
        write_yaml(self._data, filename)

    @classmethod
    def from_yaml(cls, filename):
        p = cls()
        p._data = read_yaml(filename)
        return p

    @classmethod
    def from_interim_results(cls, interim_results, worksheet=DWA):
        """
        get calculation parameters
        """
        p = cls()
        p.worksheet = worksheet
        p.interim_results = interim_results
        p._calc()
        return p

    @classmethod
    def from_series(cls, series, duration_steps, series_kind, worksheet):
        p = cls()
        p.worksheet = worksheet
        p.interim_results = calculate_u_w(series, duration_steps, series_kind)
        p._calc()
        return p

    def _calc(self):
        self.get_approaches()
        self.split_interim_results()
        self._calc_params()
        self._balance_parameter_change()

    def get_duration_steps(self):
        """
        duration step boundary for the various distribution functions in minutes

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
        }[self.worksheet]

    def get_approaches(self):
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
        # acc. to ATV-A 121 chap. 5.2.1
        if self.worksheet == ATV:
            self._data.append(OrderedDict({PARAM_COL.FROM: None,
                                           PARAM_COL.TO: None,
                                           PARAM_COL.U: LOG2,
                                           PARAM_COL.W: LOG1}))

        elif self.worksheet == DWA:
            duration_bound_1, duration_bound_2 = self.get_duration_steps()

            self._data.append(OrderedDict({PARAM_COL.FROM: 0,
                                           PARAM_COL.TO: duration_bound_1,
                                           PARAM_COL.U: HYP,
                                           PARAM_COL.W: LOG2}))
            self._data.append(OrderedDict({PARAM_COL.FROM: duration_bound_1,
                                           PARAM_COL.TO: duration_bound_2,
                                           PARAM_COL.U: LOG2,
                                           PARAM_COL.W: LOG2}))
            self._data.append(OrderedDict({PARAM_COL.FROM: duration_bound_2,
                                           PARAM_COL.TO: np.inf,
                                           PARAM_COL.U: LIN,
                                           PARAM_COL.W: LIN}))

        else:
            raise NotImplementedError

    def split_interim_results(self):
        for i, row in enumerate(self._data):
            it_res = self.interim_results.loc[row[PARAM_COL.FROM]: row[PARAM_COL.TO]]
            if it_res.empty or it_res.index.size == 1:
                continue
            self._data[i][COL.DUR] = list(map(int, it_res.index.values))
            for p in PARAM.U_AND_W:  # u or w
                self._data[i][PARAM_COL.VALUES(p)] = list(map(float, it_res[p].values))

    def get_interim_results(self):
        if self.interim_results is None:
            """
            extract interim results from parameters
            """
            interim_results = dict()
            for p in self._data:
                for col in [COL.DUR, PARAM_COL.VALUES(PARAM.U), PARAM_COL.VALUES(PARAM.W)]:
                    if col in interim_results:
                        interim_results[col] += p[col]
                    else:
                        interim_results[col] = p[col]

            interim_results = pd.DataFrame.from_dict(interim_results, orient='columns').set_index(
                COL.DUR).drop_duplicates()
            self.interim_results = interim_results.rename({PARAM_COL.VALUES(PARAM.U): PARAM.U,
                                                           PARAM_COL.VALUES(PARAM.W): PARAM.W}, axis=1)

        return self.interim_results

    def _calc_params(self, params_mean=None, duration_mean=None):
        """
        calculate parameters a_u, a_w, b_u and b_w and add it to the dict

        Args:
            parameter (list[dict]):
            params_mean (dict[float]):
            duration_mean (float):

        Returns:
            list[dict]: parameters
        """
        for row in self._data:
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
                                                                        param_mean=param_mean,
                                                                        duration_mean=duration_mean)

                # ----------------------------
                elif approach == HYP:
                    a_start = 20.0
                    if a_label in row and not np.isnan(row[a_label]):
                        a_start = row[a_label]

                    b_start = 15.0
                    if b_label in row and not np.isnan(row[b_label]):
                        b_start = row[b_label]

                    row[a_label], row[b_label] = hyperbolic_formulation(dur, param, a_start=a_start, b_start=b_start,
                                                                        param_mean=param_mean,
                                                                        duration_mean=duration_mean)

                # ----------------------------
                elif approach == LIN:
                    pass

                # ----------------------------
                else:
                    raise NotImplementedError

                # ----------------------------

    def _balance_parameter_change(self):
        # the balance between the different duration ranges acc. to DWA-A 531 chap. 5.2.4
        duration_step = self._data[0][PARAM_COL.TO]
        durations = np.array([duration_step - 0.001, duration_step + 0.001])

        # if the interim results end here
        if any(durations < self.interim_results.index.values.min()) or \
                any(durations > self.interim_results.index.values.max()):
            return

        u, w = self.get_u_w(durations)
        self._calc_params(params_mean=dict(u=np.mean(u), w=np.mean(w)), duration_mean=duration_step)

    def get_row(self, duration):
        for row in self._data:
            if row[PARAM_COL.FROM] <= duration <= row[PARAM_COL.TO]:
                return row

    def get_scalar_param(self, p, duration):
        """

        Args:
            p (str): name of the parameter 'u' or 'w'
            duration (float | int): in minutes

        Returns:
            (float, float): u, w
        """
        row = self.get_row(duration)

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

        Args:
            duration (numpy.ndarray| list | float | int): in minutes

        Returns:
            (float, float): u, w
        """
        if isinstance(duration, (list, np.ndarray)):
            func = self.get_array_param
        else:
            func = self.get_scalar_param

        return (func(p, duration) for p in PARAM.U_AND_W)

    @classmethod
    def from_interim_results_file(cls, interim_results_fn, worksheet=DWA):
        """DEPRECIATED: for compatibility reasons. To use the old file and convert it to the new parameters file"""
        interim_results = pd.read_csv(interim_results_fn, index_col=0, header=0)
        p = cls.from_interim_results(interim_results, worksheet=worksheet)
        return p
