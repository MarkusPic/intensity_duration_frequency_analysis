import math
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

from .definitions import SERIES, METHOD, PARAM
from .extrem_value_series import ExtremValueSeries
from .idf_backend_abstract import IdfParametersABC
from .in_out import read_yaml
from .intensity_evaluation import IntensitiesExtractor
from .little_helpers import duration_steps_readable, minutes_readable
from .parameter_formulas import FORMULA_REGISTER, _Formula, LinearFormula


class IdfParameters(IdfParametersABC):
    def __init__(self, series_kind: str or Literal['partial', 'annual'] = SERIES.PARTIAL,
                 worksheet: str or Literal['KOSTRA', 'convective_vs_advective', 'ATV-A_121'] = METHOD.KOSTRA,
                 extended_durations=False):
        super().__init__(series_kind, worksheet, extended_durations)

        self.parameters_series = {}  # parameters u and w (distribution function) from the event series analysis
        self.parameters_final = {}  # parameters of the distribution function after the regression
        # {lower duration bound in minutes: {parameter u or w: {'function': function-name}}}

        if worksheet is not None:
            self.set_parameter_approaches_from_worksheet(worksheet)

        self._extrem_value_series = None  # type: ExtremValueSeries

    def calculation_done(self):
        return self.parameters_series

    def calc_from_series(self, series):
        self._intensity_extractor = IntensitiesExtractor(series, self.durations)
        self._extrem_value_series = ExtremValueSeries.from_intensity_extractor(self._intensity_extractor)

        self.parameters_series[PARAM.U] = []
        self.parameters_series[PARAM.W] = []
        for duration_integer, data in self._extrem_value_series.iter_duration_steps(self.series_kind):
            data  # type: dict # keys [x, y, series_kind]

            res = sps.linregress(data['x'], data['y'])
            self.parameters_series[PARAM.U].append(res.intercept)
            self.parameters_series[PARAM.W].append(res.slope)

        self.parameters_series[PARAM.U] = np.array(self.parameters_series[PARAM.U])
        self.parameters_series[PARAM.W] = np.array(self.parameters_series[PARAM.W])

        self._calc_params()  # sets self.parameters_final

    def reverse_engineering(self, idf_table, linear_interpolation=False):
        durations = idf_table.index.values
        u = idf_table[1].values
        w_dat = idf_table.sub(u, axis=0)
        # --
        w = []
        for dur in durations:
            dat = w_dat.loc[dur].dropna()
            log_tn_i = np.log(dat.index.values)
            w.append(sum(dat.values * log_tn_i) / sum(log_tn_i ** 2))

        # ---
        self.durations = durations
        self.parameters_series[PARAM.U] = np.array(u)
        self.parameters_series[PARAM.W] = np.array(w)

        if linear_interpolation:
            self.clear_parameter_approaches()
            self.add_parameter_approach(0, 'linear', 'linear')
            self._calc_params()  # from u,w to a_u, a_w, ...
        else:
            self._calc_params()

    def limit_duration(self, lowest=None, highest=None):
        bool_array = super().limit_duration(lowest, highest)
        for param in PARAM.U_AND_W:
            if param in self.parameters_series:
                self.parameters_series[param] = self.parameters_series[param][bool_array]

    def set_parameter_approaches_from_worksheet(self, worksheet):
        """
        Set approaches depending on the duration and the parameter.

        Args:
            worksheet (str): worksheet name for the analysis:
                - 'DWA-A_531'
                - 'ATV-A_121'
                - 'DWA-A_531_advektiv' (yet not implemented)

        Returns:
            list[dict]: table of approaches depending on the duration and the parameter
        """
        self.parameters_final = read_yaml(Path(__file__).parent / 'approaches' / (worksheet + '.yaml'))
        if self.parameters_series:
            self._calc_params()

    def add_parameter_approach(self, duration_bound, approach_u, approach_w):
        self.parameters_final[duration_bound] = {
            PARAM.U: {PARAM.FUNCTION: approach_u},
            PARAM.W: {PARAM.FUNCTION: approach_w},
        }

    def clear_parameter_approaches(self):
        self.parameters_final = {}

    def _iter_params(self):
        # parameters_final has the lower bound duration step as a key
        # making a zip with lower bound duration step and upper bound duration step
        # last upper bound duration step is infinity
        lower_bounds = list(self.parameters_final.keys())
        return zip(lower_bounds, lower_bounds[1:] + [np.inf])

    def _calc_params(self):
        """
        Calculate parameters a_u, a_w, b_u and b_w and add it to the dict.
        """
        for dur_lower_bound, dur_upper_bound in self._iter_params():

            # bool array for this duration range
            param_part = (self.durations >= dur_lower_bound) & (self.durations <= dur_upper_bound)

            if param_part.sum() <= 1:
                # Only one duration step in this duration range.
                # Only one value available in the series for this regression.
                del self.parameters_final[dur_lower_bound]
                continue

            # {'u': {'function': ...}, 'w': {'function': ...}}
            params_dur = self.parameters_final[dur_lower_bound]  # type: dict[str, (_Formula or dict)]

            # selected duration steps for the duration range
            dur = self.durations[param_part]

            for p in PARAM.U_AND_W:  # u or w
                # array of parameter values u|w
                values_series = self.parameters_series[p][param_part]

                # ---
                # convert dict with approach-string to Formula-object
                if not isinstance(params_dur[p], _Formula):
                    approach = params_dur[p][PARAM.FUNCTION]
                    if approach not in FORMULA_REGISTER:
                        # if approach is not defined in code - raise Error
                        raise NotImplementedError(f'{approach=}')

                    params_dur[p] = _Formula.from_dict(params_dur[p])  # init object

                if not params_dur[p].is_fitted:
                    # if fit was not already run
                    params_dur[p].fit(dur, values_series)

        # ---
        self._balance_parameter_change()

    def _balance_parameter_change(self):
        last = {PARAM.U: None, PARAM.W: None}
        for dur_lower_bound, dur_upper_bound in self._iter_params():
            duration_change = dur_upper_bound

            for p in PARAM.U_AND_W:  # u or w
                formula_this = self.parameters_final[dur_lower_bound][p]  # type: _Formula

                if dur_upper_bound not in self.parameters_final:
                    # last range
                    if last[p] is None:  # last was linear OR only one formula
                        continue
                    formula_this.fit(formula_this.durations, formula_this.values, *last[p])
                    continue

                formula_next = self.parameters_final[dur_upper_bound][p]  # type: _Formula

                if isinstance(formula_next, LinearFormula) and isinstance(formula_this, LinearFormula):
                    last[p] = None
                    continue
                elif isinstance(formula_next, LinearFormula):
                    value_mean = formula_this.get_param(duration_change)
                elif isinstance(formula_this, LinearFormula):
                    value_mean = formula_next.get_param(duration_change)
                else:
                    value_mean = (formula_this.get_param(duration_change) + formula_next.get_param(duration_change)) / 2

                if last[p] is None:
                    formula_this.fit(formula_this.durations, formula_this.values, duration_change, value_mean)
                else:
                    formula_this.fit(formula_this.durations, formula_this.values, [last[p][0], duration_change],
                                     [last[p][1], value_mean])
                last[p] = (duration_change, value_mean)

    def measured_points(self, return_periods, max_duration=None):
        """
        Get the calculation results of the rainfall with u and w without the estimation of the formula.

        Args:
            return_periods (float | np.ndarray | list | pd.Series): return period in [a]
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
                         data=interim_results[PARAM.U] + interim_results[PARAM.W] * np.log(return_periods))

    def interim_plot_parameters(self):
        dur = self.durations
        fig, axes = plt.subplots(2, sharex=True)
        for ax, label in zip(axes, PARAM.U_AND_W):
            y = self.parameters_series[label]
            ax.plot(dur, y, lw=0, marker='.')

            for lower, upper in self._iter_params():
                dur_ = dur[(dur >= lower) & (dur <= upper)]
                y_pred = self.parameters_final[lower][label].get_param(dur_)
                # y_pred = self.get_array_param(label, dur)
                ax.plot(dur_, y_pred, lw=1, label=f'${self.parameters_final[lower][label].latex_formula}$')

            ax.grid(ls=':', lw=0.5)

            ax.legend(title='Formula')
            for d in self.parameters_final:
                ax.axvline(d, color='black', lw=0.7, ls='--')
            ax.set_xscale('log', base=math.e)
            ax.set_xticks(dur)
            ax.set_xticklabels(duration_steps_readable(dur), rotation=90)
            ax.set_title(f'Parameter: {label}')
        return fig

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
        param = self.get_duration_section(duration, p)  # type: _Formula

        if param is None:
            return np.nan

        return param.get_param(duration)

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

    def get_depth_of_rainfall(self, duration, return_period):
        """
        calculate the height of the rainfall h in L/m² = mm (respectively the unit of the series)

        Args:
            duration (int | float | list | numpy.ndarray | pandas.Series): duration: in minutes
            return_period (float): in years

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: height of the rainfall h in L/m² = mm (respectively the unit of the series)
        """
        if self.series_kind == SERIES.ANNUAL:
            if return_period < 5:
                print('WARNING: Using an annual series and a return period < 5 a will result in faulty values!')

            if return_period <= 10:
                return_period = np.exp(1.0 / return_period) / (np.exp(1.0 / return_period) - 1.0)

            log_tn = -np.log(np.log(return_period / (return_period - 1.0)))

        else:
            log_tn = np.log(return_period)

        u, w = self.get_u_w(duration)
        return u + w * log_tn

    def get_return_period(self, height_of_rainfall, duration):
        """
        calculate the return period, when the height of rainfall and the duration are given

        Args:
            height_of_rainfall (float): in [mm]
            duration (int | float | list | numpy.ndarray | pandas.Series): in minutes

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: return period in years
        """
        u, w = self.get_u_w(duration)
        return np.exp((height_of_rainfall - u) / w)

    def to_dict(self):
        def to_basic(a):
            return list([round(float(i), 4) for i in a])

        return {
            PARAM.SERIES: self.series_kind,
            PARAM.DUR: to_basic(self.durations),
            PARAM.PARAMS_SERIES: {p: to_basic(l) for p, l in self.parameters_series.items()},
            PARAM.PARAMS_FINAL: self.parameters_final
        }

    @classmethod
    def from_yaml(cls, filename, worksheet=None):
        data = read_yaml(filename)
        p = cls(series_kind=data[PARAM.SERIES])
        p.durations = data[PARAM.DUR]
        # list to numpy.array
        p.parameters_series = {p: np.array(l) for p, l in data[PARAM.PARAMS_SERIES].items()}
        if PARAM.PARAMS_FINAL in data:
            p.parameters_final = data[PARAM.PARAMS_FINAL]
        else:
            p.set_parameter_approaches_from_worksheet(worksheet)

        p._calc_params()
        return p

    def interim_plot_series(self, ncols=3):
        n_plots = len(self.durations)
        n_cols = min(ncols, n_plots)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.suptitle(f'Interim Results of {self.series_kind}-series Plot')

        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, duration_integer in zip(axes, self.durations):
            data = self._extrem_value_series.data[duration_integer]
            x = data['x']  # probability
            y = data['y']  # intensity

            i_dur = list(self.durations).index(duration_integer)

            u = self.parameters_series[PARAM.U][i_dur]
            w = self.parameters_series[PARAM.W][i_dur]

            line = u + x * w

            ax.axhline(u, lw=0.7, color='black')
            ax.axvline(0, lw=0.7, color='black')

            ax.plot(x, line, label='Fitted model', color='C1', linewidth=2)
            ax.scatter(x, y, s=30, label='Observed data', color='C2', alpha=0.7)

            dur_plot = minutes_readable(duration_integer)

            ax.set_ylabel(f'Intensity (mm/{dur_plot})')
            ax.set_xlabel(r'$ln(T_n) = ln(\frac{L + 0.2}{k-0.4}*\frac{M}{L})$')
            ax.set_title(f'Duration: {dur_plot}\nu={u:0.2f}, w={w:0.2f}')

            ax2 = ax.twiny()
            xlim = ax.get_xlim()
            ax2.set_xlim(math.exp(xlim[0]), math.exp(xlim[1]))

            ax.legend()
            ax.grid(linestyle='--', alpha=0.7)

        # Hide empty subplots
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        return fig
