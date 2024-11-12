import math
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

from .definitions import SERIES, METHOD, PARAM
from .in_out import write_yaml, read_yaml
from .little_helpers import duration_steps_readable, minutes_readable, get_progress_bar
from .parameter_formulas import FORMULA_REGISTER, _Formula, register_formulas_to_yaml, LinearFormula
from .sww_utils import year_delta, guess_freq, rain_events, agg_events


class IdfParameters:
    def __init__(self, series_kind: str or Literal['partial', 'annual'] = SERIES.PARTIAL,
                 worksheet: str or Literal['KOSTRA', 'convective_vs_advective', 'ATV-A_121'] = METHOD.KOSTRA,
                 extended_durations=False):
        self.series_kind = series_kind

        self._durations = None
        self._set_default_durations(extended_durations)

        self.parameters_series = {}  # parameters u and w (distribution function) from the event series analysis
        self.parameters_final = {}  # parameters of the distribution function after the regression
        # {lower duration bound in minutes: {parameter u or w: {'function': function-name}}}

        if worksheet is not None:
            self.set_parameter_approaches_from_worksheet(worksheet)

        self._interims = None  # type: ExtremeValueParameters

    def calc_from_series(self, series):
        self._interims = ExtremeValueParameters(series, self.durations)

        self._interims.evaluate(self.series_kind)

        self.parameters_series[PARAM.U] = np.array(self._interims.u)
        self.parameters_series[PARAM.W] = np.array(self._interims.w)

        self._calc_params()

    def reverse_engineering(self, idf_table, linear_interpolation=False):
        durations = idf_table.index.values
        u = idf_table[1].values
        w_dat = idf_table.sub(u, axis=0)
        # --
        w = []
        for dur in durations:
            dat = w_dat.loc[dur]
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
        self.limit_durations_from_freq(freq_minutes)

    def limit_duration(self, lowest=None, highest=None):
        # bool_array = self.durations == False
        bool_array = np.array([False]*self.durations.size)
        if lowest is not None:
            bool_array |= self.durations >= lowest
        if highest is not None:
            bool_array |= self.durations <= highest

        self.durations = self.durations[bool_array]
        for param in PARAM.U_AND_W:
            if param in self.parameters_series:
                self.parameters_series[param] = self.parameters_series[param][bool_array]

    def limit_durations_from_freq(self, freq_minutes):
        # only multiple of freq
        # Aus DWA-M 531 Abschnitt 2:
        # > Da Niederschlagsmesser Tageswerte liefern, sind hier nur Auswertungen für Regendauern möglich, die ein Vielfaches von 24 h betragen.
        self.durations = self.durations[(self.durations % freq_minutes) == 0]

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
                    value_mean = (formula_this.get_param(duration_change) + formula_next.get_param(duration_change))/2

                if last[p] is None:
                    formula_this.fit(formula_this.durations, formula_this.values, duration_change, value_mean)
                else:
                    formula_this.fit(formula_this.durations, formula_this.values, [last[p][0], duration_change], [last[p][1], value_mean])
                last[p] = (duration_change, value_mean)

    def measured_points(self, return_periods, max_duration=None):
        """
        Get the calculation results of the rainfall with u and w without the estimation of the formula.

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

    def to_yaml(self, filename):
        register_formulas_to_yaml()

        def to_basic(a):
            return list([round(float(i), 4) for i in a])
        data = {
            PARAM.SERIES: self.series_kind,
            PARAM.DUR: to_basic(self.durations),
            PARAM.PARAMS_SERIES: {p: to_basic(l) for p, l in self.parameters_series.items()},
            PARAM.PARAMS_FINAL: self.parameters_final
        }
        write_yaml(data, filename)

    def pprint(self):
        from pprint import pprint

        def to_basic(a):
            return list([round(float(i), 4) for i in a])

        pprint({
            PARAM.SERIES: self.series_kind,
            PARAM.DUR: to_basic(self.durations),
            PARAM.PARAMS_SERIES: {p: to_basic(l) for p, l in self.parameters_series.items()},
            PARAM.PARAMS_FINAL: self.parameters_final
        })

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


class ExtremeValueParameters:
    def __init__(self, series, duration_steps):
        self.series = series
        self.duration_steps = duration_steps

        self.base_frequency = guess_freq(self.series.index)  # DateOffset/Timedelta
        self.base_delta = pd.Timedelta(self.base_frequency)

        # measuring time in years
        measurement_start, measurement_end = self.series.index[[0, -1]]
        self.measurement_period = (measurement_end - measurement_start) / year_delta(years=1)
        if round(self.measurement_period, 1) < 10:
            warnings.warn("The measurement period is too short. The results may be inaccurate! "
                          "It is recommended to use at least ten years. "
                          f"(-> Currently {self.measurement_period}a used)")

        # acc. to DWA-A 531 chap. 4.2:
        # The values must be independent of each other for the statistical evaluations.
        # estimated four hours acc. (Schilling, 1984)
        # for larger durations - use the duration as minimal gap
        self.min_event_gap = pd.Timedelta(hours=4)

        # Use only the (2-3 multiplied with the number of measuring years) of the biggest
        # values in the database (-> acc. to ATV-A 121 chap. 4.3; DWA-A 531 chap. 4.4).
        # As a requirement for the extreme value distribution.
        self.threshold_sample_size = int(math.floor(self.measurement_period * math.e))

        self._data = {}
        self.u = []
        self.w = []
        self._intensities = {}

    @staticmethod
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

    @staticmethod
    def _plotting_formula(k, l, m):
        """
        plotting function acc. to DWA-A 531 chap. 5.1.3 for the partial series

        Args:
            k (float or np.ndarray): running index
            l (float or np.ndarray): sample size
            m (float or np.ndarray): measurement period

        Returns:
            float: estimated empirical return period
        """
        return (l + 0.2) * m / ((k - 0.4) * l)

    def evaluate(self, series_kind: str or Literal['partial', 'annual'] = SERIES.PARTIAL):
        """
        Statistical analysis for each duration step.

        acc. to DWA-A 531 chap. 5.1

        Save the parameters of the distribution function as interim results.

        Args:
            series_kind (str): which kind of series should be used to evaluate the extreme values.
        """
        pbar = get_progress_bar(self.duration_steps, desc='Calculating Parameters u and w')

        for duration_integer in pbar:
            try:
                pbar.set_description(f'Calculating Parameters u and w for duration {duration_integer:0.0f}')
            except: ...

            if series_kind == SERIES.ANNUAL:
                x, y = self.annual_series(duration_integer)
            elif series_kind == SERIES.PARTIAL:
                x, y = self.partial_series(duration_integer)
            else:
                raise NotImplementedError(f"Unknown series kind {series_kind}")

            res = sps.linregress(x, y)
            self.u.append(res.intercept)
            self.w.append(res.slope)

            self._data[duration_integer] = {'x': x, 'y': y, 'u': res.intercept, 'w': res.slope, 'series_kind': series_kind}

    def get_intensities(self, duration_integer: int):
        if duration_integer not in self._intensities:
            duration = pd.Timedelta(minutes=duration_integer)

            if duration < self.base_delta:
                return

            if duration < self.min_event_gap:
                min_gap = self.min_event_gap
            else:
                min_gap = duration

            events = rain_events(self.series, min_gap=min_gap)

            # Correction factor acc. to DWA-A 531 chap. 4.3
            improve = self._improve_factor(duration / self.base_delta)

            roll_sum = self.series.rolling(duration).sum()

            events['i'] = agg_events(events, roll_sum, 'max') * improve
            events['ix'] = agg_events(events, roll_sum, 'idxmax')
            self._intensities[duration_integer] = events.set_index('ix')['i']
        return self._intensities[duration_integer]

    def annual_series(self, duration_integer: int):
        """
        Create an annual series of the maximum overlapping sum per year and calculate the "u" and "w" parameters.

        acc. to DWA-A 531 chap. 5.1.5

        Gumbel distribution | https://en.wikipedia.org/wiki/Gumbel_distribution

        Args:
            duration_integer (int): Duration step in minutes.

        Returns:
            dict[str, float]: Parameter u and w from the annual series for a specific duration step as a tuple.
        """
        annually_series = self.get_intensities(duration_integer).resample('YE').max().sort_values(ascending=False).values
        x = -np.log(np.log((annually_series.size + 0.2) / (annually_series.size - sps.rankdata(annually_series)[::-1] + 0.6)))
        return x, annually_series

    def partial_series(self, duration_integer: int):
        """
        Create a partial series of the largest overlapping sums and calculate the "u" and "w" parameters.

        acc. to DWA-A 531 chap. 5.1.4

        Exponential distribution | https://en.wikipedia.org/wiki/Exponential_distribution

        Args:
            duration_integer (int): Duration step in minutes.

        Returns:
            dict[str, float]: parameter u and w from the partial series for a specific duration step as a tuple
        """
        partially_series = self.get_intensities(duration_integer).sort_values(ascending=False).values

        if partially_series.size < self.threshold_sample_size:
            warnings.warn('Fewer events in series than recommended for extreme value analysis. Use the results with mindfulness.')
        else:
            partially_series = partially_series[:self.threshold_sample_size]

        x = np.log(self._plotting_formula(sps.rankdata(partially_series)[::-1], partially_series.size, self.measurement_period))

        return x, partially_series

    def plot_series(self, ncols=3):
        n_plots = len(self.duration_steps)
        n_cols = min(ncols, n_plots)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.suptitle(f'Interim Results of {self._data[self.duration_steps[0]]["series_kind"]}-series Plot')

        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, duration_integer in zip(axes, self.duration_steps):
            data = self._data[duration_integer]
            x = data['x']  # probability
            y = data['y']  # intensity

            line = data['u'] + x * data['w']

            ax.axhline(data["u"], lw=0.7, color='black')
            ax.axvline(0, lw=0.7, color='black')

            ax.plot(x, line, label='Fitted model', color='C1', linewidth=2)
            ax.scatter(x, y, s=30, label='Observed data', color='C2', alpha=0.7)

            dur_plot = minutes_readable(duration_integer)

            ax.set_ylabel(f'Intensity (mm/{dur_plot})')
            ax.set_xlabel(r'$ln(T_n) = ln(\frac{L + 0.2}{k-0.4}*\frac{M}{L})$')
            ax.set_title(f'Duration: {dur_plot}\nu={data["u"]:0.2f}, w={data["w"]:0.2f}')

            ax2 = ax.twiny()
            xlim = ax.get_xlim()
            ax2.set_xlim(math.exp(xlim[0]), math.exp(xlim[1]))

            ax.legend()
            ax.grid(linestyle='--', alpha=0.7)

        # Hide empty subplots
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        return fig
