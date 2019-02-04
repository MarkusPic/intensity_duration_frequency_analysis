__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import os
import warnings
from math import floor
from os import path
from webbrowser import open as show_file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .arg_parser import heavy_rain_parser
from .calculation_methods import get_u_w, get_parameter, calculate_u_w, depth_of_rainfall, minutes_readable
from .definitions import *
from .in_out import csv_args, import_series
from .sww_utils import remove_timezone, guess_freq, year_delta


########################################################################################################################
class IntensityDurationFrequencyAnalyse(object):
    """
    heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)

    This program reads the measurement data of the rainfall
    and calculates the distribution of the rainfall as a function of the return period and the duration
    
    for duration steps up to 12 hours (and more) and return period in a range of '0.5a <= T_n <= 100a'
    """

    def __init__(self, series_kind=PARTIAL, worksheet=DWA, output_path=None, extended_durations=False,
                 output_filename=None, auto_save=False, unix=False, **kwargs):
        """
        
        :param str series_kind:
        :param str worksheet:
        :param str output_path:
        :param bool extended_durations:
        :param str output_filename:
        """
        self.series_kind = series_kind
        self.worksheet = worksheet

        self.data_base = output_filename  # id/label/name of the series
        self.series = None

        self._parameter = None
        self._interim_results = None

        self._auto_save = auto_save

        self._unix = unix

        if not output_path:
            out_path = ''
        else:
            if path.isfile(output_path):
                output_path = path.dirname(output_path)
            out_path = path.join(output_path, '' if self.data_base is None else (self.data_base + '_') + 'data')

            if not path.isdir(out_path):
                os.mkdir(out_path)

        self._output_path = out_path
        self._output_filename = output_filename

        # sampling points of the duration steps
        self.duration_steps = np.array([5, 10, 15, 20, 30, 45, 60, 90, 180, 270, 360, 450, 600, 720])
        if extended_durations:
            duration_steps_extended = np.array([720, 1080, 1440, 2880, 4320, 5760, 7200, 8640])
            self.duration_steps = np.append(self.duration_steps, duration_steps_extended)

        # self.duration_steps = pd.to_timedelta(self.duration_steps, unit='m')

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def file_stamp(self):
        return '_'.join(
            [self.data_base, self.worksheet, self.series_kind])  # , "{:0.0f}a".format(self.measurement_period)])

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def measurement_period(self):
        """
        :return: measuring time in years
        :rtype: float
        """
        if self.series is None:
            return np.NaN
        datetime_index = self.series.index
        return (datetime_index[-1] - datetime_index[0]) / year_delta(years=1)

    @property
    def output_filename(self):
        if self._output_filename is None:
            return path.join(self._output_path, self.file_stamp)
        else:
            return path.join(self._output_path, self._output_filename)

    # ------------------------------------------------------------------------------------------------------------------
    def set_series(self, series, name=None):
        self.series = series
        if self.data_base is None:
            self.data_base = name

        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError('The series has to have a DatetimeIndex.')

        if series.index.tz is not None:
            self.series = remove_timezone(self.series)

        base_freq = guess_freq(self.series.index)
        base_min = base_freq / pd.Timedelta(minutes=1)
        self.duration_steps = self.duration_steps[self.duration_steps >= base_min]

        if round(self.measurement_period, 1) < 10:
            warnings.warn("The measurement period is too short. The results may be inaccurate! "
                          "It is recommended to use at least ten years. "
                          "(-> Currently {}a used)".format(self.measurement_period))

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def interim_results(self):
        if self._interim_results is None:
            inter_file = self.output_filename + '_interim_results.csv'
            if path.isfile(inter_file):
                self._interim_results = pd.read_csv(inter_file, index_col=0, **csv_args(self._unix))
            else:
                # save the parameter of the distribution function in the interim results
                if self.series is None:
                    raise ImportError('No Series was defined!')
                self._interim_results = calculate_u_w(self.series, self.duration_steps, self.measurement_period,
                                                      self.series_kind)
                if self._auto_save:
                    self._interim_results.to_csv(inter_file, **csv_args(self._unix))

        return self._interim_results

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def parameter(self):
        if self._parameter is None:
            self._parameter = get_parameter(self.interim_results, self.worksheet)
        return self._parameter

    def save_parameters(self):
        par_file = self.output_filename + '_parameter.csv'
        if not path.isfile(par_file):
            self.parameter.to_csv(par_file, index=False, **csv_args(self._unix))

    # ------------------------------------------------------------------------------------------------------------------
    def get_u_w(self, duration):
        return get_u_w(duration, self.parameter, self.interim_results)

    # ------------------------------------------------------------------------------------------------------------------
    def save_u_w(self, durations=None):
        if durations is None:
            durations = self.duration_steps

        fn = self.output_filename + '_results_u_w.csv'
        u, w = self.get_u_w(durations)
        df = pd.DataFrame(index=durations)
        df.index.name = 'duration'
        df['u'] = u
        df['w'] = w
        df.to_csv(fn, **csv_args(self._unix))

    # ------------------------------------------------------------------------------------------------------------------
    def depth_of_rainfall(self, duration, return_period):
        """
        calculate the height of the rainfall in [l/m² = mm]
        
        :param duration: in [min]
        :type duration: float | np.array | pd.Series

        :param return_period:
        :type return_period: float
        :return: height of the rainfall in [l/m² = mm]
        """
        u, w = self.get_u_w(duration)
        return depth_of_rainfall(u, w, self.series_kind, return_period)

    # ------------------------------------------------------------------------------------------------------------------
    def print_depth_of_rainfall(self, duration, return_period):
        """
        calculate and print the height of the rainfall in [l/m² = mm]

        :param duration: in minutes
        :type duration: float
        :param return_period: in years
        :type return_period: float
        """
        print('Resultierende Regenhöhe h_N(T_n={:0.1f}a, D={:0.1f}min) = {:0.2f} mm'
              ''.format(return_period, duration, self.depth_of_rainfall(duration, return_period)))

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def h2r(height_of_rainfall, duration):
        """
        calculate the specific rain flow rate in [l/(s*ha)]
        if 2 array-like parameters are give, a element-wise calculation will be made.
        So the length of the array must be the same.
        
        :param height_of_rainfall: in [mm]
        :type height_of_rainfall: float | np.array | pd.Series
        
        :param duration:
        :type duration: float | np.array | pd.Series
        
        :return: specific rain flow rate in [l/(s*ha)]
        :rtype: float | np.array | pd.Series
        """
        return height_of_rainfall / duration * (1000 / 6)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def r2h(rain_flow_rate, duration):
        """
        convert the rain flow rate to the height of rainfall in [mm]
        if 2 array-like parameters are give, a element-wise calculation will be made.
        So the length of the array must be the same.

        :param rain_flow_rate: in [l/(s*ha)]
        :type rain_flow_rate: float | np.array | pd.Series

        :param duration:
        :type duration: float | np.array | pd.Series

        :return: height of rainfall in [mm]
        :rtype: float | np.array | pd.Series
        """
        return rain_flow_rate * duration / (1000 / 6)

    def rain_flow_rate(self, duration, return_period):
        """
        convert the height of rainfall to the specific rain flow rate in [l/(s*ha)]
        if 2 array-like parameters are give, a element-wise calculation will be made.
        So the length of the array must be the same.

        :param duration:
        :type duration: float | np.array | pd.Series

        :param return_period:
        :type return_period: float

        :return: specific rain flow rate in [l/(s*ha)]
        :rtype: float | np.array | pd.Series
        """
        return self.h2r(height_of_rainfall=self.depth_of_rainfall(duration=duration, return_period=return_period),
                        duration=duration)

    def print_rain_flow_rate(self, duration, return_period):
        """
        calculate and print the flow rate of the rainfall in [l/(s*ha)]

        :param duration: in minutes
        :type duration: float

        :param return_period: in years
        :type return_period: float
        """
        print('Resultierende Regenspende r_N(T_n={:0.1f}a, D={:0.1f}min) = {:0.2f} L/(s*ha)'
              ''.format(return_period, duration, self.rain_flow_rate(duration, return_period)))

    def r_720_1(self):
        return self.rain_flow_rate(duration=720, return_period=1)

    # ------------------------------------------------------------------------------------------------------------------
    def get_return_period(self, height_of_rainfall, duration):
        """
        calculate the return period, when the height of rainfall and the duration are given

        :param height_of_rainfall: in [mm]
        :type height_of_rainfall: float

        :param duration:
        :type duration: float

        :return: return period
        :rtype: float
        """
        u, w = self.get_u_w(duration)
        return np.exp((height_of_rainfall - u) / w)

    # ------------------------------------------------------------------------------------------------------------------
    def get_duration(self, height_of_rainfall, return_period):
        """
        calculate the return period, when the height of rainfall and the duration are given

        :param height_of_rainfall: in [mm]
        :type height_of_rainfall: float

        :param duration:
        :type duration: float

        :return: return period
        :rtype: float
        """
        durs = np.arange(min(self.duration_steps), max(self.duration_steps), 0.5)
        h = self.depth_of_rainfall(durs, return_period)
        duration = np.interp(height_of_rainfall, h, durs)
        return duration

    # ------------------------------------------------------------------------------------------------------------------
    def result_table(self, durations=None, return_periods=None, add_names=False):
        if durations is None:
            durations = self.duration_steps

        if return_periods is None:
            return_periods = [0.5, 1, 2, 3, 5, 10, 15, 50, 100]

        result_table = pd.DataFrame(index=durations)
        for t in return_periods:
            result_table[t] = self.depth_of_rainfall(result_table.index, t)

        if add_names:
            result_table.index.name = 'duration (min)'
            result_table.columns = pd.MultiIndex.from_tuples([(rp, round(1 / rp, 3)) for rp in result_table.columns])
            result_table.columns.names = ['return period (a)', 'frequency (1/a)']
        return result_table

    def write_table(self, durations=None, return_periods=None, add_names=False):
        table = self.result_table(durations=durations, return_periods=return_periods, add_names=add_names)
        fn = self.output_filename + '_results_h_N.csv'

        print(table.round(1).to_string())

        table.to_csv(fn, **csv_args(self._unix), float_format='%0.2f')

    # ------------------------------------------------------------------------------------------------------------------
    def measured_points(self, return_time, interim_results=None, max_duration=None):
        """
        get the calculation results of the rainfall with u and w without the estimation of the formulation
        
        :param return_time: return period in [a]
        :type return_time: float | np.array | list | pd.Series
        
        :param interim_results: data with duration as index and u & w as data
        :type interim_results: pd.DataFrame
        
        :param max_duration: max duration in [min]
        :type max_duration: float
        
        :return: series with duration as index and the height of the rainfall as data
        :rtype: pd.Series
        """
        if interim_results is None:
            interim_results = self.interim_results.copy()

        if max_duration is not None:
            interim_results = interim_results.loc[:max_duration].copy()

        return pd.Series(index=interim_results.index,
                         data=interim_results['u'] + interim_results['w'] * np.log(return_time))

    # ------------------------------------------------------------------------------------------------------------------
    def result_figure(self, min_duration=5.0, max_duration=720.0, logx=False, return_periods=None, color=False):
        duration_steps = np.arange(min_duration, max_duration + 1, 1)
        plt.style.use('bmh')

        if return_periods is None:
            return_periods = [0.5, 1, 10, 50, 100]
            return_periods = [1, 2, 5, 10, 50]

        table = self.result_table(durations=duration_steps, return_periods=return_periods)
        ax = table.plot(color=(None if color else 'black'), logx=logx, legend=color)

        for _, return_time in enumerate(return_periods):
            p = self.measured_points(return_time, max_duration=max_duration)
            ax.plot(p, 'k' + 'x')

            if not color:
                x, y = list(p.tail(1).items())[0]
                ax.text(x + 10, y, '{} a'.format(return_time), verticalalignment='center', horizontalalignment='left',
                        # bbox=dict(facecolor='white', alpha=1.0, lw=1)
                        )

        ax.tick_params(axis='both', which='both', direction='out')
        ax.set_xlabel('Duration D in (min)')
        ax.set_ylabel('Rainfall h$\mathsf{_N}$ in (mm)')
        ax.set_title('IDF curves')

        fig = ax.get_figure()

        cm_to_inch = 2.54
        fig.set_size_inches(h=21 / cm_to_inch, w=29.7 / cm_to_inch)  # (11.69, 8.27)
        fig.tight_layout()
        return fig

    def result_plot(self, min_duration=5.0, max_duration=720.0, logx=False, fmt='png', show=False):
        fig = self.result_figure(min_duration=min_duration, max_duration=max_duration, logx=logx)
        fn = self.output_filename + '_idf_plot.' + fmt
        fig.savefig(fn, dpi=260)
        plt.close(fig)
        if show:
            show_file(fn)
        return fn

    def result_plot_XXX(self, min_duration=5.0, max_duration=720.0, logx=False, fmt='png', show=False):
        duration_steps = np.arange(min_duration, max_duration + 1, 1)
        colors = ['r', 'g', 'b', 'y', 'm']

        plt.style.use('bmh')

        return_periods = [0.5, 1, 10, 50, 100]
        return_periods = [1, 2, 5, 10, 50]

        table = self.result_table(durations=duration_steps, return_periods=return_periods)
        table.index = pd.to_timedelta(table.index, unit='m')
        ax = table.plot(color=colors, logx=logx)

        ax.tick_params(axis='both', which='both', direction='out')

        for i in range(len(return_periods)):
            return_time = return_periods[i]
            color = colors[i]
            p = self.measured_points(return_time, max_duration=max_duration)
            p.index = pd.to_timedelta(p.index, unit='m')
            ax.plot(p, color + 'x')

            # plt.text(max_duration * ((10 - offset) / 10), depth_of_rainfall(max_duration * ((10 - offset) / 10),
            #                                                                 return_time, parameter_1,
            #                                                                 parameter_2) + offset, '$T_n=$' + str(return_time))

        ax.set_xlabel('Dauerstufe $D$ in $[min]$')
        ax.set_ylabel('Regenhöhe $h_N$ in $[mm]$')
        ax.set_title('Regenhöhenlinien')
        ax.legend(title='$T_n$= ... [a]')

        if max_duration > 1.5 * 60:
            pass
        else:
            pass

        major_ticks = pd.to_timedelta(self.interim_results.loc[:max_duration].index, unit='m').total_seconds() * 1.0e9
        # minor_ticks = pd.date_range("00:00", "23:59", freq='15T').time
        # print(major_ticks)
        # exit()
        ax.set_xticks(major_ticks)
        # print(ax.get_xticks())
        from matplotlib import ticker

        def timeTicks(x, pos):
            x = pd.to_timedelta(x, unit='ns').total_seconds() / 60
            h = int(x / 60)
            m = int(x % 60)
            s = ''
            if h:
                s += '{}h'.format(h)
            if m:
                s += '{}min'.format(m)
            return s

        formatter = ticker.FuncFormatter(timeTicks)
        ax.xaxis.set_major_formatter(formatter)
        # print(ax.get_xticks())
        # plt.axis([0, max_duration, 0, depth_of_rainfall(max_duration,
        #                                                 return_periods[len(return_periods) - 1],
        #                                                 parameter_1, parameter_2) + 10])

        fig = ax.get_figure()

        fn = self.output_filename + '_plot.' + fmt

        cm_to_inch = 2.54
        fig.set_size_inches(h=21 / cm_to_inch, w=29.7 / cm_to_inch)  # (11.69, 8.27)
        fig.tight_layout()
        fig.savefig(fn, dpi=260)
        plt.close(fig)
        if show:
            show_file(fn)
        return fn

    def return_periods_frame(self, series, durations=None):
        if durations is None:
            durations = self.duration_steps

        df = pd.DataFrame(index=series.index)

        for d in durations:
            ts_sum = series.rolling(d, center=True, min_periods=1).sum()
            df[minutes_readable(d)] = self.get_return_period(height_of_rainfall=ts_sum, duration=d)

        return df

    @classmethod
    def command_line_tool(cls):

        def _not_none(*args):
            return all(a is not None for a in args)

        user = heavy_rain_parser()

        # --------------------------------------------------------------------------------------------------------------
        d = user.duration
        h = user.height_of_rainfall
        t = user.return_period
        out = user.output
        name = path.basename('.'.join(user.input.split('.')[:-1]))
        if out is None:
            out = ''
            # out = path.dirname(user.input)

        auto_save = False
        if user.plot or user.export_table:
            auto_save = True

        # --------------------------------------------------------------------------------------------------------------
        idf = cls(series_kind=user.series_kind, worksheet=user.worksheet, output_path=out,
                  extended_durations=user.extended_duration, output_filename=name,
                  auto_save=auto_save, unix=user.unix)

        # --------------------------------------------------------------------------------------------------------------
        if user.r_720_1:
            d = 720
            t = 1

        # --------------------------------------------------------------------------------------------------------------
        ts = import_series(user.input)
        # ts.to_frame().to_parquet('.'.join(user.input.split('.')[:-1] + ['parquet']),engine='pyarrow')

        # --------------------------------------------------------------------------------------------------------------
        if _not_none(d) and not user.plot and not user.export_table:
            new_freq = floor(d / 4)
            ts = ts.resample('{:0.0f}T'.format(new_freq)).sum()

        # --------------------------------------------------------------------------------------------------------------
        idf.set_series(ts)

        # --------------------------------------------------------------------------------------------------------------
        if _not_none(d, t):
            idf.print_depth_of_rainfall(duration=d, return_period=t)
            idf.print_rain_flow_rate(duration=d, return_period=t)
            pass
        elif _not_none(d, h):
            t = idf.get_return_period(h, d)
            print('The return period is {:0.1f} years.'.format(t))
            idf.print_depth_of_rainfall(duration=d, return_period=t)
            idf.print_rain_flow_rate(duration=d, return_period=t)

        elif _not_none(h, t):
            d = idf.get_duration(h, t)
            print('The duration is {:0.1f} minutes.'.format(d))
            idf.print_depth_of_rainfall(duration=d, return_period=t)
            idf.print_rain_flow_rate(duration=d, return_period=t)

        # --------------------------------------------------------------------------------------------------------------
        if user.plot:
            idf.result_plot(show=True)

        # --------------------------------------------------------------------------------------------------------------
        if user.export_table:
            idf.write_table()
