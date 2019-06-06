from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

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
from .sww_utils import remove_timezone, guess_freq, year_delta, rain_events, agg_events, event_duration, \
    resample_rain_series, rain_bar_plot
from idf_analysis.plot_helpers import idf_bar_axes


########################################################################################################################
class IntensityDurationFrequencyAnalyse:
    """
    heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)

    This program reads the measurement data of the rainfall
    and calculates the distribution of the rainfall as a function of the return period and the duration
    
    for duration steps up to 12 hours (and more) and return period in a range of '0.5a <= T_n <= 100a'
    """

    def __init__(self, series_kind=PARTIAL, worksheet=DWA, output_path=None, extended_durations=False,
                 output_filename=None, auto_save=False, unix=False, **kwargs):
        """
        heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)

        This program reads the measurement data of the rainfall
        and calculates the distribution of the rainfall as a function of the return period and the duration

        for duration steps up to 12 hours (and more) and return period in a range of '0.5a <= T_n <= 100a'

        Args:
            series_kind (str): ['partial', 'annual']
            worksheet (str): ['DWA-A_531', 'ATV-A_121', 'DWA-A_531_advektiv']
            output_path (str): path to directory where the (interim-)results get saved
            extended_durations (bool): add [720, 1080, 1440, 2880, 4320, 5760, 7200, 8640] minutes to the calculation
            output_filename (str): id/label/name of the series
            auto_save (bool): if the interim-results get saved
            unix (bool): using ',' (comma) for .csv-files else ';' (semicolon)
            **kwargs: not in use
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

        self._my_return_periods_frame = None
        # self.duration_steps = pd.to_timedelta(self.duration_steps, unit='m')

    # __________________________________________________________________________________________________________________
    @property
    def file_stamp(self):
        """
        Returns:
            str: default filename for the (interim-)results
        """
        return '_'.join(
            [self.data_base, self.worksheet, self.series_kind])  # , "{:0.0f}a".format(self.measurement_period)])

    # __________________________________________________________________________________________________________________
    @property
    def output_filename(self):
        """
        Returns:
            str: filename/-path for the (interim-)results
        """
        if self._output_filename is None:
            return path.join(self._output_path, self.file_stamp)
        else:
            return path.join(self._output_path, self._output_filename)

    # __________________________________________________________________________________________________________________
    @property
    def measurement_period(self):
        """
        Returns:
            float: measuring time in years
        """
        if self.series is None:
            return np.NaN
        datetime_index = self.series.index
        return (datetime_index[-1] - datetime_index[0]) / year_delta(years=1)

    # __________________________________________________________________________________________________________________
    def set_series(self, series, name=None):
        """
        set the series for the analysis

        Args:
            series (pandas.Series): precipitation time-series
            name (str): name of the series (used for the result filenames)
        """
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

    # __________________________________________________________________________________________________________________
    @property
    def interim_results(self):
        """
        get the interim results
        if previously saved read the file, else calculate the interim results
        and if `auto_save` if TRUE save the results to a file

        Returns:
            pandas.DataFrame: interim results
        """
        if self._interim_results is None:
            inter_file = self.output_filename + '_interim_results.csv'
            if path.isfile(inter_file):
                self._interim_results = pd.read_csv(inter_file, index_col=0, skipinitialspace=True,
                                                    **csv_args(self._unix))
            else:
                # save the parameter of the distribution function in the interim results
                if self.series is None:
                    raise ImportError('No Series was defined!')
                self._interim_results = calculate_u_w(self.series, self.duration_steps, self.measurement_period,
                                                      self.series_kind)
                if self._auto_save:
                    self._interim_results.to_csv(inter_file, **csv_args(self._unix))

        return self._interim_results

    # __________________________________________________________________________________________________________________
    @property
    def parameter(self):
        """
        Returns:
            pandas.DataFrame: calculation parameters
        """
        if self._parameter is None:
            self._parameter = get_parameter(self.interim_results, self.worksheet)
        return self._parameter

    def save_parameters(self):
        """
        save parameters as .csv-file to the workspace/output_path
        """
        par_file = self.output_filename + '_parameter.csv'
        if not path.isfile(par_file):
            self.parameter.to_csv(par_file, index=False, **csv_args(self._unix))

    # __________________________________________________________________________________________________________________
    def get_u_w(self, duration):
        """
        calculate the u and w parameters depending on the durations

        Args:
            duration (numpy.ndarray | float | int): in minutes

        Returns:

        """
        return get_u_w(duration, self.parameter, self.interim_results)

    # __________________________________________________________________________________________________________________
    def save_u_w(self, durations=None):
        if durations is None:
            durations = self.duration_steps

        fn = self.output_filename + '_results_u_w.csv'
        u, w = self.get_u_w(durations)
        df = pd.DataFrame(index=durations)
        df.index.name = COL.DUR
        df['u'] = u
        df['w'] = w
        df.to_csv(fn, **csv_args(self._unix))

    # __________________________________________________________________________________________________________________
    def depth_of_rainfall(self, duration, return_period):
        """
        calculate the height of the rainfall h in L/m² = mm

        Args:
            duration (float | np.array | pd.Series): duration: in minutes
            return_period (float): in years

        Returns:
            float: height of the rainfall h in L/m² = mm
        """
        u, w = self.get_u_w(duration)
        return depth_of_rainfall(u, w, self.series_kind, return_period)

    # __________________________________________________________________________________________________________________
    def print_depth_of_rainfall(self, duration, return_period):
        """
        calculate and print the height of the rainfall in [l/m² = mm]

        Args:
            duration (float): in minutes
            return_period (float): in years
        """
        print('Resultierende Regenhöhe h_N(T_n={:0.1f}a, D={:0.1f}min) = {:0.2f} mm'
              ''.format(return_period, duration, self.depth_of_rainfall(duration, return_period)))

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
    def r_720_1(self):
        return self.rain_flow_rate(duration=720, return_period=1)

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
    def write_table(self, durations=None, return_periods=None, add_names=False):
        table = self.result_table(durations=durations, return_periods=return_periods, add_names=add_names)
        fn = self.output_filename + '_results_h_N.csv'

        print(table.round(1).to_string())

        table.to_csv(fn, **csv_args(self._unix), float_format='%0.2f')

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
    def result_figure(self, min_duration=5.0, max_duration=720.0, logx=False, return_periods=None, color=False):
        duration_steps = np.arange(min_duration, max_duration + 1, 1)
        plt.style.use('bmh')

        if return_periods is None:
            return_periods = [0.5, 1, 10, 50, 100]
            return_periods = [1, 2, 5, 10, 50]

        table = self.result_table(durations=duration_steps, return_periods=return_periods)
        if color:
            table.columns.name = 'T$\\mathsf{_N}$ in (a)'
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
        ax.set_ylabel('Rainfall h$\\mathsf{_N}$ in (mm)')
        ax.set_title('IDF curves')

        fig = ax.get_figure()

        cm_to_inch = 2.54
        fig.set_size_inches(h=21 / cm_to_inch, w=29.7 / cm_to_inch)  # (11.69, 8.27)
        fig.tight_layout()
        return fig

    # __________________________________________________________________________________________________________________
    def result_plot(self, min_duration=5.0, max_duration=720.0, logx=False, fmt='png', show=False, color=False):
        fig = self.result_figure(min_duration=min_duration, max_duration=max_duration, logx=logx, color=color)
        fn = self.output_filename + '_idf_plot.' + fmt
        fig.savefig(fn, dpi=260)
        plt.close(fig)
        if show:
            show_file(fn)
        return fn

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
    def return_periods_frame(self, series, durations=None, printable_names=True):
        """

        Args:
            series (pandas.Series):
            durations (list, optional): list of durations in minutes which are of interest,
                                        default: pre defined durations
            printable_names (bool): if durations should be as readable in dataframe, else in minutes

        Returns:
            pandas.DataFrame: return periods depending of the duration per datetimeindex
        """
        if durations is None:
            durations = self.duration_steps

        df = pd.DataFrame(index=series.index)

        for d in durations:
            ts_sum = series.rolling(d, center=True, min_periods=1).sum()
            if printable_names:
                col = minutes_readable(d)
            else:
                col = d
            df[col] = self.get_return_period(height_of_rainfall=ts_sum, duration=d)

        return df

    @property
    def my_return_periods_frame_filename(self):
        return path.join(self.output_filename + '_return_periods.parquet')

    def my_return_periods_frame(self, durations=None, printable_names=True):
        fn = self.my_return_periods_frame_filename
        if self._my_return_periods_frame is None:
            if path.isfile(fn):
                self._my_return_periods_frame = pd.read_parquet(fn)
                self._my_return_periods_frame.columns = self._my_return_periods_frame.columns.to_series().astype(int)

            else:
                if durations is None:
                    durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320]

                self._my_return_periods_frame = self.return_periods_frame(self.series, durations,
                                                                          printable_names=printable_names)

                self._my_return_periods_frame.columns = self._my_return_periods_frame.columns.to_series().astype(str)
                self._my_return_periods_frame.to_parquet(fn, compression='brotli')
                self._my_return_periods_frame.columns = self._my_return_periods_frame.columns.to_series().astype(int)

        return self._my_return_periods_frame

    # __________________________________________________________________________________________________________________
    @classmethod
    def command_line_tool(cls):

        def _not_none(*args):
            return all(a is not None for a in args)

        user = heavy_rain_parser()

        # --------------------------------------------------
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

        # --------------------------------------------------
        idf = cls(series_kind=user.series_kind, worksheet=user.worksheet, output_path=out,
                  extended_durations=user.extended_duration, output_filename=name,
                  auto_save=auto_save, unix=user.unix)

        # --------------------------------------------------
        if user.r_720_1:
            d = 720
            t = 1

        # --------------------------------------------------
        ts = import_series(user.input)
        # ts.to_frame().to_parquet('.'.join(user.input.split('.')[:-1] + ['parquet']),engine='pyarrow')

        # --------------------------------------------------
        if _not_none(d) and not user.plot and not user.export_table:
            new_freq = floor(d / 4)
            ts = ts.resample('{:0.0f}T'.format(new_freq)).sum()

        # --------------------------------------------------
        idf.set_series(ts)

        # --------------------------------------------------
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

        # --------------------------------------------------
        if user.plot:
            idf.result_plot(show=True)

        # --------------------------------------------------
        if user.export_table:
            idf.write_table()

    @property
    def event_table_filename(self):
        return path.join(self.output_filename + '_events.csv')

    def get_events(self):
        if path.isfile(self.event_table_filename):
            events = pd.read_csv(self.event_table_filename, skipinitialspace=True)
            events[COL.START] = pd.to_datetime(events[COL.START])
            events[COL.END] = pd.to_datetime(events[COL.END])
            events[COL.DUR] = pd.to_timedelta(events[COL.DUR])
        else:
            series = self.series.resample('T').sum().fillna(0)
            events = rain_events(series)
            events[COL.LP] = agg_events(events, series, 'sum').round(1)
            events[COL.DUR] = event_duration(events)
            events = events.sort_values(by=COL.LP, ascending=False)
            events.to_csv(self.event_table_filename, index=False)

        return events

    def event_report(self, min_event_rain_sum=25, min_return_period=0.5, out_path=None, durations=None):
        """
        create pdf file with the biggest rain events
        for each event is represented by a plot of the rain series
        and a IDF analysis where the return periods are calculated

        Args:
            min_event_rain_sum (float): only events with a bigger rain sum will be created
            min_return_period (float): only events with a bigger return period will be analysed
                                       (the plot will be created anyway)
            out_path (str): path and filename of the final report
            durations (list[int]): analysed durations
                        (default: [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320])
        """
        if out_path is None:
            out_path = path.join(self.output_filename + '_idf_events.pdf')

        if durations is None:
            durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320]

        events = self.get_events()

        main_events = events[events[COL.LP] > min_event_rain_sum].copy()

        unit = 'mm'
        column_name = 'Precipitation'

        pdf = PdfPages(out_path)

        for _, event in main_events.iterrows():
            fig, caption = self.event_plot(event, durations=durations, min_return_period=min_return_period,
                                           unit=unit, column_name=column_name)

            # -------------------------------------
            fig.get_axes()[0].set_title(caption + '\n\n\n')

            # DIN A4
            fig.set_size_inches(w=8.27, h=11.69)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        pdf.close()

    def event_plot(self, event, durations=None, unit='mm', column_name='Precipitation', min_return_period=0.5):
        start = event[COL.START]
        end = event[COL.END]

        if durations is None:
            durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320]

        caption = 'rain event\nbetween {} and {}\nwith a total sum of {:0.1f} {}\nand a duration of {}'.format(
            start.strftime('%Y-%m-%d %H:%M'),
            end.strftime('%Y-%m-%d %H:%M'),
            event[COL.LP],
            unit,
            end - start)

        ts = self.series[start:end].resample('T').sum().fillna(0).copy()
        fig: plt.Figure = plt.figure()

        # -------------------------------------
        idf_table = self.my_return_periods_frame(printable_names=True)[start:end]

        if not (idf_table > min_return_period).any().any():
            max_period, duration = idf_table.max().max(), idf_table.max().idxmax()
            rain_ax = fig.add_subplot(111)
            caption += '\nThe maximum return period was {:0.2f}a\nat a duration of {}.'.format(max_period, duration)

        else:
            idf_bar_ax = fig.add_subplot(211)
            idf_bar_ax = idf_bar_axes(idf_bar_ax, idf_table, durations)
            rain_ax = fig.add_subplot(212, sharex=idf_bar_ax)

        # -------------------------------------
        ts_sum, minutes = resample_rain_series(ts)
        rain_ax = rain_bar_plot(ts_sum, rain_ax)
        rain_ax.set_ylabel('{} in [{}/{}min]'.format(column_name, unit, minutes if minutes != 1 else ''))
        rain_ax.set_xlim(ts.index[0], ts.index[-1])

        return fig, caption

    def return_period_scatter(self, min_return_period=0.5, out_path=None, durations=None):
        if out_path is None:
            out_path = path.join(self.output_filename + '_all_events_max_return_period.pdf')

        if durations is None:
            durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320]

        dur_short = durations[:durations.index(90)]
        dur_long = durations[durations.index(90):]

        events = self.get_events()
        events = events[events[COL.LP] > 25].copy()

        tn_long_list = dict()
        tn_short_list = dict()

        from my_helpers import check
        check()

        for _, event in events.iterrows():
            start = event[COL.START]
            end = event[COL.END]
            idf_table = self.my_return_periods_frame(durations, printable_names=False)[start:end]
            # idf_table[idf_table < min_return_period] = np.NaN

            tn = idf_table.loc[start:end]
            tn_short = tn[dur_short].max().max()
            tn_long = tn[dur_long].max().max()

            if tn_long > tn_short:
                tn_long_list[start] = tn_long
            else:
                tn_short_list[start] = tn_short

        print(tn_short_list)
        print(tn_long_list)

        check()
        fig, ax = plt.subplots()

        ax.scatter(x=list(tn_short_list.keys()), y=list(tn_short_list.values()), color='red')
        ax.scatter(x=list(tn_long_list.keys()), y=list(tn_long_list.values()), color='blue')
        fig: plt.Figure = ax.get_figure()

        ax.set_ylabel('Return Period in a')

        def line_in_legend(color=None, marker=None, lw=None, ls=None, **kwargs):
            return Line2D([0], [0], color=color, marker=marker, linewidth=lw, linestyle=ls, **kwargs)

        custom_lines = list()
        custom_lines.append(line_in_legend(color='red', marker='o', lw=0))
        custom_lines.append(line_in_legend(color='blue', marker='o', lw=0))
        # -----------------
        l1 = ax.legend(custom_lines, ['< 60 min', '> 60 min'], loc='best', title='max Duration')
        ax.add_artist(l1)

        # -----------------
        # DIN A4
        fig.set_size_inches(w=7, h=5)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
