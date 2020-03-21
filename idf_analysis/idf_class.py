__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import warnings
from math import floor
from os import path
from webbrowser import open as show_file
from scipy.optimize import newton

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from tqdm import tqdm

from .arg_parser import heavy_rain_parser
from .calculation_methods import get_u_w, get_parameters, calculate_u_w, depth_of_rainfall
from .little_helpers import minutes_readable, height2rate, delta2min
from .definitions import *
from .in_out import import_series, write_yaml, read_yaml
from .sww_utils import (remove_timezone, guess_freq, rain_events, agg_events, event_duration,
                        resample_rain_series, rain_bar_plot, IdfError, )
from .plot_helpers import idf_bar_axes
from .additional_scripts import measured_points


########################################################################################################################
class IntensityDurationFrequencyAnalyse:
    """
    heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)

    This program reads the measurement data of the rainfall
    and calculates the distribution of the rainfall as a function of the return period and the duration
    
    for duration steps up to 12 hours (and more) and return period in a range of '0.5a <= T_n <= 100a'
    """

    def __init__(self, series_kind=PARTIAL, worksheet=DWA, extended_durations=False):
        """
        heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)

        This program reads the measurement data of the rainfall
        and calculates the distribution of the rainfall as a function of the return period and the duration

        for duration steps up to 12 hours (and more) and return period in a range of '0.5a <= T_n <= 100a'

        Args:
            series_kind (str): ['partial', 'annual']
            worksheet (str): ['DWA-A_531', 'ATV-A_121', 'DWA-A_531_advektiv']
            extended_durations (bool): add [720, 1080, 1440, 2880, 4320, 5760, 7200, 8640] minutes to the calculation
        """
        self.series_kind = series_kind
        self.worksheet = worksheet

        self._series = None  # type: pd.Series # rain time-series
        self._freq = None  # frequency of the rain series

        self._parameter = None  # how to calculate the idf curves
        self._return_periods_frame = None  # type: pd.DataFrame # with return periods of all given durations
        self._rain_events = None

        # sampling points of the duration steps in minutes
        self._duration_steps = [5, 10, 15, 20, 30, 45, 60]
        self._duration_steps += [i * 60 for i in [1.5, 3, 4.5, 6, 7.5, 10, 12]]  # duration steps in hours
        if extended_durations:
            self._duration_steps += [i * 60 * 24 for i in
                                     [0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]  # duration steps in hours

    # __________________________________________________________________________________________________________________
    @property
    def series(self):
        if self._series is None:
            raise IdfError('No Series defined for IDF-Analysis!')
        return self._series

    @series.setter
    def series(self, series):
        self._series = series

    def set_series(self, series):
        """
        set the series for the analysis

        Args:
            series (pandas.Series): precipitation time-series
        """
        if not isinstance(series, pd.Series):
            raise IdfError('The series has to be a pandas Series.')

        if not isinstance(series.index, pd.DatetimeIndex):
            raise IdfError('The series has to have a DatetimeIndex.')

        if series.index.tz is not None:
            series = remove_timezone(series)

        series = series.replace(0, np.NaN).dropna()
        self._freq = guess_freq(series.index)
        freq_minutes = delta2min(self._freq)
        self.duration_steps = list(filter(lambda d: d >= freq_minutes, self.duration_steps))
        self.series = series

    # __________________________________________________________________________________________________________________
    @property
    def duration_steps(self):
        """
        get duration steps (in minutes) for the parameter calculation and basic evaluations
        Returns:
            list | numpy.ndarray: duration steps in minutes
        """
        if self._duration_steps is None:
            raise IdfError('No Series defined for IDF-Analysis!')
        return self._duration_steps

    @duration_steps.setter
    def duration_steps(self, durations):
        """
        set duration steps (in minutes) for the parameter calculation and basic evaluations
        Args:
            durations (list | numpy.ndarray): duration steps in minutes
        """
        if not isinstance(durations, (list, np.ndarray)):
            raise IdfError('Duration steps have to be {} got "{}"'.format((list, np.ndarray), type(durations)))
        self._duration_steps = durations

    # __________________________________________________________________________________________________________________
    @property
    def parameters(self):
        """
        get the calculation parameters

        calculation method depending on the used worksheet and on the duration
        also the parameters for each method

        to save some time and save the parameters with
        :func:`IntensityDurationFrequencyAnalyse.write_parameters`
        and read them later with :func:`IntensityDurationFrequencyAnalyse.read_parameters`

        Returns:
            list[dict]: calculation parameters
        """
        if self._parameter is None:
            interim_results = calculate_u_w(self.series, self.duration_steps, self.series_kind)
            self._parameter = get_parameters(interim_results, self.worksheet)
        return self._parameter

    def write_parameters(self, filename):
        """
        save parameters as yaml-file to save computation time.

        Args:
            filename (str): filename for the parameters yaml-file
        """
        write_yaml(self.parameters, filename)

    def read_parameters(self, filename):
        """
        read parameters from a .yaml-file to save computation time.
        extract interim results from parameters

        Args:
            filename (str): filename of the parameters yaml-file
        """
        self._parameter = read_yaml(filename)

    def auto_save_parameters(self, filename):
        """auto-save the parameters as a yaml-file to save computation time."""
        if path.isfile(filename):
            self.read_parameters(filename)
        else:
            self.write_parameters(filename)

    # __________________________________________________________________________________________________________________
    def get_u_w(self, duration):
        """
        calculate the u and w parameters depending on the durations

        Args:
            duration (int | float | list | numpy.ndarray | pandas.Series): in minutes

        Returns:
            (numpy.ndarray, numpy.ndarray) | (float, float): u and w
        """
        return get_u_w(duration, self.parameters)

    # __________________________________________________________________________________________________________________
    def depth_of_rainfall(self, duration, return_period):
        """
        calculate the height of the rainfall h in L/m² = mm

        Args:
            duration (int | float | list | numpy.ndarray | pandas.Series): duration: in minutes
            return_period (float): in years

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: height of the rainfall h in L/m² = mm
        """
        u, w = self.get_u_w(duration)
        return depth_of_rainfall(u, w, return_period, series_kind=self.series_kind)

    # __________________________________________________________________________________________________________________
    def rain_flow_rate(self, duration, return_period):
        """
        convert the height of rainfall to the specific rain flow rate in [l/(s*ha)]
        if 2 array-like parameters are give, a element-wise calculation will be made.
        So the length of the array must be the same.

        Args:
            duration (int | float | list | numpy.ndarray | pandas.Series): in minutes
            return_period (float): in years

        Returns:
                int | float | list | numpy.ndarray | pandas.Series: specific rain flow rate in [l/(s*ha)]
        """
        return height2rate(height_of_rainfall=self.depth_of_rainfall(duration=duration, return_period=return_period),
                           duration=duration)

    # __________________________________________________________________________________________________________________
    def r_720_1(self):
        """
        rain flow rate in [l/(s*ha)] for a duration of 12h and a return period of 1 year

        Returns:
            float: rain flow rate in [l/(s*ha)]
        """
        return self.rain_flow_rate(duration=720, return_period=1)

    # __________________________________________________________________________________________________________________
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

    # __________________________________________________________________________________________________________________
    def get_duration(self, height_of_rainfall, return_period):
        """
        calculate the duration, when the height of rainfall and the return period are given

        Args:
            height_of_rainfall (float): in [mm]
            return_period (float): in years

        Returns:
            float: duration in minutes
        """
        return newton(lambda d: self.depth_of_rainfall(d, return_period) - height_of_rainfall, x0=1)

    # __________________________________________________________________________________________________________________
    def result_table(self, durations=None, return_periods=None, add_names=False):
        """
        get a standard idf table of rainfall depth with return periods as columns and durations as rows

        Args:
            durations (list | numpy.ndarray): list of durations in minutes for the table
            return_periods (list): list of return periods in years for the table
            add_names (bool): weather to use expressive names as index-&column-label

        Returns:
            pandas.DataFrame: idf table
        """
        if durations is None:
            durations = self.duration_steps

        if return_periods is None:
            return_periods = [0.5, 1, 2, 3, 5, 10, 15, 50, 100]

        result_table = dict()
        for t in return_periods:
            result_table[t] = self.depth_of_rainfall(durations, t)

        result_table = pd.DataFrame(result_table, index=durations)

        if add_names:
            result_table.index.name = 'duration (min)'
            result_table.columns = pd.MultiIndex.from_tuples([(rp, round(1 / rp, 3)) for rp in result_table.columns])
            result_table.columns.names = ['return period (a)', 'frequency (1/a)']
        return result_table

    ####################################################################################################################
    def result_figure(self, min_duration=5.0, max_duration=720.0, logx=False, return_periods=None, color=False):
        duration_steps = np.arange(min_duration, max_duration + 1, 1)
        plt.style.use('bmh')

        if return_periods is None:
            # return_periods = [0.5, 1, 10, 50, 100]
            return_periods = [1, 2, 5, 10, 50]

        table = self.result_table(durations=duration_steps, return_periods=return_periods)
        if color:
            table.columns.name = 'T$\\mathsf{_N}$ in (a)'
        ax = table.plot(color=(None if color else 'black'), logx=logx, legend=color)

        for _, return_time in enumerate(return_periods):
            p = measured_points(self, return_time, max_duration=max_duration)
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
        return fig, ax

    ####################################################################################################################
    def get_return_periods_frame(self, series, durations=None):
        """

        Args:
            series (pandas.Series):
            durations (list): list of durations in minutes which are of interest (default: pre defined durations)

        Returns:
            pandas.DataFrame: return periods depending of the duration per datetimeindex
        """
        if durations is None:
            durations = self.duration_steps

        df = pd.DataFrame(index=series.index)

        freq = delta2min(guess_freq(series.index))
        for d in durations:
            if d % freq != 0:
                warnings.warn('Using durations (= {} minutes), '
                              'which are not a multiple of the base frequency (= {} minutes) of the series, '
                              'will lead to misinterpretations.'.format(d, freq))
            ts_sum = series.rolling(pd.Timedelta(minutes=d)).sum()
            df[d] = self.get_return_period(height_of_rainfall=ts_sum, duration=d)

        # printable_names (bool): if durations should be as readable in dataframe, else in minutes
        # df = df.rename(minutes_readable, axis=0)

        return df.round(1)

    @property
    def return_periods_frame(self):
        """
        get the return periods over the whole time-series for the default duration steps.

        Returns:
            pandas.DataFrame: data-frame of return periods where the columns are the duration steps
        """
        if self._return_periods_frame is None:
            self._return_periods_frame = self.get_return_periods_frame(self.series)
        return self._return_periods_frame

    def write_return_periods_frame(self, filename, **kwargs):
        """save the return-periods dataframe as a parquet-file to save computation time."""
        df = self.return_periods_frame.copy()
        df.columns = df.columns.to_series().astype(str)
        df.to_parquet(filename, **kwargs)

    def read_return_periods_frame(self, filename, **kwargs):
        """read the return-periods dataframe as a parquet-file to save computation time."""
        df = pd.read_parquet(filename, **kwargs)
        df.columns = df.columns.to_series().astype(int)
        self._return_periods_frame = df

    def auto_save_return_periods_frame(self, filename):
        """auto-save the return-periods dataframe as a parquet-file to save computation time."""
        if path.isfile(filename):
            self.read_return_periods_frame(filename)
        else:
            self.write_return_periods_frame(filename)

    ####################################################################################################################
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
        fn = path.join(out, '{label}_data', '{label}_{file_name}')
        # --------------------------------------------------
        idf = cls(series_kind=user.series_kind, worksheet=user.worksheet, extended_durations=user.extended_duration,
                  output_directory=out, output_label=name)

        # --------------------------------------------------
        if user.r_720_1:
            d = 720
            t = 1

        # --------------------------------------------------
        ts = import_series(user.input)

        # --------------------------------------------------
        # for faster computation
        if _not_none(d) and not user.plot and not user.export_table:
            new_freq = floor(d / 4)
            ts = ts.resample('{:0.0f}T'.format(new_freq)).sum().replace(0, np.Nan).dropna()

        # --------------------------------------------------
        idf.set_series(ts)

        # --------------------------------------------------
        if user.plot or user.export_table:
            idf.auto_save_parameters(fn.format(label=name, file_name='parameter.yaml'))

        # --------------------------------------------------
        if _not_none(d, t):
            pass

        elif _not_none(d, h):
            t = idf.get_return_period(h, d)
            print('The return period is {:0.1f} years.'.format(t))

        elif _not_none(h, t):
            d = idf.get_duration(h, t)
            print('The duration is {:0.1f} minutes.'.format(d))

        print('Resultierende Regenhöhe h_N(T_n={t:0.1f}a, D={d:0.1f}min) = {h:0.2f} mm'
              ''.format(t=t, d=d, h=idf.depth_of_rainfall(d, t)))
        print('Resultierende Regenspende r_N(T_n={t:0.1f}a, D={d:0.1f}min) = {r:0.2f} L/(s*ha)'
              ''.format(t=t, d=d, r=idf.rain_flow_rate(d, t)))
        # --------------------------------------------------
        if user.plot:
            fig, ax = idf.result_figure()
            plot_fn = fn.format(label=name, file_name='_idf_plot.png')
            fig.savefig(plot_fn, dpi=260)
            plt.close(fig)
            show_file(plot_fn)

        # --------------------------------------------------
        if user.export_table:
            table = idf.result_table(add_names=True)
            print(table.round(1).to_string())
            table.to_csv(fn.format(label=name, file_name='results_h_N.csv'), sep=';', decimal='.', float_format='%0.2f')

    ####################################################################################################################
    @property
    def rain_events(self):
        """
        get the all the rain events of the time-series

        Returns:
            pandas.DataFrame: data-frame of events with start-, end-time and duration
        """
        if self._rain_events is None:
            events = rain_events(self.series)
            events[COL.DUR] = event_duration(events)
            events[COL.LP] = agg_events(events, self.series, 'sum').round(1)
            # events = events.sort_values(by=COL.LP, ascending=False)
            self._rain_events = events

        return self._rain_events

    def write_rain_events(self, filename, sep=';', decimal='.'):
        """save the rain-events dataframe as a csv-file for external use or to save computation time."""
        self.rain_events.to_csv(filename, index=False, sep=sep, decimal=decimal)

    def read_rain_events(self, filename, sep=';', decimal='.'):
        """read the rain-events dataframe as a csv-file to save computation time."""
        events = pd.read_csv(filename, skipinitialspace=True, sep=sep, decimal=decimal)
        events[COL.START] = pd.to_datetime(events[COL.START])
        events[COL.END] = pd.to_datetime(events[COL.END])
        events[COL.DUR] = pd.to_timedelta(events[COL.DUR])
        self._rain_events = events

    def auto_save_rain_events(self, filename, sep=';', decimal='.'):
        """auto-save the rain-events dataframe as a csv-file to save computation time."""
        if path.isfile(filename):
            self.read_rain_events(filename, sep=sep, decimal=decimal)
        else:
            self.write_rain_events(filename, sep=sep, decimal=decimal)

    ####################################################################################################################
    def event_report(self, filename, min_event_rain_sum=25, min_return_period=0.5, durations=None):
        """
        create pdf file with the biggest rain events
        for each event is represented by a plot of the rain series
        and a IDF analysis where the return periods are calculated

        Args:
            filename (str): path (directory + filename) for the created pdf-report
            min_event_rain_sum (float): only events with a bigger rain sum will be created
            min_return_period (float): only events with a bigger return period will be analysed
                                       (the plot will be created anyway)
            out_path (str): path and filename of the final report
            durations (list[int]): analysed durations
                        (default: [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320])
        """
        if durations is None:
            durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320]

        events = self.rain_events
        events[COL.LP] = agg_events(events, self.series, 'sum')

        main_events = events[events[COL.LP] > min_event_rain_sum].to_dict(orient='index')
        main_events = main_events.sort_values(by=COL.LP, ascending=False)

        unit = 'mm'
        column_name = 'Precipitation'

        pdf = PdfPages(filename)

        for _, event in tqdm(main_events.items()):
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

        freq = guess_freq(self.series.index)
        pstart = start - pd.Timedelta(freq)
        pend = end + pd.Timedelta(freq)
        ts = self.series[pstart:pend].resample(freq).sum().fillna(0).copy()

        fig = plt.figure()

        # -------------------------------------
        idf_table = self.return_periods_frame[pstart:pend]
        idf_table = idf_table.rename(minutes_readable, axis=0)

        # print(idf_table > min_return_period)

        max_period, duration = idf_table.max().max(), idf_table.max().idxmax()
        caption += '\nThe maximum return period was {:0.2f}a\nat a duration of {}.'.format(max_period, duration)

        if not (idf_table > min_return_period).any().any():
            # max_period, duration = idf_table.max().max(), idf_table.max().idxmax()
            # caption += '\nThe maximum return period was {:0.2f}a\nat a duration of {}.'.format(max_period, duration)
            rain_ax = fig.add_subplot(111)

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
