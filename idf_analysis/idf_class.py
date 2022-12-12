__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import warnings
from os import path, mkdir
from webbrowser import open as show_file

from scipy.optimize import newton

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .arg_parser import heavy_rain_parser
from .idf_backend import IdfParameters
from .little_helpers import minutes_readable, height2rate, delta2min, rate2height, frame_looper, event_caption
from .definitions import *
from .in_out import import_series
from .sww_utils import (remove_timezone, guess_freq, rain_events, agg_events, event_duration, resample_rain_series,
                        rain_bar_plot, IdfError, )
from .plot_helpers import idf_bar_axes
from .synthetic_rainseries import _BlockRain, _EulerRain


########################################################################################################################
class IntensityDurationFrequencyAnalyse:
    """
    heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)

    This program reads the measurement data of the rainfall
    and calculates the distribution of the rainfall as a function of the return period and the duration
    
    for duration steps up to 12 hours (and more) and return period in a range of '0.5a <= T_n <= 100a'

    Attributes:
        _series (pandas.Series): rain time-series
        _freq (pandas.DateOffset): frequency of the rain series
        _return_periods_frame (pandas.DataFrame): with return periods of all given durations
        _rain_events (pandas.DataFrame):
        _rainfall_sum_frame (pandas.DataFrame): with rain sums of all given durations

    """
    def __init__(self, series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=False):
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
        self._series = None
        self._freq = None

        #  how to calculate the idf curves
        # IdfParameters.__init__(self, series_kind=series_kind, worksheet=worksheet,
        #                                 extended_durations=extended_durations)
        self._parameter = IdfParameters(series_kind=series_kind, worksheet=worksheet,
                                        extended_durations=extended_durations)
        # self._parameter = None  # type: IdfParameters #  how to calculate the idf curves

        self._return_periods_frame = None
        self._rain_events = None
        self._rainfall_sum_frame = None

    # __________________________________________________________________________________________________________________
    @property
    def series(self) -> pd.Series:
        if self._series is None:
            raise IdfError('No Series defined for IDF-Analysis!')
        return self._series

    @series.setter
    def series(self, series: pd.Series):
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

        self._freq = guess_freq(series.index)
        freq_minutes = delta2min(self._freq)
        self._parameter.filter_durations(freq_minutes)
        self.series = series.replace(0, np.NaN).dropna()
        self._return_periods_frame = None
        self._rain_events = None
        self._rainfall_sum_frame = None

    # __________________________________________________________________________________________________________________
    @property
    def duration_steps(self):
        """
        get duration steps (in minutes) for the parameter calculation and basic evaluations

        Returns:
            list | numpy.ndarray: duration steps in minutes
        """
        # if not self.parameters:
        #     raise IdfError('No Series defined for IDF-Analysis!')
        return self.parameters.durations

    @duration_steps.setter
    def duration_steps(self, durations):
        """
        set duration steps (in minutes) for the parameter calculation and basic evaluations
        Args:
            durations (list | numpy.ndarray): duration steps in minutes
        """
        if not isinstance(durations, (list, np.ndarray)):
            raise IdfError('Duration steps have to be {} got "{}"'.format((list, np.ndarray), type(durations)))
        self.parameters.durations = durations

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
            IdfParameters: calculation parameters
        """
        if not self._parameter:
            self._parameter.calc_from_series(self.series)
        return self._parameter

    def write_parameters(self, filename):
        """
        save parameters as yaml-file to save computation time.

        Args:
            filename (str): filename for the parameters yaml-file
        """
        self.parameters.to_yaml(filename)

    def read_parameters(self, filename):
        """
        Read parameters from a .yaml-file to save computation time.

        Extract interim results from parameters.

        Args:
            filename (str, Path): filename of the parameters yaml-file
        """
        self._parameter = IdfParameters.from_yaml(filename)

    def auto_save_parameters(self, filename):
        """Auto-save the parameters as a yaml-file to save computation time."""
        if path.isfile(filename):
            self.read_parameters(filename)
        else:
            from os import makedirs
            parent = path.dirname(filename)
            if parent and not path.isdir(parent):
                makedirs(parent)
            self.write_parameters(filename)

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
        if self.parameters.series_kind == SERIES.ANNUAL:
            if return_period < 5:
                print('WARNING: Using an annual series and a return period < 5 a will result in faulty values!')

            if return_period <= 10:
                return_period = np.exp(1.0 / return_period) / (np.exp(1.0 / return_period) - 1.0)

            log_tn = -np.log(np.log(return_period / (return_period - 1.0)))

        else:
            log_tn = np.log(return_period)

        u, w = self.parameters.get_u_w(duration)
        return u + w * log_tn

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
        height_of_rainfall = self.depth_of_rainfall(duration=duration, return_period=return_period)
        return height2rate(height_of_rainfall=height_of_rainfall, duration=duration)

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
        u, w = self.parameters.get_u_w(duration)
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
        Get an idf-table of rainfall depth with return periods as columns and durations as rows.

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
            return_periods = [1, 2, 3, 5, 10, 20, 25, 30, 50, 75, 100]

        result_table = {}
        for t in return_periods:
            result_table[t] = self.depth_of_rainfall(durations, t)

        result_table = pd.DataFrame(result_table, index=durations)

        if add_names:
            result_table.index.name = 'duration (min)'
            result_table.columns = pd.MultiIndex.from_tuples([(rp, round(1 / rp, 3)) for rp in result_table.columns])
            result_table.columns.names = ['return period (a)', 'frequency (1/a)']
        return result_table

    ####################################################################################################################
    def result_figure(self, min_duration=5.0, max_duration=720.0, logx=False, return_periods=None, color=True, ax=None,
                      linestyle=None, add_interim=False):
        """
        Create a plot with the idf-curves with duration on the x-axis and rainfall depth on the y-axis.

        Args:
            min_duration (float): Shortest duration on the plot.
            max_duration (float): Longest duration on the plot.
            logx (bool): Use a logarithmic scale on the x-axis.
            return_periods (list[int] | Optional): List of return periods to plot. Default = [1, 2, 5, 10, 50, 100]
            color (bool): Use color and a legend to differentiate between the return periods (default).
                Otherwise, annotation text and black lines.
            ax (plt.Axes): Axes to plot on. Default = create new one.
            linestyle (str): Line-style for the plotted lines.
            add_interim (bool): Add interim results from the series analysis as scatter points.

        Returns:
            (plt.Figure, plt.Axes): figure and axes of the plot.
        """
        duration_steps = np.arange(min_duration, max_duration + 1, 1)

        if return_periods is None:
            return_periods = [1, 2, 5, 10, 50, 100]

        table = self.result_table(durations=duration_steps, return_periods=return_periods)
        if color:
            table.columns.name = 'T$\\mathsf{_N}$ in a'
        ax = table.plot(color=(None if color else 'black'), logx=logx, legend=color, ax=ax, ls=linestyle)

        for _, return_time in enumerate(return_periods):
            if add_interim:
                p = self.parameters.measured_points(return_time, max_duration=max_duration)
                # p = measured_points(self, return_time, max_duration=max_duration)
                ax.plot(p, 'k' + 'x')

            if not color:
                x, y = list(table[return_time].tail(1).items())[0]
                ax.text(x + 10, y, '{} a'.format(return_time), verticalalignment='center', horizontalalignment='left',
                        # bbox=dict(facecolor='white', alpha=1.0, lw=1)
                        )

        ax.tick_params(axis='both', which='both', direction='out')
        ax.set_xlabel('Duration D in min')
        ax.set_ylabel('Rainfall h$\\mathsf{_N}$ in mm')
        ax.set_title('IDF curves')

        fig = ax.get_figure()

        cm_to_inch = 2.54
        fig.set_size_inches(h=21 / cm_to_inch, w=29.7 / cm_to_inch)  # (11.69, 8.27)
        fig.tight_layout()
        return fig, ax

    ####################################################################################################################
    @classmethod
    def command_line_tool(cls):
        user = heavy_rain_parser()

        # --------------------------------------------------
        # use the same directory as the input file and make as subdir with the name of the input_file + "_idf_data"
        out = '{label}_idf_data'.format(label='.'.join(user.input.split('.')[:-1]))

        if not path.isdir(out):
            mkdir(out)
            action = 'Creating'
        else:
            action = 'Using'

        print('{} the subfolder "{}" for the interim- and final-results.'.format(action, out))

        fn_pattern = path.join(out, 'idf_{}')

        # --------------------------------------------------
        idf = cls(series_kind=user.series_kind, worksheet=user.worksheet, extended_durations=True)

        # --------------------------------------------------
        parameters_fn = fn_pattern.format('parameters.yaml')

        if path.isfile(parameters_fn):
            print('Found existing interim-results in "{}" and using them for calculations.'.format(parameters_fn))
        else:
            print('Start reading the time-series {} for the analysis.'.format(user.input))
            ts = import_series(user.input).replace(0, np.NaN).dropna()
            # --------------------------------------------------
            idf.set_series(ts)
            print('Finished reading.')

        # --------------------------------------------------
        idf.auto_save_parameters(parameters_fn)

        # --------------------------------------------------
        h = user.height_of_rainfall
        r = user.flow_rate_of_rainfall
        d = user.duration
        t = user.return_period

        if r is not None:
            if h is None and d is not None:
                h = rate2height(rain_flow_rate=r, duration=d)

            elif d is None and h is not None:
                d = h / r * 1000 / 6

        if user.r_720_1:
            d = 720
            t = 1

        if any((h, d, t)):
            if all((d, t)):
                pass

            elif all((d, h)):
                t = idf.get_return_period(h, d)
                print('The return period is {:0.1f} years.'.format(t))

            elif all((h, t)):
                d = idf.get_duration(h, t)
                print('The duration is {:0.1f} minutes.'.format(d))

            print('Resultierende Regenhöhe h_N(T_n={t:0.1f}a, D={d:0.1f}min) = {h:0.2f} mm'
                  ''.format(t=t, d=d, h=idf.depth_of_rainfall(d, t)))
            print('Resultierende Regenspende r_N(T_n={t:0.1f}a, D={d:0.1f}min) = {r:0.2f} L/(s*ha)'
                  ''.format(t=t, d=d, r=idf.rain_flow_rate(d, t)))

        # --------------------------------------------------
        if user.plot:
            fig, ax = idf.result_figure()
            plot_fn = fn_pattern.format('curves_plot.png')
            fig.savefig(plot_fn, dpi=260)
            plt.close(fig)
            show_file(plot_fn)
            print('Created the IDF-curves-plot and saved the file as "{}".'.format(plot_fn))

        # --------------------------------------------------
        if user.export_table:
            table = idf.result_table(add_names=True)
            print(table.round(2).to_string())
            table_fn = fn_pattern.format('table.csv')
            table.to_csv(table_fn, sep=';', decimal=',', float_format='%0.2f')
            print('Created the IDF-curves-plot and saved the file as "{}".'.format(table_fn))

    ####################################################################################################################
    def get_rainfall_sum_frame(self, series=None, durations=None):
        """
        Get a rainfall sum frame for any series with the duration steps as columns.

        Default: The time-series and the duration-steps of the analysis.

        Args:
            series (pandas.Series, Optional): rainfall time-series
            durations (list, Optional): list of durations in minutes which are of interest (default: pre-defined durations)

        Returns:
            pandas.DataFrame: Rain sum depending on the duration per datetime-index.
        """
        if durations is None:
            durations = self.duration_steps

        if series is None:
            if self._rainfall_sum_frame is not None:
                return self._rainfall_sum_frame
            ts = self.series.copy()
            freq = self._freq
        else:
            freq = guess_freq(series.index)
            ts = series.copy()
            ts = ts.asfreq(freq).fillna(0)
            # ts = series.replace(0, np.NaN).dropna()

        df = pd.DataFrame(index=ts.index)
        # df = {}

        freq_num = delta2min(freq)

        for d in frame_looper(ts.index.size, columns=durations, label='rainfall_sum'):
            if d % freq_num != 0:
                warnings.warn('Using durations (= {} minutes), '
                              'which are not a multiple of the base frequency (= {} minutes) of the series, '
                              'will lead to misinterpretations.'.format(d, freq_num))
            df[d] = ts.rolling(pd.Timedelta(minutes=d)).sum()

        # printable_names (bool): if durations should be as readable in dataframe, else in minutes
        # df = df.rename(minutes_readable, axis=0)

        return df#.round(2)

    @property
    def rainfall_sum_frame(self):
        """
        Get the rainfall sum over the whole time-series for the default duration steps.

        Returns:
            pandas.DataFrame: Rain sum depending on the duration per datetime-index.
        """
        if self._rainfall_sum_frame is None:
            self._rainfall_sum_frame = self.get_rainfall_sum_frame()
        return self._rainfall_sum_frame

    ####################################################################################################################
    def get_return_periods_frame(self, series=None, durations=None):
        """
        Get the return periods for any time-series with the duration steps as columns.

        Default: The time-series and the duration-steps of the analysis.

        Args:
            series (pandas.Series, Optional): rainfall time-series
            durations (list, Optional): Durations in minutes which are of interest (default: pre-defined durations)

        Returns:
            pandas.DataFrame: Return periods depending on the duration per datetime-index.
        """
        sums = self.get_rainfall_sum_frame(series=series, durations=durations)
        df = pd.DataFrame(index=sums.index)
        # df = {}
        for d in frame_looper(sums.index.size, columns=sums.columns, label='return_periods'):
            df[d] = self.get_return_period(height_of_rainfall=sums[d][sums[d] >= 0.1], duration=d)
        return df#.fillna(0)#.round(2)

    @property
    def return_periods_frame(self):
        """
        Get the return periods over the whole time-series for the default duration steps.

        Returns:
            pandas.DataFrame: data-frame of return periods where the columns are the duration steps
        """
        if self._return_periods_frame is None:
            self._return_periods_frame = self.get_return_periods_frame()
        return self._return_periods_frame

    def write_return_periods_frame(self, filename, **kwargs):
        """save the return-periods dataframe as a parquet-file to save computation time."""
        df = self.return_periods_frame.copy()
        df.columns = df.columns.to_series().astype(str)
        df.round(2).to_parquet(filename, **kwargs)

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
    @property
    def rain_events(self):
        """
        get the all the rain events of the time-series

        default minimal gap between events is 4 hours

        Returns:
            pandas.DataFrame: data-frame of events with start-, end-time and duration
        """
        if self._rain_events is None:
            events = rain_events(self.series)
            events[COL.DUR] = event_duration(events)
            events[COL.LP] = agg_events(events, self.series, 'sum').round(1)
            events[COL.LAST] = events[COL.START] - events[COL.END].shift()
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
        events[COL.LAST] = pd.to_timedelta(events[COL.LAST])
        self._rain_events = events

    def auto_save_rain_events(self, filename, sep=';', decimal='.'):
        """auto-save the rain-events dataframe as a csv-file to save computation time."""
        if path.isfile(filename):
            self.read_rain_events(filename, sep=sep, decimal=decimal)
        else:
            self.write_rain_events(filename, sep=sep, decimal=decimal)

    def add_max_return_periods_to_events(self, events):
        if COL.MAX_PERIOD not in events:
            return_periods_frame = self.return_periods_frame
            max_periods = return_periods_frame.max(axis=1).fillna(0)  # fill NaN -> weil < 0.1 gefiltert wurde
            max_periods_duration = return_periods_frame.idxmax(axis=1)
            datetime_max = agg_events(events, max_periods, 'idxmax')
            datetime_max = np.where(np.isnan(datetime_max), events[COL.START].values, datetime_max)
            # alternative:
            # [xv if c else yv for c, xv, yv in zip(np.isnan(datetime_max), events[COL.START].values, datetime_max)]
            events[COL.MAX_PERIOD] = max_periods[datetime_max].values
            events[COL.MAX_PERIOD_DURATION] = max_periods_duration[datetime_max].values

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
            durations (list[int]): analysed durations
                        (default: [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320])
        """
        if durations is None:
            durations = self.parameters.durations

        events = self.rain_events
        self.add_max_return_periods_to_events(events)

        main_events = events[(events[COL.LP] > min_event_rain_sum) & (events[COL.MAX_PERIOD] > min_return_period)].sort_values(by=COL.MAX_PERIOD, ascending=False).to_dict(
            orient='index')

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

    def event_plot(self, event, durations=None, unit='mm', column_name='Precipitation', min_return_period=1.):
        if isinstance(event, pd.Series):
            event = event.to_dict()

        plot_range = slice(event[COL.START] - pd.Timedelta(self._freq), event[COL.END] + pd.Timedelta(self._freq))

        if COL.MAX_PERIOD not in event:
            return_periods_frame = self.return_periods_frame[plot_range]
            event[COL.MAX_PERIOD] = return_periods_frame.max().max()
            event[COL.MAX_PERIOD_DURATION] = return_periods_frame.max().idxmax()

        ts = self.series[plot_range].resample(self._freq).sum().fillna(0).copy()

        # -------------------------------------
        fig = plt.figure()

        if event[COL.MAX_PERIOD] < min_return_period:
            rain_ax = fig.add_subplot(111)

        else:
            if durations is not None:
                max_dur = max(durations)
            else:
                max_dur = max(self.duration_steps)

            return_periods_frame_extended = self.get_return_periods_frame(
                self.series[event[COL.START] - pd.Timedelta(minutes=max_dur):
                            event[COL.END] + pd.Timedelta(self._freq)].resample(self._freq).sum(),
                durations=durations
            ).round(1)

            idf_bar_ax = fig.add_subplot(211)
            idf_bar_ax = idf_bar_axes(idf_bar_ax, return_periods_frame_extended)
            rain_ax = fig.add_subplot(212, sharex=idf_bar_ax)

        # -------------------------------------
        ts_sum, minutes = resample_rain_series(ts)
        rain_ax = rain_bar_plot(ts_sum, rain_ax)
        rain_ax.set_ylabel('{} in {}/{}min'.format(column_name, unit, minutes if minutes != 1 else ''))
        rain_ax.set_xlim(ts.index[0], ts.index[-1])

        return fig, event_caption(event, unit)

    ####################################################################################################################
    def event_return_period_report(self, filename, min_return_period=1):
        events = self.rain_events
        self.add_max_return_periods_to_events(events)

        main_events = events[events[COL.MAX_PERIOD] > min_return_period].sort_values(by=COL.MAX_PERIOD, ascending=False)

        pdf = PdfPages(filename)

        for _, event in tqdm(main_events.to_dict(orient='index').items()):
            fig, ax = self.return_period_event_figure(event)

            # -------------------------------------
            # DIN A4
            fig.set_size_inches(h=11.69 / 2, w=8.27)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        pdf.close()

    def return_period_event_figure(self, event):
        period_line = self.return_periods_frame[event[COL.START]:event[COL.END]].max()

        # period_line[period_line < 0.75] = np.NaN
        period_line = period_line.dropna()

        ax = period_line.plot()  # type: plt.Axes

        ax.set_title('rain event\n'
                     'between {} and {}\n'
                     'with a total sum of {:0.1f} mm\n'
                     'and a duration of {}\n'
                     'The maximum return period was {:0.2f}a\n'
                     'at a duration of {}.'.format(event[COL.START].strftime('%Y-%m-%d %H:%M'),
                                                   event[COL.END].strftime('%Y-%m-%d %H:%M'),
                                                   event[COL.LP],
                                                   event[COL.END] - event[COL.START],
                                                   period_line.max(),
                                                   period_line.idxmax()))
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # print(ax.get_ylim())
        # ax.set_ylim(0.01, 300)
        # exit()
        ax.set_xticks(period_line.index)
        ax.set_xticklabels([minutes_readable(m) for m in period_line.index])
        ax.set_xlabel('duration steps')
        ax.set_ylabel('return period in years')
        return ax.get_figure(), ax

    @classmethod
    def from_idf_table(cls, idf_table, worksheet=METHOD.KOSTRA):
        """
        Create an IDF-analysis-object based on an idf-tabel (i.e. from a given KOSTRA table)

        Args:
            idf_table (pandas.DataFrame): idf-table with index=Durations and columns=return Period and values=Rainheight
            worksheet (str | optional): name of the worksheet to use.  default: 'KOSTRA'

        Returns:
            IntensityDurationFrequencyAnalyse: idf-object
        """
        idf = cls(worksheet=worksheet)
        idf._parameter.reverse_engineering(idf_table)
        return idf

    @property
    def model_rain_block(self):
        """
        Create a model block rain class.

        Returns:
            _BlockRain: Synthetic model block rain.
        """
        return _BlockRain(self)

    @property
    def model_rain_euler(self):
        """
        Create a model Euler rain class.

        Returns:
            _EulerRain: Synthetic model Euler rain.
        """
        return _EulerRain(self)
