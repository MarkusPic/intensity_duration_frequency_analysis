__author__ = "Markus Pichler, Marlies Hierzer"
__credits__ = ["Markus Pichler", "Marlies Hierzer"]
__maintainer__ = "Markus Pichler, Marlies Hierzer"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .definitions import COL
from .idf_class import IntensityDurationFrequencyAnalyse
from .little_helpers import duration_steps_readable, minutes_readable, frame_looper, event_caption
from .plot_helpers import _bar_axes, RETURN_PERIOD_COLORS, _set_xlim
from .sww_utils import guess_freq, rain_events, event_duration, resample_rain_series, rain_bar_plot

# from matplotlib.colors import ListedColormap, Normalize
# from matplotlib.ticker import NullFormatter

COL.MAX_SRI = 'max_SRI_{}'
COL.MAX_SRI_DURATION = 'max_SRI_duration_{}'


####################################################################################################################
def grisa_factor(tn):
    """
    calculates the grisa-factor according to Grisa's formula

    Args:
        tn (float): in [years]

    Returns:
        float: factor
    """
    return 1 + (np.log(tn) / np.log(2))


def next_bigger(v, l):
    return l[next(x for x, val in enumerate(l) if val >= v)]


class SCHMITT:
    # Zuweisung nach Schmitt des SRI über der Wiederkehrperiode
    SRI_TN = {
        1: 1,
        2: 1,
        3: 2,
        5: 2,
        10: 3,
        20: 4,
        25: 4,
        30: 5,
        50: 6,
        75: 6,
        100: 7
    }

    # Erhöhungsfaktoren nach Schmitt für SRI 8,9,10,11,12 basierend auf SRI 7
    # untere und obere Grenze
    MULTI_FACTOR = {
        8: (1.2, 1.39),
        9: (1.4, 1.59),
        10: (1.6, 2.19),
        11: (2.2, 2.78),
        12: (2.8, 2.8),
    }

    VERBAL = {
        (1, 2): 'Starkregen',
        (3, 5): 'intensiver Starkregen',
        (6, 7): 'außergewöhnlicher Starkregen',
        (8, 12): 'extremer Starkregen'
    }

    INDICES_COLOR = {1: (0.69, 0.9, 0.1),
                     2: (0.8, 1, 0.6),
                     3: (0.9, 1, 0.3),
                     4: (1, 0.96, 0),
                     5: (1, 0.63, 0),
                     6: (1, 0.34, 0),
                     7: (1, 0.16, 0),
                     8: (0.97, 0.12, 0.24),
                     9: (1, 0.10, 0.39),
                     10: (0.97, 0.03, 0.51),
                     11: (0.92, 0.08, 0.75),
                     12: (0.66, 0.11, 0.86)}

    INDICES_COLOR_RGB = {1: (176, 230, 25),
                         2: (204, 255, 153),
                         3: (230, 255, 77),
                         4: (255, 244, 0),
                         5: (255, 160, 0),
                         6: (255, 86, 0),
                         7: (255, 40, 0),
                         8: (247, 30, 61),
                         9: (255, 26, 99),
                         10: (247, 9, 130),
                         11: (235, 21, 191),
                         12: (189, 28, 220)}

    INDICES_COLOR_HEX = {1: "#b0e619",
                         2: "#ccff99",
                         3: "#e6ff4d",
                         4: "#fff400",
                         5: "#ffa000",
                         6: "#ff5600",
                         7: "#ff2800",
                         8: "#f71e3d",
                         9: "#ff1a63",
                         10: "#f70982",
                         11: "#eb15bf",
                         12: "#bd1cdc"}


krueger_pfister_verbal = {
    (1, 4): 'moderat',
    (5, 7): 'stark',
    (8, 10): 'heftig',
    (11, 12): 'extrem'
}

grisa_verbal = {
    (1, 2): 'Minor',
    (3, 4): 'Moderate',
    (5, 6): 'Major',
    (7, 8): 'Extreme',
    (9, 10): 'Catastrophic'
}


def cat_dict(cat):
    res = {}
    for num_range, verbal in cat.items():
        for i in range(num_range[0], num_range[1]+1):
            res[i] = verbal
    return res


####################################################################################################################
class HeavyRainfallIndexAnalyse(IntensityDurationFrequencyAnalyse):
    indices = list(range(1, 13))

    class METHODS:
        SCHMITT = 'Schmitt'
        SCHMITT2015 = 'Schmitt_2015'  # ortsunabhängig
        KRUEGER_PFISTER = 'KruegerPfister'
        MUDERSBACH = 'Mudersbach'

        @classmethod
        def all(cls):
            return cls.SCHMITT, cls.KRUEGER_PFISTER, cls.MUDERSBACH

    indices_color = SCHMITT.INDICES_COLOR

    @classmethod
    def color_map_index(cls, idx):
        return mcolors.ListedColormap([(1, 1, 1)] + list(cls.indices_color.values()))(mcolors.Normalize(vmin=0, vmax=12)(idx))

    def __init__(self, *args, method=METHODS.SCHMITT, **kwargs):
        IntensityDurationFrequencyAnalyse.__init__(self, *args, **kwargs)
        self.method = method
        self._sri_frame = None

    def set_series(self, series):
        IntensityDurationFrequencyAnalyse.set_series(self, series)
        self._sri_frame = None

    def get_sri(self, height_of_rainfall, duration):
        """
        calculate the heavy rain index (StarkRegenIndex), when the height of rainfall and the duration are given

        Args:
            height_of_rainfall (float): in [mm]
            duration (int | float | list | numpy.ndarray | pandas.Series): in minutes

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: heavy rain index
        """
        tn = self.get_return_period(height_of_rainfall, duration)
        # ----

        if self.method == self.METHODS.MUDERSBACH:
            if isinstance(tn, (pd.Series, np.ndarray)):
                sri = np.round(1.5 * np.log(tn) + 0.4 * np.log(duration), 0)
                sri[tn <= 1] = 1
                sri[tn >= 100] = 12
            else:
                if tn <= 1:
                    sri = 1
                elif tn >= 100:
                    sri = 12
                else:
                    sri = np.round(1.5 * np.log(tn) + 0.4 * np.log(duration), 0)

        elif self.method == self.METHODS.SCHMITT:
            if isinstance(tn, (pd.Series, np.ndarray)):
                breaks = [-np.inf] + list(SCHMITT.SRI_TN.keys()) + [np.inf]
                d = dict(zip(range(11), SCHMITT.SRI_TN.values()))
                sri = pd.cut(tn, breaks, labels=False).replace(d)

                over_100 = tn > 100
                hn_100 = self.depth_of_rainfall(duration, 100)
                breaks2 = [1] + [f[0] for f in SCHMITT.MULTI_FACTOR.values()][1:] + [np.inf]
                d2 = dict(zip(range(len(breaks2) - 1), range(8, 13)))
                sri.loc[over_100] = pd.cut(height_of_rainfall.loc[over_100] / hn_100, breaks2, labels=False).replace(d2)

            else:
                if tn >= 100:
                    hn_100 = self.depth_of_rainfall(duration, 100)
                    for sri, mul in SCHMITT.MULTI_FACTOR.items():
                        if height_of_rainfall <= hn_100 * mul[0]:
                            break
                else:
                    sri = SCHMITT.SRI_TN[next_bigger(tn, list(SCHMITT.SRI_TN.keys()))]

        elif self.method == self.METHODS.KRUEGER_PFISTER:
            h_24h = self.depth_of_rainfall(duration=24 * 60, return_period=tn)
            hn_100 = self.depth_of_rainfall(duration=duration, return_period=100)
            duration_adjustment_factor = height_of_rainfall / h_24h
            intensity_adjustment_factor = height_of_rainfall / hn_100
            sri = grisa_factor(tn) * duration_adjustment_factor * intensity_adjustment_factor
            if isinstance(sri, (pd.Series, np.ndarray)):
                sri[tn < 0.5] = 0
            else:
                if tn < 0.5:
                    sri = 0
            sri = np.clip(np.ceil(sri), 0, 12)

        elif self.method == self.METHODS.SCHMITT2015:
            # TODO: additional_files.schmitt_ortsunabhaengig_2015.ods
            if duration == 15:
                if height_of_rainfall < 10:
                    sri = 0
            elif duration == 60:
                if height_of_rainfall < 15:
                    sri = 0
            elif duration == 120:
                if height_of_rainfall < 20:
                    sri = 0
            elif duration == 240:
                if height_of_rainfall < 20:
                    sri = 0
            elif duration == 360:
                if height_of_rainfall < 25:
                    sri = 0
            else:
                sri = np.nan

        else:
            raise NotImplementedError(f'Method {self.method} not implemented!')

        if isinstance(tn, (pd.Series, np.ndarray)):
            sri[tn <= 0.3] = 0
        else:
            if tn < 0.3:
                sri = 0

        return sri

    # __________________________________________________________________________________________________________________
    def result_sri_table(self, durations=None):
        """
        get a standard idf table of rainfall depth with return periods as columns and durations as rows

        Args:
            durations (list | numpy.ndarray | None): list of durations in minutes for the table

        Returns:
            pandas.DataFrame: idf table
        """
        idf_table = self.result_table(durations)

        if self.method == self.METHODS.SCHMITT:
            sri_table = idf_table.rename(columns=SCHMITT.SRI_TN)

            for sri, mul in SCHMITT.MULTI_FACTOR.items():
                sri_table[sri] = mul[1] * sri_table[7]

            sri_table = sri_table.loc[:, ~sri_table.columns.duplicated('last')]

        elif self.method == self.METHODS.MUDERSBACH:
            # zuerst eine Tabelle mit den Wiederkehrperioden
            rp_table = pd.DataFrame(index=idf_table.index, columns=range(1, 13))

            # abhängigkeit nach dauerstufe
            a = np.log(rp_table.index.values) * 0.4
            for sri in rp_table.columns:
                rp_table[sri] = np.exp((sri + 0.5 - a) / 1.5)

            rp_table.loc[:, 1] = 1
            # rp_table.loc[:, 12] = 100

            # dann mittels Dauerstufe und Wiederkehrperiode die Regenhöhe errechnen
            sri_table = rp_table.round(1).copy()
            for dur in rp_table.index:
                sri_table.loc[dur] = self.depth_of_rainfall(dur, rp_table.loc[dur])

            # extrapolation vermutlich nicht sehr seriös
            sri_table[rp_table >= 100] = np.nan
            # sri_table.loc[:12] = self.depth_of_rainfall(sri_table.index.values, 100)
            sri_table[rp_table < 1] = np.nan
            sri_table = sri_table.astype(float).round(2)
            sri_table = sri_table.ffill(axis=1, limit=None)  # .fillna(method='ffill', axis=1, limit=None)

        elif self.method == self.METHODS.KRUEGER_PFISTER:
            # duration_adjustment_factor = idf_table.div(idf_table.loc[24 * 60])
            # intensity_adjustment_factor = idf_table.div(idf_table[100].values, axis=0)
            # sri_table = grisa_factor(
            #     idf_table.columns.values) * duration_adjustment_factor * intensity_adjustment_factor
            # sri_table = sri_table.round().astype(int).clip(0,12)

            sri_table = pd.DataFrame(index=idf_table.index)
            sri_vector = (idf_table.loc[1440, 100] * idf_table.loc[:, 100]) / (1 + (np.log(100) / np.log(2)))
            for i in self.indices:
                sri_table[i] = np.sqrt(i * sri_vector)

        else:
            raise NotImplementedError(f'Method or "{self.method}" not implemented! '
                                      f'Please contact the developer for the request to implement it.')

        sri_table.index.name = 'duration in min'
        sri_table.columns.name = 'SRI'
        return sri_table

    def interim_sri_table(self, durations=None):
        """M
        get a table of SRI with return periods as columns and durations as rows

        Args:
            durations (list | numpy.ndarray): list of durations in minutes for the table
            return_periods (list): list of return periods in years for the table

        Returns:
            pandas.DataFrame: idf table
        """

        idf_table = self.result_table(durations)
        sri_table = pd.DataFrame(index=idf_table.index, columns=idf_table.columns)

        if self.method == self.METHODS.SCHMITT:
            for col in sri_table:
                sri_table[col] = SCHMITT.SRI_TN[col]

        elif self.method == self.METHODS.MUDERSBACH:
            sri_table[1] = 1

            a = np.log(sri_table.index.values) * 0.4
            for tn in [2, 3, 5, 10, 20, 25, 30, 50, 75, 100]:
                sri_table[tn] = a + np.log(tn) * 1.5
            sri_table = sri_table.round().astype(int)

        elif self.method == self.METHODS.KRUEGER_PFISTER:
            duration_adjustment_factor = idf_table.div(idf_table.loc[24 * 60])
            intensity_adjustment_factor = idf_table.div(idf_table[100].values, axis=0)
            sri_table = grisa_factor(idf_table.columns.values) * duration_adjustment_factor * intensity_adjustment_factor
            sri_table = sri_table.round().astype(int).clip(0,12)

        else:
            raise NotImplementedError(f'Method {self.method} not implemented!')

        sri_table.index.name = 'duration in min'
        sri_table.columns.name = 'Return Period in a'
        return sri_table

    ####################################################################################################################
    def result_sri_figure(self, duration_steps=None, ax=None, grid=True):
        """
        SRI curves are generated depending on the selected procedure for SRI generation.

        Args:
            duration_steps (list | numpy.ndarray): list of durations in minutes for the table
            ax (plt.Axes): if plot is a subplot give the axes
            grid (bool): if to make a grid

        Returns:
            (matplotlib.pyplot.Figure, matplotlib.pyplot.Axes): figure and axes of the plot
        """
        sri_table = self.result_sri_table(durations=duration_steps)

        ax = sri_table.plot(color=self.indices_color, logx=True, legend=True, ax=ax)

        ax.tick_params(axis='both', which='both', direction='out')
        ax.set_xlabel('Duration D')
        ax.set_ylabel('Rainfall h$\\mathsf{_N}$ in mm')
        # ax.set_xlabel('Dauerstufe D')
        # ax.set_ylabel('Regenhöhe h$\\mathsf{_N}$ in mm')
        # ax.set_title('Starkregenindex - Kurven', fontweight='bold')

        ax.set_xticks([], minor=True)
        ax.set_xticks(sri_table.index)
        ax.set_xlim(*sri_table.index.values[[0, -1]])

        ax.set_xticklabels(duration_steps_readable(sri_table.index))

        ax.set_facecolor('w')
        if grid:
            ax.grid(color='grey', linestyle='-', linewidth=0.3)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.02, 1.), loc='upper left',
                  borderaxespad=0., title='SRI')

        fig = ax.get_figure()

        # cm_to_inch = 2.54
        # fig.set_size_inches(h=11 / cm_to_inch, w=25 / cm_to_inch)  # (11.69, 8.27)
        # fig.tight_layout()
        return fig, ax

    ####################################################################################################################
    @property
    def sri_frame(self):
        """
        get the return periods over the whole time-series for the default duration steps.

        Returns:
            pandas.DataFrame: data-frame of return periods where the columns are the duration steps
        """
        if self._sri_frame is None:
            self._sri_frame = self.get_sri_frame()
        return self._sri_frame

    def get_sri_frame(self, series=None, durations=None):
        """

        Args:
            series (pandas.Series): precipitation time-series of the time range of interest i.e. of an event
            durations (list): list of durations in minutes which are of interest (default: pre defined durations)

        Returns:
            pandas.DataFrame: index=date-time-index; columns=durations; values=SRI
        """
        # TODO: Probleme bei geringen Regenhöhen, Formel nicht dafür gemacht!!
        sums = self.get_rainfall_sum_frame(series=series, durations=durations)
        df = pd.DataFrame(index=sums.index)
        for d in frame_looper(sums.index.size, columns=sums.columns, label='sri'):
            df[d] = self.get_sri(height_of_rainfall=sums[d][sums[d] > 0.1], duration=d)
        return df#.round(1)

    def add_max_sri_to_events(self, events, series=None):
        """M
        maximum SRI is added to the table

        Args:
            events (pandas.DataFrame): list of rainfall events
        """
        if COL.MAX_SRI.format(self.method) not in events:
            events[COL.MAX_SRI.format(self.method)] = None
            events[COL.MAX_SRI_DURATION.format(self.method)] = None

            rainfall_sum_frame = self.get_rainfall_sum_frame(series=series)

            for event_no, event in events.iterrows():
                s = self.get_event_sri_max(event[COL.START], event[COL.END], rainfall_sum_frame=rainfall_sum_frame)
                events.loc[event_no, COL.MAX_SRI.format(self.method)] = s.max()
                events.loc[event_no, COL.MAX_SRI_DURATION.format(self.method)] = s.idxmax()

    def get_event_sri_max(self, start, end, rainfall_sum_frame=None):
        """M
        maximum SRI is added to the table

        Args:
            events (list): list of rainfall events

        Returns:
            pandas.DataFrame: table
        """
        if rainfall_sum_frame is None:
            d = self.rainfall_sum_frame[start:end].max().to_dict()
        else:
            d = rainfall_sum_frame[start:end].max().to_dict()
        sri = {}
        for dur, h in d.items():
            sri[dur] = self.get_sri(h, dur)
        return pd.Series(sri, name=self.method)

    def sri_bar_axes(self, ax, sri_frame):
        """
        create a bar axes for the sri event plot

        Args:
            ax (matplotlib.pyplot.Axes):
            sri_frame (pandas.DataFrame): index=DatetimeIndex and data=SRI, columns=duration steps

        Returns:
            matplotlib.pyplot.Axes:
        """
        return _bar_axes(ax, sri_frame, self.indices_color,
                         legend_kwags=dict(title='Starkregenindex', handlelength=0.7))


    @staticmethod
    def event_plot_caption(event, method, unit='mm'):
        """
        get a caption for the event

        Args:
            event (pd.Series | dict): statistics of the event
            method (str): used method for HRI estimation (i.e. SCHMITT, MUDERSBACH, KRUEGER_PFISTER)
            unit (str): unit of the observation

        Returns:
            str: caption for the plot
        """
        caption = event_caption(event, unit) + '\n'
        caption += f'The method used for the SRI calculation is: {method}.\n'

        if COL.MAX_SRI.format(method) in event:
            caption += f'The maximum SRI was {event[COL.MAX_SRI.format(method)]:0.2f}\n'

        if COL.MAX_SRI_DURATION.format(method) in event:
            caption += f'at a duration of {minutes_readable(event[COL.MAX_SRI_DURATION.format(method)])}.'
        return caption

    def event_plot_sri(self, event, durations=None, unit='mm', column_name='Precipitation'):
        """
        get a plot of the selected event

        Args:
            event (pandas.Series): event
            durations (list | numpy.ndarray): list of durations in minutes for the table
            unit (str): unit of the observation
            column_name (str): label of the observation

        Returns:
            (matplotlib.pyplot.Figure, matplotlib.pyplot.Axes): figure and axes of the plot
        """
        event = event.to_dict()
        start = event[COL.START]
        end = event[COL.END]

        plot_range = slice(start - pd.Timedelta(self._freq), end + pd.Timedelta(self._freq))

        if durations:
            max_dur = max(durations)
        else:
            max_dur = max(self.duration_steps)

        sri_frame_extended = self.get_sri_frame(
            self.series[start - pd.Timedelta(minutes=max_dur):
                        end + pd.Timedelta(self._freq)].asfreq(self._freq).fillna(0)
        )
        sri_frame = sri_frame_extended[plot_range]

        if COL.MAX_SRI.format(self.method) not in event:
            event[COL.MAX_SRI.format(self.method)] = sri_frame.max().max()
            event[COL.MAX_SRI_DURATION.format(self.method)] = sri_frame.max().idxmax()

        ts = self.series[plot_range].resample(self._freq).sum().fillna(0).copy()

        # -------------------------------------
        fig = plt.figure()

        sri_bar_ax = fig.add_subplot(211)
        sri_bar_ax = self.sri_bar_axes(sri_bar_ax, sri_frame_extended)
        rain_ax = fig.add_subplot(212, sharex=sri_bar_ax)

        # -------------------------------------
        ts_sum, minutes = resample_rain_series(ts)
        rain_ax = rain_bar_plot(ts_sum, rain_ax)
        rain_ax.set_ylabel('{} in {}/{}min'.format(column_name, unit, minutes if minutes != 1 else ''))
        rain_ax.set_xlim(ts.index[0], ts.index[-1])

        return fig, self.event_plot_caption(event, self.method)

    def event_dataframe(self, event: dict) -> pd.DataFrame:
        sri_table_event = pd.DataFrame(index=self.duration_steps_for_output)

        # self.method = self.METHODS.KRUEGER_PFISTER
        # sri_table_event[self.METHODS.KRUEGER_PFISTER] = self.get_event_sri_max(event[COL.START], event[COL.END]).astype(int)
        #
        # self.method = self.METHODS.MUDERSBACH
        # sri_table_event[self.METHODS.MUDERSBACH] = self.get_event_sri_max(event[COL.START], event[COL.END]).astype(int)

        # self.method = self.METHODS.SCHMITT
        sri_table_event[self.METHODS.SCHMITT] = self.get_event_sri_max(event[COL.START], event[COL.END]).astype(int)

        sri_table_event[COL.MAX_OVERLAPPING_SUM] = self.rainfall_sum_frame[event[COL.START]:event[COL.END]].max()
        sri_table_event[COL.MAX_PERIOD] = self.return_periods_frame[event[COL.START]:event[COL.END]].max()

        return sri_table_event.rename(minutes_readable)

    def event_sri_table_plot(self, event):
        df = self.event_dataframe(event)

        fig, axes = plt.subplots(ncols=3, sharey=True, width_ratios=[2,5,5])
        # ---
        ax_sri = axes[0]
        im = ax_sri.imshow(df[[self.METHODS.SCHMITT]].values, cmap=mcolors.ListedColormap([(1, 1, 1)] + list(self.indices_color.values())), vmin=0, vmax=12, extent=(0, 2.5, df.index.size-0.5, -0.5))

        for i, y in enumerate(df[self.METHODS.SCHMITT].values):
            ax_sri.text(1.25, i, y if y > 0 else '-', ha='center', va='center')

        ax_sri.set_title(self.method, fontsize=10, fontweight='bold')
        ax_sri.set_ylabel('Duration steps')
        ax_sri.spines['left'].set_visible(False)

        for y in range(0, df.index.size):
            ax_sri.axhline(y-.5, color='white', lw=3)
        # ---
        ax_rain_sum = axes[1]
        bars = ax_rain_sum.barh(df.index, df[COL.MAX_OVERLAPPING_SUM], align='center', height=0.6, color='#1E88E5')
        labels = ax_rain_sum.bar_label(bars, df[COL.MAX_OVERLAPPING_SUM].round(1),
                                       padding=5, color='black', transform=ax_rain_sum.transAxes)

        _set_xlim(ax_rain_sum, bars, labels)

        ax_rain_sum.set_title('max. rain sum (mm)', fontsize=10, fontweight='bold')

        # ---
        ax_return_period = axes[2]

        rp_colors = RETURN_PERIOD_COLORS.copy()
        rp_list = [0] + list(rp_colors)+ [np.inf]
        rp_colors[0] = 'black'
        for rp in rp_list:
            if rp == np.inf: continue
            c = rp_colors[rp]
            s = df[COL.MAX_PERIOD]
            si = s[(s >= rp) & (s < rp_list[rp_list.index(rp)+1])].copy()
            bars = ax_return_period.barh(si.index, si.clip(upper=150), align='center', height=0.6, color=c, edgecolor='black', lw=0.2)
            if bars:
                labels = ax_return_period.bar_label(bars, si.map(lambda x: '< 1' if x < 1 else f'{x:0.1f}' if x < 150 else '> 150' if x < 150 else '>> 150'),
                                           padding=5, color='black', transform=ax_return_period.transAxes)

                # _set_xlim(ax_return_period, bars, labels)
                ax_return_period.set_xlim(0, 200)

        ax_return_period.set_title('max. return period (a)', fontsize=10, fontweight='bold')

        for ax in axes:
            ax.set_facecolor((0,0,0,0))
            # ax.set_frame_on(False)
            ax.tick_params(axis='y', which='major', length=0)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.grid(False)
            # ax.spines['top'].set_visible(True)
        return fig, axes
