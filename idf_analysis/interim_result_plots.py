import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from webbrowser import open as show_file

from .idf_class import IntensityDurationFrequencyAnalyse
from .definitions import *
from .little_helpers import minutes_readable
from .sww_utils import agg_events


def return_period_scatter(idf: IntensityDurationFrequencyAnalyse, filename='all_events_max_return_period.pdf', min_return_period=0.5, durations=None):
    if durations is None:
        durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320]

    dur_short = durations[:durations.index(90)]
    dur_long = durations[durations.index(90):]

    events = idf.rain_events
    events[COL.LP] = agg_events(events, idf.series, 'sum')
    events = events[events[COL.LP] > 25].copy()

    tn_long_list = {}
    tn_short_list = {}

    for _, event in events.iterrows():
        start = event[COL.START]
        end = event[COL.END]
        # save true
        idf_table = idf.return_periods_frame[start:end]
        idf_table = idf_table.rename(minutes_readable, axis=0)
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

    # check()
    fig, ax = plt.subplots()

    ax.scatter(x=list(tn_short_list.keys()), y=list(tn_short_list.values()), color='red')
    ax.scatter(x=list(tn_long_list.keys()), y=list(tn_long_list.values()), color='blue')
    fig = ax.get_figure()

    ax.set_ylabel('Return Period in a')

    def line_in_legend(color=None, marker=None, lw=None, ls=None, **kwargs):
        return Line2D([0], [0], color=color, marker=marker, linewidth=lw, linestyle=ls, **kwargs)

    custom_lines = []
    custom_lines.append(line_in_legend(color='red', marker='o', lw=0))
    custom_lines.append(line_in_legend(color='blue', marker='o', lw=0))
    # -----------------
    l1 = ax.legend(custom_lines, ['< 60 min', '> 60 min'], loc='best', title='max Duration')
    ax.add_artist(l1)

    # -----------------
    # DIN A4
    fig.set_size_inches(w=7, h=5)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def result_plot_v2(idf: IntensityDurationFrequencyAnalyse, filename, min_duration=5.0, max_duration=720.0, logx=False, show=False):
    duration_steps = np.arange(min_duration, max_duration + 1, 1)
    colors = ['r', 'g', 'b', 'y', 'm']

    # return_periods = [0.5, 1, 10, 50, 100]
    return_periods = [1, 2, 5, 10, 50]

    table = idf.result_table(durations=duration_steps, return_periods=return_periods)
    table.index = pd.to_timedelta(table.index, unit='m')
    ax = table.plot(color=colors, logx=logx)

    ax.tick_params(axis='both', which='both', direction='out')

    for i in range(len(return_periods)):
        return_time = return_periods[i]
        color = colors[i]
        p = idf.parameters.measured_points(return_time, max_duration=max_duration)
        p.index = pd.to_timedelta(p.index, unit='m')
        ax.plot(p, color + 'x')

        # plt.text(max_duration * ((10 - offset) / 10), depth_of_rainfall(max_duration * ((10 - offset) / 10),
        #                                                                 return_time, parameter_1,
        #                                                                 parameter_2) + offset, '$T_n=$' + str(
        #                                                                 return_time))

    ax.set_xlabel('Dauerstufe $D$ in $[min]$')
    ax.set_ylabel('Regenhöhe $h_N$ in $[mm]$')
    ax.set_title('Regenhöhenlinien')
    ax.legend(title='$T_n$= ... [a]')

    if max_duration > 1.5 * 60:
        pass
    else:
        pass

    major_ticks = np.array([d * 60 * 1.0e9 for d in idf.duration_steps if d <= max_duration])
    # minor_ticks = pd.date_range("00:00", "23:59", freq='15T').time
    # print(major_ticks)
    # exit()
    ax.set_xticks(major_ticks)
    # print(ax.get_xticks())
    from matplotlib import ticker

    def time_ticks(x, _):
        x = pd.to_timedelta(x, unit='ns').total_seconds() / 60
        h = int(x / 60)
        m = int(x % 60)
        s = ''
        if h:
            s += '{}h'.format(h)
        if m:
            s += '{}min'.format(m)
        return s

    formatter = ticker.FuncFormatter(time_ticks)
    ax.xaxis.set_major_formatter(formatter)
    # print(ax.get_xticks())
    # plt.axis([0, max_duration, 0, depth_of_rainfall(max_duration,
    #                                                 return_periods[len(return_periods) - 1],
    #                                                 parameter_1, parameter_2) + 10])

    fig = ax.get_figure()

    cm_to_inch = 2.54
    fig.set_size_inches(h=21 / cm_to_inch, w=29.7 / cm_to_inch)  # (11.69, 8.27)
    fig.tight_layout()
    fig.savefig(filename, dpi=260)
    plt.close(fig)
    if show:
        show_file(filename)
    return filename
