__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
from matplotlib.ticker import NullFormatter

from .definitions import COL
from .little_helpers import duration_steps_readable
from .sww_utils import guess_freq, rain_events, event_duration


def idf_bar_axes(ax, idf_table, durations=None, return_periods=None):
    """
    create

    Args:
        ax (matplotlib.pyplot.Axes):
        idf_table (pandas.DataFrame):
        durations (list):
        return_periods (list):

    Returns:
        matplotlib.pyplot.Axes:
    """
    if durations is None:
        durations = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440, 2880, 4320]
    if return_periods is None:
        return_periods = [0.5, 1, 2, 5, 10, 20, 50, 100]

    color_return_period = ['lightcyan', 'cyan', 'lightblue', 'blue', 'yellow', 'orange', 'red', 'magenta']

    # legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in color_return_period]
    names = ['{}a'.format(t) for t in return_periods]
    ax.legend(custom_lines, names, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(color_return_period),
              mode="expand", borderaxespad=0., title='return periods')

    duration_size = len(durations)
    # labels for the y axis
    durations_index = range(duration_size)
    dh = 1
    ax.set_yticks([i + dh/2 for i in durations_index], minor=True)
    ax.set_yticks(list(durations_index), minor=False)

    ax.set_yticklabels(duration_steps_readable(durations), minor=True)
    ax.set_yticklabels([''] * duration_size, minor=False)
    ax.set_ylabel('duration of the design rainfall')

    # for the relative start time
    freq = guess_freq(idf_table.index)
    start_period = idf_table.index[0].to_period(freq).ordinal

    # idf_table.index = idf_table.index - idf_table.index[0]

    min_duration = pd.Timedelta(minutes=1)

    for hi, d in enumerate(idf_table.columns):
        tn = idf_table[d]

        for i, t in enumerate(return_periods):
            c = color_return_period[i]
            # not really a rain event, but the results are the same
            tab = rain_events(tn, ignore_rain_below=t, min_gap=freq)

            if tab.empty:
                continue

            if 1:
                durations = (event_duration(tab) / min_duration).tolist()
                rel_starts = ((tab[COL.START] - idf_table.index[0]) / min_duration + start_period).tolist()
                bar_x = list(zip(rel_starts, durations))
            else:
                tab[COL.DUR] = event_duration(tab) / min_duration
                bar_x = [(r[COL.START] / min_duration + start_period, r[COL.DUR]) for _, r in tab.iterrows()]

            ax.broken_barh(bar_x, (hi, dh), facecolors=c)

    ax.set_ylim(0, duration_size)
    ax.set_xticklabels([])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.axhline(0, color='k')
    ax.axhline(duration_size / 2, color='k')
    return ax
