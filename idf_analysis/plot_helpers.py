import pandas as pd

from .definitions import COL
from .little_helpers import duration_steps_readable
from .sww_utils import guess_freq, rain_events, event_duration

RETURN_PERIOD_COLORS = {
    # 0.5: '#e0ffff',  # 'lightcyan',
    1: '#00ffff',  # 'cyan',
    2: '#add8e6',  # 'lightblue',
    5: '#0000ff',  # 'blue',
    10: '#ffff00',  # 'yellow',
    20: '#ffa500',  # 'orange',
    50: '#ff0000',  # 'red',
    100: '#ff00ff',  # 'magenta',
}


def _bar_axes(ax, table, colors_dict, legend_kwags, category_formatter=None):
    """
    create

    Args:
        ax (matplotlib.pyplot.Axes):
        table (pandas.DataFrame):
        colors_dict (dict): color of each return period {return period: color}

    Returns:
        matplotlib.pyplot.Axes:
    """
    categories = list(colors_dict.keys())
    colors = list(colors_dict.values())

    # legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in colors]

    if category_formatter is None:
        category_formatter = str
    names = [category_formatter(t) for t in categories]
    ax.legend(custom_lines, names, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(colors),
              mode="expand", borderaxespad=0., **legend_kwags)

    duration_steps = table.columns.values
    duration_size = len(duration_steps)
    # labels for the y axis
    durations_index = range(duration_size)
    dh = 1
    ax.set_yticks([i + dh/2 for i in durations_index], minor=True)
    ax.set_yticks(list(durations_index), minor=False)

    ax.set_yticklabels(duration_steps_readable(duration_steps), minor=True)
    ax.set_yticklabels([''] * duration_size, minor=False)
    ax.set_ylabel('duration of the design rainfall')

    # for the relative start time
    freq = guess_freq(table.index)
    start_dt = table.index[0]
    if start_dt.tzinfo is not None:
        start_dt = start_dt.tz_localize(None)
    start_period = start_dt.to_period(freq).ordinal

    # idf_table.index = idf_table.index - idf_table.index[0]

    min_duration = pd.Timedelta(minutes=1)

    for hi, d in enumerate(duration_steps):
        tn = table[d]

        for t in categories:
            c = colors_dict[t]
            # not really a rain event, but the results are the same
            tab = rain_events(tn, ignore_rain_below=t, min_gap=freq)

            if tab.empty:
                continue

            if _ := 1:
                durations = ((event_duration(tab) + freq) / min_duration).tolist()
                rel_starts = ((tab[COL.START] - table.index[0]) / min_duration + start_period).tolist()
                bar_x = list(zip(rel_starts, durations))
            else:
                tab[COL.DUR] = event_duration(tab) / min_duration
                bar_x = [(r[COL.START] / min_duration + start_period, r[COL.DUR]) for _, r in tab.iterrows()]

            ax.broken_barh(bar_x, (hi, dh), facecolors=c)

    ax.tick_params(axis='y', which='minor', length=0)
    ax.grid(axis='y', which='major')

    ax.set_ylim(0, duration_size)
    ax.set_xticklabels([])
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_major_formatter(NullFormatter())

    # ---
    duration_steps_middle_to_long = duration_steps[duration_steps > 2*60]
    if duration_steps_middle_to_long.size:
        # (k)urzzeitige Summationen, d. h. der Dauerstufen von 5 Minuten bis 2 Stunden
        ax.axhline(duration_steps.tolist().index(duration_steps_middle_to_long[0]), color='black')
        duration_steps_long = duration_steps_middle_to_long[duration_steps_middle_to_long > 3*60*24]
        if duration_steps_long.size:
            # (m)ittelfristige Summationen, d. h. der Dauerstufen von 3 Stunden bis 3 Tagen.
            ax.axhline(duration_steps.tolist().index(duration_steps_long[0]), color='black')
    return ax



def idf_bar_axes(ax, idf_table, return_period_colors=RETURN_PERIOD_COLORS):
    """
    create a return period bar axes for the event plot

    Args:
        ax (matplotlib.pyplot.Axes):
        idf_table (pandas.DataFrame):
        return_period_colors (dict): color of each return period {return period: color}

    Returns:
        matplotlib.pyplot.Axes:
    """
    return _bar_axes(ax, idf_table, return_period_colors,
                     legend_kwags=dict(title='return periods'),
                     category_formatter='{}a'.format)


def _set_xlim(ax, bars, labels):
    fig = ax.get_figure()
    # Use the Transform to include both bar and label extents
    renderer = fig.canvas.get_renderer()

    # Calculate the data limits based on bars and labels
    max_data = max(bar.get_width() for bar in bars)
    max_label_extent = max(
        label.get_window_extent(renderer=renderer).transformed(ax.transData.inverted()).x1
        for label in labels
    )
    # print([
    #     label.get_window_extent(renderer=renderer).transformed(ax.transData.inverted()).x1
    #     for label in labels
    # ])
    # print(labels[0])
    # Update x-axis limits to fit both bars and labels
    # print(max_data, max_label_extent)
    ax.set_xlim(0, max(max_data, max_label_extent))
