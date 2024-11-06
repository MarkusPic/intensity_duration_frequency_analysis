from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.heavy_rainfall_index import HeavyRainfallIndexAnalyse
from idf_analysis.definitions import *

# sub-folder for the results
output_directory = Path('ehyd_112086_idf_data')
# initialize of the analysis class
idf = HeavyRainfallIndexAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
series = pd.read_parquet('ehyd_112086.parquet').squeeze()
series.name = 'precipitation'

# to reproduce this data run:
# from ehyd_tools.in_out import get_ehyd_data
# series2 = get_ehyd_data(identifier=112086)
# series2_sel = series2[series2 != 0]
# series2_sel = series2_sel.loc[:series.index[-1]]

# setting the series for the analysis
idf.set_series(series)

# --------
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    fig = idf.parameters._interims.plot_series()
    fig.set_size_inches(12, 24)
    # fig.savefig(output_directory / 'interims.pdf')
    fig.savefig(output_directory / f'interims_{idf.parameters.series_kind}.png')
    plt.close(fig)

# --------
# auto-save the calculated parameter so save time for a later use
idf.auto_save_parameters(output_directory / 'idf_parameters.yaml')

# --------
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    for m in (METHOD.ATV, METHOD.KOSTRA):
        idf.parameters.set_parameter_approaches_from_worksheet(m)
        fig = idf.parameters.interim_plot_parameters()
        fig.set_size_inches(8, 6)
        fig.savefig(output_directory / f'idf_parameters_plot_{m}.png')
        plt.close(fig)

# --------
# plot over time - y-axis=return period | x-axis=time | legend=duration range
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    fig, _ = idf.return_period_scatter()
    fig.set_size_inches(8, 6)
    fig.savefig(output_directory / 'idf_return_period_scatter_plot.png')
    plt.close(fig)


# --------
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    fig = idf.parameters.interim_plot_parameters()
    fig.set_size_inches(8, 6)
    # fig.savefig(output_directory / 'idf_parameters_plot_no_balance.png', dpi=200)
    fig.savefig(output_directory / 'idf_parameters_plot_balanced.png')
    # fig.savefig(output_directory / 'idf_parameters_plot.png', dpi=200)
    plt.close(fig)

# --------
# plotting the IDF curves
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    fig_size = (8, 5)
    fig, ax = idf.curve_figure(color=True, logx=True, duration_steps_ticks=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_color_logx.png')
    plt.close(fig)

    fig, ax = idf.curve_figure(color=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_color.png')
    plt.close(fig)

    fig, ax = idf.curve_figure(color=True, add_interim=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_color+interim.png')
    plt.close(fig)

    fig, ax = idf.curve_figure(color=True, logx=True, duration_steps_ticks=True, add_interim=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_color_logx+interim.png')
    plt.close(fig)

    fig, ax = idf.curve_figure(color=True, logx=True, duration_steps_ticks=True, add_range_limits=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot.png')
    plt.close(fig)

# -------
# plotting the idf curves in black and white
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    fig, ax = idf.curve_figure(color=False, add_interim=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_bw+interim.png')
    plt.close(fig)

    fig, ax = idf.curve_figure(color=False)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_bw.png')
    plt.close(fig)

# --------
# save the values of the idf curve as csv file
if _ := 1:  # this is just an on/off switch, to only run certain parts of the code.
    idf.result_table(add_names=False).to_csv(output_directory / 'idf_table_UNIX.csv',
                                             sep=',', decimal='.', float_format='%0.2f')

# --------
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    idf.model_rain_euler.get_series(return_period=.5, duration=60)

# --------
# plotting the IDF curves for JOSS paper
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    fig, ax = idf.curve_figure(color=True, logx=True, max_duration=60 * 24 * 6, duration_steps_ticks=True, add_interim=False)
    ax.grid(ls=':', lw=0.5)
    fig.set_size_inches(*fig_size)
    ax.set_title('')
    fig.savefig(output_directory.resolve().parent.parent / 'joss-paper' / 'idf_curves_plot.pdf')
    plt.close(fig)
