from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.heavy_rainfall_index import HeavyRainfallIndexAnalyse
from idf_analysis.definitions import *

# sub-folder for the results
output_directory = Path('ehyd_112086_idf_data_new')
output_directory.mkdir(parents=True, exist_ok=True)
# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.ANNUAL, worksheet=METHOD.DWA_2025, extended_durations=True)

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
# auto-save the calculated parameter so save time for a later use
idf.auto_save_parameters(output_directory / 'idf_parameters.yaml')

# --------
# plot over time - y-axis=return period | x-axis=time | legend=duration range
if _ := 1:  # this is just an on/off switch, to only run certain parts of the code.
    fig, _ = idf.return_period_scatter()
    fig.set_size_inches(8, 6)
    fig.savefig(output_directory / 'idf_return_period_scatter_plot.png')
    plt.close(fig)

# --------
fig_size = (8, 5)
# plotting the IDF curves
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    fig, ax = idf.curve_figure(color=True, logx=True, duration_steps_ticks=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_color_logx.png')
    plt.close(fig)

    fig, ax = idf.curve_figure(color=True)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_color.png')
    plt.close(fig)

    # -------
    # plotting the idf curves in black and white
    fig, ax = idf.curve_figure(color=False)
    fig.set_size_inches(*fig_size)
    fig.savefig(output_directory / 'idf_curves_plot_bw.png')
    plt.close(fig)

# --------
# save the values of the idf curve as csv file
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    idf.result_table(add_names=False).to_csv(output_directory / 'idf_table_UNIX.csv',
                                             sep=',', decimal='.', float_format='%0.2f')

# --------
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    idf.model_rain_euler.get_series(return_period=.5, duration=60)
