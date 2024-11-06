from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *

# ----------------------------------
location = 'graz'
grid_point_number = 5214

# for location, grid_point_number in {'graz': 5214, 'poellau': 4683}.items():

output_directory = Path(f'design_rain_ehyd_{grid_point_number}')
output_directory.mkdir(exist_ok=True)

fn_idf_ehyd = output_directory / 'design_rain_ehyd_5214.csv'
if fn_idf_ehyd.is_file():
    df = pd.read_csv(fn_idf_ehyd, index_col=[0,1])
    df = df.rename(columns=int)
    df = df.rename(index=int, level=0)
else:
    from ehyd_tools.design_rainfall import get_max_calculation_method, get_ehyd_design_rainfall_offline
    df = get_ehyd_design_rainfall_offline(grid_point_number, pth='')
    # idf_table = get_max_calculation_method(df)

idf_table = df.xs('ÖKOSTRA', axis=0, level='calculation method', drop_level=True)

suffix = 'LINEAR'
suffix = 'KOSTRA'
idf_reverse = IntensityDurationFrequencyAnalyse.from_idf_table(idf_table, linear_interpolation=False)

# ---
fig = idf_reverse.parameters.interim_plot_parameters()
fig.savefig(output_directory / f'idf_reverse_interim_parameters_{suffix}.png')
plt.close(fig)

# ---
idf_reverse.auto_save_parameters(output_directory / f'idf_parameters_{suffix}.yaml')

# ----------------------------------
max_duration = 2880
fig, ax = idf_reverse.curve_figure(color=True, logx=True, max_duration=max_duration, duration_steps_ticks=True)

ax = idf_table.loc[:max_duration, [1, 2, 5, 10, 50, 100]].plot(ax=ax, marker='x', lw=0, color='black', legend=False)
fig.savefig(output_directory / f'idf_reverse_curves_plot_color_{location}_{suffix}.png')
plt.close(fig)

# ----------------------------------
if _ := 0:  # this is just an on/off switch, to only run certain parts of the code.
    # sub-folder for the results
    output_directory = Path('ehyd_112086_idf_data')
    # initialize of the analysis class
    idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

    # reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
    series = pd.read_parquet('ehyd_112086.parquet')['precipitation']

    # setting the series for the analysis
    idf.set_series(series)
