from ehyd_tools.design_rainfall import get_max_calculation_method, INDICES, get_ehyd_design_rainfall_offline

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# grid_point_number = 5214
for location, grid_point_number in {'graz': 5214, 'poellau': 4683}.items():

    df = get_ehyd_design_rainfall_offline(grid_point_number, pth='')

    idf_table = get_max_calculation_method(df)
    idf_table = df.xs('ÖKOSTRA', axis=0, level=INDICES.CALCULATION_METHOD, drop_level=True).copy()

    idf_reverse = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)
    idf_reverse._parameter.reverse_engineering(idf_table)

    output_directory = path.join(f'design_rain_ehyd_{grid_point_number}')
    idf_reverse.auto_save_parameters(path.join(output_directory, 'idf_parameters.yaml'))
    # exit()
    # ----------------------------------
    max_duration = 2880
    fig, ax = idf_reverse.result_figure(color=True, logx=True, max_duration=max_duration)
    fig.set_size_inches(12, 8)

    ax = idf_table.loc[:max_duration, [1, 2, 5, 10, 50, 100]].plot(ax=ax, marker='x', lw=0)
    # ax.set_xlim(0, 720)
    fig.tight_layout()
    fig.show()
    fig.savefig(path.join(f'idf_reverse_curves_plot_color_{location}.png'), dpi=200)

exit()
# ----------------------------------
# sub-folder for the results
output_directory = path.join('ehyd_112086_idf_data')
# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
series = pd.read_parquet('ehyd_112086.parquet')['precipitation']

# setting the series for the analysis
idf.set_series(series)
