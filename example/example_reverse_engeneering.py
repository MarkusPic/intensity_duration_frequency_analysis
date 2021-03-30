from ehyd_tools.design_rainfall import (ehyd_design_rainfall_ascii_reader, get_ehyd_file, get_max_calculation_method,
                                        get_rainfall_height, INDICES, )

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# df = ehyd_design_rainfall_ascii_reader(get_ehyd_file(grid_point_number=5214))
# df.to_csv('design_rain_ehyd_5214.csv')
df = pd.read_csv('design_rain_ehyd_5214.csv', index_col=[0, 1])
df.columns = df.columns.astype(int)
idf_table = get_max_calculation_method(df)
idf_table = df.xs('ÖKOSTRA', axis=0, level=INDICES.CALCULATION_METHOD, drop_level=True).copy()

idf_reverse = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)
idf_reverse._parameter.reverse_engineering(idf_table)

output_directory = path.join('design_rain_ehyd_5214')
idf_reverse.auto_save_parameters(path.join(output_directory, 'idf_parameters.yaml'))
exit()
# ----------------------------------
max_duration = 2880
fig, ax = idf_reverse.result_figure(color=True, logx=True, max_duration=max_duration)
fig.set_size_inches(12, 8)

ax = idf_table.loc[:max_duration, [1, 2, 5, 10, 50, 100]].plot(ax=ax, marker='x', lw=0)
# ax.set_xlim(0, 720)
fig.tight_layout()
fig.show()
fig.savefig(path.join('idf_reverse_curves_plot_color.png'), dpi=200)

# ----------------------------------
# sub-folder for the results
output_directory = path.join('ehyd_112086_idf_data')
# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
series = pd.read_parquet('ehyd_112086.parquet')['precipitation']

# setting the series for the analysis
idf.set_series(series)
