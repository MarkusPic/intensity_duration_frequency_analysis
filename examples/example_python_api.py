from pathlib import Path

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd
import matplotlib.pyplot as plt

# sub-folder for the results
output_directory = Path('ehyd_112086_idf_data')
# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
series = pd.read_parquet('ehyd_112086.parquet')['precipitation']

# to reproduce this data run:
# from ehyd_tools.in_out import get_ehyd_data
# series2 = get_ehyd_data(identifier=112086)
# series2_sel = series2[series2 != 0]
# series2_sel = series2_sel.loc[:series.index[-1]]

# setting the series for the analysis
idf.set_series(series)
# idf.write_parameters(path.join(output_directory, 'idf_parameters.yaml'))
# exit()
# auto-save the calculated parameter so save time for a later use
idf.auto_save_parameters(output_directory / 'idf_parameters_new.yaml')

idf.model_rain_euler.get_series(return_period=.5, duration=60)

# exit()

# idf.result_figure(color=False)

# --------
# plotting the IDF curves
fig, ax = idf.result_figure(color=True)
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.savefig(output_directory / 'idf_curves_plot_color.png', dpi=200)
plt.close(fig)

# -------
# plotting the idf curves in black and white
fig, ax = idf.result_figure(color=False, add_interim=True)
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.savefig(output_directory / 'idf_curves_plot.png', dpi=200)
plt.close(fig)

# save the values of the idf curve as csv file
idf.result_table(add_names=False).to_csv(output_directory / 'idf_table_UNIX.csv',
                                         sep=',', decimal='.', float_format='%0.2f')
