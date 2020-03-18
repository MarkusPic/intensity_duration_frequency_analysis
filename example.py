from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd
from os import path
import matplotlib.pyplot as plt

# dictionary of the data and the results
out = 'example'
# sub-folder for the results
output_directory = path.join(out, 'EXAMPLE')
# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=PARTIAL, worksheet=DWA, extended_durations=True)

# reading the pandas series of the precipitation
series = pd.read_parquet(path.join(out, 'expample_rain_data.parquet'))['precipitation']

# setting the series for the analysis
idf.set_series(series)

# auto-save the calculated parameter so save time for a later use
idf.auto_save_parameters(path.join(output_directory, 'parameters.yaml'))

# --------
# plotting the IDF curves
fig, ax = idf.result_figure(color=True)
fig.set_size_inches(6.40, 3.20)
fig.tight_layout()
fig.savefig(path.join(output_directory, 'idf_plot.png'), dpi=200)
plt.close(fig)

# -------
# plotting the idf curves in black and white
fig, ax = idf.result_figure(color=False)
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.savefig(path.join(out, 'EXAMPLE_plot.png'), dpi=200)
plt.close(fig)

# save the values of the idf curve as csv file
idf.result_table(add_names=False).to_csv(path.join(output_directory, 'idf_table.csv'))
