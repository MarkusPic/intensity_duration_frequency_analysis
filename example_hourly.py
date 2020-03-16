from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd
from os import path
import matplotlib.pyplot as plt

out = 'example'
output_directory = path.join(out, 'EXAMPLE-HOURLY')
idf = IntensityDurationFrequencyAnalyse(series_kind=PARTIAL, worksheet=DWA, extended_durations=True)

series = pd.read_parquet(path.join(out, 'expample_rain_data.parquet'))['precipitation'].resample('H').sum()

idf.set_series(series)

idf.auto_save_parameters(path.join(output_directory, 'parameters.yaml'))

fig, ax = idf.result_figure(color=True)
fig.set_size_inches(6.40, 3.20)
fig.tight_layout()
fig.savefig(path.join(output_directory, 'idf_plot.png'), dpi=200)
plt.close(fig)

idf.result_table(add_names=False).to_csv(path.join(output_directory, 'idf_table.csv'))
