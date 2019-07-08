from idf_analysis.idf_class import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd
import matplotlib.pyplot as plt

out = 'example'
name = 'EXAMPLE-HOURLY'
idf = IntensityDurationFrequencyAnalyse(series_kind=PARTIAL, worksheet=DWA, output_path=out,
                                        extended_durations=True, output_filename=name,
                                        auto_save=True, unix=True)

data = pd.read_parquet('example/expample_rain_data.parquet').resample('H').sum()

idf.set_series(data['precipitation'])

fig = idf.result_figure(color=True)

fn = idf.output_filename + '_idf_plot.png'
fig.set_size_inches(6.40, 3.20)
fig.tight_layout()
fig.savefig(fn, dpi=200)
plt.close(fig)
