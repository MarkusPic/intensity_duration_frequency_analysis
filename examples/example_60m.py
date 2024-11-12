import pandas as pd

from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import SERIES, METHOD


# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
# You need to install `pyarrow` or `fastparquet` to read and write parquet files.
series = pd.read_parquet('ehyd_112086.parquet')['precipitation']

series60 = series.resample('60min').sum()

# setting the series for the analysis
idf.set_series(series60.iloc[:40_000])

idf.parameters

h = idf.depth_of_rainfall(60, 2)
print()