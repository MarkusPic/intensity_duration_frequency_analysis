from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *
import pandas as pd
from os import path

# sub-folder for the results
output_directory = path.join('ehyd_112086_idf_data')
# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# reading the pandas series of the precipitation (data from ehyd.gv.at - ID=112086)
# You need to install `pyarrow` or `fastparquet` to read and write parquet files.
# series = pd.read_parquet('ehyd_112086.parquet')['precipitation']

# setting the series for the analysis
# idf.set_series(series)
# idf.write_parameters(path.join(output_directory, 'idf_parameters.yaml'))
# exit()
# auto-save the calculated parameter so save time for a later use
idf.auto_save_parameters(path.join(output_directory, 'idf_parameters_OLD_test.yaml'))
