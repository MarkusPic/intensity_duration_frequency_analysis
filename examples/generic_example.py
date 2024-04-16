# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 19:33:12 2020

@author: Nitesh
"""

"""
Doc: https://pypi.org/project/idf-analysis/
Example:
https://markuspic.github.io/intensity_duration_frequency_analysis/html/example/example_python_api.html

"""

import numpy as np
import pandas as pd
from idf_analysis.definitions import *
from idf_analysis.idf_class import IntensityDurationFrequencyAnalyse

idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

# generate random rain values
date_time = pd.date_range(start='2000-01-10 06:00', end='2010-01-10 06:00', freq='min')
np.random.seed(1)
data = np.random.rand(date_time.size, 1)[:,0]
data = pd.Series(data=data, name='precipitation', index=date_time)

idf.set_series(data)

idf.auto_save_parameters('idf_parameters.yaml')
print(idf.parameters)

idf.depth_of_rainfall(duration=15, return_period=1)
