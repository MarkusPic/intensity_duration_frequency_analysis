import warnings
import math
import abc

import numpy as np
import pandas as pd

"""design rainfall event based on IDF values."""


class _AbstractModelRain(abc.ABC):
    def __init__(self, idf=None):
        """
        Abstract class for a model rain.

        Args:
            idf (idf_analysis.IntensityDurationFrequencyAnalyse): idf-analysis object
        """
        self.idf = idf

    def _get_index(self, duration, interval=5):
        return range(interval, duration + interval, interval)

    def _get_idf_value(self, duration, return_period):
        return self.idf.depth_of_rainfall(duration, return_period)

    @abc.abstractmethod
    def get_series(self, return_period, duration, interval=5, **kwargs):
        """
        Get a pandas.Series of the model rain with the passed minutes as integer-index.

        Args:
            return_period (int): Return period in years.
            duration (int): Duration step in minutes.
            interval (int): Series intervall in minutes.
            **kwargs: Other key word arguments for subclasses.

        Returns:
            pd.Series: series of the model rain.
        """
        pass

    def get_time_series(self, return_period, duration, interval=5, start_time=None, **kwargs):
        """
        Get a pandas.Series of the model rain with timestamps as index.

        Args:
            return_period (int): Return period in years.
            duration (int): Duration step in minutes.
            interval (int): Series intervall in minutes.
            start_time (str | datetime.datetime): Start timestamp of the series.
            **kwargs: Other key word arguments for subclasses.

        Returns:
            pd.Series: time-series of the model rain.
        """
        rain = self.get_series(return_period, duration, interval, **kwargs)
        if start_time is not None:
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            rain.index = start_time + pd.to_timedelta(rain.index, unit='m')
            rain = rain.asfreq(pd.infer_freq(rain.index))
        return rain


class _BlockRain(_AbstractModelRain):
    """
    Synthetic model block rain.
    """
    def get_series(self, return_period, duration, interval=5, **kwargs):
        """
        Get a pandas.Series of the model block rain with the passed minutes as integer-index.

        Args:
            return_period (int): Return period in years.
            duration (int): Duration step in minutes.
            interval (int): Series intervall in minutes.
            **kwargs: Other key word arguments for subclasses.

        Returns:
            pd.Series: series of the model rain.
        """
        index = self._get_index(duration, interval)
        height = self._get_idf_value(duration, return_period)
        intensity = height / len(index)
        r = pd.Series(index=index, data=intensity)
        r = pd.concat([
            pd.Series({0: 0}),
            r
        ])
        return r


class _EulerRain(_AbstractModelRain):
    """
    Synthetic model Euler rain.
    """
    @staticmethod
    def _get_occurrence_highest_intensity(kind=2):
        if kind == 1:
            return 0
        elif kind == 2:
            return 1 / 3
        elif kind == 3:
            return 2 / 5

    def get_series(self, return_period, duration, interval=5, kind=2, **kwargs):
        """
        Get a pandas.Series of the model rain event acc. to Euler with the passed minutes as integer-index.

        Args:
            return_period (int or float): Return period in years.
            duration (int): Duration of the rain event in minutes.
            interval (int): Series intervall in minutes.
            kind (int): Either type I (1) or type II (2) or type III (3). If number smaller than 1 one can choose a custom peak at the given fraction of the event.
            **kwargs: Other key word arguments for subclasses.

        Returns:
            pd.Series: series of the model rain.
        """
        index = self._get_index(duration, interval)
        height = pd.Series(data=self._get_idf_value(np.array(index), return_period), index=index)

        if (height != height.expanding().max()).any():
            warnings.warn('The precipitation depth does not increase with the duration step. -> Adjusting data.')
            height = height.expanding().max()

        height_diff = height.diff()
        height_diff.iloc[0] = height.iloc[0]

        max_index = math.floor(round(((self._get_occurrence_highest_intensity(kind) if kind >= 1 else kind) * duration) / interval, 3)) * interval

        # sort differences and reset index
        r = height_diff.sort_values(ascending=False)
        r.index = sorted(r.index.values)

        # reverse first <occurrence_highest_intensity> values
        r.loc[:max_index] = r.loc[max_index::-1].values

        # add Zero value at first position (for SWMM ?)
        r = pd.Series(data=[0] + r.tolist(), index=[0] + r.index.values.tolist())

        return r
