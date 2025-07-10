import math
import warnings
from typing import Literal

import numpy as np
import scipy.stats as sps

from .definitions import SERIES
from .intensity_evaluation import IntensitiesExtractor
from .little_helpers import get_progress_bar


class ExtremValueSeries:
    def __init__(self):
        self.intensities = {}
        self.measurement_period = None
        self.data = {}
        self.duration_steps = None
        self._intensity_extractor = None  # type: IntensitiesExtractor

    @classmethod
    def from_intensity_extractor(cls, intensities_extractor: IntensitiesExtractor):
        new = cls()
        new._intensity_extractor = intensities_extractor
        new.duration_steps = intensities_extractor.duration_steps
        new.intensities = new._intensity_extractor._intensities
        new.measurement_period = new._intensity_extractor.measurement_period
        return new

    @classmethod
    def from_series(cls, series, duration_steps):
        intensity_extractor = IntensitiesExtractor(series, duration_steps)
        return cls.from_intensity_extractor(intensity_extractor)

    @classmethod
    def from_intensities(cls, intensities, measurement_period):
        new = cls()
        new.duration_steps = list(intensities.keys())
        new.intensities = intensities
        new.measurement_period = measurement_period
        return new

    @property
    def threshold_sample_size(self):
        # Use only the (2-3 multiplied with the number of measuring years) of the biggest
        # values in the database (-> acc. to ATV-A 121 chap. 4.3; DWA-A 531 chap. 4.4).
        # As a requirement for the extreme value distribution.
        return int(math.floor(self.measurement_period * math.e))

    def get_intensities(self, duration_integer: int):
        if self._intensity_extractor is None:
            return self.intensities[duration_integer]
        else:
            # return self._intensity_extractor.get_intensities(duration_integer)
            return self._intensity_extractor.get_intensities_simple(duration_integer)

    @staticmethod
    def _plotting_formula(k, l, m):
        """
        plotting function acc. to DWA-A 531 chap. 5.1.3 for the partial series

        Args:
            k (float or np.ndarray): running index
            l (float or np.ndarray): sample size
            m (float or np.ndarray): measurement period

        Returns:
            float: estimated empirical return period
        """
        return (l + 0.2) * m / ((k - 0.4) * l)

    def annual_series(self, duration_integer: int):
        """
        Create an annual series of the maximum overlapping sum per year and calculate the "u" and "w" parameters.

        acc. to DWA-A 531 chap. 5.1.5

        Gumbel distribution | https://en.wikipedia.org/wiki/Gumbel_distribution

        Args:
            duration_integer (int): Duration step in minutes.

        Returns:
            dict[str, float]: Parameter u and w from the annual series for a specific duration step as a tuple.
        """
        annually_series = self.get_intensities(duration_integer).resample('YE').max().sort_values(
            ascending=False).values
        x = -np.log(
            np.log((annually_series.size + 0.2) / (annually_series.size - sps.rankdata(annually_series)[::-1] + 0.6)))
        return x, annually_series

    def partial_series(self, duration_integer: int):
        """
        Create a partial series of the largest overlapping sums and calculate the "u" and "w" parameters.

        acc. to DWA-A 531 chap. 5.1.4

        Exponential distribution | https://en.wikipedia.org/wiki/Exponential_distribution

        Args:
            duration_integer (int): Duration step in minutes.

        Returns:
            dict[str, float]: parameter u and w from the partial series for a specific duration step as a tuple
        """
        partially_series = self.get_intensities(duration_integer).sort_values(ascending=False).values

        if partially_series.size < self.threshold_sample_size:
            warnings.warn(
                'Fewer events in series than recommended for extreme value analysis. Use the results with mindfulness.')
        else:
            partially_series = partially_series[:self.threshold_sample_size]

        x = np.log(self._plotting_formula(sps.rankdata(partially_series)[::-1], partially_series.size,
                                          self.measurement_period))

        return x, partially_series

    def iter_duration_steps(self, series_kind: str or Literal['partial', 'annual']):
        """
        Statistical analysis for each duration step.

        acc. to DWA-A 531 chap. 5.1

        Save the parameters of the distribution function as interim results.

        Args:
            series_kind (str): which kind of series should be used to evaluate the extreme values.
        """
        pbar = get_progress_bar(self.duration_steps, desc='Calculating Parameters u and w')

        for duration_integer in pbar:
            try:
                pbar.set_description(f'Calculating Parameters u and w for duration {duration_integer:0.0f}')
            except:
                ...

            if series_kind == SERIES.ANNUAL:
                x, y = self.annual_series(duration_integer)
            elif series_kind == SERIES.PARTIAL:
                x, y = self.partial_series(duration_integer)
            else:
                raise NotImplementedError(f"Unknown series kind {series_kind}")

            self.data[duration_integer] = {'x': x, 'y': y, 'series_kind': series_kind}
            yield duration_integer, self.data[duration_integer]

    def evaluate(self, series_kind: str or Literal['partial', 'annual']):
        """
        Statistical analysis for each duration step.

        acc. to DWA-A 531 chap. 5.1

        Save the parameters of the distribution function as interim results.

        Args:
            series_kind (str): which kind of series should be used to evaluate the extreme values.
        """
        for _ in self.iter_duration_steps(series_kind):
            pass
        return self.data
