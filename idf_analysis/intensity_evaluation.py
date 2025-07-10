import math
import warnings

import numpy as np
import pandas as pd

from .little_helpers import get_progress_bar
from .sww_utils import year_delta, guess_freq, rain_events, agg_events


class IntensitiesExtractor:
    def __init__(self, series, duration_steps):
        self.series = series
        self.duration_steps = duration_steps

        self.base_frequency = guess_freq(self.series.index)  # DateOffset/Timedelta
        self.base_delta = pd.Timedelta(self.base_frequency)

        # measuring time in years
        measurement_start, measurement_end = self.series.index[[0, -1]]
        self.measurement_period = (measurement_end - measurement_start) / year_delta(years=1)
        if round(self.measurement_period, 1) < 10:
            warnings.warn("The measurement period is too short. The results may be inaccurate! "
                          "It is recommended to use at least ten years. "
                          f"(-> Currently {self.measurement_period}a used)")

        # acc. to DWA-A 531 chap. 4.2:
        # The values must be independent of each other for the statistical evaluations.
        # estimated four hours acc. (Schilling, 1984)
        # for larger durations - use the duration as minimal gap
        self.min_event_gap = pd.Timedelta(hours=4)

        # Use only the (2-3 multiplied with the number of measuring years) of the biggest
        # values in the database (-> acc. to ATV-A 121 chap. 4.3; DWA-A 531 chap. 4.4).
        # As a requirement for the extreme value distribution.
        self.threshold_sample_size = int(math.floor(self.measurement_period * math.e))

        self._intensities = {}  # duration: {time_of_peak: intensity, ...}

    @staticmethod
    def _improve_factor(interval):
        """
        correction factor acc. to DWA-A 531 chap. 4.3

        Args:
            interval (float): length of the interval: number of observations per duration

        Returns:
            float: correction factor
        """
        improve_factor = {1: 1.14,
                          2: 1.07,
                          3: 1.04,
                          4: 1.03,
                          5: 1.00,
                          6: 1.00}

        return np.interp(interval,
                         list(improve_factor.keys()),
                         list(improve_factor.values()))

    def get_intensities_simple(self, duration_integer: int):
        if duration_integer not in self._intensities:
            duration = pd.Timedelta(minutes=duration_integer)

            if duration < self.base_delta:
                return

            if duration < self.min_event_gap:
                min_gap = self.min_event_gap
            else:
                min_gap = duration

            events = rain_events(self.series, min_gap=min_gap)

            # Correction factor acc. to DWA-A 531 chap. 4.3
            improve = self._improve_factor(duration / self.base_delta)

            roll_sum = self.series.rolling(duration).sum()

            events['i'] = agg_events(events, roll_sum, 'max') * improve
            events['ix'] = agg_events(events, roll_sum, 'idxmax')
            self._intensities[duration_integer] = events.set_index('ix')['i']
        return self._intensities[duration_integer]

    def get_intensities(self, duration_integer: int):
        if duration_integer not in self._intensities:
            duration = pd.Timedelta(minutes=duration_integer)

            if duration < self.base_delta:
                return

            if duration < self.min_event_gap:
                min_gap = self.min_event_gap
            else:
                min_gap = duration

            ts = self.series.copy()

            intensities = []
            while ts.sum() > 0:
                events = rain_events(ts, ignore_rain_below=0.01, min_gap=min_gap)
                if events.empty:
                    break

                roll_sum = ts.rolling(duration).sum()

                peak = agg_events(events, roll_sum, 'max')
                time_of_peak = agg_events(events, roll_sum, 'idxmax')

                intensities.append(
                    pd.Series(index=time_of_peak, data=peak)
                )

                new_start = time_of_peak - min_gap - duration
                new_end = time_of_peak + min_gap + self.base_delta

                for s, e in zip(new_start, new_end):
                    ts[s:e] = 0

            intensities = pd.concat(intensities).sort_index()

            # Correction factor acc. to DWA-A 531 chap. 4.3
            improve = self._improve_factor(duration / self.base_delta)
            self._intensities[duration_integer] = intensities * improve
        return self._intensities[duration_integer]

    def iter_all_intensities(self):
        pbar = get_progress_bar(self.duration_steps, desc='Calculating Intensities')

        for duration_integer in pbar:
            try:
                pbar.set_description(f'Calculating Intensities for duration {duration_integer:0.0f}')
            except:
                ...
            yield duration_integer, self.get_intensities(duration_integer)

    def get_all_intensities(self):
        return {i: d for i, d in self.iter_all_intensities()}
