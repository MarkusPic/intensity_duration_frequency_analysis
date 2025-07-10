import abc
from typing import Literal

import numpy as np

from .definitions import SERIES, METHOD, PARAM
from .in_out import write_yaml
from .intensity_evaluation import IntensitiesExtractor
from .parameter_formulas import register_formulas_to_yaml


class IdfParametersABC(abc.ABC):
    def __init__(self, series_kind: str or Literal['partial', 'annual'] = SERIES.ANNUAL,
                 worksheet: str or Literal['KOSTRA', 'convective_vs_advective', 'ATV-A_121', 'neu'] = METHOD.DWA_2025,
                 extended_durations=False):
        self.series_kind = series_kind

        self._durations = None
        self._set_default_durations(extended_durations)

        self._intensity_extractor = None  # type: IntensitiesExtractor

    @abc.abstractmethod
    def calculation_done(self):
        ...

    @abc.abstractmethod
    def calc_from_series(self, series):
        ...

    @property
    def durations(self):
        return np.array(self._durations)

    @durations.setter
    def durations(self, durations):
        self._durations = list(durations)

    def _set_default_durations(self, extended_durations=False):
        # Suggestion from ATV-A 121 (ATV-DVWK-Regelwerk 2/2001)

        # sampling points of the duration steps in minutes
        duration_steps = [5, 10, 15, 20, 30, 45, 60, 90]
        # self._duration_steps += [i * 60 for i in [3, 4.5, 6, 7.5, 10, 12, 18]]  # duration steps in hours
        duration_steps += [i * 60 for i in [2, 3, 4, 6, 9, 12, 18]]  # duration steps in hours
        if extended_durations:
            duration_steps += [i * 60 * 24 for i in [1, 2, 3, 4, 5, 6]]  # duration steps in days

        self.durations = duration_steps

    def filter_durations(self, freq_minutes):
        self.limit_duration(lowest=freq_minutes)
        self.limit_durations_from_freq(freq_minutes)

    def limit_duration(self, lowest=None, highest=None):
        # bool_array = self.durations == False
        bool_array = np.array([False] * self.durations.size)
        if lowest is not None:
            bool_array |= self.durations >= lowest
        if highest is not None:
            bool_array |= self.durations <= highest

        self.durations = self.durations[bool_array]
        return bool_array

    def limit_durations_from_freq(self, freq_minutes):
        # only multiple of freq
        # Aus DWA-M 531 Abschnitt 2:
        # > Da Niederschlagsmesser Tageswerte liefern, sind hier nur Auswertungen für Regendauern möglich, die ein Vielfaches von 24 h betragen.
        self.durations = self.durations[(self.durations % freq_minutes) == 0]

    @abc.abstractmethod
    def get_depth_of_rainfall(self, duration, return_period):
        """
        calculate the height of the rainfall h in L/m² = mm (respectively the unit of the series)

        Args:
            duration (int | float | list | numpy.ndarray | pandas.Series): duration: in minutes
            return_period (float): in years

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: height of the rainfall h in L/m² = mm (respectively the unit of the series)
        """
        ...

    @abc.abstractmethod
    def get_return_period(self, height_of_rainfall, duration):
        """
        calculate the return period, when the height of rainfall and the duration are given

        Args:
            height_of_rainfall (float): in [mm]
            duration (int | float | list | numpy.ndarray | pandas.Series): in minutes

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: return period in years
        """
        ...

    @abc.abstractmethod
    def to_dict(self):
        def to_basic(a):
            return list([round(float(i), 4) for i in a])

        return {
            PARAM.SERIES: self.series_kind,
            PARAM.DUR: to_basic(self.durations),
        }

    def pprint(self):
        from pprint import pprint
        pprint(self.to_dict())

    def to_yaml(self, filename):
        register_formulas_to_yaml()
        write_yaml(self.to_dict(), filename)

    @classmethod
    @abc.abstractmethod
    def from_yaml(cls, filename, worksheet=None):
        ...
