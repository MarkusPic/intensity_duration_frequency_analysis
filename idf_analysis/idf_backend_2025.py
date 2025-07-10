from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as sps

from .definitions import SERIES, METHOD, PARAM
from .extreme_value_analysis_2025 import ExtremeValueParameters
from .idf_backend_abstract import IdfParametersABC
from .in_out import write_yaml, read_yaml
from .intensity_evaluation import IntensitiesExtractor


class IdfParametersNew(IdfParametersABC):
    def __init__(self, series_kind: str or Literal['partial', 'annual'] = SERIES.ANNUAL,
                 worksheet: str or Literal['DWA_A_531_2025'] = METHOD.DWA_2025,
                 extended_durations=False):
        super().__init__(series_kind, worksheet, extended_durations)

        self._parameter_values = {}  # dict with keys ['theta', 'eta', 'shape', 'loc', 'scale']

        self._intensities = None  # type: dict
        self._interims = None  # type: ExtremeValueParameters

    def calculation_done(self):
        return self._parameter_values and all(map(lambda i: i is not None, self._parameter_values.values()))

    def calc_from_series(self, series):
        self._intensity_extractor = IntensitiesExtractor(series, self.durations)
        self._intensities = self._intensity_extractor.get_all_intensities()

        self._interims = ExtremeValueParameters(self._intensities)
        self._parameter_values = self._interims.evaluate(self.series_kind)

    @staticmethod
    def fix_small_return_period(tn):
        return 1 / (np.log(tn) - np.log(tn - 1))

    @staticmethod
    def fix_small_return_period_r(tn):
        return np.exp(1 / tn) / (np.exp(1 / tn) - 1)

    @property
    def gev_dist(self):
        return sps.genextreme(c=self._parameter_values['shape'],
                              loc=self._parameter_values['loc'],
                              scale=self._parameter_values['scale'])

    def scale_factor(self, duration):
        return ExtremeValueParameters.get_scale_factor(duration, self._parameter_values['theta'],
                                                       self._parameter_values['eta'])

    def get_depth_of_rainfall(self, duration, return_period):
        """
        calculate the height of the rainfall h in L/m² = mm (respectively the unit of the series)

        Args:
            duration (int | float | list | numpy.ndarray | pandas.Series): duration: in minutes
            return_period (float): in years

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: height of the rainfall h in L/m² = mm (respectively the unit of the series)
        """
        if self.series_kind == SERIES.ANNUAL:
            if return_period < 5:
                print('WARNING: Using an annual series and a return period < 5 a will result in faulty values!')

        tn = return_period
        if isinstance(tn, (int, float)):
            if tn <= 10:
                tn = self.fix_small_return_period_r(tn)
        else:
            if isinstance(tn, (list, tuple, set)):
                tn = np.array(tn)

            tn = np.where(tn <= 10, self.fix_small_return_period_r(tn), tn)

        F = 1 - 1 / tn

        x = self.gev_dist.ppf(F)
        h = x / self.scale_factor(duration) * duration
        return h

    def get_return_period(self, height_of_rainfall, duration):
        """
        calculate the return period, when the height of rainfall and the duration are given

        Args:
            height_of_rainfall (float): in [mm]
            duration (int | float | list | numpy.ndarray | pandas.Series): in minutes

        Returns:
            int | float | list | numpy.ndarray | pandas.Series: return period in years
        """
        x = height_of_rainfall / duration * self.scale_factor(duration)
        F = self.gev_dist.cdf(x)
        tn = 1 / (1 - F)

        if isinstance(tn, (int, float)):
            if tn <= 10:
                tn = self.fix_small_return_period(tn)
        else:
            if isinstance(tn, (list, tuple, set)):
                tn = np.array(tn)

            tn = np.where(tn <= 10, self.fix_small_return_period(tn), tn)

        return tn

    def to_dict(self):
        def to_basic(a):
            return list([round(float(i), 4) for i in a])

        return {
            PARAM.SERIES: self.series_kind,
            PARAM.DUR: to_basic(self.durations),
            PARAM.INTENSITIES: self.get_intensities_output(),
            PARAM.PARAMS_FINAL: self._parameter_values
        }

    def pprint(self):
        from pprint import pprint
        pprint(self.to_dict())

    def to_yaml(self, filename):
        write_yaml(self.to_dict(), filename)

    @classmethod
    def from_yaml(cls, filename, worksheet=None):
        data = read_yaml(filename)
        p = cls(series_kind=data[PARAM.SERIES])
        p.durations = data[PARAM.DUR]

        p.load_intensities(data[PARAM.INTENSITIES])
        p._parameter_values = data[PARAM.PARAMS_FINAL]
        return p

    def get_intensities_output(self):
        di = {}
        for duration in self._intensities:
            s = self._intensities[duration].copy()
            s.index = s.index.map(str)
            di[int(duration)] = s.round(3).to_dict()
        return di

    def load_intensities(self, intensities_dict):
        self._intensities = {}
        for duration in intensities_dict:
            s = pd.Series(intensities_dict[duration])
            s.index = pd.to_datetime(s.index)
            self._intensities[duration] = s
