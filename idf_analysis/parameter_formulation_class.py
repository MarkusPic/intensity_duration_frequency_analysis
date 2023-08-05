import abc
import yaml
import numpy as np

from .definitions import APPROACH, PARAM
from .parameter_formulations import folded_log_formulation, hyperbolic_formulation


class _Formulation(abc.ABC):
    def __init__(self, durations=None, param=None, a=None, b=None):
        self.durations = durations
        self.param = param

        self.a = a
        self.b = b

        self._balanced = (a is not None) and (b is not None)

    @abc.abstractmethod
    def fit(self, duration: np.array, param: np.array, param_mean=None, duration_mean=None):
        self.durations = duration
        self.param = param

        if param_mean is not None:
            self._balanced = True

    @abc.abstractmethod
    def get_param(self, duration):
        pass

    def __str__(self):
        return f'{self.__class__.__name__}(a={self.a}, b={self.b})'

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {PARAM.FUNCTION: self.LABEL, PARAM.A: self.a, PARAM.B: self.b}

    @classmethod
    def from_dict(cls, data):
        new = FORMULATION_REGISTER[data.pop(PARAM.FUNCTION)](**data)
        # new.durations = data.get(PARAM.DUR, None)
        # new.param = data.get('param', None)
        # new.a = data.get(PARAM.A, None)
        # new.b = data.get(PARAM.B, None)
        return new


class LinearFormulation(_Formulation):
    LABEL = APPROACH.LIN

    def __str__(self):
        return f'{self.__class__.__name__}()'

    def fit(self, duration: np.array, param: np.array, param_mean=None, duration_mean=None):
        super().fit(duration, param, param_mean, duration_mean)

    def get_param(self, duration):
        return np.interp(duration, self.durations, self.param)

    def to_dict(self):
        return {PARAM.FUNCTION: self.LABEL}


class LogNormFormulation(_Formulation):
    LABEL = APPROACH.LOG1

    def fit(self, duration: np.array, param: np.array, param_mean=None, duration_mean=None):
        super().fit(duration, param, param_mean, duration_mean)
        self.a, self.b = folded_log_formulation(duration, param, case=APPROACH.LOG1, param_mean=param_mean, duration_mean=duration_mean)

    def get_param(self, duration):
        return self.a + self.b * np.log(duration)


class DoubleLogNormFormulation(_Formulation):
    LABEL = APPROACH.LOG2

    def fit(self, duration: np.array, param: np.array, param_mean=None, duration_mean=None):
        super().fit(duration, param, param_mean, duration_mean)
        self.a, self.b = folded_log_formulation(duration, param, case=APPROACH.LOG2, param_mean=param_mean, duration_mean=duration_mean)

    def get_param(self, duration):
        return np.exp(self.a) * np.power(duration, self.b)


class HyperbolicFormulation(_Formulation):
    LABEL = APPROACH.HYP

    def fit(self, duration: np.array, param: np.array, param_mean=None, duration_mean=None):
        super().fit(duration, param, param_mean, duration_mean)

        a_start = 20.0
        if self.a is not None and not np.isnan(self.a):
            a_start = self.a

        b_start = 15.0
        if self.b is not None and not np.isnan(self.b):
            b_start = self.b

        self.a, self.b = hyperbolic_formulation(duration, param, a_start=a_start,
                                                b_start=b_start,
                                                param_mean=param_mean,
                                                duration_mean=duration_mean)

    def get_param(self, duration):
        return self.a * duration / (duration + self.b)


FORMULATION_REGISTER = {
    LogNormFormulation.LABEL: LogNormFormulation,
    DoubleLogNormFormulation.LABEL: DoubleLogNormFormulation,
    HyperbolicFormulation.LABEL: HyperbolicFormulation,
    LinearFormulation.LABEL: LinearFormulation,
}


def register_formulations_to_yaml():
    for f in FORMULATION_REGISTER.values():
        yaml.add_representer(f, lambda dumper, data: dumper.represent_dict(data.to_dict().items()))
