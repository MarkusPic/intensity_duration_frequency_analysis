import abc
import inspect
import warnings

import numpy as np
import yaml

from scipy.optimize import curve_fit, OptimizeWarning

from .definitions import APPROACH, PARAM


# === === === === === === === === === === === === === ===
class _Formula(abc.ABC):
    LABEL = None

    def __init__(self, durations=None, values=None):
        self.durations = durations
        self.values = values

        self.is_fitted = False
        self._balanced = False

    @abc.abstractmethod
    def fit(self, durations: np.array, values: np.array, durations_fixed=None, values_fixed=None):
        self.durations = durations
        self.values = values
        self.is_fitted = True

        if values_fixed is not None:
            # if mean values are provided, then the balance function is in process ...
            self._balanced = True

    @abc.abstractmethod
    def get_param(self, duration):
        return ...

    def __str__(self):
        return f'{self.__class__.__name__.replace("Formula", "")}'

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {PARAM.FUNCTION: self.LABEL}

    @classmethod
    def from_dict(cls, data):
        return FORMULA_REGISTER[data.pop(PARAM.FUNCTION)](**data)

    @property
    @abc.abstractmethod
    def latex_formula(self):
        return ''


# === === === === === === === === === === === === === ===
class _Formula2Params(_Formula, abc.ABC):
    def __init__(self, durations=None, values=None, a=None, b=None):
        super().__init__(durations, values)

        # parameters for formula|function
        self.a = a
        self.b = b

        self._balanced = (a is not None) and (b is not None)

    @staticmethod
    @abc.abstractmethod
    def formula(D, a, b):
        return ...

    def get_param(self, duration):
        return self.formula(duration, self.a, self.b)

    def __str__(self):
        if self.a is None:
            a = '-'
        else:
            a = round(self.a, 2)
        if self.b is None:
            b = '-'
        else:
            b = round(self.b, 2)

        return f'{super().__str__()}({a=}, {b=})'

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {PARAM.FUNCTION: self.LABEL,
                PARAM.A: self.a,
                PARAM.B: self.b}

    @property
    @abc.abstractmethod
    def latex_formula(self):
        return ''


# === === === === === === === === === === === === === ===
class LinearFormula(_Formula):
    LABEL = APPROACH.LIN

    def __str__(self):
        return f'{super().__str__()}()'

    def fit(self, durations: np.array, values: np.array, durations_fixed=None, values_fixed=None):
        if values_fixed is not None and durations_fixed is not None:
            i = list(durations).index(durations_fixed)
            values[i] = values_fixed
        super().fit(durations, values, values_fixed, durations_fixed)

    def get_param(self, duration):
        return np.interp(duration, self.durations, self.values)

    @property
    def latex_formula(self):
        return 'Linear'


# === === === === === === === === === === === === === ===
class LogNormFormula(_Formula2Params):
    LABEL = APPROACH.LOG1

    def fit(self, durations: np.array, values: np.array, durations_fixed=None, values_fixed=None):
        super().fit(durations, values, values_fixed, durations_fixed)

        if values_fixed and durations_fixed:
            mean_ln_duration = np.log(durations_fixed)
        else:
            values_fixed = values.mean()
            mean_ln_duration = np.log(durations).mean()

        divisor = ((np.log(durations) - mean_ln_duration) ** 2).sum()

        self.b = ((values - values_fixed) * (np.log(durations) - mean_ln_duration)).sum() / divisor
        self.a = values_fixed - self.b * mean_ln_duration

    @staticmethod
    def formula(D, a, b):
        return a + b * np.log(D)

    @property
    def latex_formula(self):
        return f'LogNorm({self.a:0.2f} + {self.b:0.2f}*log(D))'


# === === === === === === === === === === === === === ===
class DoubleLogNormFormula(_Formula2Params):
    LABEL = APPROACH.LOG2

    def fit(self, durations: np.array, values: np.array, durations_fixed=None, values_fixed=None):
        super().fit(durations, values, values_fixed, durations_fixed)

        if values_fixed and durations_fixed:
            mean_ln_duration = np.log(durations_fixed)
            mean_ln_param = np.log(values_fixed)
        else:
            mean_ln_param = np.log(values).mean()
            mean_ln_duration = np.log(durations).mean()

        divisor = ((np.log(durations) - mean_ln_duration) ** 2).sum()

        self.b = ((np.log(values) - mean_ln_param) * (np.log(durations) - mean_ln_duration)).sum() / divisor
        self.a = mean_ln_param - self.b * mean_ln_duration

    @staticmethod
    def formula(D, a, b):
        return np.exp(a) * np.power(D, b)

    @property
    def latex_formula(self):
        return f'DoubleLogNorm(e^{{{self.a:0.2f}}} * D^{{{self.b:0.2f}}})'


# === === === === === === === === === === === === === ===
def hyperbolic_formula(duration: np.array, param: np.array, a_start=20.0, b_start=15.0, param_mean=None, duration_mean=None):
    """
    Computes the hyperbolic formula using the given parameters.

    Args:
        duration (np.array): Array of durations.
        param (np.array): Array of parameters.
        a_start (float): Initial value for 'a' parameter.
        b_start (float): Initial value for 'b' parameter.
        param_mean (float, optional): Mean value of parameters. Defaults to None.
        duration_mean (float, optional): Mean value of durations. Defaults to None.

    Returns:
        tuple[float, float]: Computed value of 'a' and 'b'.
    """

    def get_param(dur, par, a_, b_):
        """
        Computes 'a' and 'b' parameters based on given durations and parameters.

        Args:
            dur (np.array): Array of durations.
            par (np.array): Array of parameters.
            a_ (float): Current value of 'a' parameter.
            b_ (float): Current value of 'b' parameter.

        Returns:
            tuple[float, float]: Computed value of 'a' and 'b'.
        """
        i = -a_ / (dur + b_)
        if param_mean:
            i_mean = - param_mean / duration_mean
            param_mean_ = param_mean
        else:
            i_mean = i.mean()
            param_mean_ = par.mean()

        b_ = ((par - param_mean_) * (i - i_mean)).sum() / ((i - i_mean) ** 2).sum()
        a_ = param_mean_ - b_ * i_mean
        return a_, b_

    # ------------------------------------------------------------------------------------------------------------------
    iteration_steps = 0

    a = a_start
    b = b_start

    conditions = True
    while conditions:
        conditions = False

        a_s = a
        b_s = b
        a, b = get_param(duration, param, a, b)
        conditions = (abs(a - a_s) > 0.005) or (abs(b - b_s) > 0.005) or conditions
        a = (a + a_s) / 2
        b = (b + b_s) / 2

        iteration_steps += 1
    return float(a), float(b)


def _set_if_unknown(v, default):
    if v is not None and not np.isnan(v):
        return v
    else:
        return default


class HyperbolicFormula(_Formula2Params):
    LABEL = APPROACH.HYP

    @property
    def latex_formula(self):
        return f'Hyperbolic(\\frac{{{self.a:0.2f} * D }}{{D + {self.b:0.2f}}})'

    @staticmethod
    def formula(D, a, b):
        return a * D / (D + b)

    def fit(self, durations: np.array, values: np.array, durations_fixed=None, values_fixed=None):
        super().fit(durations, values, values_fixed, durations_fixed)

        a_start = _set_if_unknown(self.a, 20.0)
        b_start = _set_if_unknown(self.b, 15.0)

        self.a, self.b = hyperbolic_formula(durations, values, a_start=a_start,
                                            b_start=b_start,
                                            param_mean=values_fixed,
                                            duration_mean=durations_fixed)


# === === === === === === === === === === === === === ===
class AutoFormula2Params(_Formula2Params, abc.ABC):
    LABEL = 'auto'

    @property
    def latex_formula(self):
        import sympy as sp
        D, a, b = sp.symbols('D a b')
        return f'{self.LABEL}({sp.latex(self.formula(D, a, b))})'

    @staticmethod
    @abc.abstractmethod
    def formula(D, a, b):
        return ...

    def get_param(self, duration):
        return self.formula(duration, self.a, self.b)

    def fit(self, durations: np.array, values: np.array, durations_fixed=None, values_fixed=None):
        super().fit(durations, values, values_fixed, durations_fixed)

        if values_fixed is None:
            fit = curve_fit(self.formula, durations, values)
            self.a, self.b = fit[0]
        else:
            import sympy as sp

            p, D, a, b = sp.symbols('p D a b')
            sp.latex(self.formula(D, a, b))

            fa = sp.solve(self.formula(durations_fixed, a, b) - values_fixed, a)[0]

            # Convert symbolic function to a numerical function for curve fitting
            f_fixed_func = sp.lambdify((D, b), self.formula(D, fa, b), 'numpy')

            fit = curve_fit(f_fixed_func, durations, values, p0=self.b)
            self.b = fit[0][0]
            self.a = float(fa.subs(b, self.b))


# === === === === === === === === === === === === === ===
# === === === === === === === === === === === === === ===
# === === === === === === === === === === === === === ===
class _FormulaNParams(_Formula, abc.ABC):
    LABEL = None

    def __init__(self, durations=None, values=None, params=None, *params_args, **params_kwargs):
        super().__init__(durations, values)

        if params is not None:
            self.params = params
        elif params_args:
            self.params = list(params_args)
        elif params_kwargs:
            self.params = list(params_kwargs.values())

        self._balanced = bool(params)

    @staticmethod
    @abc.abstractmethod
    def formula(D, *params):
        return ...

    def get_param(self, duration):
        return self.formula(duration, *self.params)

    def __str__(self):
        return super().__str__() + '(' + ', '.join([f'{p:0.2f}' for p in self.params]) + ')'

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {PARAM.FUNCTION: self.LABEL,
                'params': [float(p) for p in self.params]}

    @property
    @abc.abstractmethod
    def latex_formula(self):
        return ''


# === === === === === === === === === === === === === ===
class AutoFormulaNParams(_FormulaNParams, abc.ABC):
    LABEL = 'autoN'

    @property
    def latex_formula(self):
        import sympy as sp
        label = ''.join(i.capitalize() for i in self.LABEL.split('_'))
        s = f'{label}({sp.latex(self.formula_sy(*self.get_sympy_variables), mul_symbol="dot")})'
        for v, n in zip(self.get_sympy_variables[1:], self.params):
            s = (s.replace(f' {v} ', f' {n:0.2f} ').
                 replace(f' {v}}}', f' {n:0.2f}}}').
                 replace(f'{{{v} ', f'{{{n:0.2f} ').
                 replace(f'({v} ', f'({n:0.2f} ').
                 replace(f'{{{v}}}', f'{{{n:0.2f}}}').
                 replace(f' {v})', f' {n:0.2f})'))
        return s
        # f'{label}({sp.latex(self.formula_sy(self.get_sympy_variables[0], *self.params), mul_symbol="dot", full_prec=False)})'
        # return f'{label}({sp.latex(self.formula_sy(self.get_sympy_variables[0], *[round(p,2) for p in self.params]), mul_symbol="dot", full_prec=False)})'

    def get_param(self, duration):
        return self.formula(duration, *self.params)

    @property
    def n_params(self):
        # Get the signature of the function
        signature = inspect.signature(self.formula)
        # Count the parameters in the signature
        return len(signature.parameters) - 1

    @staticmethod
    @abc.abstractmethod
    def formula_sy(D, *params):
        ...

    @property
    def get_sympy_variables(self):
        import sympy as sp
        import string
        D, *params = sp.symbols('D ' + ' '.join(string.ascii_lowercase[:self.n_params]))
        return D, *params

    def fit(self, durations: np.array, values: np.array, durations_fixed=None, values_fixed=None):
        super().fit(durations, values, values_fixed, durations_fixed)

        if values_fixed is None:
            fit = curve_fit(self.formula, durations, values)
            self.params = list(fit[0])
        else:
            import sympy as sp
            D, *params_sy = self.get_sympy_variables

            if isinstance(durations_fixed, list) and (len(durations_fixed) == self.n_params):
                with warnings.catch_warnings(action="ignore", category=OptimizeWarning):
                    fit = curve_fit(self.formula, durations_fixed, values_fixed)
                self.params = list(fit[0])
                return

            variants = []
            for i in range(len(params_sy)):
                if isinstance(durations_fixed, list):
                    warnings.warn('parameter balance not implemented yet')
                fi = sp.solve(self.formula_sy(durations_fixed, *params_sy) - values_fixed, params_sy[i])[0]

                params_rest_sy = list(params_sy[:i]) + list(params_sy[i+1:])
                # Convert symbolic function to a numerical function for curve fitting
                f_fixed_func = sp.lambdify((D, *params_rest_sy), self.formula_sy(D, fi, *params_rest_sy), 'numpy')

                fit = curve_fit(f_fixed_func, durations, values, p0=self.params[i])
                params_rest = list(fit[0])
                params_i = float(fi.subs(list(zip(params_rest_sy, params_rest))))

                params = list(params_rest[:i]) + [params_i] + list(params_rest[i:])
                from statsmodels.tools.eval_measures import rmse

                variants.append([params, rmse(self.formula(durations, *params), values)])

            self.params = min(variants, key=lambda x: x[1])[0]


class HyperbolicAuto(AutoFormulaNParams):
    LABEL = APPROACH.HYP

    @staticmethod
    def formula(D, a, b):
        return a * D / (D + b)

    @staticmethod
    def formula_sy(D, a, b):
        return HyperbolicAuto.formula(D, a, b)


class LogNormAuto(AutoFormulaNParams):
    LABEL = APPROACH.LOG1

    @staticmethod
    def formula(D, a, b):
        return a + b * np.log(D)

    @staticmethod
    def formula_sy(D, a, b):
        import sympy as sp
        return a + b * sp.log(D)


class DoubleLogNormAuto(AutoFormulaNParams):
    LABEL = APPROACH.LOG2

    @staticmethod
    def formula(D, a, b):
        return np.exp(a) * np.power(D, b)

    @staticmethod
    def formula_sy(D, a, b):
        import sympy as sp
        return sp.exp(a) * D**b


# === === === === === === === === === === === === === ===
FORMULA_REGISTER = {
    LogNormFormula.LABEL: LogNormAuto,
    DoubleLogNormFormula.LABEL: DoubleLogNormAuto,
    HyperbolicFormula.LABEL: HyperbolicAuto,
    LinearFormula.LABEL: LinearFormula,
    # HyperbolicAuto.LABEL: HyperbolicAuto,
    # LogNormAuto.LABEL: LogNormAuto,
    # DoubleLogNormAuto.LABEL: DoubleLogNormAuto,
}


def register_formulas_to_yaml():
    for f in FORMULA_REGISTER.values():
        yaml.add_representer(f, lambda dumper, data: dumper.represent_dict(data.to_dict().items()))
