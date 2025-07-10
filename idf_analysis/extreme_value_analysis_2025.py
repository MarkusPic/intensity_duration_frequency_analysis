from typing import Literal

import numpy as np
import pandas as pd
import scipy.optimize as spo
import scipy.special as sp
import scipy.stats as sps

from .definitions import SERIES


class ExtremeValueParameters:
    def __init__(self, intensities: dict):
        self._intensities = intensities

    def annual_intensities(self):
        df = {}
        for d in self._intensities:
            df[d] = self._intensities[d].resample('YE').max()
        return pd.DataFrame(df)

    @staticmethod
    def get_scale_factor(duration, theta, eta):
        return (duration + theta) ** eta

    @classmethod
    def get_scale_intensities(cls, df, theta, eta):
        """
        Scale rainfall intensities using the scaling factor.
        """
        b_D = cls.get_scale_factor(df.columns.values, theta, eta)
        return df * b_D

    def estimate_scale_parameters(self, df, theta0=None, eta0=None):
        """
        Estimate theta and eta parameters by minimizing the Kruskal-Wallis Test Statistic.
        """

        def test_statistic(params):
            theta, eta = params
            scaled_intensities = self.get_scale_intensities(df, theta, eta)

            # ----
            # manually
            if _ := 0:
                ranked_df = scaled_intensities.stack().rank().unstack()
                m = df.size

                # Anzahl der Jahre (n_D)
                n_D = df.index.size

                # KW statistic
                statistic = 0
                for duration in df:
                    # Mittlerer Rang für die Dauerstufe (r̄_D)
                    r_bar_D = ranked_df[duration].mean()

                    # Summenformel
                    statistic += n_D * (r_bar_D - (m + 1) / 2) ** 2

                statistic *= (12 / (m * (m + 1)))

            # else:
            statistic, p_value = sps.kruskal(*[scaled_intensities[d] for d in scaled_intensities])

            # statistic = round(statistic, 4-int(math.floor(math.log10(abs(statistic)))))
            # print(statistic)
            return statistic

        tol = None

        if theta0 is None:
            theta0 = 0.05

        if eta0 is None:
            eta0 = 0.75

        # Minimize Test Statistic
        result = spo.minimize(test_statistic, x0=[theta0, eta0], bounds=[(0, None), (0, None)], method='Powell',
                              tol=tol)
        # print(result.nfev)
        theta_opt, eta_opt = result.x

        return theta_opt, eta_opt

    @staticmethod
    def compute_lmoments(data, shape=-0.1):
        """Compute L-moments (L1, L2, and L-skew) from the dataset."""
        # Recompute L-moments manually with a focus on stability
        L1 = sps.lmoment(data, 1)
        L2 = sps.lmoment(data, 2)

        scale = (L2 * shape) / ((1 - 2 ** (-shape)) * sp.gamma(1 + shape))
        loc = L1 - scale * (1 - sp.gamma(1 + shape)) / shape

        return shape, loc, scale

    def fit_gev_distribution(self, scaled_intensities, method='LM-fixed'):
        """
        Fit the Generalized Extreme Value (GEV) distribution to scaled intensities.
        """
        data = scaled_intensities.stack().values

        if method == 'LM-open':
            from lmoments3 import distr
            gev_params = distr.gev.lmom_fit(data)
            shape, loc, scale = gev_params['c'], gev_params['loc'], gev_params['scale']
        elif method == 'MLE':
            shape, loc, scale = sps.genextreme.fit(data)
        elif method == 'LM-fixed':
            shape, loc, scale = self.compute_lmoments(data, -0.1)
        elif method == 'MM-fixed':
            loc, shape = sps.genextreme.fit_loc_scale(data, -0.1)
            shape = -0.1
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

        return shape, loc, scale

    def estimate_parameters(self, df_i, theta=None, eta=None, verbose=False):
        theta, eta = self.estimate_scale_parameters(df_i, theta, eta)

        if verbose:
            print(f"Estimated Parameters: θ = {theta}, η = {eta}")

        # 3. Scale Intensities and Fit GEV
        scaled_intensities = self.get_scale_intensities(df_i, theta, eta)

        # 4. Fit GEV Distribution
        shape, loc, scale = self.fit_gev_distribution(scaled_intensities)
        if verbose:
            print(f"GEV Parameters: Shape={shape}, Location={loc}, Scale={scale}")

        return theta, eta, shape, loc, scale

    def evaluate(self, series_kind: str or Literal['partial', 'annual'] = SERIES.ANNUAL):
        """
        Statistical analysis for each duration step.

        acc. to DWA-A 531 chap. 5.1

        Save the parameters of the distribution function as interim results.

        Args:
            series_kind (str): which kind of series should be used to evaluate the extreme values.
        """
        if series_kind == SERIES.PARTIAL:
            return NotImplementedError()

        df = self.annual_intensities()
        df_i = df / df.columns.values

        theta, eta, shape, loc, scale = self.estimate_parameters(df_i)
        return {'theta': float(theta),
                'eta': float(eta),
                'shape': float(shape),
                'loc': float(loc),
                'scale': float(scale)}

    def bootstrap_conf_intervals(self, df, theta, eta, n_iterations=1000):
        """Estimate confidence intervals using the quasi-block-bootstrap method."""
        from tqdm import trange

        boot_samples = []
        for _ in trange(n_iterations):
            boot_data = df.loc[np.random.choice(df.index, df.index.size, replace=True)]
            try:
                theta, eta, shape, loc, scale = self.estimate_parameters(boot_data, theta, eta, verbose=False)
            except ValueError:
                continue
            # shape, loc, scale = fit_gev_distribution(boot_data)
            boot_samples.append((theta, eta, shape, loc, scale))

        # boot_samples = np.array(boot_samples)
        boot_samples = pd.DataFrame(boot_samples, columns=['theta', 'eta', 'shape', 'loc', 'scale'])
        sample_stats = boot_samples.describe(percentiles=[0.025, 0.975]).round(3)
        return sample_stats
