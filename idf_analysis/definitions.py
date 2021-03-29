__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"


class METHOD:
    # DWA = 'DWA-A_531'  # (2012)
    KOSTRA = 'KOSTRA'
    CONVECTIVE_ADVECTIVE = 'convective_vs_advective'

    ATV = 'ATV-A_121'  # (1985)

    DWA_adv = CONVECTIVE_ADVECTIVE

    # ÖKOSTRA ohne Dauerstufenausgleich nach Hammer (1993) "Eine optimierte Starkniederschalgsauswertung"
    OWUNDA = 'OWUNDA'

    OPTIONS = [KOSTRA, CONVECTIVE_ADVECTIVE, ATV]


class SERIES:
    # base series for calculation
    PARTIAL = 'partial'
    ANNUAL = 'annual'

    OPTIONS = [PARTIAL, ANNUAL]


class APPROACH:
    # parameters approach
    LOG1 = 'single_logarithmic'
    LOG2 = 'double_logarithmic'
    HYP = 'hyperbolic'
    LIN = 'linear'


class PARAM:
    SERIES = 'series_kind'  # Art der Serie
    # METHOD = 'Methodik'

    DUR = 'durations'  # Dauerstufen

    # parameters of the distribution of the series
    # parameters_series
    U = 'u'
    W = 'w'
    U_AND_W = U, W

    PARAMS_SERIES = 'parameters_series'
    PARAMS_FINAL = 'parameters_final'

    A = 'a'
    B = 'b'
    FUNCTION = 'function'

    # parameters of the distribution function of the regression
    # parameters_final

    # parameters for the function
    # @staticmethod
    # def A(p):
    #     return '{}_{}'.format('a', p)
    #
    # @staticmethod
    # def B(p):
    #     return '{}_{}'.format('b', p)
    #
    # @staticmethod
    # def FUNCTION(p):
    #     return 'function_{}'.format(p)

    @staticmethod
    def VALUES(p):
        return '{}_values'.format(p)


class COL:
    # column names for the event table
    START = 'start'
    END = 'end'
    DUR = 'duration'
    LP = 'rain_sum'
    MAX_OVERLAPPING_SUM = 'max_overlapping_sum'
    MAX_PERIOD = 'max_return_period'
    MAX_PERIOD_DURATION = 'max_return_period_duration'
    LAST = 'last_event'
