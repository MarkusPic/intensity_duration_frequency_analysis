class METHOD:
    # DWA_2012 = 'DWA_A_531_2012'
    KOSTRA = 'KOSTRA'
    CONVECTIVE_ADVECTIVE = 'convective_vs_advective'

    ATV = 'ATV-A_121'  # (1985)

    DWA_adv = CONVECTIVE_ADVECTIVE

    # ÖKOSTRA ohne Dauerstufenausgleich nach Hammer (1993) "Eine optimierte Starkniederschlagsauswertung"
    OWUNDA = 'OWUNDA'

    DWA_2025 = 'DWA_A_531_2025'

    OPTIONS = [KOSTRA, CONVECTIVE_ADVECTIVE, ATV, DWA_2025]


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

    INTENSITIES = 'intensities'

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
        return f'{p}_values'


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
