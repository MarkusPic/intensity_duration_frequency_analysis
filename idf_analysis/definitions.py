__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

DWA = 'DWA-A_531'  # (2012)
ATV = 'ATV-A_121'  # (1985)
DWA_adv = DWA + '_advektiv'
OWUNDA = 'OWUNDA'  # ÖKOSTRA ohne Dauerstufenausgleich nach Hammer (1993) "Eine optimierte Starkniederschalgsauswertung"

# base series for calculation
PARTIAL = 'partial'
ANNUAL = 'annual'

# parameters approach
LOG1 = 'single_logarithmic'
LOG2 = 'double_logarithmic'
HYP = 'hyperbolic'
LIN = 'linear'

SERIES_NAME = 'Precipitation'


class PARAM:
    U = 'u'
    W = 'w'
    U_AND_W = U, W


class COL:
    """
    column names for the event table
    """
    START = 'start'
    END = 'end'
    DUR = 'duration'
    LP = 'rain_sum'
    MAX_OVERLAPPING_SUM = 'max_overlapping_sum'
    MAX_PERIOD = 'max_return_period'
    MAX_PERIOD_DURATION = 'max_return_period_duration'


# parameters for the function
A = 'a'
B = 'b'


class PARAM_COL:
    FROM = 'von'
    TO = 'bis'
    U = PARAM.U
    W = PARAM.W

    @staticmethod
    def A(p):
        return '{}_{}'.format(A, p)

    @staticmethod
    def B(p):
        return '{}_{}'.format(B, p)

    @staticmethod
    def VALUES(p):
        return '{}_values'.format(p)
