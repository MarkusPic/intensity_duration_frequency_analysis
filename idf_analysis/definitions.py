__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

DWA = 'DWA-A_531'  # (2012)
ATV = 'ATV-A_121'  # (1985)
DWA_adv = DWA + '_advektiv'

# base series for calculation
PARTIAL = 'partial'
ANNUAL = 'annual'

# parameters approach
LOG1 = 'single_logarithmic'
LOG2 = 'double_logarithmic'
HYP = 'hyperbolic'
LIN = 'linear'

SERIES_NAME = 'Precipitation'


class COL:
    """
    column names for the event table
    """
    START = 'start'
    END = 'end'
    DUR = 'duration'
    LP = 'rain_sum'
    MAX_OVERLAPPING_SUM = 'max_overlapping_sum'
