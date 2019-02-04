__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

from .idf_class import IntensityDurationFrequencyAnalyse

"""
heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)

This program reads the measurement data of the rainfall from of a file in nieda-format
and calculates the distribution of the rainfall as a function of the return period and the duration

for duration steps up to 12 hours and return period in a range of '0.5 <= T_n <= 100'
"""
