__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import argparse
from heavy_rain_analysis.definitions import DWA, ATV, DWA_adv, PARTIAL, ANNUAL


########################################################################################################################
class Borders():
    def __init__(self, min_=None, max_=None, unit=''):
        self.min_ = min_
        self.max_ = max_
        self.unit = unit

    def __str__(self):
        s = ''
        if self.min_ is not None:
            s += '>= {} {unit}'.format(self.min_, unit=self.unit)
        if self.min_ is not None and self.max_ is not None:
            s += ' and '
        if self.max_ is not None:
            s += '<= {} {unit}'.format(self.max_, unit=self.unit)
        return s

    def __contains__(self, item):
        b = True
        if self.min_ is not None:
            b &= item >= self.min_
        if self.max_ is not None:
            b &= item <= self.max_
        return b

    def __iter__(self):
        return iter([str(self)])


########################################################################################################################
def heavy_rain_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='input file with the rain time series',
                        required=False)
    parser.add_argument('-t', '--return_period',
                        help='return period in years',
                        required=False, type=float, choices=Borders(0.5, 100, 'a'))
    parser.add_argument('-d', '--duration',
                        help='duration in minutes',
                        required=False, type=int, choices=Borders(5, 12 * 60, 'min'))
    parser.add_argument('-r', '--rainfall',
                        help='rainfall in mm or Liter/m^2',
                        required=False, type=float, choices=Borders(0, unit='mm'))
    parser.add_argument('-ws', '--worksheet',
                        help='Worksheet used to calculate.',
                        default=DWA,
                        required=False, type=str, choices=[ATV, DWA, DWA_adv])
    parser.add_argument('-kind', '--series_kind',
                        help='The kind of series used for the calculation. '
                             'Calculation with partial series is more precise',
                        # '({}=annual series; {}=partial series)'.format(ANNUAL, PARTIAL),
                        default=PARTIAL,
                        required=False, type=str, choices=[PARTIAL, ANNUAL])
    return parser.parse_args()
