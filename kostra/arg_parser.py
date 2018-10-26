__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import argparse
from kostra.definitions import DWA, ATV, DWA_adv, PARTIAL, ANNUAL


########################################################################################################################
class Borders():
    def __init__(self, min_=None, max_=None, unit=''):
        self.min_ = min_
        self.max_ = max_
        self.unit = unit
    
    def __str__(self):
        s = ''
        if self.min_:
            s += '>= {} {unit}'.format(self.min_, unit=self.unit)
        if self.min_ and self.max_:
            s += ' and '
        if self.max_:
            s += '<= {} {unit}'.format(self.max_, unit=self.unit)
        return s
    
    def __contains__(self, item):
        b = True
        if self.min_:
            b &= item >= self.min_
        if self.max_:
            b &= item <= self.max_
        return b
    
    def __iter__(self):
        return iter([str(self)])


# ------------------------------------------------------------------------------------------------------------------
heavy_rain_parser = argparse.ArgumentParser()
heavy_rain_parser.add_argument('-i', '--input',
                               help='rain input file in nieda format',
                               required=False)
heavy_rain_parser.add_argument('-t', '--returnperiod',
                               help='return period in years',
                               required=False, type=float, choices=Borders(0.5, 100, 'a'))
heavy_rain_parser.add_argument('-d', '--duration',
                               help='duration in minutes',
                               required=False, type=int, choices=Borders(5, 12 * 60, 'min'))
heavy_rain_parser.add_argument('-r', '--rainfall',
                               help='rainfall in mm or Liters/m^2',
                               required=False, type=float, choices=Borders(0, unit='mm'))
heavy_rain_parser.add_argument('-ws', '--worksheet',
                               help='Worksheet used to calculate.',
                               default=DWA,
                               required=False, type=str, choices=[ATV, DWA, DWA_adv])
heavy_rain_parser.add_argument('-kind', '--series_kind',
                               help='The kind of series used for the calculation. '
                                    'Calculation with partial series is more precise',
                               default=PARTIAL,
                               required=False, type=str, choices=[PARTIAL, ANNUAL])
