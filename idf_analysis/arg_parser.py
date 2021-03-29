__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import argparse

from .definitions import SERIES, METHOD


class Borders(object):
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


def heavy_rain_parser():
    calc_help = ' (If two of the three variables ' \
                '(rainfall (height or flow-rate), duration, return period) are given, ' \
                'the third variable is calculated.)'
    parser = argparse.ArgumentParser()
    parser.description = 'heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)\n' \
                         'All files will be saved in the same directory of the input file ' \
                         'but in a subfolder called like the inputfile + "_idf_data". ' \
                         'Inside this folder a file called "idf_parameter.yaml"-file will be saved and ' \
                         'contains interim-calculation-results and will be automatically reloaded on the next call.'

    parser.add_argument('-i', '--input',
                        help='input file with the rain time-series (csv or parquet)',
                        required=True)
    # --------------------------------------------
    parser.add_argument('-ws', '--worksheet',
                        help='From which worksheet the recommendations for calculating the parameters should be taken.',
                        default=METHOD.KOSTRA,
                        required=False, type=str, choices=METHOD.OPTIONS)
    parser.add_argument('-kind', '--series_kind',
                        help='The kind of series used for the calculation. '
                             '(Calculation with partial series is more precise and recommended.)',
                        default=SERIES.PARTIAL,
                        required=False, type=str, choices=SERIES.OPTIONS)
    # --------------------------------------------
    parser.add_argument('-t', '--return_period',
                        help='return period in years' + calc_help,
                        required=False, type=float, choices=Borders(0.5, 100, 'a'))
    parser.add_argument('-d', '--duration',
                        help='duration in minutes' + calc_help,
                        required=False, type=float, choices=Borders(5, 6 * 24 * 60, 'min'))
    parser.add_argument('-r', '--flow_rate_of_rainfall',
                        help='rainfall in Liter/(s * ha)' + calc_help,
                        required=False, type=float, choices=Borders(0, unit='L/(s*ha)'))
    parser.add_argument('-h_N', '--height_of_rainfall',
                        help='rainfall in mm or Liter/m^2' + calc_help,
                        required=False, type=float, choices=Borders(0, unit='mm'))
    # --------------------------------------------
    parser.add_argument('--r_720_1',
                        help='design rainfall with a duration of 720 minutes (=12 h) and a return period of 1 year',
                        required=False, action='store_true')
    # --------------------------------------------
    parser.add_argument('--plot',
                        help='get a plot of the idf relationship',
                        required=False, action='store_true')
    # --------------------------------------------
    parser.add_argument('--export_table',
                        help='get a table of the most frequent used values',
                        required=False, action='store_true')
    # --------------------------------------------
    return parser.parse_args()
