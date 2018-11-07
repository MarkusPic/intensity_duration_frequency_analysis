from math import floor
from pprint import pprint

__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

from .arg_parser import heavy_rain_parser
from .main_class import IntensityDurationFrequencyAnalyse
from .definitions import *
from .in_out import import_series
import pandas as pd
from os import path


def not_none(*args):
    return all(a is not None for a in args)


def tool_executor():
    user = heavy_rain_parser()
    pprint(vars(user))

    # --------------------------------------------------------------------------------------------------------------
    d = user.duration
    h = user.height_of_rainfall
    t = user.return_period
    out = user.output
    name = path.basename('.'.join(user.input.split('.')[:-1]))
    if out is None:
        out = path.dirname(user.input)

    auto_save = False
    if user.plot or user.export_table:
        auto_save = True

    # --------------------------------------------------------------------------------------------------------------
    idf = IntensityDurationFrequencyAnalyse(series_kind=user.series_kind, worksheet=user.worksheet, output_path=out,
                                            extended_durations=user.extended_duration, output_filename=name,
                                            auto_save=auto_save)

    # --------------------------------------------------------------------------------------------------------------
    if user.r_720_1:
        d = 720
        t = 1

    # --------------------------------------------------------------------------------------------------------------
    ts = import_series(user.input)
    # ts.to_frame().to_parquet('.'.join(user.input.split('.')[:-1] + ['parquet']),engine='pyarrow')

    # --------------------------------------------------------------------------------------------------------------
    if not_none(d) and not user.plot and not user.export_table:
        new_freq = floor(d / 4)
        ts = ts.resample('{:0.0f}T'.format(new_freq)).sum()

    # --------------------------------------------------------------------------------------------------------------
    idf.set_series(ts)

    # --------------------------------------------------------------------------------------------------------------
    if not_none(d, t):
        idf.print_depth_of_rainfal(duration=d, return_period=t)
        idf.print_rain_flow_rate(duration=d, return_period=t)
        pass
    elif not_none(d, h):
        tn = idf.get_return_period(h, d)
        print('The return period is {:0.1f} years.'.format(tn))
        idf.print_rain_flow_rate(d, tn)

    elif not_none(h, t):
        pass

    # --------------------------------------------------------------------------------------------------------------
    if user.plot:
        idf.result_plot(show=True)

    # --------------------------------------------------------------------------------------------------------------
    if user.export_table:
        idf.write_table()


if __name__ == '__main__':
    tool_executor()