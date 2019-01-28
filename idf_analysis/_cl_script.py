__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

from os import path
from math import floor
from .arg_parser import heavy_rain_parser
from .idf_class import IntensityDurationFrequencyAnalyse
from .in_out import import_series, csv_args


def not_none(*args):
    return all(a is not None for a in args)


def tool_executor():
    user = heavy_rain_parser()

    # --------------------------------------------------------------------------------------------------------------
    d = user.duration
    h = user.height_of_rainfall
    t = user.return_period
    out = user.output
    name = path.basename('.'.join(user.input.split('.')[:-1]))
    if out is None:
        out = ''
        # out = path.dirname(user.input)

    auto_save = False
    if user.plot or user.export_table:
        auto_save = True

    # --------------------------------------------------------------------------------------------------------------
    idf = IntensityDurationFrequencyAnalyse(series_kind=user.series_kind, worksheet=user.worksheet, output_path=out,
                                            extended_durations=user.extended_duration, output_filename=name,
                                            auto_save=auto_save, unix=user.unix)

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
        idf.print_depth_of_rainfall(duration=d, return_period=t)
        idf.print_rain_flow_rate(duration=d, return_period=t)
        pass
    elif not_none(d, h):
        t = idf.get_return_period(h, d)
        print('The return period is {:0.1f} years.'.format(t))
        idf.print_depth_of_rainfall(duration=d, return_period=t)
        idf.print_rain_flow_rate(duration=d, return_period=t)

    elif not_none(h, t):
        d = idf.get_duration(h, t)
        print('The duration is {:0.1f} minutes.'.format(d))
        idf.print_depth_of_rainfall(duration=d, return_period=t)
        idf.print_rain_flow_rate(duration=d, return_period=t)

    # --------------------------------------------------------------------------------------------------------------
    if user.plot:
        idf.result_plot(show=True)

    # --------------------------------------------------------------------------------------------------------------
    if user.export_table:
        idf.write_table()
