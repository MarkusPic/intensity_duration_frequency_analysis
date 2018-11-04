from .arg_parser import heavy_rain_parser
from .main_class import IntensityDurationFrequencyAnalyse
from .definitions import *


def tool_executor():
    user = heavy_rain_parser()
    print(user)

    idf = IntensityDurationFrequencyAnalyse(series_kind=PARTIAL, worksheet=DWA, output_path=None,
                                            extended_durations=False, output_filename=None)