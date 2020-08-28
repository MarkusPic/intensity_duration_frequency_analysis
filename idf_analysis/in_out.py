__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
import yaml
from collections import OrderedDict


def import_series(filename, series_label='precipitation', index_label='datetime', csv_reader_args=None):
    """

    Args:
        filename:
        series_label:
        index_label:
        csv_reader_args: for example: sep="," or "." and decimal=";" or ","

    Returns:
        pandas.Series: precipitation series
    """
    if filename.endswith('csv'):
        if csv_reader_args is None:
            csv_reader_args = dict(sep=';', decimal=',')
        try:
            ts = pd.read_csv(filename, index_col=0, header=0, squeeze=True, **csv_reader_args)
            ts.index = pd.to_datetime(ts.index)
            ts.index.name = index_label
            ts.name = series_label
        except:
            raise UserWarning('ERROR | '
                              'Something is wrong with your csv format. The file should only include two columns. '
                              'First column is the date and time index (prefered format is "YYYY-MM-DD HH:MM:SS") '
                              'and second column the precipitation values in mm. '
                              'As a separator use "{sep}" and as decimal sign use "{decimal}".'.format(**csv_reader_args))

        return ts
    elif filename.endswith('parquet'):
        return pd.read_parquet(filename, columns=[series_label])[series_label].rename_axis(index_label, axis='index')
    elif filename.endswith('pkl'):
        return pd.read_pickle(filename).rename(series_label).rename_axis(index_label, axis='index')
    else:
        raise NotImplementedError('Sorry, but only csv, parquet and pickle files are implemented. '
                                  'Maybe there will be more options soon.')

# ------------------------------------------------------------------------------
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def _dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def _dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_representer(OrderedDict, _dict_representer)
yaml.add_constructor(_mapping_tag, _dict_constructor)


def write_yaml(data, fn):
    yaml.dump(data, open(fn, 'w'), default_flow_style=None, width=200)


def read_yaml(filename):
    return yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
