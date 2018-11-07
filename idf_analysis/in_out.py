__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd


def csv_args(unix=False):
    if unix:
        return dict(sep=',', decimal='.')
    else:
        return dict(sep=';', decimal=',')


def import_series(filename, series_label='precipitation', index_label='datetime', unix=False):
    """

    :param filename:
    :param series_label:
    :param index_label:
    :param unix: whether to use a "," as separator and a "." as decimal sign or ";" and ",".
    :type unix: bool
    :return:
    """
    if filename.endswith('csv'):
        ts = pd.read_csv(filename, index_col=0, header=None, squeeze=True, names=[series_label], **csv_args(unix))
        ts.index = pd.to_datetime(ts.index)
        ts.index.name = index_label
        return ts
    elif filename.endswith('parquet'):
        return pd.read_parquet(filename, columns=[series_label])[series_label].rename_axis(index_label, axis='index')
    elif filename.endswith('pkl'):
        return pd.read_pickle(filename).rename(series_label).rename_axis(index_label, axis='index')
    else:
        raise NotImplementedError('Sorry, but only csv files are implemented. Maybe there will be more options soon.')
