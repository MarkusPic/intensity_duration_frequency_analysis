from pathlib import Path

import pandas as pd
import yaml  # pip install PyYAML


def import_series(filename, series_label='precipitation', index_label='datetime', csv_reader_args=None):
    """
    Import series data from csv, parquet of pickle file.

    Args:
        filename (str or pathlib.Path):
        series_label (str): name of series in file.
        index_label (str): prefered index name. (just used for plotting)
        csv_reader_args (dict): for example: sep="," or "." and decimal=";" or ","

    Returns:
        pandas.Series: precipitation series
    """
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.suffix == '.csv':
        if csv_reader_args is None:
            csv_reader_args = dict(sep=';', decimal=',')
        try:
            ts = pd.read_csv(filename, index_col=0, header=0, **csv_reader_args).squeeze('columns')
            ts.index = pd.to_datetime(ts.index)
            ts.index.name = index_label
            ts.name = series_label
        except Exception as e:
            print(e)
            raise UserWarning('ERROR | '
                              'Something is wrong with your csv format. The file should only include two columns. '
                              'First column is the date and time index (prefered format is "YYYY-MM-DD HH:MM:SS") '
                              'and second column the precipitation values in mm. '
                              'As a separator use "{sep}" and as decimal sign use "{decimal}".'.format(
                **csv_reader_args))

        return ts
    elif filename.suffix == 'parquet':
        # You need to install `pyarrow` or `fastparquet` to read and write parquet files.
        return pd.read_parquet(filename, columns=[series_label])[series_label].rename_axis(index=index_label)
    elif filename.suffix == 'pkl':
        return pd.read_pickle(filename).rename(series_label).rename_axis(index=index_label)
    else:
        raise NotImplementedError('Sorry, but only csv, parquet and pickle files are implemented. '
                                  'Maybe there will be more options soon.')


_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def write_yaml(data, filename):
    """
    Write dict to yaml file.

    Args:
        data (dict): dict to write
        filename (str or pathlib.Path): path to yaml file.
    """
    yaml.dump(data, open(filename, 'w'), default_flow_style=None, width=200)


def read_yaml(filename):
    """
    Read yaml file.

    Args:
        filename (str or pathlib.Path): path to yaml file.

    Returns:
        dict: dict to read
    """
    return yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
