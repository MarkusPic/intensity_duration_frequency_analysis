__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
# def read(file_path, skiprows=2, date_format='%Y-%m-%d %H:%M:00', separator='  '):
#     def dateparse(x):
#         return pd.datetime.strptime(x, date_format)
#
#     df = pd.read_csv(file_path, skiprows=skiprows, header=None, index_col=0, sep=separator,
#                      parse_dates=True, date_parser=dateparse, names=('time', COLUMN_NAME),
#                      engine='python')
#     return df


########################################################################################################################
# printing and plotting functions
def plot_partial_series(events_sort, name, param_u_p, param_w_p):
    x_min = events_sort.log_return_periods[-1:]
    x_max = events_sort.log_return_periods[:1]
    plot = events_sort.plot.scatter(x='log_return_periods', y='max_osum')
    plt.plot([x_min, x_max], [param_u_p + param_w_p * x_min, param_u_p + param_w_p * x_max], 'k--',
             color='Grey')
    figure = plot.get_figure()
    figure.savefig(name + ".jpg")


########################################################################################################################
def plot_annual_series(annually_series, param_u_j, param_w_j, name="annual_plot"):
    x_min = annually_series.x[-1:]
    x_max = annually_series.x[:1]
    plot = annually_series.plot.scatter(x='x', y='max_osum')
    plt.plot([x_min, x_max], [param_u_j + param_w_j * x_min, param_u_j + param_w_j * x_max], 'k--',
             color='Grey')
    figure = plot.get_figure()
    global TIMESTAMP
    figure.savefig(TIMESTAMP+name+".jpg")


########################################################################################################################
########################################################################################################################
if __name__ == '__main__':
    pass
    # input_filename = pre_path + 'min_16412'

    # from sww.libs.timeseries.io.nieda import convert_min_to_pickle
    # convert_min_to_pickle(pre_path, pre_path)
    # exit()

    # input_filename = path.join(pre_path, 'Graz_1Jahr.txt')
    # file_input = read(input_filename)

    # from sww.libs.timeseries.io.nieda import read_light_csv
    # file_input = read_light_csv(input_filename, column_name=COLUMN_NAME)
    # file_input[COLUMN_NAME][file_input[COLUMN_NAME] >= 6.5] = 0