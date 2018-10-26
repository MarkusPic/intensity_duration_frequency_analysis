__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import numpy as np
import pandas as pd
from .kostra import Kostra
from plotly.graph_objs import Contour
from .plotly_interface import PlotlyAxes, Ax, Scatter


class PlotlyKostra(Kostra):
    def __init__(self, **kwags):
        Kostra.__init__(self, **kwags)

    # ------------------------------------------------------------------------------------------------------------------
    def result_plotly(self, min_duration=1.0, max_duration=8640, xscale="linear"):
        duration_steps = np.arange(min_duration, max_duration, 1)
        colors = ['red', 'green', 'blue', 'yellow', 'magenta']

        # return_periods = [1, 2, 5, 10, 50]
        return_periods = [0.5, 1, 10, 50, 100]
        # offset = 0.0

        table = self.result_table(durations=duration_steps, return_periods=return_periods)

        axes = PlotlyAxes()

        # axes.plot(data=table, row=1, color=colors)

        for i in range(len(return_periods)):
            return_time = return_periods[i]
            color = colors[i]

            axes.plot(data=table[return_time], row=1, color=color)
            points = self.measured_points(return_time, max_duration=max_duration)
            axes.append(Ax(row=1, traces=Scatter(x=points.index, y=points,
                                                 mode='markers',
                                                 showlegend=False,
                                                 hoverinfo='none',
                                                 marker=dict(color=color))))

        # ax.set_xlabel('Dauerstufe $D$ in $[min]$')
        # ax.set_ylabel('Regenhöhe $h_N$ in $[mm]$')
        # ax.legend(title='$T_n$= ... [a]')

        fig = axes.get_figure()
        fig.set_size(w=1800, h=1000)
        fig.set_title('Regenhöhenlinien')
        file = self.output_filename + '_plot'
        fig.save(file)

    # ------------------------------------------------------------------------------------------------------------------
    def result_plotly_contour(self, min_duration=1.0, max_duration=8640, xscale="linear"):
        # duration_steps = np.arange(min_duration, max_duration, 1)
        # duration_steps = np.arange(5, 60, 1)
        duration_steps = np.logspace(0, 3, num=300)

        result_table = pd.DataFrame(index=duration_steps)
        heights = np.logspace(0, 2, num=300)
        # print(heights)
        # exit()
        # heights = range(1, 180, 1)
        # heights = range(0, 60, 1)

        for h in heights:
            result_table[h] = self.get_return_period(h, result_table.index)

        result_table[result_table < 0.1] = 0
        result_table[result_table > 200] = 200

        s = result_table.stack()
        z = s.values.tolist()
        y = s.index.get_level_values(1).tolist()
        x = s.index.get_level_values(0).tolist()

        axes = PlotlyAxes()
        axes.append(Ax(row=1, traces=Contour(z=z,
                                             x=x,
                                             y=y,
                                             contours=dict(
                                                 coloring='heatmap',
                                                 showlabels=True,
                                                 labelfont=dict(
                                                     family='Raleway',
                                                     size=12,
                                                     color='white')
                                             ),
                                             name='Wiederkehrperiode in [Jahre]',
                                             colorscale='Rainbow',
                                             colorbar=dict(title='Wiederkehrperiode in [Jahre]',
                                                           titleside='right'),
                                             hoverinfo='x+y+z+name'),
                       ylabel='Regenhöhe in [mm]',
                       xlabel='Dauerstufe in [min]'))

        fig = axes.get_figure()
        fig.set_size(w=1800, h=1000)
        fig.set_title('Regenhöhenlinien')
        file = self.output_filename + '_countour_plot'
        fig.save(file)