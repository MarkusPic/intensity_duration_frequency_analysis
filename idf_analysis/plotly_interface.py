__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

import pandas as pd
from webbrowser import open as open_file
from plotly.graph_objs import Figure as plotlyFigure
from plotly.offline import plot as plotly_plot
from plotly.tools import make_subplots as plotly_subplots
from plotly.graph_objs import Scatter, Bar


########################################################################################################################
def vspace(x0, x1, color='#d3d3d3', y0=0, y1=1, opacity=0.3, border_width=0, yref='paper'):
    """

    :param x0:
    :param x1:
    :param color:
    :param y0:
    :param y1:
    :param opacity:
    :param border_width:
    :param yref: 'paper' or 'y'
    :return:
    """
    return {
        'type': 'rect',
        'yref': yref,
        'x0': x0,
        'y0': y0,
        'x1': x1,
        'y1': y1,
        'fillcolor': color,
        'opacity': opacity,
        'line': {
            'width': border_width,
        }
    }


########################################################################################################################
def html_bold(*input):
    """
    creates a Latex-bold interface around all input items
    :param input: _list_ of all items
    :return: _list_ or _str_ depend on input
    """
    sting = '<b>{}</b>'
    if len(input) == 1:
        return sting.format(input[0])
    return [sting.format(x) for x in input]


########################################################################################################################
class PlotlyFigure(object):
    def __init__(self, fig=None, n_rows=None):
        if fig:
            self.figure = fig
        else:
            self.figure = plotlyFigure()

        self.n_rows = n_rows

        self.set_background_color('#E5E5E5')
        self.set_size_auto()

    def set_title(self, title, fontweight='bold'):
        if isinstance(title, list):
            title = '<br>'.join(title)
        title = title.replace('\n', '<br>')
        if fontweight == 'bold':
            title = html_bold(title)
        self.figure.layout.update(title=title)

    def set_background_color(self, color):
        """

        :type color: str
        """
        self.figure.layout.update(plot_bgcolor=color)

    def set_size(self, h=None, w=1200):
        """

        :type h: int
        """
        if not h:
            # n_rows = len(self.figure['data'])
            h = self.n_rows * 500
        self.figure.layout.update(height=h, width=w)

    def set_size_auto(self):
        self.figure.layout.update(autosize=True)

    def save(self, fname, auto_open=False):
        plotly_plot(self.figure, filename=fname + '.html', auto_open=False)
        if auto_open:
            open_file(fname + '.html')
        print('>> {} <<'.format(fname + '.html'))

    def add_shape(self, shape):
        self.figure.layout.shapes += tuple([shape])

    def add_vspace(self, timetable, color='#ffff00', y0=0, y1=1, opacity=0.3, border_width=0, yref='paper'):
        for start, end in zip(timetable.start, timetable.end):
            self.add_shape(vspace(start, end, color=color, y0=y0, y1=y1, opacity=opacity, border_width=border_width,
                                  yref=yref))

    def add_vspace2(self, timetable, color='#ffff00', y0=0, y1=1, opacity=0.3, border_width=0):
        for start, end in zip(timetable.start, timetable.end):
            self.add_shape(vspace(start, end, color=color, y0=y0, y1=y1, opacity=opacity, border_width=border_width))

    def get_fig(self):
        return self.figure

    def config_axis(self, ax, no, title=None, rangemode=None, side=None, showgrid=None, zeroline=None):
        """

        :param str ax: x | y
        :param int no: von 1 bis ...
        :param str title:
        :param str rangemode: 'nonnegative' | ...
        :param str side: 'left' | 'right'
        :param bool showgrid:
        :param bool zeroline:
        :return:
        """
        ax_name = '{}axis{}'.format(ax, no if no != 1 else '')
        di = dict(title=title,
                  overlaying=ax if no != 1 else None,
                  rangemode=rangemode,
                  side=side,
                  showgrid=showgrid,
                  zeroline=zeroline)

        if ax_name not in self.figure.layout:
            self.figure.layout.update({ax_name: dict()})

        for key, value in di.items():
            if value is not None:
                self.figure.layout[ax_name].update({key: value})

    def set_second_y_axis(self, no=2, title=None, rangemode='nonnegative', side='right', showgrid=False,
                          zeroline=False):
        self.config_axis(ax='y', no=no, title=title, rangemode=rangemode, side=side, showgrid=showgrid,
                         zeroline=zeroline)


# ROW = 1


########################################################################################################################
class PlotlyAxes(list):
    def __init__(self):
        list.__init__(self)
        self.objects = []
        self.rows = set()

    def append(self, object_):
        """

        :type object_: Ax
        """
        self.objects.append(object_)
        self.rows.add(object_.row)

    def get_figure(self, grid_color='#fff', vertical_spacing=0.1, spaces=None):
        """

        :type grid_color: str
        :rtype: PlotlyFigure
        """
        n_rows = len(self.rows)

        if n_rows == 1:
            if len(self.objects) == 1:
                ax = self.objects[0]
                data = ax.traces
            else:
                data = []
                for ax in self.objects:
                    if isinstance(ax.traces, list):
                        data += ax.traces
                    else:
                        data.append(ax.traces)

            if isinstance(data, list):
                fig = plotlyFigure(data=data)
            else:
                fig = plotlyFigure(data=[data])

            fig['layout'].update(title=ax.title)
            fig['layout']['yaxis'].update(title=ax.ylabel)
            fig['layout']['yaxis'].update(gridcolor=grid_color)
            fig['layout']['yaxis'].update(range=ax.ylim)
            fig['layout']['xaxis'].update(gridcolor=grid_color)
            fig['layout']['xaxis'].update(title=ax.xlabel)

        else:
            titles = dict()
            for ax in self.objects:
                titles[ax.row] = ax.title

            titles = list(titles.values())

            fig = plotly_subplots(rows=n_rows, cols=1, shared_xaxes=True, subplot_titles=titles)

            custom_domain = False

            if spaces is not None and len(spaces) == n_rows:
                custom_domain = True

                l1 = 1 - (n_rows - 1) * vertical_spacing
                li = l1 / sum(spaces)
                domains = {}
                y1 = 1
                row = 1
                for s in spaces:
                    y0 = y1 - li * s
                    domains[row] = [round(y0, 2), round(y1, 2)]
                    y1 = y0 - vertical_spacing
                    row += 1

            for ax in self.objects:
                data = ax.traces

                if isinstance(data, list):
                    for trace in data:
                        fig.append_trace(trace, ax.row, 1)
                else:
                    fig.append_trace(ax.traces, ax.row, 1)

                fig['layout']['yaxis{}'.format(ax.row)].update(title=ax.ylabel)
                fig['layout']['yaxis{}'.format(ax.row)].update(gridcolor=grid_color)
                fig['layout']['yaxis{}'.format(ax.row)].update(range=ax.ylim)
                if ax.tickvals:
                    fig['layout']['yaxis{}'.format(ax.row)].update(tickvals=ax.tickvals)
                if ax.ticktext:
                    fig['layout']['yaxis{}'.format(ax.row)].update(ticktext=ax.ticktext)

                if custom_domain:
                    fig['layout']['yaxis{}'.format(ax.row)].update(domain=domains[ax.row])

                if ax.row == 1:
                    fig['layout']['xaxis'].update(gridcolor=grid_color)
                    fig['layout']['xaxis'].update(title=ax.xlabel)
                    fig['layout']['xaxis1'].update(gridcolor=grid_color)
                    fig['layout']['xaxis1'].update(title=ax.xlabel)
        return PlotlyFigure(fig, n_rows=n_rows)

    def plot(self, data, kind='line', **kwargs):
        """

        :param data: to plot
        :type data:  list | pd.DataFrame | pd.Series
        :param kind: plot type (line, ...)
        :param kwargs: from LinePlot or Ax
        """
        if kind == 'line':
            self.append(LinePlot(data, **kwargs))
        elif kind == 'bar':
            self.append(Ax(Bar(x=data.index.values, y=data.values, name=data.name), **kwargs))
        else:
            self.append(Ax(data, **kwargs))


########################################################################################################################
class Ax(object):
    def __init__(self, traces, title=None, ylabel=None, ylim=None, row=None, xlabel=None, tickvals=None, ticktext=None):
        self.traces = traces
        self.title = title
        self.ylabel = ylabel
        self.ylim = ylim
        self.row = row
        self.xlabel = xlabel
        self.tickvals = tickvals
        self.ticktext = ticktext


def series_scatter(series, line_settings=dict(), yaxis=None, connectgaps=None):
    return Scatter(x=series.index, y=series.values, name=series.name, line=line_settings, yaxis=yaxis,
                   connectgaps=connectgaps)


########################################################################################################################
class LinePlot(Ax):
    def __init__(self, data, name=None, color=None, step=False, step_shape='hv', spline=False,
                 title=None, ylabel=None, xlabel=None, ylim=None, row=None, yaxis=None, dash=None, tickvals=None,
                 ticktext=None, width=None, connectgaps=None):
        """

        :type data: list | pd.DataFrame | pd.Series
        :param str dash: dot | dash
        """
        # step_shape is how a step is plotted {'hv', 'vh'}
        #   'hv'= fist horizontal than vertival line (only has an effect when step=True)
        line_settings = dict()

        if step:
            line_settings['shape'] = step_shape

        if spline:
            line_settings['shape'] = 'spline'

        if color:
            line_settings['color'] = color

        if dash:
            line_settings['dash'] = dash

        if width:
            line_settings['width'] = width

        if isinstance(data, list):
            traces = []
            for d in data:
                if isinstance(d, pd.Series):
                    traces.append(series_scatter(d, line_settings=line_settings, yaxis=yaxis, connectgaps=connectgaps))
                    # traces.append(Scatter(x=d.index, y=d, name=d.name, line=line_settings, yaxis=yaxis))

        elif isinstance(data, pd.DataFrame):
            traces = []
            # if not color:
            #     colors = get_color_dict(data.columns.tolist())
            for column in data:
                # s = data[column].copy()
                # if not color:
                #     line_settings['color'] = colors[column]
                traces.append(
                    series_scatter(data[column], line_settings=line_settings, yaxis=yaxis, connectgaps=connectgaps))
                # traces.append(Scatter(x=s.index, y=s.values, name=column, line=line_settings))

        elif isinstance(data, pd.Series):
            traces = [series_scatter(data, line_settings=line_settings, yaxis=yaxis, connectgaps=connectgaps)]

        else:
            raise NotImplementedError('{} is not implemented in plotly interface'.format(type(data)))
            # if not name:
            #     name = data.name
            # traces = Scatter(x=data.index, y=data, name=name, line=line_settings, yaxis=yaxis)

        Ax.__init__(self, traces, title=title, ylabel=ylabel, ylim=ylim, row=row, xlabel=xlabel, tickvals=tickvals,
                    ticktext=ticktext)

