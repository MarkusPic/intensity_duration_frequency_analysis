from .idf_class import IntensityDurationFrequencyAnalyse


def command_line_tool():
    import matplotlib.pyplot as plt
    plt.style.use('bmh')

    IntensityDurationFrequencyAnalyse.command_line_tool()
