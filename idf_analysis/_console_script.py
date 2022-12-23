from .idf_class import IntensityDurationFrequencyAnalyse


def command_line_tool():
    """
    Execute the command line tool
    """
    import matplotlib.pyplot as plt
    plt.style.use('bmh')

    IntensityDurationFrequencyAnalyse.command_line_tool()
