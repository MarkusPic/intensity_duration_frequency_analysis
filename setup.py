__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

from setuptools import setup

setup(
    name='idf_analysis',
    version='0.1',
    packages=['idf_analysis'],
    url='https://github.com/MarkusPic/intensity_duration_frequency_analysis',
    license='MIT',
    author='Markus Pichler',
    author_email='markus.pichler@tugraz.at',
    description='heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)',
    scripts=['bin/idf_analysis'],
    install_requires=['numpy', 'pandas', 'matplotlib', 'tzlocal', 'pytz'],
)
