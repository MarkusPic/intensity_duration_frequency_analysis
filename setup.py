__author__ = "Markus Pichler"
__credits__ = ["Markus Pichler"]
__maintainer__ = "Markus Pichler"
__email__ = "markus.pichler@tugraz.at"
__version__ = "0.1"
__license__ = "MIT"

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='idf_analysis',
    version='0.1',
    packages=find_packages(),
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.json', '*.', '*.png', '*.pdf', '*.csv', '*.tex', '*.xslx'],
    },
    url='https://github.com/MarkusPic/intensity_duration_frequency_analysis',
    license='MIT',
    author='pichler',
    author_email='markus.pichler@tugraz.at',
    description='heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)',
    scripts=['bin/idf_analysis'],
    install_requires=requirements,
)
