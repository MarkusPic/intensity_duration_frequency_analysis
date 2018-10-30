from setuptools import setup, find_packages

setup(
    name='lib',
    version='0.1',
    packages=find_packages(),
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.json', '*.', '*.png', '*.pdf', '*.csv', '*.tex','*.xslx'],
    },
    url='',
    license='',
    author='pichler',
    author_email='markus.pichler@tugraz.at',
    description='heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)',
    scripts=['bin/kostra'],
)
