© [Institute of Urban Water Management and Landscape Water Engineering](https://www.sww.tugraz.at), [Graz University of Technology](https://www.tugraz.at/home/) and [Markus Pichler](mailto:markus.pichler@tugraz.at)


# Intensity duration frequency analysis (based on KOSTRA)


[![license](https://img.shields.io/github/license/markuspic/intensity_duration_frequency_analysis.svg?style=flat)](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/idf-analysis)](https://pypi.python.org/pypi/idf-analysis)
[![DOI](https://zenodo.org/badge/142560436.svg)](https://zenodo.org/doi/10.5281/zenodo.10559991)
[![Buymeacoffee](https://badgen.net/badge/icon/buymeacoffee?icon=buymeacoffee&label=donate)](https://www.buymeacoffee.com/MarkusP)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.07607/status.svg)](https://doi.org/10.21105/joss.07607)
[![contributing](https://img.shields.io/badge/Contributing-red?style=flat)](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/CONTRIBUTING.md)
[![docs](https://img.shields.io/badge/Documentation-purple?style=flat&logo=readthedocs)](https://markuspic.github.io/intensity_duration_frequency_analysis/)
[![code-of-conduct](https://img.shields.io/badge/Code_of_Conduct-grey?style=flat)](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/CODE_OF_CONDUCT.md)
[![Tests](https://github.com/MarkusPic/intensity_duration_frequency_analysis/actions/workflows/tests.yaml/badge.svg)](https://github.com/MarkusPic/intensity_duration_frequency_analysis/actions/workflows/tests.yaml)
[![Coverage](https://codecov.io/gh/MarkusPic/intensity_duration_frequency_analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/MarkusPic/intensity_duration_frequency_analysis)
[![documentation build](https://github.com/MarkusPic/intensity_duration_frequency_analysis/actions/workflows/sphinx_docs_to_gh_pages.yaml/badge.svg)](https://github.com/MarkusPic/intensity_duration_frequency_analysis/actions/workflows/sphinx_docs_to_gh_pages.yaml)

[![PyPI - Downloads](https://img.shields.io/pypi/dd/idf-analysis)](https://pypi.python.org/pypi/idf-analysis)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/idf-analysis)](https://pypi.python.org/pypi/idf-analysis)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/idf-analysis)](https://pypi.python.org/pypi/idf-analysis)

Heavy rainfall intensity as a function of duration and return period is defined according to [DWA-A 531 (2012)](https://shop.dwa.de/DWA-A-531-Starkregen-in-Abhaengigkeit-von-Wiederkehrzeit-und-Dauer-September-2012-Stand-korrigierte-Fassung-Mai-2017/A-531-Hauptprodukt-12-main).
This program reads rainfall measurement data 
and calculates the distribution of design rainfall as a function of both return period and duration, for durations 
up to 12 hours (and beyond) and return periods in the range 0.5 a ≤ Tₙ ≤ 100 a.

The guideline was also applied in the [KOSTRA-DWD](https://www.dwd.de/DE/leistungen/kostra_dwd_rasterwerte/kostra_dwd_rasterwerte.html) application.

Since version 0.4, the updated guidelines of [DWA-A 531 (2025)](https://shop.dwa.de/DWA-A-531-Starkregen-in-Abhaengigkeit-von-Wiederkehrzeit-und-Dauer-Juni-2025/A-531-Hauptprodukt-25-main) have also been implemented.

----

> Heavy rainfall data are among the most important planning parameters in water management and hydraulic engineering practice. In urban areas, for example, they are required as initial parameters for the design of rainwater drainage systems and in watercourses for the dimensioning of hydraulic structures. The accuracy of the target values of the corresponding calculation methods and models depends crucially on their accuracy. Their overestimation can lead to considerable additional costs in the structural implementation, their underestimation to an unacceptable, excessive residual risk of failure during the operation of water management and hydraulic engineering facilities. Despite the area-wide availability of heavy rainfall data through "Coordinated Heavy Rainfall Regionalisation Analyses" (KOSTRA), there is still a need for local station analyses, e.g. to evaluate the now extended data series, to evaluate recent developments or to classify local peculiarities in comparison to the KOSTRA data. However, this is only possible without restrictions if the methodological approach recommended in the worksheet is followed. 

**[DWA-A 531 (2012)](http://www.dwa.de/dwa/shop/shop.nsf/Produktanzeige?openform&produktid=P-DWAA-8XMUY2) Translated with www.DeepL.com/Translator**

----

> An intensity-duration-frequency (IDF) curve is a mathematical function that relates the rainfall intensity with its duration and frequency of occurrence. These curves are commonly used in hydrology for flood forecasting and civil engineering for urban drainage design. However, the IDF curves are also analysed in hydrometeorology because of the interest in the time concentration or time-structure of the rainfall.

**[Wikipedia](https://en.wikipedia.org/wiki/Intensity-duration-frequency_curve)**

----

This package was developed by [Markus Pichler](mailto:markus.pichler@tugraz.at) during his bachelor thesis and finalised it in the course of his employment at the [Institute of Urban Water Management and Landscape Water Engineering](https://www.sww.tugraz.at).

## Documentation

Read the docs [here 📖](https://markuspic.github.io/intensity_duration_frequency_analysis).

## Please cite as

Pichler, M. (2025). idf-analysis: Heavy rainfall intensity as a function of duration and return period. Journal of Open Source Software, 10(106), 7607. https://doi.org/10.21105/joss.07607

## Installation

This package is written in Python3. (use a version > 3.5)

```
pip install idf-analysis
```

Add the following tags to the command for special options:

- ```--user```: To install the package only for the local user account (no admin rights needed)
- ```--upgrade```: To update the package

### Windows

You have to install python first (i.e. the original python from the [website](https://www.python.org/downloads/)).

To use the command-line-tool, it is advisable to add the path to your Python binary to the environment variables [^path1].
There is also an option during the installation to add python to the PATH automatically. [^path2]

[^path1]: https://geek-university.com/python/add-python-to-the-windows-path/
[^path2]: https://datatofish.com/add-python-to-windows-path/


### Linux/Unix

Python is pre-installed on most operating systems (as you probably knew).

### Dependencies

Packages required for this program will be installed with pip during the installation process and can be seen 
in the [`requirements.txt`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/requirements.txt) file.


```mermaid
flowchart TD
    contourpy["contourpy<br>1.3.1"]
    cycler["cycler<br>0.12.1"]
    fonttools["fonttools<br>4.54.1"]
    kiwisolver["kiwisolver<br>1.4.7"]
    matplotlib["matplotlib<br>3.9.2"]
    mpmath["mpmath<br>1.3.0"]
    numpy["numpy<br>2.1.3"]
    packaging["packaging<br>24.2"]
    pandas["pandas<br>2.2.3"]
    pillow["pillow<br>11.0.0"]
    pyparsing["pyparsing<br>3.2.0"]
    python-dateutil["python-dateutil<br>2.9.0.post0"]
    pytz["pytz<br>2024.2"]
    pyyaml["PyYAML<br>6.0.2"]
    scipy["scipy<br>1.14.1"]
    six["six<br>1.16.0"]
    sympy["sympy<br>1.13.3"]
    tqdm["tqdm<br>4.67.0"]
    tzdata["tzdata<br>2024.2"]
    contourpy -- "&ge;1.23" --> numpy
    matplotlib -- "&ge;0.10" --> cycler
    matplotlib -- "&ge;1.0.1" --> contourpy
    matplotlib -- "&ge;1.23" --> numpy
    matplotlib -- "&ge;1.3.1" --> kiwisolver
    matplotlib -- "&ge;2.3.1" --> pyparsing
    matplotlib -- "&ge;2.7" --> python-dateutil
    matplotlib -- "&ge;20.0" --> packaging
    matplotlib -- "&ge;4.22.0" --> fonttools
    matplotlib -- "&ge;8" --> pillow
    pandas -- "&ge;1.26.0" --> numpy
    pandas -- "&ge;2.8.2" --> python-dateutil
    pandas -- "&ge;2020.1" --> pytz
    pandas -- "&ge;2022.7" --> tzdata
    python-dateutil -- "&ge;1.5" --> six
    scipy -- "&ge;1.23.5,&lt;2.3" --> numpy
    sympy -- "&ge;1.1.0,&lt;1.4" --> mpmath
```

## Usage

The documentation of the Python-package can be found [here](https://markuspic.github.io/intensity_duration_frequency_analysis/api.html).

One basic usage could be:

```python
import pandas as pd
from idf_analysis import IntensityDurationFrequencyAnalyse
from idf_analysis.definitions import *

# initialize of the analysis class
idf = IntensityDurationFrequencyAnalyse(series_kind=SERIES.PARTIAL, worksheet=METHOD.KOSTRA, extended_durations=True)

series = pd.Series(index=pd.DatetimeIndex(...), data=...)  # this is just a placeholder

# setting the series for the analysis
idf.set_series(series)
# auto-save the calculated parameter so save time for a later use, as the parameters doesn't have to be calculated again.
idf.auto_save_parameters('idf_parameters.yaml')
```

If you only want to analyse an already existing IDF-table

```python
import pandas as pd
from idf_analysis import IntensityDurationFrequencyAnalyse

idf_table = pd.DataFrame(...)  # this is just a placeholder
# index: Duration Steps in minutes as int or float
# columns: Return Periods in years as int or float
# values: rainfall height in mm
idf = IntensityDurationFrequencyAnalyse.from_idf_table(idf_table)
```

## Commandline tool

The following commands show the usage for Linux/Unix systems.
To use these features on Windows you have to add ```python -m``` before each command.

To start the script use following commands in the terminal/Prompt

```idf_analysis```

> ```idf_analysis -h```

```
usage: __main__.py [-h] -i INPUT
                   [-ws {ATV-A_121,KOSTRA,convective_vs_advective}]
                   [-kind {partial,annual}] [-t {>= 0.5 a and <= 100 a}]
                   [-d {>= 5 min and <= 8640 min}] [-r {>= 0 L/s*ha}]
                   [-h_N {>= 0 mm}] [--r_720_1] [--plot] [--export_table]

heavy rain as a function of the duration and the return period acc. to DWA-A
531 (2012) All files will be saved in the same directory of the input file but
in a subfolder called like the inputfile + "_idf_data". Inside this folder a
file called "idf_parameter.yaml"-file will be saved and contains interim-
calculation-results and will be automatically reloaded on the next call.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file with the rain time-series (csv or parquet)
  -ws {ATV-A_121,KOSTRA,convective_vs_advective}, --worksheet {ATV-A_121,KOSTRA,convective_vs_advective}
                        From which worksheet the recommendations for
                        calculating the parameters should be taken.
  -kind {partial,annual}, --series_kind {partial,annual}
                        The kind of series used for the calculation.
                        (Calculation with partial series is more precise and
                        recommended.)
  -t {>= 0.5 a and <= 100 a}, --return_period {>= 0.5 a and <= 100 a}
                        return period in years (If two of the three variables
                        (rainfall (height or flow-rate), duration, return
                        period) are given, the third variable is calculated.)
  -d {>= 5 min and <= 8640 min}, --duration {>= 5 min and <= 8640 min}
                        duration in minutes (If two of the three variables
                        (rainfall (height or flow-rate), duration, return
                        period) are given, the third variable is calculated.)
  -r {>= 0 L/(s*ha)}, --flow_rate_of_rainfall {>= 0 L/(s*ha)}
                        rainfall in Liter/(s * ha) (If two of the three
                        variables (rainfall (height or flow-rate), duration,
                        return period) are given, the third variable is
                        calculated.)
  -h_N {>= 0 mm}, --height_of_rainfall {>= 0 mm}
                        rainfall in mm or Liter/m^2 (If two of the three
                        variables (rainfall (height or flow-rate), duration,
                        return period) are given, the third variable is
                        calculated.)
  --r_720_1             design rainfall with a duration of 720 minutes (=12 h)
                        and a return period of 1 year
  --plot                get a plot of the idf relationship
  --export_table        get a table of the most frequent used values
```

## Example

In these examples you can see the usage in a reproducible way. This examples uses open data provided from the Austrian government. You can also find a link to the script used to download this data.

[Example Jupyter notebook for the commandline](examples/example_commandline.ipynb) (or in the [docs](https://markuspic.github.io/intensity_duration_frequency_analysis/examples/example_python_api.html))

[Example Jupyter notebook for the python api](examples/example_python_api.ipynb) (or in the [docs](https://markuspic.github.io/intensity_duration_frequency_analysis/examples/example_python_api.html))

[Example Jupyter notebook for the python api of the DWA 2025 methodology](examples/example_python_api_2025.ipynb) (or in the [docs](https://markuspic.github.io/intensity_duration_frequency_analysis/examples/example_python_api_2025.html))

[Example python skript](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/examples/example_python_api.py)


### Example Files

[Interim Results of the idf analysis](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/examples/ehyd_112086_idf_data/idf_parameters.yaml)

### Example Plot using the DWA 2012 methodology

![IDF-Curves-Plot](https://raw.githubusercontent.com/MarkusPic/intensity_duration_frequency_analysis/refs/heads/main/examples/ehyd_112086_idf_data/idf_curves_plot.png)

### Example Plot using the DWA 2025 methodology

![IDF-Curves-Plot](https://raw.githubusercontent.com/MarkusPic/intensity_duration_frequency_analysis/refs/heads/main/examples/ehyd_112086_idf_data_new/idf_curves_plot_color_logx.png)


### Example IDF table

[IDF-Table](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/examples/ehyd_112086_idf_data/idf_table_UNIX.csv)


| return period in a<br>duration in min |     1 |      2 |      3 |      5 |     10 |     20 |     25 |     30 |     50 |     75 |    100 |
|--------------------------------------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
|                                     5 |  9.39 |  10.97 |  11.89 |  13.04 |  14.61 |  16.19 |  16.69 |  17.11 |  18.26 |  19.18 |  19.83 |
|                                    10 | 15.15 |  17.62 |  19.06 |  20.88 |  23.35 |  25.82 |  26.62 |  27.27 |  29.09 |  30.54 |  31.56 |
|                                    15 | 19.03 |  22.25 |  24.13 |  26.51 |  29.72 |  32.94 |  33.98 |  34.83 |  37.20 |  39.08 |  40.42 |
|                                    20 | 21.83 |  25.71 |  27.99 |  30.85 |  34.73 |  38.62 |  39.87 |  40.89 |  43.75 |  46.02 |  47.63 |
|                                    30 | 25.60 |  30.66 |  33.62 |  37.35 |  42.41 |  47.47 |  49.10 |  50.43 |  54.16 |  57.12 |  59.22 |
|                                    45 | 28.92 |  35.51 |  39.37 |  44.23 |  50.83 |  57.42 |  59.54 |  61.28 |  66.14 |  69.99 |  72.73 |
|                                    60 | 30.93 |  38.89 |  43.54 |  49.40 |  57.36 |  65.31 |  67.88 |  69.97 |  75.83 |  80.49 |  83.79 |
|                                    90 | 33.37 |  41.74 |  46.64 |  52.80 |  61.17 |  69.54 |  72.23 |  74.43 |  80.60 |  85.49 |  88.96 |
|                                   180 | 38.01 |  47.13 |  52.46 |  59.18 |  68.30 |  77.42 |  80.36 |  82.76 |  89.48 |  94.81 |  98.60 |
|                                   270 | 41.01 |  50.60 |  56.21 |  63.28 |  72.87 |  82.46 |  85.55 |  88.07 |  95.14 | 100.75 | 104.73 |
|                                   360 | 43.29 |  53.23 |  59.04 |  66.37 |  76.31 |  86.25 |  89.45 |  92.06 |  99.39 | 105.20 | 109.33 |
|                                   450 | 45.14 |  55.36 |  61.33 |  68.87 |  79.08 |  89.30 |  92.59 |  95.28 | 102.81 | 108.79 | 113.03 |
|                                   600 | 47.64 |  58.23 |  64.43 |  72.23 |  82.82 |  93.41 |  96.82 |  99.61 | 107.42 | 113.61 | 118.01 |
|                                   720 | 49.29 |  60.13 |  66.47 |  74.45 |  85.29 |  96.12 |  99.61 | 102.46 | 110.44 | 116.78 | 121.28 |
|                                  1080 | 54.41 |  64.97 |  71.15 |  78.94 |  89.50 | 100.06 | 103.46 | 106.24 | 114.02 | 120.20 | 124.58 |
|                                  1440 | 58.02 |  67.72 |  73.39 |  80.54 |  90.24 |  99.93 | 103.05 | 105.61 | 112.75 | 118.42 | 122.45 |
|                                  2880 | 66.70 |  77.41 |  83.68 |  91.57 | 102.29 | 113.00 | 116.45 | 119.26 | 127.16 | 133.42 | 137.87 |
|                                  4320 | 71.93 |  85.72 |  93.78 | 103.95 | 117.73 | 131.52 | 135.96 | 139.58 | 149.75 | 157.81 | 163.53 |
|                                  5760 | 78.95 |  95.65 | 105.42 | 117.72 | 134.43 | 151.13 | 156.50 | 160.89 | 173.20 | 182.97 | 189.90 |
|                                  7200 | 83.53 | 101.38 | 111.82 | 124.98 | 142.83 | 160.68 | 166.43 | 171.12 | 184.28 | 194.72 | 202.13 |
|                                  8640 | 85.38 | 104.95 | 116.40 | 130.82 | 150.38 | 169.95 | 176.25 | 181.40 | 195.82 | 207.27 | 215.39 |

## Contributing and Support

If you're interested in contributing or need help, check out the [CONTRIBUTING.md](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/main/CONTRIBUTING.md) file for guidelines.


## Testing

The code is written in pure python so no compiling is needed.

Most of the code is (or at least will be) automatically tested using the test suite in the [./tests](./tests) folder. To run the tests, use:

```sh
pytest tests
```

## Background

Pseudocode for the parameter calculation.

```
For every duration step
    calculating maximum rainfall intensity for each event for the corresponding duration step
    
    if using annual event series:  # only recommeded for measurements longer than 20 year
        converting max event rainfall intensities per year to a series
        calculating parameters u and w using the gumbel distribution
        
    elif using partial event series:
        converting the n (approximatly 2.72 x measurement duration in years) biggest event rainfall intensities to a series
        calculating parameters u and w using the exponential distribution
    
Splitting IDF curve formulation in to several duration ranges
For each duration range:
    For each parameter (u and w):
        balancing the parameter over all duation steps (in the range) using a given formulation (creating parameters a and b)
        # one-folded-logaritmic | two-folded-logarithmic | hyperbolic

u(D) = f(a_u, b_u, D)
w(D) = f(a_w, b_w, D)

h(D,Tn) = u(D) + w(D) * ln(Tn)
```
