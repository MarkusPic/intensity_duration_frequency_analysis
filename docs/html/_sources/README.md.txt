© [Institute of Urban Water Management and Landscape Water Engineering](https://www.sww.tugraz.at), [Graz University of Technology](https://www.tugraz.at/home/) and [Markus Pichler](mailto:markus.pichler@tugraz.at)


# Intensity duration frequency analysis (based on KOSTRA)


[![license](https://img.shields.io/github/license/markuspic/intensity_duration_frequency_analysis.svg?style=flat)](https://github.com/MarkusPic/intensity_duration_frequency_analysis/blob/master/LICENSE)
[![docs ](https://img.shields.io/badge/docs-good-brightgreen.svg?style=flat)](https://markuspic.github.io/intensity_duration_frequency_analysis)
[![PyPI](https://img.shields.io/pypi/v/idf-analysis.svg)](https://pypi.python.org/pypi/idf-analysis)

[![PyPI - Downloads](https://img.shields.io/pypi/dd/idf-analysis)](https://pypi.python.org/pypi/idf-analysis)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/idf-analysis)](https://pypi.python.org/pypi/idf-analysis)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/idf-analysis)](https://pypi.python.org/pypi/idf-analysis)

heavy rain as a function of the duration and the return period acc. to [DWA-A 531 (2012)](http://www.dwa.de/dwa/shop/shop.nsf/Produktanzeige?openform&produktid=P-DWAA-8XMUY2)
This program reads the measurement data of the rainfall
and calculates the distribution of the design rainfall as a function of the return period and the duration
for duration steps up to 12 hours (and more) and return period in a range of '0.5a &le; T_n &le; 100a'

The guideline was used in the application [KOSTRA-DWD](https://www.dwd.de/DE/leistungen/kostra_dwd_rasterwerte/kostra_dwd_rasterwerte.html).

----

> Heavy rainfall data are among the most important planning parameters in water management and hydraulic engineering practice. In urban areas, for example, they are required as initial parameters for the design of rainwater drainage systems and in watercourses for the dimensioning of hydraulic structures. The accuracy of the target values of the corresponding calculation methods and models depends crucially on their accuracy. Their overestimation can lead to considerable additional costs in the structural implementation, their underestimation to an unacceptable, excessive residual risk of failure during the operation of water management and hydraulic engineering facilities. Despite the area-wide availability of heavy rainfall data through "Coordinated Heavy Rainfall Regionalisation Analyses" (KOSTRA), there is still a need for local station analyses, e.g. to evaluate the now extended data series, to evaluate recent developments or to classify local peculiarities in comparison to the KOSTRA data. However, this is only possible without restrictions if the methodological approach recommended in the worksheet is followed. In the DWA-A 531 worksheet, the main features of the ATVA 121 worksheet published in 1985 and of the identical DVWK-R 124 booklet of the DVWK Rules for Water Management "Heavy Rain Evaluation after Return Time and Duration" are retained. The aim of the revision is to take account of current developments without, however, calling into question the standardisation of the procedure for statistical heavy rain analyses which was intended at the time.

**[DWA-A 531 (2012)](http://www.dwa.de/dwa/shop/shop.nsf/Produktanzeige?openform&produktid=P-DWAA-8XMUY2) Translated with www.DeepL.com/Translator**

----

> An intensity-duration-frequency curve (IDF curve) is a mathematical function that relates the rainfall intensity with its duration and frequency of occurrence. These curves are commonly used in hydrology for flood forecasting and civil engineering for urban drainage design. However, the IDF curves are also analysed in hydrometeorology because of the interest in the time concentration or time-structure of the rainfall.

**[Wikipedia](https://en.wikipedia.org/wiki/Intensity-duration-frequency_curve)**

----

This package developed [Markus Pichler](mailto:markus.pichler@tugraz.at) during his bachelor thesis and finalised it in the course of his employment at the [Institute of Urban Water Management and Landscape Water Engineering](https://www.sww.tugraz.at).


# Install

The script is written in Python3. (use a version > 3.5)

## Windows

You have to install python (i.e. the original python from the [website](https://www.python.org/downloads/)).

The following commands show the usage for Linux/Unix systems. 

To use these features on Windows you have to add ```python -m``` before each command 
and you have to add the path to your python binary to the environment variables [^path1].

[^path1]: https://geek-university.com/python/add-python-to-the-windows-path/

There is also an option during the installation to add python to the PATH automatically. [^path2]

[^path2]: https://datatofish.com/add-python-to-windows-path/

![python_install](https://datatofish.com/wp-content/uploads/2018/10/0001_add_Python_to_Path.png)

## Linux/Unix

Python is pre-installed on most operating systems (as you probably knew).

## Required python packages

Packages required for this program will be installed with pip during the installation process and can be seen in the 'requirements.txt' file.

## Fresh install

```
pip install idf-analysis
```

Add the following tags to the command for special options:

- ```--user```: To install the package only for the local user account (no admin rights needed)
- ```--upgrade```: To update the package

# Usage

To start the script use following commands in the terminal/Prompt

```idf_analysis```

The documentation of the python-API can be found [here](https://markuspic.github.io/intensity_duration_frequency_analysis/api.html).

# Commandline tool 

> ```idf_analysis -h```

```
usage: __main__.py [-h] -i INPUT
                   [-ws {ATV-A_121,DWA-A_531,DWA-A_531_advektiv}]
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
  -ws {ATV-A_121,DWA-A_531,DWA-A_531_advektiv}, --worksheet {ATV-A_121,DWA-A_531,DWA-A_531_advektiv}
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
