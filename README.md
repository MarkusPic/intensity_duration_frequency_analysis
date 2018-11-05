# Heavy Rain Analyysis based on KOSTRA
heavy rain as a function of the duration and the return period acc. to DWA-A 531 (2012)
This program reads the measurement data of the rainfall
and calculates the distribution of the rainfall as a function of the return period and the duration
for duration steps up to 12 hours (and more) and return period in a range of '0.5a &lt;= T_n &lt;= 100a'

# Install

See the python packages requirements in the 'requirements.txt'.

To install the packages via github, git must be installed first. (see [https://git-scm.com/](https://git-scm.com/))

Otherwise download the package manually via your browser and replace git+xxx.git with the path to the unzipped folder.


## Fresh install

```
pip3 install git+https://github.com/maxipi/kostra.git
```

## Update package

```
pip3 install git+https://github.com/maxipi/kostra.git --upgrade 
```

# Usage

# Commandline tool 

> ```idf_relation -h```

```
usage: idf_relation [-h] -i INPUT [-out OUTPUT] [-t {>= 0.5 a and <= 100 a}]
                    [-d {>= 5 min and <= 720 min}] [-h_N {>= 0 mm}]
                    [-ws {ATV-A_121,DWA-A_531,DWA-A_531_advektiv}]
                    [-kind {partial,annual}] [--r_720_1] [--plot]
                    [--extended_duration] [--export_table]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file with the rain time series
  -out OUTPUT, --output OUTPUT
                        output path, where to write the results / default:
                        same as input
  -t {>= 0.5 a and <= 100 a}, --return_period {>= 0.5 a and <= 100 a}
                        return period in years (If two of the three variables
                        (rainfall, duration, return period) are given, the
                        third variable is calculated.)
  -d {>= 5 min and <= 720 min}, --duration {>= 5 min and <= 720 min}
                        duration in minutes (If two of the three variables
                        (rainfall, duration, return period) are given, the
                        third variable is calculated.)
  -h_N {>= 0 mm}, --height_of_rainfall {>= 0 mm}
                        rainfall in mm or Liter/m^2 (If two of the three
                        variables (rainfall, duration, return period) are
                        given, the third variable is calculated.)
  -ws {ATV-A_121,DWA-A_531,DWA-A_531_advektiv}, --worksheet {ATV-A_121,DWA-A_531,DWA-A_531_advektiv}
                        Worksheet used to calculate.
  -kind {partial,annual}, --series_kind {partial,annual}
                        The kind of series used for the calculation.
                        Calculation with partial series is more precise
  --r_720_1             design rainfall with a duration of 720 minutes (=12h)
                        and a return period of 1 day
  --plot                get a plot of the idf relationship
  --extended_duration   add [720, 1080, 1440, 2880, 4320, 5760, 7200, 8640]
                        (in minutes) to the duration steps which will be
                        calculated
  --export_table        get a table of the most frequent used values
```

