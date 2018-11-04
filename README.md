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
usage: idf_relation [-h] [-i INPUT] [-t {>= 0.5 a and <= 100 a}]
                    [-d {>= 5 min and <= 720 min}] [-r {>= 0 mm}]
                    [-ws {ATV-A 121,DWA-A 531,DWA-A 531 advektiv}]
                    [-kind {partial,annual}]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file with the rain time series
  -t {>= 0.5 a and <= 100 a}, --return_period {>= 0.5 a and <= 100 a}
                        return period in years
  -d {>= 5 min and <= 720 min}, --duration {>= 5 min and <= 720 min}
                        duration in minutes
  -r {>= 0 mm}, --rainfall {>= 0 mm}
                        rainfall in mm or Liter/m^2
  -ws {ATV-A 121,DWA-A 531,DWA-A 531 advektiv}, --worksheet {ATV-A 121,DWA-A 531,DWA-A 531 advektiv}
                        Worksheet used to calculate.
  -kind {partial,annual}, --series_kind {partial,annual}
                        The kind of series used for the calculation.
                        Calculation with partial series is more precise
```