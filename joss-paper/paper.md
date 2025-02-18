---
title: 'idf-analysis: Heavy rainfall intensity as a function of duration and return period'
tags:
  - Python
  - engineering
  - hydrology
  - precipitation
  - rainfall
authors:
  - name: Markus Pichler
    orcid: 0000-0002-8101-2163
    corresponding: true
    affiliation: 1
affiliations:
 - name: Institute of Urban Water Management and Landscape Water Engineering, Graz University of Technology, Graz, Austria
   index: 1
   ror: 00d7xrm67
date: 10 February 2025
bibliography: paper.bib
---

# Summary

Heavy rainfall data are among the most important planning parameters in water management and hydraulic engineering practice. They are required as initial parameters for the dimensioning of rainwater drainage systems in urban areas and for the dimensioning of hydraulic structures on watercourses. 

The accuracy of the calculation methods and models significantly impacts both costs and risks. Overestimation can lead to considerable additional construction costs, while underestimation can result in an excessive residual risk of failure during the operation of water management and hydraulic engineering facilities.

An intensity-duration-frequency (IDF) curve is a mathematical function (see \autoref{eq:idf}) that relates the rainfall intensity with its duration and frequency of occurrence. An example can be seen in \autoref{fig:idf_curves}. These curves are commonly used in hydrology for flood forecasting and civil engineering for urban drainage design. However, the IDF curves are also analysed in hydrometeorology because of the interest in the time concentration or time-structure of the rainfall.

The package follows the guidance recommendations in @dwa:2012.

![Example IDF curves for the city of Graz. For each return period T$_N$ (in the legend) a corresponding rainfall depth in mm (y axis) depending on the duration (x axis) is shown. \label{fig:idf_curves}](idf_curves_plot.pdf)


# Statement of need

`idf-analysis` is a Python package designed to analyze local changes in rainfall behavior and assess variations in return periods of specific rainfall events. It processes time-series rainfall measurements to derive parameters for intensity-duration-frequency (IDF) curves. This enables estimations of return periods for events of different durations. Additionally, the package allows for comparisons of rainfall behavior across different locations or time periods. This makes it useful for studying climate change effects.

The analysis begins by identifying independent rainfall events within the dataset. For each event, the maximum intensity is determined across a wide range of durations. When analyzing long-term datasets spanning at least 20 years (ideally over 50 years), an annual series analysis is performed using the Gumbel distribution, where only the maximum intensity per year is considered. For medium-length datasets spanning at least 10 years (preferably over 20 years), a partial series is constructed by selecting the largest intensities based on an exponential distribution. The number of selected events is determined as 2.7 times the recording duration in years (e.g., for a 10-year dataset, the 27 highest intensities are analyzed).

The derived parameters for the extreme value distribution are categorized into short-, medium-, and long-term duration ranges. Within each range, the parameters are fitted using a function dependent on duration. The formulation follows the plotting approach proposed by @cunnane:1978, with currently implemented options including one-folded logarithmic, two-folded logarithmic, and hyperbolic models, though additional formulations can be incorporated.

The general equation for calculating rainfall depth as a function of duration and return period is given by:

\begin{equation}\label{eq:idf}
h(D,T_n) = u(D) + w(D) \cdot \ln(T_n)
\end{equation}

where $h$ is the rainfall depth, $u(D)$ and $w(D)$ are duration ($D$)-dependent parameters, and $T_n$ represents the return period. The parameter formulations can also vary based on duration range (short-, medium-, or long-term).

When first published on GitHub in 2018, no other tool existed that could derive IDF curves directly from observed rainfall data. This package was developed to standardize the workflow and eliminate the need for individual, custom implementations.

While @mendez:2024 introduced a similar tool, it has limitations: it only supports hourly data and applies a single formulation across all durations, which can lead to inaccuracies in IDF curve representation. The `idf-analysis` package overcomes these constraints by offering more flexibility in duration-specific parameterization.

In the broader context of hydrological analysis, the `wetterdienst` Python package (@gutzmann:2024) provides access to weather data, including precipitation records that can be used as input for IDF analysis. This makes it a valuable preparatory tool for IDF curve estimation.

The significance of IDF curve estimation is well recognized in the scientific community. @schardong:2020 developed a web-based tool for estimating IDF curves at both gauged and ungauged locations. @lanciotti:2022 reviewed various methodologies and data sources used for IDF curve development. However, as neither study mentioned an existing standardized tool for deriving IDF curves from observed data, it is likely that researchers rely on custom scripts. The `idf-analysis` package provides a structured, efficient alternative, reducing workload and ensuring consistency in IDF curve estimation.

# Acknowledgements

I am very thankful to David Chamy for his guidance in programming the package. I am also grateful to Günter Gruber for giving me the opportunity to research this topic as part of my bachelor project.

Furthermore, I would like to thank the Institute of Urban Water Management and Landscape Water Engineering at Graz University of Technology, where my work allowed me to further develop the code for this package.

# References