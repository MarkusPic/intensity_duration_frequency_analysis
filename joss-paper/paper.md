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
date: 6 November 2024
bibliography: paper.bib
---

# Summary

Heavy rainfall data are among the most important planning parameters in water management and hydraulic engineering practice. They are required as initial parameters for the dimensioning of rainwater drainage systems in urban areas and for the dimensioning of hydraulic structures on watercourses. 

The accuracy of the target values of the corresponding calculation methods and models depends crucially on their accuracy. Their overestimation can lead to considerable additional costs in the structural implementation, their underestimation to an unacceptable, excessive residual risk of failure during the operation of water management and hydraulic engineering facilities. 

 It should be noted that overestimating them can lead to considerable additional costs during construction, while underestimating them can lead to an unacceptable, excessive residual risk of failure during the operation of water management and hydraulic engineering facilities.

An intensity-duration-frequency (IDF) curve is a mathematical function (see \autoref{eq:idf}) that relates the rainfall intensity with its duration and frequency of occurrence. An example can be seen in \autoref{fig:idf_curves}. These curves are commonly used in hydrology for flood forecasting and civil engineering for urban drainage design. However, the IDF curves are also analysed in hydrometeorology because of the interest in the time concentration or time-structure of the rainfall.

The package follows the guidance recommendations in @dwa:2012.

![Example IDF curves for the city of Graz.\label{fig:idf_curves}](idf_curves_plot.pdf)

# Statement of need

To assess the characteristics of rainfall and its implications for water management, local station analyses are essential for evaluating data series, recent trends, and identifying unique local features.

`idf-analysis` is an advanced Python package designed to process rainfall measurement time series and derive the necessary parameters for IDF curves. This enables reliable estimations of return periods for specific rainfall events. Additionally, it facilitates comparative studies of rainfall behavior across different stations or time periods, which is particularly valuable for assessing potential impacts of climate change.

The analysis begins by splitting the rainfall data into independent rainfall events. The intensity for various durations is calculated, and the maximum intensity of each event is extracted. For lengthy time series, an annual series analysis is conducted using the Gumbel distribution, focusing on the maximum intensity per year. For medium-length time series, a partial series analysis employs the exponential distribution, using the largest $2.7$ times the number of measurement years’ events.

Parameters from the extreme value distribution are categorized into short-, medium-, and long-term duration ranges. These parameters are then fitted into functions that vary with duration. The package currently supports one-folded logarithmic, two-folded logarithmic, and hyperbolic formulations, with the flexibility for future extensions.

The general equation for calculating rainfall depth based on duration and return period is:
\begin{equation}\label{eq:idf}
h(D,T_n) = u(D) + w(D) * ln(T_n)
\end{equation}

where $h$ represents the rainfall depth, $u(D)$ and $w(D)$ are duration-dependent parameters, and $T_n$ is the return period. The formulation of $u$ and $w$ may further depend on the duration range being considered.

To support this analysis, several tools have been developed by various researchers. @gutzmann:2024 introduced the wetterdienst Python package for downloading and analyzing weather data, including precipitation for IDF curve generation. @Schardong:2020 developed a web-based tool for estimating IDF curves for both gauged and ungauged sites, and @Lanciotti:2022 provided a comprehensive review of existing formulations and data types. In contrast, @Mendez:2024 presented a similar code that relies solely on hourly data and applies a single formulation across all durations.

# Acknowledgements

I would like to extend my gratitude to the Institute of Urban Water Management and Landscape Water Engineering at Graz University of Technology, where my work allowed me to further develop the code for this package.

I am also deeply thankful to David Chamy for his invaluable guidance in programming the package, and to Günter Gruber for providing the opportunity to work on this topic during my bachelor project.

# References