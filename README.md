# Climate Warming Pattern Recognition using CMIP6

## Overview

This project analyzes historical and projected future climate data from the Coupled Model Intercomparison Project Phase 6 (CMIP6) to characterize spatiotemporal warming patterns and quantify long-term climate shifts across global regions.

The analysis uses near-surface air temperature (`tas`) data from the NOAA-GFDL GFDL-ESM4 climate model. Annual temperature distributions are characterized using quantiles at each global grid cell, and long-term warming rates are calculated to quantify changes in temperature distributions over time. K-means clustering is then applied to identify distinct spatial patterns of climate change across the globe. Together, these analyses provide insight into long-term climate shifts and their spatial variability across the globe.


## Data

The project uses CMIP6 data from the NOAA Geophysical Fluid Dynamics Laboratory (NOAA-GFDL) GFDL-ESM4 model.

### Historical Simulation

* Dataset: `CMIP.NOAA-GFDL.GFDL-ESM4.esm-hist.3hr.gr1`
* Experiment: `esm-hist`
* Variable: Near-urface air temperature (`tas`)
* Frequency: 3-hourly
* Period: 1948-2014

### Future Projection

* Dataset: `ScenarioMIP.NOAA-GFDL.GFDL-ESM4.ssp126.3hr.gr1`
* Experiment: `ssp126`
* Variable: Near-surface air temperature (`tas`)
* Frequency: 3-hourly
* Period: 2015-2100

The data are accessed through the Pangeo CMIP6 data catalog using `intake-esm`. 

> **Note:** CMIP6 datasets contain large volumes of high-frequency climate data. The analysis therefore uses `xarray` and `Dask` to perform lazy and chunked computations and reduce unnecessary memory usage.

## Project Structure

```text
cwpr-cmip6/
│
├── cmip_analysis.py
├── warming_pattern_recognition.ipynb
├── README.md
└── ...
```

`cmip_analysis.py`: Provides the reusable analysis functions that support the project.

`warming_pattern_recognition.ipynb`: Presents the main analysis workflow, including historical and future data processing, annual temperature-quantile analysis, warming-rate estimation, K-means pattern recognition, spatial analysis, and visualization.

