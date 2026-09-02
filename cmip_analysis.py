import numpy as np
import pandas as pd
import xarray as xr


def reshape_by_year(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    if 'time' not in ds.coords:
        raise ValueError("Dataset must have a 'time' coordinate.")
    
    years = range(start_year, end_year)
    n_years = len(years)
    n_times = len(ds.time) // n_years

    index = pd.MultiIndex.from_product([years, range(n_times)], names=['year', 'time'])

    return (ds.assign_coords(time=index).unstack('time'))


def prepare_variable(ds: xr.Dataset, variable: str) -> xr.DataArray:
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    da = ds[variable]
    da = da.squeeze(drop=True)
    
    return da.transpose('year', 'time', 'lat', 'lon')


def calculate_quantiles(da: xr.DataArray, n_quantiles: int = 101) -> xr.DataArray:
    q = np.linspace(0, 1, n_quantiles)

    return da.quantile(q=q, dim='time')


def calculate_warming_slopes(da: xr.DataArray) -> xr.DataArray:
    if 'year' not in da.coords:
        raise ValueError("DataArray must have a 'year' coordinate.")

    years = da.year

    x_mean = years.mean()
    y_mean = da.mean(dim='year')
    n_years = da.sizes['year']

    s_xx = (years * years).sum() - n_years * x_mean ** 2
    s_xy = da.dot(years) - n_years * x_mean * y_mean

    return s_xy / s_xx