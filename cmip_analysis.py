import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from collections.abc import Iterable


def reshape_by_year(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have a 'time' coordinate.")
    
    years = range(start_year, end_year)
    n_years = len(years)
    n_times = len(ds.time) // n_years

    index = pd.MultiIndex.from_product([years, range(n_times)], names=['year', 'time_index'])
    reshaped = ds.assign_coords(time=index).unstack('time')

    return reshaped.rename({'time_index': 'time'})


def prepare_variable(ds: xr.Dataset, variable: str) -> xr.DataArray:
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    da = ds[variable]
    da = da.squeeze(drop=True)
    
    return da.transpose('year', 'time', 'lat', 'lon')


def calculate_quantiles(da: xr.DataArray, n_quantiles: int = 101) -> xr.DataArray:
    q = np.linspace(0, 1, n_quantiles)

    return da.quantile(q=q, dim='time')


def calculate_warming_slopes(quantiles: xr.DataArray) -> xr.DataArray:
    if 'year' not in quantiles.dims:
        raise ValueError("DataArray must have a 'year' coordinate.")

    years = quantiles.year

    x_mean = years.mean()
    y_mean = quantiles.mean(dim='year')
    n_years = quantiles.sizes['year']

    s_xx = (years * years).sum() - n_years * x_mean ** 2
    s_xy = quantiles.dot(years) - n_years * x_mean * y_mean

    return s_xy / s_xx


def prepare_spatial_data(slopes: xr.DataArray, lat_block_size: int = 20) -> tuple[np.ndarray, pd.DataFrame]:
    if 'lat' not in slopes.coords or 'lon' not in slopes.coords:
        raise ValueError("DataArray must have 'lat' and 'lon' coordinates.")

    n_lat = slopes.sizes['lat']
    n_lon = slopes.sizes['lon']
    n_quantiles = slopes.sizes['quantile']

    slope_values = np.empty((n_lat * n_lon, n_quantiles), dtype=np.float32)

    coordinates = pd.DataFrame({'lat': np.repeat(slopes.lat.values, n_lon), 
                                'lon': np.tile(slopes.lon.values, n_lat)})
    coordinates['lon'] = np.where(coordinates['lon'] > 180, coordinates['lon'] - 360, coordinates['lon'])

    for start in range(0, n_lat, lat_block_size):
        end = min(start + lat_block_size, n_lat)

        slopes_block = slopes.isel(lat=slice(start, end)).compute()

        block_values = (slopes_block.transpose('lat', 'lon', 'quantile').values.reshape(-1, n_quantiles))

        start_idx = start * n_lon
        end_idx = end * n_lon

        slope_values[start_idx:end_idx] = block_values

    return slope_values, coordinates


def fit_kmeans(data: np.ndarray, n_clusters: int, init: str = 'random', n_init: int = 10, max_iter: int = 1000, 
               tol: float = 1e-5, random_state: int = 4) -> tuple[KMeans, np.ndarray]:

    model = KMeans(n_clusters=n_clusters, 
                   init=init, 
                   n_init=n_init, 
                   max_iter=max_iter, 
                   tol=tol, 
                   random_state=random_state
                   )

    labels = model.fit_predict(data)

    return model, labels


def calculate_wcss(data: np.ndarray, cluster_range: Iterable[int], init: str = 'random', 
                   n_init: int = 10, max_iter: int = 1000, tol: float = 1e-5, 
                   random_state: int = 4) -> list[float]:
    wcss = []

    for n_clusters in cluster_range:
        model, _ = fit_kmeans(data, n_clusters, init=init, n_init=n_init,
                              max_iter=max_iter, tol=tol, random_state=random_state)

        wcss.append(model.inertia_)

    return wcss


def calculate_cluster_areas(slopes: xr.DataArray, coordinates: pd.DataFrame, labels: np.ndarray,
                            n_clusters: int) -> list[float]:

    lon_width = abs(float(slopes.lon[1] - slopes.lon[0]))
    lat_half_width = abs(float(slopes.lat[1] - slopes.lat[0])) / 2

    areas = [0.0] * n_clusters

    for cluster in range(n_clusters):
        cluster_latitudes = coordinates.loc[labels == cluster, 'lat'].to_numpy()

        upper = np.radians(np.clip(cluster_latitudes + lat_half_width, -90, 90))
        lower = np.radians(np.clip(cluster_latitudes - lat_half_width, -90, 90))

        areas[cluster] = float(np.sum((lon_width / 360) * (np.sin(upper) - np.sin(lower)) / 2))

    return areas

