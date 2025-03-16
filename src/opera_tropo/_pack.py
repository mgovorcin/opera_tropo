import numpy as np
import xarray as xr

from .product_info import TROPO_PRODUCTS, GLOBAL_ATTRS
from .utils import round_mantissa

# NOTE: check if it is better to add attributes at the end to
#       leave this empty and more light

def pack_ztd(wet_ztd: np.ndarray, hydrostatic_ztd: np.ndarray,
             lons: np.ndarray, lats: np.ndarray,
             zs: np.ndarray, model_time: np.ndarray,
             chunk_size={"longitude": 128, "latitude": 128,
                         "height": -1, "time": 1},
             keep_bits: bool = True):
    """
    Package Zenith Total Delay (ZTD) data into an xarray Dataset.

    Parameters:
    - wet_ztd (array-like): Array of wet zenith total delays.
    - hydrostatic_ztd (array-like): Array of hydrostatic zenith total delays.
    - lons (array-like): Array of longitudes.
    - lats (array-like): Array of latitudes.
    - zs (array-like): Array of heights.
    - model_time (array-like): Array of model times.
    - chunk_size (dict): Chunk size for the dataset.

    Returns:
    - ds (xarray.Dataset): Packaged ZTD data.
    """

    dim = ["height", "latitude", "longitude"]
    reference_time = model_time.astype('datetime64[s]').astype('O')[0]
    reference_time = reference_time.strftime('%Y-%m-%d %H:%M:%S')   

    #total_zenith_delay = hydrostatic_ztd + wet_ztd 
    wet_ztd = wet_ztd.astype(TROPO_PRODUCTS.wet_delay.dtype)
    hydrostatic_ztd = hydrostatic_ztd.astype(TROPO_PRODUCTS.hydrostatic_delay.dtype)  
    zs = zs.astype('float64')

    # Rounding
    if keep_bits:
        if TROPO_PRODUCTS.wet_delay.keep_bits:
            round_mantissa(wet_ztd,
                           keep_bits=int(TROPO_PRODUCTS.wet_delay.keep_bits))
        if TROPO_PRODUCTS.hydrostatic_delay.keep_bits:
            round_mantissa(hydrostatic_ztd,
                            keep_bits=int(TROPO_PRODUCTS.hydrostatic_delay.keep_bits))
    
    ds = xr.Dataset(
        data_vars=dict(
            wet_delay=(dim, wet_ztd.transpose(2, 0, 1),
                       TROPO_PRODUCTS.wet_delay.to_dict()),
            hydrostatic_delay=(dim, hydrostatic_ztd.transpose(2, 0, 1),
                               TROPO_PRODUCTS.hydrostatic_delay.to_dict()), 
        ),

        # normalizing longitudes to the range [-180, 180] from [0, 360]
        # GDAL expects coordinates to be float64
        coords=dict(height=zs,
                    latitude=np.float64(lats),
                    longitude=(np.float64(lons) + 180) % 360 - 180), 
        attrs=GLOBAL_ATTRS | {"reference_time": reference_time}
    )

    # Add coords attrs
    ds['height'].attrs.update(TROPO_PRODUCTS.coords.height.get_attr)
    ds['latitude'].attrs.update(TROPO_PRODUCTS.coords.latitude.get_attr)
    ds['longitude'].attrs.update(TROPO_PRODUCTS.coords.longitude.get_attr)

    # Add time
    ds = ds.expand_dims({"time": model_time})
    ds['time'].attrs.update(TROPO_PRODUCTS.coords.time.get_attr)
    # Remove time units due to conflicts with encoding
    del ds['time'].attrs['units']
    ds['time'].encoding.update(TROPO_PRODUCTS.coords.time.encoding)

    # Add spatial reference
    ds.rio.write_crs("EPSG:4326", inplace=True)

    # Data Variables
    wet_fill = TROPO_PRODUCTS.wet_delay.fillvalue
    hydro_fill = TROPO_PRODUCTS.hydrostatic_delay.fillvalue
    ds["wet_delay"].attrs["_FillValue"] = wet_fill 
    ds["hydrostatic_delay"].attrs["_FillValue"] = hydro_fill 

    # Add chunks to data variables
    if chunk_size is not None:
        for key in ["wet_delay", "hydrostatic_delay"]:
            ds[key] = ds[key].chunk(chunk_size)

        # Ensure that chunking is applied to the entire dataset
        ds = ds.chunk(chunk_size)
    return ds
