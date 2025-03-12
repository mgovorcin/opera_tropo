from __future__ import annotations

import time
import shutil
import psutil
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from dask.distributed import Client

from .log.loggin_setup import log_runtime 
from .core import calculate_ztd
from ._pack import pack_ztd
from .utils import round_mantissa_xr
from .product_info import TROPO_PRODUCTS

try:
    from RAiDER.models.model_levels import A_137_HRES, LEVELS_137_HEIGHTS
except ImportError as e:
    print(f"RAiDER is not properly installed or accessible. Error: {e}")

logger = logging.getLogger(__name__)

def _rounding_mantissa_blocks(ds: xr.DataArray, keep_bits:int):
    """
    Rounds the mantissa blocks of a given xr.DataArray.

    Parameters:
    - ds (xr.DataArray): The input xr.DataArray to be processed.
    - keep_bits (int): The number of bits to keep in the mantissa blocks.

    Returns:
    - xr.DataArray: The processed xr.DataArray with rounded mantissa blocks.
    """
    return ds.map(round_mantissa_xr, keep_bits=keep_bits)


@log_runtime
def tropo(file_path: str,
          output_file: str,
          *,
          out_heights: list = [],
          lat_chunk_size: int = 128,
          lon_chunk_size: int = 128,
          num_workers: int = 1,
          num_threads: int = 1,
          max_memory: int = 4,  # GB
          compression_options: dict = {},
          temp_dir: str = None,
          keep_bits: bool = True) -> None:
    """
    Calculate TROPO delay and save the output to a NetCDF file.

    Parameters:
    - file_path (str): Path to the input dataset file.
    - output_file (str): Path to the output NetCDF file.
    - out_heights (list, optional): List of output heights. Default is an empty list.
    - lat_chunk_size (int, optional): Chunk size for latitude dimension. Default is 128.
    - lon_chunk_size (int, optional): Chunk size for longitude dimension. Default is 128.
    - num_workers (int, optional): Number of Dask workers. Default is 1.
    - num_threads (int, optional): Number of threads per Dask worker. Default is 1.
    - max_memory (int, optional): Maximum memory limit for Dask workers. Default is 4 GB.
    - compression_options (dict, optional): Compression options for the output NetCDF file. Default is None.
    - temp_dir (str, optional): Temporary directory for Dask intermediate files. Default is None.
    - keep_bits (bool, optional): Flag to indicate whether to keep the bits. Default is True.

    Returns:
    - None

    Raises:
    - ValueError: If the input dataset file cannot be opened.

    """
    process = psutil.Process()
    logger.info("Calculating TROPO delay")    
    # Set default compression options if not provided
    compression_defaults = {
        "zlib": True,
        "compression_flag": False,
        "complevel": 4,
        "shuffle": True
    }
    compression_options = {**compression_defaults, **compression_options}
    compress = compression_options.pop("compression_flag")
    
    # Setup Dask Client and temp. directory
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    client = Client(
        n_workers=num_workers,
        threads_per_worker=num_threads,
        memory_limit=f"{max_memory}GB",
        local_directory=temp_dir,
    )
    logger.info(f'Dask server link: {client.dashboard_link}')

    # Open the dataset
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open the dataset file: {file_path}."
                          "Ensure the file exists and is a valid dataset."
                          "Original error: {e}") 
    
    # Rechunk
    if not ds.chunks:
        logger.info(f"Rechunking {file_path}")
        chunks = {
            'longitude': lon_chunk_size, 
            'latitude': lat_chunk_size,
            'time': 1, 
            'level': len(A_137_HRES) - 1
        }
        ds = ds.chunk(chunks) 

    # Get output size
    cols = ds.sizes.get('latitude')
    rows = ds.sizes.get('longitude')
    
    if out_heights is not None and len(out_heights) > 0:
        zlevels = np.array(out_heights)
    else:
        zlevels = np.flipud(LEVELS_137_HEIGHTS)
    out_size = np.empty((cols, rows, len(zlevels)),
                        dtype=np.float32)
    
    # To skip interpolation if out_heights are same as default
    if np.array_equal(out_heights, np.flipud(LEVELS_137_HEIGHTS)):
        out_heights = None

    # Get output template
    # NOTE: translate dim order to follow CF 1.8
    # time, height, latitude, longitude 
    # to be compliant with gdal
    template = pack_ztd(
        wet_ztd=out_size, 
        hydrostatic_ztd=out_size,
        lons=ds.longitude.values, 
        lats=ds.latitude.values,
        zs=zlevels, 
        model_time=ds.time.values,
        chunk_size={"longitude": lat_chunk_size,
                    "latitude": lon_chunk_size,
                    "height": -1, "time": 1},
        keep_bits=False)

    # Calculate ZTD
    # NOTE: Peak in RAM is 40GB after ingesting map_block output
    # with default 145 height level, specifying out_heights can
    # lower mem usage 
    mem = process.memory_info().rss / 1e6 
    logger.info(f"Estimating ZTD delay, mem usage {mem:.2f} GB")
    t1 = time.time()
    out_ds = ds.map_blocks(calculate_ztd,
                kwargs={'out_heights': out_heights}, 
                        template=template).compute()
    t2 = time.time()
    mem = process.memory_info().rss / 1e6 
    logger.info(f"ZTD calculation took {t2 - t1:.2f} seconds.")
    logger.info(f"Mem usage {mem:.2f} GB")

    # Clean up
    del template, ds 

    # Note, apply again rounding as interpolation can change
    # output, double check if needed
    if out_heights is not None and len(out_heights)>0 and keep_bits:
        # use one keep_bits setting, need to figure how to apply
        # different rounding for each data_var in xr.Dataset
        keep_bit_kwargs = {'keep_bits': TROPO_PRODUCTS.wet_delay.keep_bits}
        out_ds = out_ds.map_blocks(_rounding_mantissa_blocks,
                                    kwargs=keep_bit_kwargs) 
    
    # Save and compress output
    t1 = time.time()
    msg = 'and Compressing' if compress else ''
    encoding = {var: compression_options if compress else {} for var in out_ds.data_vars}
    out_ds.to_netcdf(output_file, encoding=encoding, mode='w')
    t2 = time.time()
    logger.info(f"Saving {msg} took {t2 - t1:.2f} seconds.")
    mem = process.memory_info().rss / 1e6
    logger.info(f"Mem usage {mem:.2f} GB") 
    client.close()
    shutil.rmtree(temp_dir)
