###### DEVELOPEMENT STAGE ############
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone
logger = logging.getLogger(__name__)

import RAiDER

from .run import tropo 
from .utils import get_hres_datetime, get_max_memory_usage
from .config import runconfig, pge_runconfig
from .log.loggin_setup import setup_logging, log_runtime
from  .browse_image import make_browse_image_from_nc
from opera_tropo import __version__

@log_runtime
def run(
    cfg: runconfig.TropoWorkflow, 
    pge_runconfig: pge_runconfig.RunConfig,
    debug: bool = False,
) -> None:
    """Run the troposphere ZTD on Global Weather Model input.

    Parameters
    ----------
    cfg : TropoWorkflow
        `TropoWorkflow` object for controlling the workflow.
    pge_runconfig : RunConfig
        PGE-specific metadata for the output product.
    debug : bool, optional
        Enable debug logging.
        Default is False.

    """

    setup_logging(logger_name="opera_tropo", debug=debug, filename=cfg.log_file)
    setup_logging(logger_name="RAiDER",  debug=debug, filename=cfg.log_file)
    setup_logging(logger_name="dask",  debug=debug, filename=cfg.log_file)       

    #Save the start for a metadata field
    #processing_start_datetime = datetime.now(timezone.utc)
    cfg.work_directory.mkdir(exist_ok=True, parents=True)

    # Get output filename
    hres_date, hres_hour = get_hres_datetime(cfg.input_options.input_file_path)
    output_filename = cfg.output_options.get_output_filename(hres_date, hres_hour)

    # Run dolphin's displacement workflow
    tropo(file_path = cfg.input_options.input_file_path,
          output_file = Path(cfg.work_directory) / output_filename,
          out_heights = cfg.output_options.output_heights,
          lat_chunk_size = cfg.worker_settings.block_shape[0],
          lon_chunk_size = cfg.worker_settings.block_shape[1],
          num_workers = cfg.worker_settings.n_workers,
          num_threads = cfg.worker_settings.threads_per_worker,
          max_memory = cfg.worker_settings.max_memory,
          compression_options = cfg.output_options.compression_kwargs,
          temp_dir = cfg.worker_settings.dask_temp_dir
         )

    # Generate output browse image
    logger.info(f"Output file: {Path(cfg.work_directory) / output_filename}")
    output_png = Path(cfg.work_directory) / output_filename
    output_png = output_png.with_suffix(".png")
    logger.info(f"Output browse image: {output_png}")
    make_browse_image_from_nc(output_png, Path(cfg.work_directory) / output_filename)

    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file RAIDER version: {RAiDER.__version__}")
    logger.info(f"Current running opera_tropo version: {__version__}")
