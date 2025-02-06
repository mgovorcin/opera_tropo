import json
import time
import psutil
import os
import sys
import logging
import logging.config
from pathlib import Path
from functools import wraps
from collections.abc import Callable
from typing import (TypeVar, ParamSpec)

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

# Used for callable types
T = TypeVar("T")
P = ParamSpec("P")


def setup_logging(*, logger_name: str = "opera_tropo", debug: bool = False, filename: str = None):
    config_file = Path(__file__).parent / "log-config.json"

    with open(config_file) as f_in:
        config = json.load(f_in)

    if logger_name not in config["loggers"]:
        config["loggers"][logger_name] = {"level": "INFO", "handlers": ["stderr"]}

    if debug:
        config["loggers"][logger_name]["level"] = "DEBUG"

    if filename:
        if "file" not in config["loggers"][logger_name]["handlers"]:
            config["loggers"][logger_name]["handlers"].append("file")
        config["handlers"]["file"]["filename"] = os.fspath(filename)
        print(config["handlers"]["file"]["filename"] )
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if "filename" not in config["handlers"]["file"]:
        config["handlers"].pop("file", None)

    logging.config.dictConfig(config)

def log_runtime(f: Callable[P, T]) -> Callable[P, T]:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = logging.getLogger('tropo') # change name to __name__
    print(f.__module__, __name__)
    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        # Get memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1e6

        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_seconds = t2 - t1
        elapsed_minutes = elapsed_seconds / 60.0

        # Get memory usage after function call
        memory_after = process.memory_info().rss / 1e6 

        time_string = (
            f"Total elapsed time for {f.__module__}.{f.__name__}: "
            f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)"
        )   

        memory_string = (
            f" Memory before: {memory_before:.2f} MB, "
            f" after: {memory_after:.2f} MB, "
            f" total: {memory_after - memory_before:.2f} MB" 
        )

        logger.info(time_string + '\n' + memory_string)

        return result

    return wrapper