from __future__ import annotations

import numpy as np
from datetime import datetime, timezone
from numpy.typing import DTypeLike
from dataclasses import dataclass, field

import RAiDER

GLOBAL_ATTRS = {
    # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#standard-name
    "Conventions": "CF-1.8",
    "title": "OPERA_L4_ZENITH_TROPO",
    "institution": "NASA Jet Propulsion Laboratory (JPL)",
    "contact" : "opera-sds-ops@jpl.nasa.gov", 
    "source" : "ECMWF",
    "platform" : "HRES Model",
    "mission_name": 'OPERA',
    "source_url": "https://www.ecmwf.int/en/forecasts/datasets/set-i",
    "references":   "https://raider.readthedocs.io/en/latest/", 
    "description": "OPERA One-way Tropospheric Zenith Delay for Synthetic Aperture Radar",
    "comment" : ("Intersect/interpolate"
                 " with DEM and multiple with -4pi/radar wavelength (2 way)"
                 " to get SAR correction"),
    "software": "RAiDER",
    "software_version": f"{RAiDER.__version__}",
    # Audit trail. date/time 0f day/ user name/ program name/command arguments
    "history": f"Created on: {str(datetime.now(timezone.utc))}",
    }


### DATASET COORDS ###
@dataclass
class ProductCoords:
    """Container for tropo xarray coord info."""
    axis : str
    units : str
    standard_name : str
    long_name: str
    description: str
    encoding : dict[str, str] = field(default_factory=dict) 
    
    @property
    def get_attr(self) -> dict:
        """Return all tropo coord  attrs excluding encoding"""
        out_dict = self.__dict__.copy()
        del out_dict['encoding']
        return out_dict
    

@dataclass
class TropoCoordAttrs:
    latitude: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="Y",
            units="degrees_north",
            standard_name="latitude",
            long_name="Latitude",
            description=("Angular distance of a point north or south"
                         " of the equator."),
            encoding={}
        )
    )
    longitude: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="X",
            units="degrees_east",
            standard_name="longitude",
            long_name="Longitude",
            description=("Angular distance of a point east or west"
                         " of the Prime Meridian."),
            encoding={}
        )
    )
    height: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="Z",
            units="meters",
            standard_name="height",
            long_name="Ellipsoidal Height",
            description="Height above ellipsoid WGS84",
            encoding={}
        )
    )
    time: ProductCoords = field(
        default_factory=lambda: ProductCoords(
            axis="T",
            units=None, # units specified in encoding
            standard_name="time",
            long_name="UTC time",
            description="Numerical Weather Model base time",
            encoding={
                "units": "hours since 1900-01-01",
                "calendar": "gregorian"
            }
        )
    )
    @property
    def names(self) -> list[str]:
        """Return all tropo coord  as a list."""
        return [v for v in self.__dict__.keys()]
        

### DATASET VARIABLES ###
@dataclass
class ProductInfo:
    """Information about a troposphere product dataset."""

    name: str
    long_name: str
    description: str
    positive: str
    fillvalue: DTypeLike
    dtype: DTypeLike
    attrs: dict[str, str] = field(default_factory=dict)
    keep_bits: int | None = None

    def to_dict(self):
        desc_dict = dict(
            standard_name=self.name,
            long_name=self.long_name,
            description=self.description
        )
        return self.attrs | desc_dict


@dataclass
class TropoProducts:
    """Container for tropo product dataset info."""

    wet_delay: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="wet_delay",
            long_name="Zenith Wet Delay",
            description=(
                "One-way Zenith Wet Delay."
            ),
            fillvalue=np.nan,
            positive='down',
            # Note sure should I keep grid_mapping here
            attrs={"units": "meters",
                   "grid_mapping": "spatial_ref"},
            # 10 bits, has a max quantization error of 
            # about 0.1 millimeters
            keep_bits=10,
            dtype=np.float32,
        )
    )

    hydrostatic_delay: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="hydrostatic_delay",
            long_name="Zenith Hydrostatic Delay",
            description=(
                "One-way Zenith Wet Delay."
            ),
            fillvalue=np.nan,
            positive='down', 
            # Note sure should I keep grid_mapping here
            attrs={"units": "meters",
                   "grid_mapping": "spatial_ref"},
            # 12 bits, has a max quantization error of 
            # about 0.2 millimeters
            keep_bits=12,
            dtype=np.float32,
        )
    )

    coords: TropoCoordAttrs = field(default_factory=TropoCoordAttrs,
                                    init=False)

    def __post_init__(self):
        self.coords = TropoCoordAttrs()

    def __iter__(self):
        """Return all tropo dataset info as an iterable."""
        return iter(self.__dict__.values())

    @property
    def names(self) -> list[str]:
        """Return all tropo dataset names as a list."""
        return [v.name for v in self.__dict__.values()]

# Create a single instance to be used throughout the application
TROPO_PRODUCTS = TropoProducts()
