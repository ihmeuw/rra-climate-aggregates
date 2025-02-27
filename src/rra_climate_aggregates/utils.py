import numpy as np
import rasterra as rt
import xarray as xr
from affine import Affine


def to_raster(
    ds: xr.DataArray,
    no_data_value: float = np.nan,
    crs: str = "EPSG:4326",
) -> rt.RasterArray:
    """Convert an xarray DataArray to a RasterArray.

    Parameters
    ----------
    ds
        The xarray DataArray to convert.
    no_data_value
        The value to use for missing data. This should be consistent with the dtype of the data.
    crs
        The coordinate reference system of the data.

    Returns
    -------
    rt.RasterArray
        The RasterArray representation of the input data.
    """
    lat, lon = ds["latitude"].data, ds["longitude"].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-dlat,
        f=lat[-1],
    )
    return rt.RasterArray(
        data=ds.data[::-1],
        transform=transform,
        crs=crs,
        no_data_value=no_data_value,
    )
