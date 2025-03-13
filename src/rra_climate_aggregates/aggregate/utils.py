import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterra as rt
from rasterio.features import MergeAlg, rasterize
from shapely import MultiPolygon, Polygon

from rra_climate_aggregates.data import (
    PopulationModelData,
)


def build_location_masks(
    hierarchy: str,
    pm_data: PopulationModelData,
) -> tuple[dict[int, tuple[slice, slice]], npt.NDArray[np.uint32]]:
    """Build location masks for each location in the hierarchy.

    Parameters
    ----------
    hierarchy
        The name of the hierarchy to build location masks for. Must be one of
        of the keys of the HIERARCHY_MAP constant.
    pm_data
        PopulationModelData object to load the population model data.

    Returns
    -------
    tuple[dict[int, tuple[slice, slice]], npt.NDArray[np.uint32]]
        The first element is a dictionary mapping location IDs to a tuple of
        slices representing the bounds of the location in the mask. This is useful
        for subseting the mask and data arrays before processing as downstream
        operations scale with the number of pixels in the mask. The second element
        is the mask itself, a 2D array of uint32 values where each location ID is
        represented by a unique integer value.
    """
    template = pm_data.load_results("2020q1")
    raking_shapes = pm_data.load_raking_shapes(hierarchy)

    shape_values = [
        (shape, loc_id)
        for loc_id, shape in raking_shapes.set_index("location_id")
        .geometry.to_dict()
        .items()
    ]
    bounds_map = build_bounds_map(template, shape_values)

    location_mask = np.zeros_like(template, dtype=np.uint32)
    location_mask = rasterize(
        shape_values,
        out=location_mask,
        transform=template.transform,
        merge_alg=MergeAlg.replace,
    )
    return bounds_map, location_mask


def build_bounds_map(
    raster_template: rt.RasterArray,
    shape_values: list[tuple[Polygon | MultiPolygon, int]],
) -> dict[int, tuple[slice, slice]]:
    """Build a map of location IDs to buffered slices of the raster template.

    Parameters
    ----------
    raster_template
        The raster template to build the bounds map for.
    shape_values
        A list of tuples where the first element is a shapely Polygon or MultiPolygon
        in the CRS of the raster template and the second element is the location ID
        of the shape.

    Returns
    -------
    dict[int, tuple[slice, slice]]
        A dictionary mapping location IDs to a tuple of slices representing the bounds
        of the location in the raster template. The slices are buffered by 10 pixels
        to ensure that the entire shape is included in the mask.
    """
    # The tranform maps pixel coordinates to the CRS coordinates.
    # This mask is the inverse of that transform.
    to_pixel = ~raster_template.transform

    bounds_map = {}
    for shp, loc_id in shape_values:
        xmin, ymin, xmax, ymax = shp.bounds
        pxmin, pymin = to_pixel * (xmin, ymax)
        pixel_buffer = 10
        pxmin = max(0, int(pxmin) - pixel_buffer)
        pymin = max(0, int(pymin) - pixel_buffer)
        pxmax, pymax = to_pixel * (xmax, ymin)
        pxmax = min(raster_template.width, int(pxmax) + pixel_buffer)
        pymax = min(raster_template.height, int(pymax) + pixel_buffer)
        bounds_map[loc_id] = (slice(pymin, pymax), slice(pxmin, pxmax))

    return bounds_map


def aggregate_pop_to_hierarchy(
    data: pd.DataFrame, hierarchy: pd.DataFrame
) -> pd.DataFrame:
    """Create all aggregate populations for a given hierarchy from most-detailed data.

    Parameters
    ----------
    data
        The most-detailed population data to aggregate.
    hierarchy
        The hierarchy to aggregate the data to.

    Returns
    -------
    pd.DataFrame
        The population data with values for all levels of the hierarchy.
    """
    results = (
        data.drop(columns=["weighted_climate", "scenario", "value"])
        .rename(columns={"population": "value"})
        .set_index("location_id")
        .copy()
    )
    # Most detailed locations can be at multiple levels of the hierarchy,
    # so we loop over all levels from most detailed to global, aggregating
    # level by level and appending the results to the data.
    for level in reversed(list(range(1, hierarchy.level.max() + 1))):
        level_mask = hierarchy.level == level
        parent_map = hierarchy.loc[level_mask].set_index("location_id").parent_id
        subset = results.loc[parent_map.index]
        subset["parent_id"] = parent_map

        parent_values = (
            subset.groupby(["year_id", "parent_id"])[["value"]]
            .sum()
            .reset_index()
            .rename(columns={"parent_id": "location_id"})
            .set_index("location_id")
        )
        results = pd.concat([results, parent_values])

    results = (
        results.reset_index()
        .sort_values(["location_id", "year_id"])
        .reset_index(drop=True)
    )
    return results


def aggregate_climate_to_hierarchy(
    data: pd.DataFrame, hierarchy: pd.DataFrame
) -> pd.DataFrame:
    """Create all aggregate climate values for a given hierarchy from most-detailed data.

    Parameters
    ----------
    data
        The most-detailed climate data to aggregate.
    hierarchy
        The hierarchy to aggregate the data to.

    Returns
    -------
    pd.DataFrame
        The climate data with values for all levels of the hierarchy.
    """
    results = data.set_index("location_id").copy()

    # Most detailed locations can be at multiple levels of the hierarchy,
    # so we loop over all levels from most detailed to global, aggregating
    # level by level and appending the results to the data.
    for level in reversed(list(range(1, hierarchy.level.max() + 1))):
        level_mask = hierarchy.level == level
        parent_map = hierarchy.loc[level_mask].set_index("location_id").parent_id

        subset = results.loc[parent_map.index]
        subset["parent_id"] = parent_map

        parent_values = (
            subset.groupby(["year_id", "scenario", "parent_id"])[
                ["weighted_climate", "population"]
            ]
            .sum()
            .reset_index()
            .rename(columns={"parent_id": "location_id"})
            .set_index("location_id")
        )
        parent_values["value"] = (
            parent_values.weighted_climate / parent_values.population
        )
        results = pd.concat([results, parent_values])
    results = (
        results.drop(columns=["weighted_climate", "population"])
        .reset_index()
        .sort_values(["location_id", "year_id"])
        .reset_index(drop=True)
    )
    return results
