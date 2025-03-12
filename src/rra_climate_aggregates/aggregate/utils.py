import numpy as np
import numpy.typing as npt
import pandas as pd
from rasterio.features import MergeAlg, rasterize

from rra_climate_aggregates.data import (
    PopulationModelData,
)


def build_location_masks(
    hierarchy: str,
    pm_data: PopulationModelData,
) -> tuple[dict[int, tuple[slice, slice]], npt.NDArray[np.uint32]]:
    template = pm_data.load_results("2020q1")

    raking_shapes = pm_data.load_raking_shapes(hierarchy)

    shape_values = [
        (shape, loc_id)
        for loc_id, shape in raking_shapes.set_index("location_id")
        .geometry.to_dict()
        .items()
    ]
    bounds_map = {}
    to_pixel = ~template.transform
    for shp, loc_id in shape_values:
        xmin, ymin, xmax, ymax = shp.bounds
        pxmin, pymin = to_pixel * (xmin, ymax)
        pixel_buffer = 100
        pxmin = max(0, int(pxmin) - pixel_buffer)
        pymin = max(0, int(pymin) - pixel_buffer)
        pxmax, pymax = to_pixel * (xmax, ymin)
        pxmax = min(template.width, int(pxmax) + pixel_buffer)
        pymax = min(template.height, int(pymax) + pixel_buffer)
        bounds_map[loc_id] = (slice(pymin, pymax), slice(pxmin, pxmax))

    location_mask = np.zeros_like(template, dtype=np.uint32)
    location_mask = rasterize(
        shape_values,
        out=location_mask,
        transform=template.transform,
        merge_alg=MergeAlg.replace,
    )
    return bounds_map, location_mask


def aggregate_pop_to_hierarchy(
    data: pd.DataFrame, hierarchy: pd.DataFrame
) -> pd.DataFrame:
    results = (
        data.drop(columns=["weighted_climate", "value"])
        .rename(columns={"population": "value"})
        .set_index("location_id")
        .copy()
    )
    for level in reversed(list(range(1, hierarchy.level.max() + 1))):
        level_mask = hierarchy.level == level
        parent_map = hierarchy.loc[level_mask].set_index("location_id").parent_id
        subset = results.loc[parent_map.index]
        subset["parent_id"] = parent_map

        parent_values = (
            subset.groupby(["year", "scenario", "parent_id"])[["value"]]
            .sum()
            .reset_index()
            .rename(columns={"parent_id": "location_id"})
            .set_index("location_id")
        )
        results = pd.concat([results, parent_values])

    results = (
        results.reset_index()
        .sort_values(["location_id", "year"])
        .reset_index(drop=True)
    )
    return results


def aggregate_climate_to_hierarchy(
    data: pd.DataFrame, hierarchy: pd.DataFrame
) -> pd.DataFrame:
    results = data.set_index("location_id").copy()

    for level in reversed(list(range(1, hierarchy.level.max() + 1))):
        level_mask = hierarchy.level == level
        parent_map = hierarchy.loc[level_mask].set_index("location_id").parent_id

        subset = results.loc[parent_map.index]
        subset["parent_id"] = parent_map

        parent_values = (
            subset.groupby(["year", "scenario", "parent_id"])[
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
        .sort_values(["location_id", "year"])
        .reset_index(drop=True)
    )
    return results
