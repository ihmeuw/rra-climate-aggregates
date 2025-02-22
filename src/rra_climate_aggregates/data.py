from pathlib import Path
from typing import TypeAlias

import geopandas as gpd
import rasterra as rt
import xarray as xr
import shapely

from rra_climate_aggregates import constants as cac


# Type aliases
Polygon: TypeAlias = shapely.Polygon | shapely.MultiPolygon
BBox: TypeAlias = tuple[float, float, float, float]
Bounds: TypeAlias = BBox | Polygon

class PopulationModelData:

    _modeling_frame_filename = "modeling_frame.parquet"

    def __init__(
        self,
        root: str | Path = cac.POPULATION_MODEL_ROOT,
    ) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def modeling(self) -> Path:
        return self.root / "modeling"

    def resolution_root(self, resolution: str) -> Path:
        return self.modeling / f"{resolution}m"

    def modeling_frame_path(self, resolution: str) -> Path:
        return self.resolution_root(resolution) / self._modeling_frame_filename

    def load_modeling_frame(self, resolution: str) -> gpd.GeoDataFrame:
        path = self.modeling_frame_path(resolution)
        return gpd.read_parquet(path)

    @property
    def results(self) -> Path:
        return Path(self.root, "results")

    def results_root(self, target_resolution: str) -> Path:
        return self.results / target_resolution

    def results_path(self, target_resolution: str, time_point: str) -> Path:
        return self.results_root(target_resolution) / f"{time_point}.tif"

    def load_results(self, target_resolution: str, time_point: str, **kwargs) -> rt.RasterArray:
        path = self.results_path(target_resolution, time_point)
        return rt.load_raster(path, **kwargs)


class ClimateData:

    def __init__(
        self,
        root: str | Path = cac.CLIMATE_DATA_ROOT,
    ) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def annual_results(self) -> Path:
        return self.results / "annual"

    def annual_results_path(self, scenario: str, measure: str, draw: int) -> Path:
        raw_path = self.annual_results / scenario / measure / f"{draw:>03}.nc"
        resolved = raw_path.resolve()
        gcm = resolved.stem
        actual_path = self.annual_results / "raw" / "v2" / "compiled" / scenario / measure / f"{gcm}.tif"
        return actual_path

    def load_annual_results(self, scenario: str, measure: str, draw: int) -> xr.DataSet:
        path = self.annual_results_path(scenario, measure, draw)
        return xr.open_dataset(path)


class ClimateAggregateData:

    def __init__(
        self,
        root: str | Path = cac.MODEL_ROOT,
    ) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root
