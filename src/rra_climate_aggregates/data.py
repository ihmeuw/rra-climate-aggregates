from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterra as rt
import shapely
import xarray as xr
import yaml
from rra_tools.shell_tools import mkdir, touch

from rra_climate_aggregates import constants as cac

# Type aliases
type Polygon = shapely.Polygon | shapely.MultiPolygon
type BBox = tuple[float, float, float, float]
type Bounds = BBox | Polygon


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

    def modeling_frame_path(self) -> Path:
        resolution_root = self.modeling / f"{cac.RESOLUTION}m"
        return resolution_root / self._modeling_frame_filename

    def load_modeling_frame(self) -> gpd.GeoDataFrame:
        path = self.modeling_frame_path()
        return gpd.read_parquet(path)

    def load_block_frame(self, block_key: str) -> gpd.GeoDataFrame:
        path = self.modeling_frame_path()
        filters = [("block_key", "==", block_key)]
        return gpd.read_parquet(path, filters=filters)

    @property
    def results(self) -> Path:
        return Path(self.root, "results")

    def results_path(self, block_key: str, time_point: str) -> Path:
        results_root = self.results / "current"
        spec_path = results_root / "specification.yaml"
        with spec_path.open("r") as f:
            spec = yaml.safe_load(f)
        true_root = Path(spec["output_root"])
        path = true_root / "raked_predictions" / time_point / f"{block_key}.tif"
        return path

    def load_results(self, block_key: str, time_point: str) -> rt.RasterArray:
        path = self.results_path(block_key, time_point)
        return rt.load_raster(path)

    def load_real_results(self, time_point: str) -> rt.RasterArray:
        path = self.results / "current" / "wgs84_0p01" / f"{time_point}.tif"
        return rt.load_raster(path)

    @property
    def raking_shapes(self) -> Path:
        return self.root / "admin-inputs" / "raking"

    def load_raking_shapes(
        self, hierarchy: str, bbox: shapely.Polygon | None = None
    ) -> gpd.GeoDataFrame:
        if "gbd" in hierarchy or "fhs" in hierarchy:
            shape_path = self.raking_shapes / f"shapes_{hierarchy}_wpp_2022.parquet"
            gdf = gpd.read_parquet(shape_path, bbox=bbox)
            pop_path = self.raking_shapes / f"population_{hierarchy}_wpp_2022.parquet"
            pop = pd.read_parquet(pop_path)
            year_mask = pop.year_id == 2020  # noqa: PLR2004
            keep_cols = ["location_id", "location_name", "most_detailed", "parent_id"]
            out = gdf.merge(pop.loc[year_mask, keep_cols], on="location_id", how="left")
            out = out[out.most_detailed == 1]
        else:
            assert "lsae" in hierarchy
            shape_path = (
                self.raking_shapes / "gbd-inputs" / f"shapes_{hierarchy}.parquet"
            )
            out = gpd.read_parquet(shape_path, bbox=bbox)
        return out


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

    def annual_results_path(self, scenario: str, measure: str, draw: str) -> Path:
        raw_path = self.annual_results / scenario / measure / f"{draw}.nc"
        resolved = raw_path.resolve()
        gcm = resolved.stem
        actual_path = (
            self.annual_results
            / "raw"
            / "v2"
            / "compiled"
            / scenario
            / measure
            / f"{gcm}.nc"
        )
        return actual_path

    def load_annual_results(self, scenario: str, measure: str, draw: str) -> xr.Dataset:
        path = self.annual_results_path(scenario, measure, draw)
        ds = xr.open_dataset(path, decode_coords="all")
        ds = ds.rio.write_crs("EPSG:4326")
        return ds


class ClimateAggregateData:
    def __init__(
        self,
        root: str | Path = cac.MODEL_ROOT,
    ) -> None:
        self._root = Path(root)
        self._create_model_root()

    def _create_model_root(self) -> None:
        mkdir(self.root, exist_ok=True)
        mkdir(self.logs, exist_ok=True)

        mkdir(self.raw_results, exist_ok=True)
        mkdir(self.results, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def log_dir(self, step_name: str) -> Path:
        return self.logs / step_name

    @property
    def raw_results(self) -> Path:
        return self.root / "raw-results"

    def raw_results_path(
        self, version: str, hierarchy: str, measure: str, draw: str
    ) -> Path:
        return self.raw_results / version / hierarchy / measure / f"{draw}.parquet"

    def save_raw_results(
        self,
        df: pd.DataFrame,
        version: str,
        hierarchy: str,
        measure: str,
        draw: str,
    ) -> None:
        path = self.raw_results_path(version, hierarchy, measure, draw)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        df.to_parquet(path)

    @property
    def results(self) -> Path:
        return self.root / "results"
