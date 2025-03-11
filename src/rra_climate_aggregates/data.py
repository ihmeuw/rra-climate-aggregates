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

    def __init__(
        self,
        root: str | Path = cac.POPULATION_MODEL_ROOT,
    ) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def results(self) -> Path:
        return Path(self.root, "results") / "current" / "wgs84_0p01"

    def load_results(self, time_point: str) -> rt.RasterArray:
        path = self.results / f"{time_point}.tif"
        return rt.load_raster(path)

    @property
    def raking_data(self) -> Path:
        return self.root / "admin-inputs" / "raking"

    def load_raking_shapes(self, pixel_hierarchy: str) -> gpd.GeoDataFrame:
        if pixel_hierarchy == "gbd_2021":
            shape_path = self.raking_data / f"shapes_{pixel_hierarchy}_wpp_2022.parquet"
            gdf = gpd.read_parquet(shape_path)

            # We're using population data here instead of a hierarchy because
            # The populations include extra locations we've supplemented that aren't
            # modeled in GBD (e.g. locations with zero popoulation or places that
            # GBD uses population scalars from WPP to model)
            pop_path = self.raking_data / f"population_{pixel_hierarchy}_wpp_2022.parquet"
            pop = pd.read_parquet(pop_path)

            keep_cols = ["location_id", "location_name", "most_detailed", "parent_id"]
            keep_mask = (
                (pop.year_id == pop.year_id.max())  # Year doesn't matter
                & (pop.most_detailed == 1)
            )
            out = gdf.merge(pop.loc[keep_mask, keep_cols], on="location_id", how="left")
        elif pixel_hierarchy in ["lsae_1209", "lsae_1285"]:
            # This is only a2 geoms, so already most detailed
            shape_path = (
                self.raking_data / "gbd-inputs" / f"shapes_{pixel_hierarchy}_a2.parquet"
            )
            out = gpd.read_parquet(shape_path)
        else:
            msg = f"Unknown pixel hierarchy: {pixel_hierarchy}"
            raise ValueError(msg)
        return out

    def load_hierarchy(self, admin_hierarchy: str) -> pd.DataFrame:
        allowed_hierarchies = ["gbd_2021", "fhs_2021", "lsae_1209", "lsae_1285"]
        if admin_hierarchy not in allowed_hierarchies:
            msg = f"Unknown admin hierarchy: {admin_hierarchy}"
            raise ValueError(msg)
        path = self.raking_data / "gbd-inputs" / f"hierarchy_{admin_hierarchy}.parquet"
        return pd.read_parquet(path)


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
        return self.annual_results / scenario / measure / f"{draw}.nc"

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

    @property
    def root(self) -> Path:
        return self._root

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def log_dir(self, step_name: str) -> Path:
        return self.logs / step_name

    def version_root(self, version: str) -> Path:
        return self.root / version

    def raw_results_root(self, version: str) -> Path:
        return self.version_root(version) / "raw-results"

    def raw_results_path(
        self, version: str, hierarchy: str, scenario: str, measure: str, draw: str
    ) -> Path:
        root = self.raw_results_root(version)
        return root / hierarchy / scenario / measure / f"{draw}.parquet"

    def save_raw_results(
        self,
        df: pd.DataFrame,
        version: str,
        hierarchy: str,
        scenario: str,
        measure: str,
        draw: str,
    ) -> None:
        path = self.raw_results_path(version, hierarchy, scenario, measure, draw)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        df.to_parquet(path)

    def load_raw_results(
        self, version: str, hierarchy: str, scenario: str, measure: str, draw: str
    ) -> pd.DataFrame:
        path = self.raw_results_path(version, hierarchy, scenario, measure, draw)
        return pd.read_parquet(path)

    def results_root(self, version: str) -> Path:
        return self.version_root(version) / "results"

    def population_path(self, version: str, hierarchy: str) -> Path:
        return self.results_root(version) / hierarchy / "population.parquet"

    def save_population(self, df: pd.DataFrame, version: str, hierarchy: str) -> None:
        path = self.population_path(version, hierarchy)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        df.to_parquet(path)

    def load_population(self, version: str, hierarchy: str) -> pd.DataFrame:
        path = self.population_path(version, hierarchy)
        return pd.read_parquet(path)

    def results_path(
        self, version: str, hierarchy: str, scenario: str, measure: str
    ) -> Path:
        return self.results_root(version) / hierarchy / f"{measure}_{scenario}.parquet"

    def save_results(
        self,
        df: pd.DataFrame,
        version: str,
        hierarchy: str,
        scenario: str,
        measure: str,
    ) -> None:
        path = self.results_path(version, hierarchy, scenario, measure)
        mkdir(path.parent, exist_ok=True, parents=True)
        touch(path, clobber=True)
        df.to_parquet(path)

    def load_results(
        self, version: str, hierarchy: str, scenario: str, measure: str
    ) -> pd.DataFrame:
        path = self.results_path(version, hierarchy, scenario, measure)
        return pd.read_parquet(path)

