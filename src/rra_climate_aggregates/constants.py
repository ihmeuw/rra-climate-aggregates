from pathlib import Path

RRA_ROOT = Path("/mnt/team/rapidresponse/pub/")
POPULATION_MODEL_ROOT = RRA_ROOT / "population-model"
MODEL_ROOT = RRA_ROOT / "climate-aggregates"

CLIMATE_DATA_ROOT = Path("/mnt/share/erf/climate_downscale")

RESOLUTION = "100"
TARGET_RESOLUTION = f"world_cylindrical_{RESOLUTION}"

SCENARIOS = [
    "ssp126",
    "ssp245",
    "ssp585",
]

MEASURES = [
    "mean_temperature",
    "mean_high_temperature",
    "mean_low_temperature",
    "days_over_30C",
    "malaria_suitability",
    "dengue_suitability",
    "wind_speed",
    "relative_humidity",
    "total_precipitation",
    "precipitation_days",
]

DRAWS = [f"{d:>03}" for d in range(100)]

# Mapping between pixel aggregation hierarchies to location aggregation hierarchies.
# The pixel aggregation hierarchies are the most detailed shapes used to
# aggregate the pixel data to the location level.
# The location aggregation hierarchies correspond to particular modeling datasets.
# Their most-detailed shapes may be a subset of the pixel aggregation hierarchy they
# map to. They are used to produce results from the most detailed level up to the global
# level.
HIERARCHY_MAP = {
    "gbd_2021": ["gbd_2021", "fhs_2021"],
    "lsae_1209": ["lsae_1209"],
}
