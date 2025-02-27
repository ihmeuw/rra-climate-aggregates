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

SHAPE_HIERARCHIES = [
    # "gbd_2021",
    "lsae_a2",
]

AGG_HIERARCHIES = [
    "gbd_2021",
    "fhs_2021",
    "lsae",
]
