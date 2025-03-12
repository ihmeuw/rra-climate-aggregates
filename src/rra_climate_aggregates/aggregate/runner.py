import itertools

import click
import numpy as np
import pandas as pd
import tqdm
from rra_tools import jobmon

from rra_climate_aggregates import cli_options as clio
from rra_climate_aggregates import constants as cac
from rra_climate_aggregates.aggregate import utils
from rra_climate_aggregates.data import (
    ClimateAggregateData,
    ClimateData,
    PopulationModelData,
)
from rra_climate_aggregates.utils import to_raster


def aggregate_main(
    version: str,
    scenario: str,
    measure: str,
    draw: str,
    hierarchy: str,
    population_model_root: str,
    climate_data_root: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
) -> None:
    print("Aggregating", scenario, measure, draw)
    pm_data = PopulationModelData(population_model_root)
    cd_data = ClimateData(climate_data_root)
    ca_data = ClimateAggregateData(output_dir)

    print("Loading climate data")
    ds = cd_data.load_annual_results(scenario, measure, draw)

    subset_hierarchies = cac.HIERARCHY_MAP[hierarchy]

    print("Processing", hierarchy)
    print("Building location masks")
    bounds_map, mask = utils.build_location_masks(hierarchy, pm_data)
    results = []
    print(f"Aggregating data with {len(bounds_map)} locations")
    for year in tqdm.trange(1950, 2101, disable=not progress_bar):
        pop_raster = pm_data.load_results(f"{year}q1")
        pop_arr = pop_raster._ndarray  # noqa: SLF001
        clim_var = ds.sel(year=year)["value"]
        clim_raster = (
            to_raster(clim_var).resample_to(pop_raster, "nearest").astype(np.float32)
        )
        pop_weighted = pop_raster * clim_raster
        pop_weighted_arr = pop_weighted._ndarray  # noqa: SLF001

        for location_id, (rows, cols) in tqdm.tqdm(
            list(bounds_map.items()), disable=not progress_bar
        ):
            loc_mask = mask[rows, cols] == location_id
            loc_clim = pop_weighted_arr[rows, cols]
            loc_pop = pop_arr[rows, cols]
            wc = np.nansum(loc_clim[loc_mask])
            pop = np.nansum(loc_pop[loc_mask])
            r = wc / pop if pop else np.nan
            results.append((location_id, year, scenario, wc, pop, r))

    h_results = pd.DataFrame(
        results,
        columns=[
            "location_id",
            "year_id",
            "scenario",
            "weighted_climate",
            "population",
            "value",
        ],
    ).sort_values(by=["location_id", "year_id"])

    agg_h = pm_data.load_hierarchy(hierarchy)
    if scenario == "ssp245" and measure == "mean_temperature" and draw == "000":
        pop = utils.aggregate_pop_to_hierarchy(h_results, agg_h)
        for subset_hierarchy in subset_hierarchies:
            subset_h = pm_data.load_hierarchy(subset_hierarchy)
            subset_pop = pop[pop.location_id.isin(subset_h.location_id)]
            ca_data.save_population(subset_pop, version, subset_hierarchy)

    climate = utils.aggregate_climate_to_hierarchy(h_results, agg_h)
    for subset_hierarchy in subset_hierarchies:
        subset_h = pm_data.load_hierarchy(subset_hierarchy)
        subset_climate = climate[climate.location_id.isin(subset_h.location_id)]
        ca_data.save_raw_results(
            subset_climate,
            version,
            subset_hierarchy,
            scenario,
            measure,
            draw,
        )


@click.command()
@clio.with_version()
@clio.with_scenario()
@clio.with_measure()
@clio.with_draw()
@clio.with_hierarchy()
@clio.with_input_directory("population-model", cac.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cac.CLIMATE_DATA_ROOT)
@clio.with_output_directory(cac.MODEL_ROOT)
@clio.with_progress_bar()
def aggregate_task(
    version: str,
    scenario: str,
    measure: str,
    draw: str,
    hierarchy: str,
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    aggregate_main(
        version,
        scenario,
        measure,
        draw,
        hierarchy,
        population_model_dir,
        climate_data_dir,
        output_dir,
        progress_bar=progress_bar,
    )


@click.command()
@clio.with_version()
@clio.with_scenario(allow_all=True)
@clio.with_measure(allow_all=True)
@clio.with_draw(allow_all=True)
@clio.with_hierarchy(allow_all=True)
@clio.with_input_directory("population-model", cac.POPULATION_MODEL_ROOT)
@clio.with_input_directory("climate-data", cac.CLIMATE_DATA_ROOT)
@clio.with_output_directory(cac.MODEL_ROOT)
@clio.with_queue()
def aggregate(
    version: str,
    scenario: list[str],
    measure: list[str],
    draw: list[str],
    hierarchy: list[str],
    population_model_dir: str,
    climate_data_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    ca_data = ClimateAggregateData(output_dir)

    jobs = []
    for s, m, j, h in itertools.product(scenario, measure, draw, hierarchy):
        if not ca_data.raw_results_path(version, h, s, m, j).exists():
            jobs.append((s, m, j, h))
    jobs = list(set(jobs))

    print(f"Running {len(jobs)} jobs")

    jobmon.run_parallel(
        runner="catask",
        task_name="aggregate",
        flat_node_args=(
            ("scenario", "measure", "draw", "hierarchy"),
            jobs,
        ),
        task_args={
            "version": version,
            "population-model-dir": population_model_dir,
            "climate-data-dir": climate_data_dir,
            "output-dir": output_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "30G",
            "runtime": "400m",
            "project": "proj_rapidresponse",
        },
        log_root=ca_data.log_dir("aggregate"),
        max_attempts=3,
    )
