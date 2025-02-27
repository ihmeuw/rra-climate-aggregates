from collections.abc import Callable

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    convert_choice,
    process_choices,
    with_choice,
    with_debugger,
    with_dry_run,
    with_input_directory,
    with_num_cores,
    with_output_directory,
    with_overwrite,
    with_progress_bar,
    with_queue,
    with_verbose,
)

from rra_climate_aggregates import constants as cac


def with_version[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--version",
        help="Run version to generate.",
        required=True,
    )


def with_block_key[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--block-key",
        "-b",
        type=click.STRING,
        required=True,
        help="Block key to run.",
    )


def with_scenario[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "scenario",
        allow_all=allow_all,
        choices=cac.SCENARIOS,
        help="Scenario to process.",
        convert=allow_all,
    )


def with_measure[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "measure",
        allow_all=allow_all,
        choices=cac.MEASURES,
        help="Variable to generate.",
        convert=allow_all,
    )


def with_draw[**P, T](
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "draw",
        allow_all=allow_all,
        choices=cac.DRAWS,
        help="Draw to process.",
        convert=allow_all,
    )


__all__ = [
    "RUN_ALL",
    "convert_choice",
    "process_choices",
    "with_choice",
    "with_debugger",
    "with_dry_run",
    "with_input_directory",
    "with_num_cores",
    "with_output_directory",
    "with_overwrite",
    "with_progress_bar",
    "with_queue",
    "with_verbose",
]
