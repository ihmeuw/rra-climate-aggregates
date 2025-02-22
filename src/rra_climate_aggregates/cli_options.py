from collections.abc import Collection
from typing import ParamSpec, TypeVar

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
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

__all__ = [
    "RUN_ALL",
    "ClickOption",
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
