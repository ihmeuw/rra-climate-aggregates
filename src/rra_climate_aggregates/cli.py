import click

from rra_climate_aggregates import (
    aggregate,
)


@click.group()
def carun() -> None:
    """Run a stage of the population modeling pipeline."""


@click.group()
def catask() -> None:
    """Run an individual modeling task in the population modeling pipeline."""


for module in [aggregate]:
    runner = getattr(module, "RUNNER", None)
    task_runner = getattr(module, "TASK_RUNNER", None)

    if not runner or not task_runner:
        continue

    command_name = module.__name__.split(".")[-1]

    carun.add_command(runner, command_name)
    catask.add_command(task_runner, command_name)
