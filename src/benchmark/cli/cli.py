import datetime

import click

from .benchmark import cli_benchmark
from .stats import cli_stats

start_datetime: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_json", type=str)
@click.option(
    "--label",
    "-l",
    type=str,
    default=start_datetime,
    help="Specify the label for the benchmark.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    help=(
        "Specify the directory where benchmark results are to be saved. "
        "You can also use the S3 protocol URL to save to an S3 bucket."
    ),
)
@click.option(
    "--parallel",
    "-p",
    type=int,
    default=1,
    help="Specifies the number of parallel executions.",
)
@click.option(
    "--aws-profile",
    type=str,
    help=(
        "Specify the aws profile. This option is referenced " "when using the S3 protocol with the `--output` option."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=("It even builds the QUBO model based on the input json configuration. " "It does not run on the machine."),
)
def benchmark(input_json: str, label: str, output: str, parallel: int, aws_profile: str, dry_run: bool):
    """QUBO Benchmark"""
    print(f"input_json: {input_json}")
    print(f"label: {label}")
    print(f"output: {output}")
    print(f"parallel: {parallel}")
    print(f"aws_profile: {aws_profile}")
    print(f"dry_run: {dry_run}")

    cli_benchmark(input_json, label, output, parallel, aws_profile, dry_run)


@cli.command()
@click.argument("input_jsons", nargs=-1)
@click.option(
    "--output",
    "-o",
    type=str,
    help="Specify the directory/file where stats file saved."
    "If this option is not specified, the output will be `stats.json` directly"
    "under the directory where it was executed.",
)
@click.option(
    "--aws-profile",
    type=str,
    help="Specify the aws profile. This option is referenced when using the S3 protocol with the input_jsons.",
)
def stats(input_jsons: str | list[str], output: str, aws_profile: str):
    """Generate QUBO benchmark stats data."""
    print(f"input_json: {input_jsons}")
    print(f"output: {output}")
    print(f"aws_profile: {aws_profile}")

    cli_stats(input_jsons, output, aws_profile)
