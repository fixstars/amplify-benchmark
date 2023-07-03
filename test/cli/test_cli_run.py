from pathlib import Path

import pytest

from amplify_bench.cli.cli import cli


def test_benchmark_dry_run(runner):
    input_json_filename = "benchmark.yml"
    input_json_path = Path(__file__).parent / ".." / "data" / input_json_filename

    command = [
        "run",
        "-l",
        "test run",
        str(input_json_path),
        "-o",
        "test_output",
        "--dry-run",
    ]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0


@pytest.mark.slow
def test_benchmark(runner):
    input_json_filename = "benchmark.yml"
    input_json_path = Path(__file__).parent / ".." / "data" / input_json_filename

    command = [
        "run",
        "-l",
        "test run",
        str(input_json_path),
        "-o",
        "test_output",
    ]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0
