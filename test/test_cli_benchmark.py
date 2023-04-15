from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from benchmark.cli import cli_benchmark, parse_args


def test_parse_args():
    input_json_path = "/input/cli_input.json"
    test_label = "test_label"
    output_path = "/hoge/fuga"
    n_parallel = 4

    args = parse_args(
        [
            input_json_path,
            "--label",
            test_label,
            "-o",
            output_path,
            "-p",
            str(n_parallel),
            "--dry-run",
        ]
    )

    assert args.input_json == input_json_path
    assert args.label == test_label
    assert args.output == output_path
    assert args.parallel == n_parallel
    assert args.dry_run


@pytest.mark.parametrize(
    "input_json_filename",
    [
        "input_ae.json",
        "input_nec.json",
        "input_sqbm.json",
        "input_da4_light_benchmark.json",
    ],
)
def test_cli_benchmark_dry_run(input_json_filename: str):
    # benchmark -l 'test run' input_sample.json -o 'test_output' --dry-run
    input_json_path = Path(__file__).parent.parent / "test" / "data" / input_json_filename
    with patch(
        "sys.argv",
        ["benchmark", "-l", "test run", str(input_json_path), "-o", "test_output", "--dry-run"],
    ):
        cli_benchmark()


@pytest.mark.skip
@pytest.mark.parametrize(
    "input_json_filename",
    [
        "input_ae.json",
    ],
)
def test_cli_benchmark(input_json_filename: str):
    # benchmark -l 'test run' input_sample.json -o 'test_output' --dry-run
    input_json_path = Path(__file__).parent.parent / "test" / "data" / input_json_filename

    with TemporaryDirectory() as dname:
        output_path = Path(dname) / "test_output"

        with patch(
            "sys.argv",
            ["benchmark", "-l", "test run", str(input_json_path), "-o", str(output_path)],
        ):
            output_path.mkdir(mode=0o755, parents=True, exist_ok=True)
            cli_benchmark()

        with patch(
            "sys.argv",
            ["benchmark", "-l", "test run", str(input_json_path), "-o", str(output_path), "-p", "4"],
        ):
            output_path.mkdir(mode=0o755, parents=True, exist_ok=True)
            cli_benchmark()

        # check file
        file_list = list(output_path.glob("*"))
        assert len(file_list) == 2
