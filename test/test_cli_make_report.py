import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import jsonschema

from benchmark.cli import cli_make_report, cli_make_report_impl


def test_cli_make_report_impl():
    # benchmark make_report input_sample.json -o 'sample_output'
    input_jsons = [
        str(Path(__file__).parent / "data" / "input_ae_20230310_172902.json"),
    ]

    with TemporaryDirectory() as dname:
        output_path = Path(dname) / "sample_output"

        cli_make_report_impl(input_jsons, str(output_path), None)

        with open(Path(__file__).parent / ".." / "src" / "benchmark" / "cli" / "schemas" / "report_data.json") as f:
            json_schema = json.load(f)
        with open(output_path / "data" / "data.json") as f:
            instnance = json.load(f)
        jsonschema.validate(instnance, json_schema)


def test_cli_make_report():
    input_jsons = [
        str(Path(__file__).parent / "data" / "input_ae_20230310_172902.json"),
    ]
    with TemporaryDirectory() as dname:
        output_path = Path(dname) / "sample_output"
        output_path.mkdir(mode=0o755, parents=True, exist_ok=True)

        with patch(
            "sys.argv",
            ["make_report", input_jsons[0], "-o", str(output_path)],
        ):
            cli_make_report()

        with open(
            Path(__file__).parent / ".." / "src" / "benchmark" / "cli" / "schemas" / "report_data.json", "r"
        ) as f:
            json_schema = json.load(f)
        with open(output_path / "data" / "data.json", "r") as f:
            instnance = json.load(f)
        jsonschema.validate(instnance, json_schema)
