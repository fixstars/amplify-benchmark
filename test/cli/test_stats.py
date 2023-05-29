from pathlib import Path

from benchmark.cli.cli import cli


def test_stats(runner):
    input_jsons = "benchmark_20230419_102731.json"
    input_json_path = Path(__file__).parent / ".." / "data" / input_jsons

    command = [
        "stats",
        str(input_json_path),
        "-o",
        "test_stats",
    ]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0
