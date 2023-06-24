import json
import os
import shutil
from pathlib import Path

from benchmark.cli.cli import cli
from benchmark.cli.stats import format_result_json_to_stats_json


def test_stats_output(runner):
    input_jsons = "benchmark_20230419_102731.json"
    input_json_path = Path(__file__).parent / ".." / "data" / input_jsons

    # test case 0
    command = ["stats", str(input_json_path)]
    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    expected = Path().cwd() / "stats.json"
    assert expected.exists() is True
    os.remove(expected)

    # test case 1
    command = ["stats", str(input_json_path), "-o", "test_stats"]
    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    expected = Path().cwd() / "test_stats" / "stats.json"
    assert expected.exists() is True
    shutil.rmtree(expected.parent)

    # test case 2
    command = ["stats", str(input_json_path), "-o", "test_stats/hoge.json"]
    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    expected = Path().cwd() / "test_stats" / "hoge.json"
    assert expected.exists() is True
    shutil.rmtree(expected.parent)


def test_format_result_json_to_stats_json():
    input_json_path = Path(__file__).parent / ".." / "data" / "benchmark_20230419_102731.json"
    with open(input_json_path, mode="rt", encoding="utf-8") as file:
        data = json.load(file)
    stats = format_result_json_to_stats_json(data)
    output_json_path = Path(__file__).parent / ".." / "data" / "benchmark_20230419_102731_stats.json"
    with open(output_json_path, mode="rt", encoding="utf-8") as file:
        output = json.load(file)
    assert stats == output
