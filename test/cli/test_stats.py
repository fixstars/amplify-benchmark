import os
import shutil
from pathlib import Path

from benchmark.cli.cli import cli


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
