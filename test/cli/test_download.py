import re

import pytest

from benchmark.cli.cli import cli
from benchmark.cli.download_all import cli_download_clean


@pytest.mark.slow
def test_cvrp_download(runner, data):
    input_class = "Cvrp"
    command = ["download", input_class]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    # downloadコマンド実行後に、downloadしたファイルが存在するか確認する
    instance_dir = data / "CVRPLIB"

    file_list = list(instance_dir.glob("*.vrp"))

    # instance_listに含まれるインスタンスがdownloadされていることを確認する
    with open(instance_dir / "instance_list.txt", "r") as f:
        instance_set = set(line.strip() for line in f.readlines())
        assert set(file.stem for file in file_list) == instance_set

    # downloadしたファイルを削除する
    result = runner.invoke(cli, ["download", "--clean", input_class])
    assert result.exit_code == 0

    assert len(list(instance_dir.glob("*.vrp"))) == 0


def test_cvrp_clean(runner, data):
    instance_dir = data / "CVRPLIB"

    # instance_dir に空のファイルを作成する
    (instance_dir / "hoge.vrp").touch()
    assert len(list(instance_dir.glob("*.vrp"))) != 0

    input_class = "Cvrp"
    result = runner.invoke(cli, ["download", "--clean", input_class])
    assert result.exit_code == 0

    assert len(list(instance_dir.glob("*.vrp"))) == 0


@pytest.mark.slow
def test_maxcut_download(runner, data):
    input_class = "MaxCut"
    command = ["download", input_class]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    # downloadコマンド実行後に、downloadしたファイルが存在するか確認する
    instance_dir = data / "GSET"

    # ファイル名にドットを含まないファイルのみを取得する
    pattern = r"^[^.]*$"
    file_list = [p for p in instance_dir.iterdir() if re.search(pattern, str(p.name))]
    assert len(file_list) != 0

    # instance_listに含まれるインスタンスがdownloadされていることを確認する
    with open(instance_dir / "instance_list.txt", "r") as f:
        instance_list = [line.strip() for line in f.readlines()]

        for file in file_list:
            assert file.stem in instance_list

    # downloadしたファイルを削除する
    result = runner.invoke(cli, ["download", "--clean", input_class])
    assert result.exit_code == 0

    file_list = [p for p in instance_dir.iterdir() if re.search(pattern, str(p))]
    assert len(file_list) == 0


def test_maxcut_clean(runner, data):
    instance_dir = data / "GSET"

    # instance_dir に空のファイルを作成する
    (instance_dir / "hoge").touch()

    # ファイル名にドットを含まないファイルのみを取得する
    pattern = r"^[^.]*$"
    file_list = [p for p in instance_dir.iterdir() if re.search(pattern, str(p.name))]
    assert len(file_list) != 0

    input_class = "MaxCut"
    result = runner.invoke(cli, ["download", "--clean", input_class])
    assert result.exit_code == 0

    file_list = [p for p in instance_dir.iterdir() if re.search(pattern, str(p))]
    assert len(file_list) == 0


@pytest.mark.slow
def test_qap_download(runner, data):
    input_class = "Qap"
    command = ["download", input_class]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    # downloadコマンド実行後に、downloadしたファイルが存在するか確認する
    instance_dir = data / "QAPLIB"
    file_list = list(instance_dir.glob("*.dat"))

    # instance_listに含まれるインスタンスがdownloadされていることを確認する
    with open(instance_dir / "instance_list.txt", "r") as f:
        instance_set = set(line.strip() for line in f.readlines())
        assert set(file.stem for file in file_list) == instance_set

    # downloadしたファイルを削除する
    cli_download_clean(input_class)
    assert len(list(instance_dir.glob("*.dat"))) == 0


def test_qap_clean(runner, data):
    # instance_dir に空のファイルを作成する
    instance_dir = data / "QAPLIB"
    (instance_dir / "hoge.dat").touch()
    assert len(list(instance_dir.glob("*.dat"))) != 0

    input_class = "Qap"
    result = runner.invoke(cli, ["download", "--clean", input_class])
    assert result.exit_code == 0

    assert len(list(instance_dir.glob("*.dat"))) == 0


@pytest.mark.slow
def test_qplib_download(runner, data):
    input_class = "Qplib"
    command = ["download", input_class]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    # downloadコマンド実行後に、downloadしたファイルが存在するか確認する
    instance_dir = data / "QPLIB"
    file_list = list(instance_dir.glob("*.qplib"))
    assert len(file_list) != 0

    # instance_listに含まれるインスタンスがdownloadされていることを確認する
    with open(instance_dir / "instance_list.txt", "r") as f:
        instance_set = set(line.strip() for line in f.readlines())
        assert set(file.stem for file in file_list) == instance_set

    # downloadしたファイルを削除する
    cli_download_clean(input_class)
    assert len(list(instance_dir.glob("*.qplib"))) == 0


def test_qplib_clean(runner, data):
    # instance_dir に空のファイルを作成する
    instance_dir = data / "QPLIB"
    (instance_dir / "hoge.qplib").touch()
    assert len(list(instance_dir.glob("*.qplib"))) != 0

    input_class = "Qplib"
    result = runner.invoke(cli, ["download", "--clean", input_class])
    assert result.exit_code == 0

    assert len(list(instance_dir.glob("*.qplib"))) == 0


@pytest.mark.slow
def test_tsp_download(runner, data):
    input_class = "Tsp"
    command = ["download", input_class]

    result = runner.invoke(cli, command)
    assert result.exit_code == 0

    # downloadコマンド実行後に、downloadしたファイルが存在するか確認する
    instance_dir = data / "TSPLIB"
    file_list = list(instance_dir.glob("*.tsp"))
    assert len(file_list) != 0

    # instance_listに含まれるインスタンスがdownloadされていることを確認する
    with open(instance_dir / "instance_list.txt", "r") as f:
        instance_set = set(line.strip() for line in f.readlines())
        assert set(file.stem for file in file_list) == instance_set

    # downloadしたファイルを削除する
    cli_download_clean(input_class)
    assert len(list(instance_dir.glob("*.tsp"))) == 0


def test_tsp_clean(runner, data):
    # instance_dir に空のファイルを作成する
    instance_dir = data / "TSPLIB"
    (instance_dir / "hoge.tsp").touch()
    assert len(list(instance_dir.glob("*.tsp"))) != 0

    input_class = "Tsp"
    result = runner.invoke(cli, ["download", "--clean", input_class])
    assert result.exit_code == 0

    assert len(list(instance_dir.glob("*.tsp"))) == 0
