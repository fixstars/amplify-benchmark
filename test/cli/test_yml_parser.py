from itertools import product
from pathlib import Path
from unittest import TestCase

import jsonschema
import pytest

from amplify_bench.cli.parser import parse_input_data
from amplify_bench.client_config.base import get_client_config
from amplify_bench.problem.base import gen_problem


def test_yml_parser():
    # test for parse_input_data

    # test normal case
    input_filename = "benchmark.yml"
    input_path = Path(__file__).parent / ".." / "data" / input_filename
    problem_list = [
        gen_problem("Tsp", "eil51", constraint_weight=1),
        gen_problem("Tsp", "eil51", constraint_weight=2),
        gen_problem("Tsp", "eil51", constraint_weight=3),
        gen_problem("Tsp", "berlin52", constraint_weight=1),
        gen_problem("Tsp", "berlin52", constraint_weight=2),
        gen_problem("Tsp", "berlin52", constraint_weight=3),
        gen_problem("Tsp", "pr76", constraint_weight=1),
        gen_problem("Tsp", "pr76", constraint_weight=2),
        gen_problem("Tsp", "pr76", constraint_weight=3),
    ]
    client_list = [
        get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 1000}, "FixstarsClient"),
        get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 2000}, "FixstarsClient"),
        get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 3000}, "FixstarsClient"),
        get_client_config({}, {"timeout": 1000}, "GurobiClient"),
        get_client_config({}, {"timeout": 2000}, "GurobiClient"),
        get_client_config({}, {"timeout": 3000}, "GurobiClient"),
    ]
    num_samples_list = [10, 20]
    expected = list(product(problem_list, client_list, num_samples_list))

    problem_list = [
        gen_problem("Qap", "chr12a", constraint_weight=1),
        gen_problem("Qap", "chr12a", constraint_weight=2),
        gen_problem("Qap", "chr12a", constraint_weight=3),
        gen_problem("Qap", "esc32a", constraint_weight=1),
        gen_problem("Qap", "esc32a", constraint_weight=2),
        gen_problem("Qap", "esc32a", constraint_weight=3),
    ]
    client_list = [
        get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 1000}, "FixstarsClient"),
        get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 2000}, "FixstarsClient"),
        get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 3000}, "FixstarsClient"),
        get_client_config({}, {"timeout": 1000}, "GurobiClient"),
        get_client_config({}, {"timeout": 2000}, "GurobiClient"),
        get_client_config({}, {"timeout": 3000}, "GurobiClient"),
    ]
    num_samples_list = [10, 20]
    expected.extend(list(product(problem_list, client_list, num_samples_list)))
    case = TestCase()
    ret = parse_input_data(input_path)
    case.assertCountEqual(ret, expected)

    # test no matrix case
    input_filename = "benchmark_without_matrix.yml"
    input_path = Path(__file__).parent / ".." / "data" / input_filename
    expected = [
        (
            gen_problem("Tsp", "eil51", constraint_weight=1),
            get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 1000}, "FixstarsClient"),
            10,
        ),
        (
            gen_problem("Tsp", "eil51", constraint_weight=1),
            get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 2000}, "FixstarsClient"),
            10,
        ),
        (
            gen_problem("Tsp", "eil51", constraint_weight=2),
            get_client_config({"url": "https://HOGEHOGE", "token": "aiueo"}, {"timeout": 2000}, "FixstarsClient"),
            10,
        ),
    ]
    case = TestCase()
    ret = parse_input_data(input_path)
    case.assertCountEqual(ret, expected)

    # test circular reference
    with pytest.raises(ValueError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_circular_reference.yml")
    assert str(e.value) == "detect circular reference in input data"

    # test empty file
    with pytest.raises(jsonschema.exceptions.ValidationError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_empty_input.yml")

    # test invalid file
    with pytest.raises(jsonschema.exceptions.ValidationError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_invalid_without_problem.yml")

    with pytest.raises(jsonschema.exceptions.ValidationError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_invalid_without_client.yml")

    with pytest.raises(jsonschema.exceptions.ValidationError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_invalid_without_numsample.yml")

    with pytest.raises(jsonschema.exceptions.ValidationError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_invalid_client.yml")

    with pytest.raises(jsonschema.exceptions.ValidationError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_invalid_client_variable.yml")

    # test KeyError
    with pytest.raises(KeyError) as _:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_ref.yml")

    # test invalid file extension
    with pytest.raises(ValueError) as e:
        parse_input_data(Path(__file__).parent / ".." / "data" / "error_invalid_file_extension.txt")

        assert str(e.value) == "invalid file extension: .txt must be .json or .yml or .yaml"

    # test variable
    input_filename = "variable.yml"
    input_path = Path(__file__).parent / ".." / "data" / input_filename
    expected = [
        (
            gen_problem("Tsp", "burma14"),
            get_client_config({}, {"outputs": {"feasibilities": True}, "timeout": 1000}, "FixstarsClient"),
            100,
        ),
        (
            gen_problem("Tsp", "ulysses16"),
            get_client_config({}, {"outputs": {"feasibilities": True}, "timeout": 3000}, "FixstarsClient"),
            100,
        ),
        (
            gen_problem("Tsp", "bayg29"),
            get_client_config({}, {"outputs": {"feasibilities": True}, "timeout": 3000}, "FixstarsClient"),
            100,
        ),
    ]
    case = TestCase()
    ret = parse_input_data(input_path)
    case.assertCountEqual(ret, expected)

    # deprecated matrix
    input_filename = "deprecated_matrix.yml"
    input_path = Path(__file__).parent / ".." / "data" / input_filename
    expected = [
        (
            gen_problem("MaxCut", "G1"),
            get_client_config({}, {"timeout": 1000}, "FixstarsClient"),
            1,
        ),
        (
            gen_problem("MaxCut", "G1"),
            get_client_config({}, {"timeout": 2000}, "FixstarsClient"),
            1,
        ),
        (
            gen_problem("MaxCut", "G11"),
            get_client_config({}, {"timeout": 1000}, "FixstarsClient"),
            1,
        ),
        (
            gen_problem("MaxCut", "G11"),
            get_client_config({}, {"timeout": 2000}, "FixstarsClient"),
            1,
        ),
        (
            gen_problem("MaxCut", "G1"),
            get_client_config(
                {"token": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, {"time_limit_sec": 1}, "FujitsuDA3SolverClient"
            ),
            1,
        ),
        (
            gen_problem("MaxCut", "G1"),
            get_client_config(
                {"token": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, {"time_limit_sec": 2}, "FujitsuDA3SolverClient"
            ),
            1,
        ),
        (
            gen_problem("MaxCut", "G11"),
            get_client_config(
                {"token": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, {"time_limit_sec": 1}, "FujitsuDA3SolverClient"
            ),
            1,
        ),
        (
            gen_problem("MaxCut", "G11"),
            get_client_config(
                {"token": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, {"time_limit_sec": 2}, "FujitsuDA3SolverClient"
            ),
            1,
        ),
    ]
    case = TestCase()
    ret = parse_input_data(input_path)
    case.assertCountEqual(ret, expected)
