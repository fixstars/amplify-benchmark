from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import requests

from amplify_bench.problem.base import gen_problem
from amplify_bench.problem.tsp import (
    Tsp,
    gen_random_tsp_instance,
    get_instance_file,
    load_tsp_file,
    load_tsp_opt_distance,
)


def test_load_tsp_file_error(data, cleanup):
    TSP_DIR = data / "TSPLIB"
    with pytest.raises(FileNotFoundError):
        load_tsp_file(str(TSP_DIR / "file_not_found.tsp"))
    with pytest.raises(RuntimeError):
        instance_file = get_instance_file("pr2392")
        load_tsp_file(instance_file)
        cleanup(instance_file)
    with pytest.raises(requests.HTTPError):
        get_instance_file("hoge")


@pytest.mark.parametrize(
    "instance, n, best_known",
    [
        ("burma14", 14, 3323),
        ("ulysses16", 16, 6859),
        ("bayg29", 29, 1610),
        ("eil51", 51, 426),
        ("kroA100", 100, 21282),
    ],
)
def test_load_tsp_file(instance: str, n: int, best_known: int, cleanup):
    instance_file = get_instance_file(instance)
    assert n == load_tsp_file(instance_file)[0]
    assert best_known == load_tsp_opt_distance(instance)

    kwargs: dict = {"instance": instance, "constraint_weight": 0.5}
    problem = Tsp(**kwargs)
    assert instance == problem.get_input_parameter()["instance"]
    assert 0.5 == problem.get_input_parameter()["parameters"]["constraint_weight"]

    problem.make_model()  # call `make_model()` for loading TSPLIB file
    assert best_known == problem.get_input_parameter()["best_known"]

    cleanup(instance_file)


def test_gen_random_tsp_instance():
    assert 10 == gen_random_tsp_instance("random10")[0]
    assert 20 == gen_random_tsp_instance("random20")[0]
    assert 30 == gen_random_tsp_instance("random30")[0]

    with pytest.raises(RuntimeError):
        gen_random_tsp_instance("hoge10")

    fixed_seed_instance_list = [gen_random_tsp_instance("random10", seed=5) for _ in range(2)]
    assert fixed_seed_instance_list[0][0] == fixed_seed_instance_list[1][0]
    assert np.all(fixed_seed_instance_list[0][1] == fixed_seed_instance_list[1][1])
    assert np.all(fixed_seed_instance_list[0][2] == fixed_seed_instance_list[1][2])


@pytest.mark.parametrize(
    "instance, seed",
    [
        ("random10", 0),
        ("random20", 1),
        ("random30", 2),
        ("random40", 3),
        ("random50", 4),
    ],
)
def test_random_tsp_problem(instance: str, seed: int):
    kwargs: dict = {"instance": instance, "seed": seed}
    problem = Tsp(**kwargs)
    assert instance == problem.get_input_parameter()["instance"]
    assert seed == problem.get_input_parameter()["parameters"]["seed"]


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "rand_10.tsp"
    instance = filepath.stem
    problem = Tsp(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Tsp", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate(cleanup, solver_solution_mock):
    expected: Dict[str, Any] = dict()

    # infeasible case
    problem = Tsp(instance := "burma14")
    problem.make_model()
    expected = {"label": "total distances", "value": None, "path": ""}
    assert expected == problem.evaluate(solver_solution_mock(is_feasible=False))

    # feasible case
    best_known = 3323
    best_known_path = "6 12 7 10 8 9 0 1 13 2 3 4 5 11"
    expected = {
        "label": "total distances",
        "value": best_known,
        "path": best_known_path,
    }

    path_list = list(map(lambda x: int(x), best_known_path.split(" ")))
    assert [6, 12, 7, 10, 8, 9, 0, 1, 13, 2, 3, 4, 5, 11] == path_list
    values = np.zeros((14, 14), dtype=int)
    for i, city in enumerate(path_list):
        values[i][city] = 1
    result = problem.evaluate(solver_solution_mock(is_feasible=True, values=values.reshape((14 * 14)).tolist()))
    assert expected == result

    cleanup(get_instance_file(instance))
