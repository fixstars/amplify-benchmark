import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from benchmark.problem.base import gen_problem
from benchmark.problem.tsp import (
    Tsp,
    calc_tour_dist,
    calc_tour_dist_from_problem,
    gen_random_tsp_instance,
    load_tsp_file,
    load_tsp_opt_distance,
    load_tsp_opt_tour,
)

from ..common import SolverSolutionSimulator as SolverSolution


def test_load_tsp_file(data):
    TSP_DIR = data / "TSPLIB"
    assert 14 == load_tsp_file(str(TSP_DIR / "burma14.tsp"))[0]
    assert 16 == load_tsp_file(str(TSP_DIR / "ulysses16.tsp"))[0]
    assert 29 == load_tsp_file(str(TSP_DIR / "bayg29.tsp"))[0]
    assert 51 == load_tsp_file(str(TSP_DIR / "eil51.tsp"))[0]


def test_load_tsp_file_error(data):
    TSP_DIR = data / "TSPLIB"
    with pytest.raises(FileNotFoundError):
        load_tsp_file(str(TSP_DIR / "file_not_found.tsp"))
    with pytest.raises(RuntimeError):
        load_tsp_file(str(TSP_DIR / "pr2392.tsp"))


def test_load_tsp_opt(data):
    assert 3323 == load_tsp_opt_distance("burma14")
    assert 6859 == load_tsp_opt_distance("ulysses16")
    assert 1610 == load_tsp_opt_distance("bayg29")
    assert 426 == load_tsp_opt_distance("eil51")


@pytest.mark.parametrize(
    "instance, best_known, kp",
    [
        ("burma14", 3323, 0.25),
        ("burma14", 3323, 1.0),
        ("ulysses16", 6859, 0.25),
        ("bayg29", 1610, 0.25),
        ("eil51", 426, 0.25),
    ],
)
def test_tsp_problem(instance: str, best_known: float, kp: float):
    kwargs: dict = {"instance": instance, "constraint_weight": kp}
    problem = Tsp(**kwargs)
    assert instance == problem.get_input_parameter()["instance"]
    assert kp == problem.get_input_parameter()["parameters"]["constraint_weight"]

    problem.make_model()  # call `make_model()` for loading TSPLIB file
    assert best_known == problem.get_input_parameter()["best_known"]


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
    filepath = Path(__file__).parent / "data" / "burma14.tsp"
    instance = filepath.stem
    problem = Tsp(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Tsp", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate():
    expected: Dict[str, Any] = dict()

    # infeasible case
    problem = Tsp(instance="burma14")
    problem.make_model()
    expected = {"label": "total distances", "value": None, "path": ""}
    assert expected == problem.evaluate(SolverSolution(is_feasible=False))

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
    result = problem.evaluate(SolverSolution(is_feasible=True, values=values.reshape((14 * 14)).tolist()))
    assert expected == result


@pytest.mark.parametrize(
    "instance",
    ["ulysses16", "bayg29", "eil51", "kroA100", "gr96", "gr202"],
)
def test_calc_tour_dist(instance: str, data):
    TSP_DIR = data / "TSPLIB"
    problem_file = str(TSP_DIR / (instance + ".tsp"))
    sol_file = str(TSP_DIR / (instance + ".opt.tour"))
    ncity_ = int(re.sub(r"[^0-9]", "", instance))
    ncity, distances, locations = load_tsp_file(problem_file)
    assert ncity_ == ncity
    assert ncity_ == distances.shape[0]

    tour, dist = load_tsp_opt_tour(str(problem_file), str(sol_file))
    opt_dist = load_tsp_opt_distance(instance)

    assert len(tour) == ncity
    assert dist == opt_dist

    tour_0_index = [i - 1 for i in tour]
    assert dist == calc_tour_dist(tour_0_index, distances)
    assert opt_dist == calc_tour_dist(tour_0_index, distances)

    assert dist == calc_tour_dist_from_problem(tour, problem_file)
    assert opt_dist == calc_tour_dist_from_problem(tour, problem_file)
