from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from benchmark.problem.base import gen_problem
from benchmark.problem.cvrp import Cvrp, load_cvrp_file, load_cvrp_opt_distance_and_nvehicle

from ..common import SolverSolutionSimulator as SolverSolution


def test_load_tsp_file(data):
    CVRP_DIR = data / "CVRPLIB"
    assert 22 == load_cvrp_file(CVRP_DIR / "E-n22-k4.vrp")[1]
    assert 32 == load_cvrp_file(CVRP_DIR / "A-n32-k5.vrp")[1]
    assert 45 == load_cvrp_file(CVRP_DIR / "F-n45-k4.vrp")[1]
    assert 200 == load_cvrp_file(CVRP_DIR / "M-n200-k17.vrp")[1]


def test_load_tsp_file_error(data):
    CVRP_DIR = data / "CVRPLIB"
    with pytest.raises(FileNotFoundError):
        load_cvrp_file(CVRP_DIR / "file_not_found.vrp")


def test_load_tsp_opt(data):
    CVRP_DIR = data / "CVRPLIB"
    assert 375, 4 == load_cvrp_opt_distance_and_nvehicle(CVRP_DIR / "E-n22-k4.sol")
    assert 784, 5 == load_cvrp_opt_distance_and_nvehicle(CVRP_DIR / "A-n32-k5.sol")
    assert 724, 4 == load_cvrp_opt_distance_and_nvehicle(CVRP_DIR / "F-n45-k4.sol")
    assert 1275, 17 == load_cvrp_opt_distance_and_nvehicle(CVRP_DIR / "M-n200-k17.sol")


def test_load_tsp_opt_error(data):
    CVRP_DIR = data / "CVRPLIB"
    with pytest.raises(FileNotFoundError):
        assert 3323 == load_cvrp_opt_distance_and_nvehicle("file_not_found.vrp")
    with pytest.raises(FileNotFoundError):
        assert 3323 == load_cvrp_opt_distance_and_nvehicle(CVRP_DIR / "unknown.vrp")


@pytest.mark.parametrize(
    "instance, best_known, kp",
    [
        ("E-n22-k4", 375, 0.25),
        ("E-n22-k4", 375, 1.0),
        ("A-n32-k5", 784, 0.25),
        ("F-n45-k4", 724, 0.25),
        ("M-n200-k17", 1275, 0.25),
    ],
)
def test_cvrp_problem(instance: str, best_known: float, kp: float):
    kwargs: dict = {"instance": instance, "constraint_weight": kp}
    problem = Cvrp(**kwargs)
    assert instance == problem.get_input_parameter()["instance"]
    assert best_known == problem.get_input_parameter()["best_known"]
    assert kp == problem.get_input_parameter()["parameters"]["constraint_weight"]
    assert "default" == problem.get_input_parameter()["parameters"]["method"]
    assert 2 == problem.get_input_parameter()["parameters"]["scale"]


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "A-n32-k5.vrp"
    instance = filepath.stem
    problem = Cvrp(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Cvrp", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate():
    expected: Dict[str, Any] = dict()

    # infeasible case
    problem = Cvrp(instance="E-n22-k4")
    problem.make_model()
    expected = {"label": "total distances", "value": None, "path": ""}
    assert expected == problem.evaluate(SolverSolution(is_feasible=False))

    # feasible case
    best_known = 375
    best_known_path = " 0 17 20 18 15 12 0 16 19 21 14 0 13 11 4 3 8 10 0 9 7 5 2 1 6 0"
    expected = {
        "label": "total distances",
        "value": best_known,
        "path": best_known_path,
    }

    path_list = {
        0: [0, 17, 20, 18, 15, 12, 0],
        1: [0, 16, 19, 21, 14, 0],
        2: [0, 13, 11, 4, 3, 8, 10, 0],
        3: [0, 9, 7, 5, 2, 1, 6, 0],
    }

    values = np.zeros((13, 22, 4), dtype=int)
    for v, city_list in path_list.items():
        for i, city in enumerate(city_list):
            values[i][city][v] = 1
    result = problem.evaluate(
        SolverSolution(
            is_feasible=True,
            values=values.reshape((13 * 22 * 4)).tolist(),
        )
    )
    assert expected == result
