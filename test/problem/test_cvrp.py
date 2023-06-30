from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from amplify_bench.problem.base import gen_problem
from amplify_bench.problem.cvrp import (
    Cvrp,
    get_instance_file,
    get_sol_file,
    load_cvrp_file,
    load_cvrp_opt_distance_and_nvehicle,
)


@pytest.mark.parametrize(
    "instance, ncity, best_known",
    [
        ("E-n22-k4", 22, 375),
        ("A-n32-k5", 32, 784),
        ("F-n45-k4", 45, 724),
        ("M-n200-k17", 200, 1275),
    ],
)
def test_load_cvrp_file(instance: str, ncity: int, best_known: int, data, cleanup):
    instance_file = get_instance_file(instance)

    problem_dir = data / "CVRPLIB"
    assert problem_dir.resolve() == Path(instance_file).parent.resolve()
    assert ncity == load_cvrp_file(instance_file)[1]
    cleanup(instance_file)


def test_load_cvrp_file_error(data):
    CVRP_DIR = data / "CVRPLIB"
    with pytest.raises(FileNotFoundError):
        load_cvrp_file(CVRP_DIR / "file_not_found.vrp")


@pytest.mark.parametrize(
    "instance, opt_nvehicle, opt_cost",
    [
        ("E-n22-k4", 4, 375),
        ("A-n32-k5", 5, 784),
        ("F-n45-k4", 4, 724),
        ("M-n200-k17", 17, 1275),
    ],
)
def test_load_cvrp_opt(instance: str, opt_nvehicle: int, opt_cost: int, cleanup):
    sol_file = get_sol_file(instance)
    opt_solution = load_cvrp_opt_distance_and_nvehicle(sol_file)
    assert opt_cost == opt_solution[0]
    assert opt_nvehicle == opt_solution[1]
    cleanup(sol_file)


def test_load_cvrp_opt_error(data):
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
def test_cvrp_problem(instance: str, best_known: float, kp: float, cleanup):
    kwargs: dict = {"instance": instance, "constraint_weight": kp}
    problem = Cvrp(**kwargs)
    assert instance == problem.get_input_parameter()["instance"]
    assert best_known == problem.get_input_parameter()["best_known"]
    assert kp == problem.get_input_parameter()["parameters"]["constraint_weight"]
    assert "default" == problem.get_input_parameter()["parameters"]["method"]
    assert 2 == problem.get_input_parameter()["parameters"]["scale"]

    cleanup(get_instance_file(instance))
    cleanup(get_sol_file(instance))


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "A-n32-k5.vrp"
    instance = filepath.stem
    problem = Cvrp(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Cvrp", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate(cleanup, solver_solution_mock):
    expected: Dict[str, Any] = dict()

    # infeasible case
    problem = Cvrp(instance := "E-n22-k4")
    problem.make_model()
    expected = {"label": "total distances", "value": None, "path": ""}
    assert expected == problem.evaluate(solver_solution_mock(is_feasible=False))

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
        solver_solution_mock(
            is_feasible=True,
            values=values.reshape((13 * 22 * 4)).tolist(),
        )
    )
    assert expected == result

    cleanup(get_instance_file(instance))
    cleanup(get_sol_file(instance))
