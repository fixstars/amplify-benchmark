from pathlib import Path

import numpy as np
import pytest

from amplify_bench.problem.base import gen_problem
from amplify_bench.problem.qap import Qap, get_instance_file, get_sol_file, load_qap_file, load_qap_opt


def test_load_qap_file_error(data):
    QAP_DIR = data / "QAPLIB"
    with pytest.raises(FileNotFoundError):
        load_qap_file(str(QAP_DIR / "file_not_found.dat"))


@pytest.mark.parametrize(
    "instance, n, best_known",
    [
        ("chr12a", 12, 9552),
        ("chr18a", 18, 11098),
        ("esc32a", 32, 130),
        ("esc32e", 32, 2),
        ("sko100a", 100, 152002),
        ("tai150b", 150, 498896643),
        ("tai256c", 256, 44759294),
    ],
)
def test_load_qap(instance, n, best_known, cleanup):
    instance_file = get_instance_file(instance)
    N, dist, flow = load_qap_file(instance_file)
    assert n == N

    # test for load opt solution
    sol_file = get_sol_file(instance)
    load_best_known = load_qap_opt(sol_file)

    assert best_known == load_best_known

    # test for problem class
    kwargs: dict = {"instance": instance}
    problem = Qap(**kwargs)
    assert instance == problem.get_input_parameter()["instance"]
    assert best_known == problem.get_input_parameter()["best_known"]

    cleanup(instance_file)
    if instance != "esc32a":
        cleanup(sol_file)


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "qap_10.dat"
    instance = filepath.name
    problem = Qap(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Qap", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate(cleanup, solver_solution_mock):
    # infeasible case
    problem = Qap(instance := "chr12a")
    problem.make_model()
    assert {"label": "cost", "values": None, "placement": ""} == problem.evaluate(
        solver_solution_mock(is_feasible=False)
    )

    # feasible case
    best_known = 9552
    best_known_sol = "6 4 11 1 0 2 8 10 9 5 7 3"
    expected = {"label": "cost", "values": best_known, "placement": best_known_sol}

    solution_list = list(map(lambda x: int(x), best_known_sol.split(" ")))
    assert [6, 4, 11, 1, 0, 2, 8, 10, 9, 5, 7, 3] == solution_list
    values = np.zeros((12, 12), dtype=int)
    for i, city in enumerate(solution_list):
        values[i][city] = 1

    assert expected == problem.evaluate(
        solver_solution_mock(is_feasible=True, values=values.reshape((12 * 12)).tolist(), energy=9552)
    )

    cleanup(get_instance_file(instance))
    cleanup(get_sol_file(instance))
