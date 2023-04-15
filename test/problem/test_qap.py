from pathlib import Path

import numpy as np
import pytest

from benchmark.problem.base import gen_problem
from benchmark.problem.qap import Qap, load_qap_file, load_qap_opt

from ..common import SolverSolutionSimulator as SolverSolution


def test_load_qap_file(data):
    QAP_DIR = data / "QAPLIB"
    assert 12 == load_qap_file(str(QAP_DIR / "chr12a.dat"))[0]
    assert 18 == load_qap_file(str(QAP_DIR / "chr18a.dat"))[0]
    assert 32 == load_qap_file(str(QAP_DIR / "esc32a.dat"))[0]
    assert 100 == load_qap_file(str(QAP_DIR / "sko100a.dat"))[0]
    assert 150 == load_qap_file(str(QAP_DIR / "tai150b.dat"))[0]
    assert 256 == load_qap_file(str(QAP_DIR / "tai256c.dat"))[0]


def test_load_qap_file_error(data):
    QAP_DIR = data / "QAPLIB"
    with pytest.raises(FileNotFoundError):
        load_qap_file(str(QAP_DIR / "file_not_found.dat"))


def test_load_qap_opt(data):
    assert 9552 == load_qap_opt("chr12a")
    assert 11098 == load_qap_opt("chr18a")
    assert 130 == load_qap_opt("esc32a")
    assert 152002 == load_qap_opt("sko100a")
    assert 498896643 == load_qap_opt("tai150b")
    assert 44759294 == load_qap_opt("tai256c")


@pytest.mark.parametrize(
    "instance, best_known, kp",
    [
        ("chr12a", 9552, 0.25),
        ("chr12a", 9552, 0.25),
        ("chr18a", 11098, 0.25),
        ("esc32a", 130, 0.25),
        ("sko100a", 152002, 0.25),
    ],
)
def test_qap_problem(instance: str, best_known: float, kp: float):
    kwargs: dict = {"instance": instance, "constraint_weight": kp}
    problem = Qap(**kwargs)
    assert instance == problem.get_input_parameter()["instance"]
    assert best_known == problem.get_input_parameter()["best_known"]


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "chr12a.dat"
    instance = filepath.name
    problem = Qap(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Qap", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate():
    # infeasible case
    problem = Qap(instance="chr12a")
    problem.make_model()
    assert {"label": "cost", "values": None, "placement": ""} == problem.evaluate(SolverSolution(is_feasible=False))

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
        SolverSolution(is_feasible=True, values=values.reshape((12 * 12)).tolist(), energy=9552)
    )
