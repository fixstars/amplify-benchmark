from pathlib import Path

import pytest

from benchmark.problem.base import gen_problem
from benchmark.problem.qplib import Qplib

from ..common import SolverSolutionSimulator as SolverSolution


@pytest.mark.parametrize(
    "instance, best_known",
    [
        ("QPLIB_0067", -110942.0),
        ("QPLIB_5971", 2377.0),
        ("QPLIB_10070", -25.332849867902599),
    ],
)
def test_qplib_problem(instance: str, best_known: int):
    problem = Qplib(instance)
    assert instance == problem.get_input_parameter()["instance"]
    assert best_known == problem.get_input_parameter()["best_known"]


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "QPLIB_0067.qplib"
    instance = filepath.stem
    problem = Qplib(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Qplib", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate():
    problem = Qplib("QPLIB_0067")
    best_known = -110942
    expected = {"label": "objvar", "value": best_known}
    assert expected == problem.evaluate(SolverSolution(energy=best_known, is_feasible=True))
