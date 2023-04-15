from pathlib import Path

import pytest

from benchmark.problem.base import gen_problem
from benchmark.problem.maxcut import MaxCut, load_gset_matrix, load_gset_opt

from ..common import SolverSolutionSimulator as SolverSolution


def test_load_gset_matrix(data):
    MAXCUT_DIR = data / "GSET"
    assert 800 * 800 == load_gset_matrix(MAXCUT_DIR / "G1").size
    assert 800 * 800 == load_gset_matrix(MAXCUT_DIR / "G11").size


def test_load_gset_matrix_error(data):
    MAXCUT_DIR = data / "GSET"
    with pytest.raises(FileNotFoundError):
        load_gset_matrix(MAXCUT_DIR / "G0")


def test_load_gset_opt():
    assert -11624 == load_gset_opt("G1")
    assert -564 == load_gset_opt("G11")


@pytest.mark.parametrize(
    "instance, best_known",
    [
        ("G1", -11624),
        ("G11", -564),
        ("G21", -931),
    ],
)
def test_maxcut_problem(instance: str, best_known: int):
    problem = MaxCut(instance)
    assert instance == problem.get_input_parameter()["instance"]
    assert best_known == problem.get_input_parameter()["best_known"]


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "G1"
    instance = filepath.stem
    problem = MaxCut(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("MaxCut", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate():
    problem = MaxCut("G1")
    best_known = 11624
    expected = {"label": "Maximum Cuts", "value": best_known}
    assert expected == problem.evaluate(SolverSolution(energy=-best_known))
