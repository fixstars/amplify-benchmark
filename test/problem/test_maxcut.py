from pathlib import Path

import pytest

from benchmark.problem.base import gen_problem
from benchmark.problem.maxcut import MaxCut, get_instance_file, load_gset_matrix, load_gset_opt

from ..common import SolverSolutionSimulator as SolverSolution


@pytest.mark.parametrize(
    "instance, N, best_known",
    [
        ("G1", 800, -11624),
        ("G11", 800, -564),
        ("G32", 2000, -1410),
    ],
)
def test_load_gset_instance(instance, N, best_known, cleanup):
    instance_file = get_instance_file(instance)
    problem_dir = Path(__file__) / "../../../src/benchmark/problem/data/GSET/"
    assert problem_dir.resolve() == Path(instance_file).parent.resolve()
    assert N * N == load_gset_matrix(instance_file).size
    assert best_known == load_gset_opt(instance)

    cleanup(instance_file)


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
def test_maxcut_problem(instance: str, best_known: int, cleanup):
    problem = MaxCut(instance)
    assert instance == problem.get_input_parameter()["instance"]
    assert best_known == problem.get_input_parameter()["best_known"]
    cleanup(get_instance_file(instance))


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "G1"
    instance = filepath.stem
    problem = MaxCut(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("MaxCut", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate(cleanup):
    problem = MaxCut(instance := "G1")
    best_known = 11624
    expected = {"label": "Maximum Cuts", "value": best_known}
    assert expected == problem.evaluate(SolverSolution(energy=-best_known))

    cleanup(get_instance_file(instance))
