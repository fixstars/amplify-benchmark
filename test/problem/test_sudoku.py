from pathlib import Path

import numpy as np
import pytest

from benchmark.problem.base import gen_problem
from benchmark.problem.sudoku import Sudoku, get_instance_file, load_sudoku_file

from ..common import SolverSolutionSimulator as SolverSolution


def test_load_sudoku_file(data):
    instance_file = get_instance_file(instance := "9x9_h17_055")
    problem_dir = data / "SUDOKULIB"
    assert problem_dir.resolve() == Path(instance_file).parent.resolve()
    assert problem_dir.resolve() / instance == Path(instance_file).resolve()

    actual_hints = ".................1....23.4......4.2...2.....5..61......3......6.7.5....894......."
    assert actual_hints == load_sudoku_file(instance_file)


def test_load_sudoku_file_error(data):
    SUDOKU_DIR = data / "SUDOKULIB"
    with pytest.raises(FileNotFoundError):
        load_sudoku_file(SUDOKU_DIR / "9x9_h17_XXX")


@pytest.mark.parametrize(
    "instance, num_hints",
    [
        ("9x9_h17_055", 17),
        ("9x9_h39_022", 39),
        ("16x16_1_000", 55),
        ("16x16_easy_233", 111),
        ("16x16_hard_194", 98),
        ("25x25_000", 146),
    ],
)
def test_sudoku_problem(instance: str, num_hints: int):
    problem = Sudoku(instance)
    assert instance == problem.get_input_parameter()["instance"]
    assert 0 == problem.get_input_parameter()["best_known"]
    assert num_hints == problem.get_input_parameter()["parameters"]["num_hints"]


def test_load_local_file():
    filepath = Path(__file__).parent / "data" / "9x9_h17_055"
    instance = filepath.stem
    problem = Sudoku(instance, path=str(filepath))
    assert problem.get_input_parameter()["instance"] == instance

    problem2 = gen_problem("Sudoku", instance, path=str(filepath))
    assert problem2.get_input_parameter()["instance"] == instance


def test_evaluate():
    # infeasible case
    problem = Sudoku("9x9_h17_055")
    values = np.zeros(729, dtype=int).tolist()
    assert "" == problem.evaluate(SolverSolution(is_feasible=False, values=values))["answer"]

    # feasible case
    expected_answer = "593461782824975361167823549719654823482397615356182974235718496671549238948236157"
    values = np.zeros(729, dtype=int).tolist()
    for i in range(9):
        for j in range(9):
            v = expected_answer[9 * i + j]
            values[81 * i + 9 * j + int(v) - 1] = 1

    result = problem.evaluate(SolverSolution(is_feasible=True, values=values))
    assert expected_answer == result["answer"]
    assert 0 == result["value"]
