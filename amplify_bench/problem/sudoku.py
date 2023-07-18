# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from amplify import BinaryQuadraticModel, BinarySymbolGenerator, SolverSolution  # type: ignore
from amplify.constraint import one_hot  # type: ignore

from ..timer import print_log, timer
from .base import Problem


class Sudoku(Problem):
    def __init__(
        self,
        instance: str,
        path: Optional[str] = None,
    ):
        super().__init__()
        self._instance: str = instance

        grid = self.__load(instance, path)
        num_hints = len(grid) - grid.count(".")
        self.grid = grid
        self._symbols = None
        self._best_known = 0
        self._problem_parameters["num_hints"] = num_hints

    def make_model(self):
        print_log(f"make model of {self._instance}")
        symbols, model = make_sudoku_model(self.grid)
        self._symbols = symbols
        self._model = model

    def evaluate(self, solution: SolverSolution) -> dict:
        checks = self.model.check_constraints(solution.values)
        satisfied = len(list(filter(None, map(lambda x: x[1], checks))))
        num_constraints = len(checks)
        num_broken = num_constraints - satisfied
        answer = ""

        if solution.is_feasible:
            decoded = self._symbols.decode(solution.values)  # type: ignore
            N = int(len(self.grid) ** 0.5)
            answer = show_grid(N, decoded)
        else:
            pass

        return {
            "label": "Number of contradictions",
            "value": num_broken,
            "answer": answer,
        }

    @staticmethod
    def __load(instance: str, path: Optional[str] = None) -> str:
        initial: str

        if path is not None:
            initial = load_sudoku_file(path)
        else:
            initial = load_sudoku_file(get_instance_file(instance))

        return initial


def get_instance_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    sudoku_dir = cur_dir / "data" / "SUDOKULIB"
    instance_file = sudoku_dir / instance
    if not instance_file.exists():
        raise FileNotFoundError(f"instance: {instance} is not found.")
    return str(instance_file)


def show_grid(N, decoded) -> str:
    def to_str(n):
        if n == 0:
            return "."
        if 0 < n and n < 10:
            return str(n)
        else:
            return chr(n + 87)

    answer = ""
    for i in range(N):
        answer += "".join(
            map(
                to_str,
                np.where(np.array(decoded[i]) != 0)[1] + 1,
            )
        )
    return answer


def load_sudoku_file(problem_file: str) -> str:
    with open(problem_file) as sudoku_file:
        initial = sudoku_file.readlines()[0].rstrip()
    return initial


@timer
def make_sudoku_model(initial: str) -> Tuple[np.ndarray, BinaryQuadraticModel]:
    def to_dec(c):
        if c.isdecimal():
            return int(c)
        else:
            return int(ord(c) - ord("a") + 10)

    n = int(len(initial) ** 0.25)
    N = n**2

    # 変数テーブルの作成
    q = BinarySymbolGenerator().array(N, N, N)

    # initialにより変数を削減
    for s in np.where(np.array(list(initial)) != ".")[0]:
        i = s // N
        j = s % N
        k = to_dec(initial[s]) - 1

        q[i, :, k] = 0  # 制約(a)
        q[:, j, k] = 0  # 制約(b)
        q[i, j, :] = 0  # 制約(d)
        for m in range(N):
            q[(n * (i // n) + m // n), (n * (j // n) + m % n), k] = 0  # 制約(c)

        q[i, j, k] = 1

    # (a): 各行には同じ数字が入らない制約条件
    row_constraints = [one_hot(q[i, :, k]) for i in range(N) for k in range(N)]

    # (b): 各列には同じ数字が入らない制約条件
    col_constraints = [one_hot(q[:, j, k]) for j in range(N) for k in range(N)]

    # (d): 一つのマスには一つの数字しか入らない制約条件
    num_constraints = [one_hot(q[i, j, :]) for i in range(N) for j in range(N)]

    # (c): nxnブロック内には同じ数字が入らない制約条件
    block_constraints = [
        one_hot(sum([q[i + m // n, j + m % n, k] for m in range(N)]))
        for i in range(0, N, n)
        for j in range(0, N, n)
        for k in range(N)
    ]

    model = BinaryQuadraticModel(
        sum(row_constraints) + sum(col_constraints) + sum(num_constraints) + sum(block_constraints)
    )

    return q, model
