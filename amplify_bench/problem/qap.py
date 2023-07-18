# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from amplify import BinaryMatrix, BinaryQuadraticModel, BinarySymbolGenerator, SolverSolution  # type: ignore
from amplify.constraint import one_hot  # type: ignore

from ..downloader import download_instance_file
from ..timer import print_log, timer
from .base import Problem


class Qap(Problem):
    def __init__(
        self,
        instance: str,
        constraint_weight: float = 1.0,
        path: Optional[str] = None,
    ):
        super().__init__()
        self._instance: str = instance
        self._problem_parameters["constraint_weight"] = constraint_weight
        self._symbols = None

        ncity, distances, flows, best_known = self.__load(instance, path)
        self._ncity = ncity
        self._distances = distances
        self._flows = flows
        self._best_known = best_known

    def make_model(self):
        print_log(f"make model of {self._instance}")
        symbols, model = make_qap_model(self._distances, self._flows, self._problem_parameters["constraint_weight"])
        self._symbols = symbols
        self._model = model

    def evaluate(self, solution: SolverSolution) -> dict:
        value: Optional[float] = None
        placement: str = ""

        if solution.is_feasible:
            spins = solution.values
            variables = np.array(self._symbols.decode(spins))  # type: ignore
            index = np.where(variables == 1)[1]
            index_str = [str(idx) for idx in index]
            value = solution.energy
            placement = " ".join(index_str)
        else:
            pass

        return {"label": "cost", "values": value, "placement": placement}

    @staticmethod
    def __load(instance: str, path: Optional[str] = None):
        ncity: int
        distances: np.ndarray
        flows: np.ndarray
        best_known: Optional[float] = None

        if path is not None:
            ncity, distances, flows = load_qap_file(path)
        else:
            instance_file = get_instance_file(instance)
            ncity, distances, flows = load_qap_file(str(instance_file))
            sol_file = get_sol_file(instance)
            best_known = load_qap_opt(sol_file)

        return ncity, distances, flows, best_known


def get_instance_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    qap_dir = cur_dir / "data" / "QAPLIB"
    if not qap_dir.exists():
        qap_dir.mkdir(parents=True)

    instance_file = qap_dir / (instance + ".dat")
    if not instance_file.exists():
        download_instance_file("Qap", instance, dest=str(instance_file))
    return str(instance_file)


def get_sol_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    qap_dir = cur_dir / "data" / "QAPLIB"
    if not qap_dir.exists():
        qap_dir.mkdir(parents=True)

    sol_file = qap_dir / (instance + ".sln")
    if not sol_file.exists():
        pass
    return str(sol_file)


def load_qap_file(problem_file: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """Return the instance information tuple (n, dist, flow).

    Args:
        problem_file (str): QAPLIB instance filepath.

    Returns:
        Tuple[int, np.array, np.array] : The instance information (n, dist, flow).
            n : number of city/factory/etc...
            dist : distances between each cities.
            flow : flows between each cities.
    """
    data = list()

    def parse_numbers(string):
        numbers = string.split()
        return [int(num) for num in numbers]

    with open(problem_file) as f:
        lines = f.read().splitlines()

        for line in lines:
            numbers = parse_numbers(line)
            data.extend(numbers)

    N = data[0]
    assert len(data) == 2 * N * N + 1, RuntimeError(f"parse error. \nfile path: {problem_file}")

    dist = np.array(data[1 : N * N + 1]).reshape((N, N))
    flow = np.array(data[N * N + 1 :]).reshape((N, N))

    return N, dist, flow


def load_qap_opt(sol_file: str) -> Optional[int]:
    best_known: Optional[int] = None
    if Path(sol_file).exists():
        with open(sol_file) as f:
            lines = f.read().splitlines()
            first = re.split(" +", re.sub("^ +", "", lines[0]))
            best_known = int(first[1])

    return best_known


@timer
def make_qap_model(distances: np.ndarray, flows: np.ndarray, kp=1.0) -> Tuple[np.ndarray, BinaryQuadraticModel]:
    N = distances.shape[0]

    # 変数テーブルの作成
    x = BinarySymbolGenerator().array(N, N)

    # 目的関数
    Q = np.einsum("ij,kl->ikjl", distances, flows).reshape(N**2, N**2)
    qmatrix = np.triu(Q) + np.triu(Q.T)  # 対角成分0と仮定
    cost = BinaryMatrix(qmatrix)

    # 行, 列に対する制約
    row_constraints = [one_hot(x[n]) for n in range(N)]
    col_constraints = [one_hot(x[:, i]) for i in range(N)]

    # 制約の重み
    D = np.abs(distances).max()
    F = np.abs(flows).max()
    weight = kp * D * F * (N - 1)

    constraints = sum(row_constraints) + sum(col_constraints)
    model = cost + weight * constraints
    return x, model
