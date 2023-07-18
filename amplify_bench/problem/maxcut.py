# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from amplify import BinaryMatrix, BinaryQuadraticModel, SolverSolution  # type: ignore

from ..downloader import download_instance_file
from ..timer import print_log, timer
from .base import Problem


class MaxCut(Problem):
    def __init__(
        self,
        instance: str,
        path: Optional[str] = None,
    ):
        super().__init__()
        self._instance: str = instance
        self._matrix = self.__load(instance, path)
        self._best_known = load_gset_opt(instance)

    def make_model(self):
        print_log(f"make model of {self._instance}")
        model = make_maxcut_model(self._matrix)
        self._model = model

    def evaluate(self, solution: SolverSolution) -> dict:
        return {"label": "Maximum Cuts", "value": -solution.energy}

    @staticmethod
    def __load(instance: str, path: Optional[str] = None):
        if path is not None:
            instance_file = path
        else:
            instance_file = get_instance_file(instance)
        return load_gset_matrix(instance_file)


def get_instance_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    maxcut_dir = cur_dir / "data" / "GSET"

    if not maxcut_dir.exists():
        maxcut_dir.mkdir(parents=True)

    instance_file = maxcut_dir / instance
    if not instance_file.exists():
        download_instance_file("MaxCut", instance, dest=str(instance_file))
    return str(instance_file)


def load_gset_matrix(problem_file: str) -> np.ndarray:
    with open(problem_file) as f:
        problem = f.readlines()
    head = problem[0].rstrip("\n").split(" ")
    node_size = int(head[0])
    matrix = np.zeros((node_size, node_size))

    for line in problem[1:]:
        ln = line.split(" ")
        i, j = int(ln[0]) - 1, int(ln[1]) - 1
        w_ij = int(ln[2])

        matrix[i][j] += 2.0 * w_ij
        matrix[i][i] -= 1.0 * w_ij
        matrix[j][j] -= 1.0 * w_ij
    return matrix


def load_gset_opt(instance) -> Optional[int]:
    cur_dir = Path(__file__).parent
    maxcut_dir = cur_dir / "data" / "GSET"
    sol_file = maxcut_dir / "best_solutions.csv"

    best_dict: Dict[str, int] = {}
    with open(sol_file, "r") as f:
        for line in f.readlines():
            ln = line.split(",")
            best_dict[ln[0]] = int(ln[1])

    best_known: Optional[int]
    if instance in best_dict:
        best_known = -1 * best_dict[instance]
    else:
        best_known = None

    return best_known


@timer
def make_maxcut_model(matrix) -> BinaryQuadraticModel:
    return BinaryQuadraticModel(BinaryMatrix(matrix))
