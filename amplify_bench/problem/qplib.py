# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from amplify import BinaryQuadraticModel, InequalityFormulation, SolverSolution, load_qplib  # type: ignore

from ..downloader import download_instance_file
from ..timer import print_log, timer
from .base import Problem

method_dict = {
    "default": InequalityFormulation.Default,
    "unary": InequalityFormulation.Unary,
    "binary": InequalityFormulation.Binary,
    "linear": InequalityFormulation.Linear,
    "relaxation": InequalityFormulation.Relaxation,
    "relaxation_linear": InequalityFormulation.RelaxationLinear,
    "relaxation_quadra": InequalityFormulation.RelaxationQuadra,
}


class Qplib(Problem):
    def __init__(
        self,
        instance: str,
        inequality_formulation_method: str = "default",
        constraint_weights: List[float] = [1.0],
        path: Optional[str] = None,
    ):
        super().__init__()
        self._instance = instance
        self._problem_parameters["inequality_formulation_method"] = inequality_formulation_method
        self._problem_parameters["constraint_weights"] = constraint_weights
        self._symbols = None

        instance_file, best_known = self.__load(instance, path)
        self._instance_file = instance_file
        self._best_known = best_known

    def make_model(self):
        print_log(f"make model of {self._instance}")
        symbols, model = make_qplib_model(
            self._instance_file,
            inequality_formulation_method=method_dict[self._problem_parameters["inequality_formulation_method"]],
            constraint_weights=self._problem_parameters["constraint_weights"],
        )
        self._symbols = symbols
        self._model = model

    def evaluate(self, solution: SolverSolution) -> Dict[str, Union[None, float, str]]:
        value: Optional[float] = None

        if solution.is_feasible:
            value = solution.energy
        else:
            pass

        return {"label": "objvar", "value": value}

    @staticmethod
    def __load(instance: str, path: Optional[str] = None) -> Tuple[str, Optional[float]]:
        if path is not None:
            instance_file = path
        else:
            instance_file = get_instance_file(instance)

        best_known = load_best_known(instance)

        return str(instance_file), best_known


def load_best_known(instance: str) -> Optional[float]:
    cur_dir = Path(__file__).parent
    qplib_dir = cur_dir / "data" / "QPLIB"

    best_dict: dict[str, float] = dict()
    with open(qplib_dir / "best_solutions.csv", "r") as f:
        for line in f.readlines():
            ln = line.strip().split(",")
            best_dict[ln[0]] = float(ln[1])

    best_known: Optional[float] = best_dict.get(instance, None)
    return best_known


def get_instance_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    qplib_dir = cur_dir / "data" / "QPLIB"
    instance_file = qplib_dir / (instance + ".qplib")
    if not instance_file.exists():
        download_instance_file("Qplib", instance, dest=str(instance_file))
    return str(instance_file)


@timer
def make_qplib_model(
    instance_file: str, inequality_formulation_method=InequalityFormulation.Default, constraint_weights=[1.0]
) -> Tuple[np.ndarray, BinaryQuadraticModel]:
    model, variables = load_qplib(instance_file, inequality_formulation_method=inequality_formulation_method)
    for i, c in enumerate(model.input_constraints):
        if i < len(constraint_weights):
            c[1] = constraint_weights[i]
        else:
            c[1] = constraint_weights[-1]

    return variables, model
