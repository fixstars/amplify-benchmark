# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
from abc import ABC, abstractmethod
from typing import Union

from amplify import (
    BinaryIntPolyArray,
    BinaryIntQuadraticModel,
    BinaryPolyArray,
    BinaryQuadraticModel,
    IsingIntPolyArray,
    IsingIntQuadraticModel,
    IsingPolyArray,
    IsingQuadraticModel,
    SolverSolution,
)

from ..util import dict_to_hash

AmplifyModel = Union[
    BinaryQuadraticModel,
    BinaryIntQuadraticModel,
    IsingQuadraticModel,
    IsingIntQuadraticModel,
]

AmplifyPolyArrayType = Union[
    BinaryPolyArray,
    BinaryIntPolyArray,
    IsingPolyArray,
    IsingIntPolyArray,
]


class Problem(ABC):
    def __init__(self):
        self._instance = None
        self._best_known = None
        self._model = None
        self._problem_parameters = dict()  # TODO: set default value

    @abstractmethod
    def make_model(self):
        pass

    @abstractmethod
    def evaluate(self, solution: SolverSolution) -> dict:
        pass

    @property
    def model(self) -> AmplifyModel:
        if self._model is None:
            self.make_model()
        return self._model

    def get_input_parameter(self) -> dict:
        problem_info = {
            "class": type(self).__name__,
            "instance": self._instance,
            "parameters": copy.deepcopy(self._problem_parameters),
            "best_known": self._best_known,
        }
        return problem_info

    def get_id(self) -> str:
        return dict_to_hash(self.get_input_parameter())


def gen_problem(name: str, instance: str, **kargs):
    """Generate Problem Class instance.

    Args:
        name (str): Problem Class name

    Raises:
        RuntimeError: problem class {name} is not supported.

    Returns:
        _type_: python class
    """
    module = importlib.import_module("..", __name__)
    if hasattr(module, name):
        problem_class = getattr(module, name)
        return problem_class(instance, **kargs)
    else:
        raise RuntimeError(f"Problem class {name} is not supported.")
