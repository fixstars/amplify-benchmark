# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from amplify import BinaryQuadraticModel, BinarySymbolGenerator, SolverSolution, einsum  # type: ignore
from amplify.constraint import one_hot  # type: ignore

from amplify_bench import Problem


class MyTsp(Problem):
    def __init__(
        self,
        instance: str,
        constraint_weight: float = 1.0,
        seed: int = 0,
        path: Optional[str] = None,
    ):
        super().__init__()
        self._instance: str = instance
        self._problem_parameters["constraint_weight"] = constraint_weight
        if instance.startswith("random"):
            self._problem_parameters["seed"] = seed
        self._symbols: Optional[np.ndarray] = None

        ncity, distances, locations, best_known = self.__load(self._instance, seed, path)
        self._ncity = ncity
        self._distances = distances
        self._locations = locations  # not used
        self._best_known = best_known

    def make_model(self):
        symbols, model = make_tsp_model(self._ncity, self._distances, self._problem_parameters["constraint_weight"])
        self._symbols = symbols
        self._model = model

    def evaluate(self, solution: SolverSolution) -> Dict[str, Union[None, float, str]]:
        value: Optional[float] = None
        path: str = ""

        if solution.is_feasible:
            spins = solution.values
            variables = np.array(self._symbols.decode(spins))  # type: ignore
            index = np.where(variables == 1)[1]
            index_str = [str(idx) for idx in index]
            value = calc_tour_dist(list(index), self._distances)
            path = " ".join(index_str)
        else:
            pass

        return {"label": "total distances", "value": value, "path": path}

    @staticmethod
    def __load(
        instance: str, seed: int = 0, path: Optional[str] = None
    ) -> Tuple[int, np.ndarray, Optional[np.ndarray], Optional[float]]:
        ncity: int
        distance_matrix: np.ndarray
        locations: Optional[np.ndarray]
        best_known: Optional[float] = None

        ncity, distance_matrix, locations = gen_random_tsp_instance(instance, seed)

        return ncity, distance_matrix, locations, best_known


def gen_random_tsp_instance(instance: str, seed: int = 0) -> Tuple[int, np.ndarray, np.ndarray]:
    if not instance.startswith("random"):
        raise RuntimeError(f"instance:{instance} is not random instance name.")

    ncity = int(instance[6:])  # instance = random{ncity}
    np.random.seed(seed)

    locations = np.random.uniform(size=(ncity, 2)).tolist()
    all_diffs = np.expand_dims(locations, axis=1) - np.expand_dims(locations, axis=0)
    distances: np.ndarray = np.sqrt(np.sum(all_diffs**2, axis=-1))
    return ncity, distances.astype(float), locations


# tourの総距離を返す
def calc_tour_dist(tour: list, distances: np.ndarray) -> int:
    """Return the sum of weights for the given tour. (city index start from 0)"""
    ncity = len(tour)
    best_dist = sum([distances[tour[i]][tour[(i + 1) % ncity]] for i in range(ncity)])

    return int(best_dist)


def make_tsp_model(ncity, distances, kp) -> Tuple[np.ndarray, BinaryQuadraticModel]:
    np.fill_diagonal(distances, 0)

    # 各行の非ゼロ最小値をリストで取得
    d_min = np.min(np.where(distances == 0, distances.max(), distances), axis=0).reshape(ncity, 1)

    # シフトした距離行列を生成
    dij = distances - d_min
    np.fill_diagonal(dij, 0)

    # 変数テーブルの作成
    q = BinarySymbolGenerator().array(ncity, ncity)  # ncity x ncity 訪問順と訪問先を表現

    # コスト関数の係数を改変し定数項を加算
    cost = einsum("ij,ni,nj->", dij, q, q.roll(-1, axis=0)) + sum(d_min)

    # 行, 列に対する制約
    row_constraints = [one_hot(q[n]) for n in range(ncity)]
    col_constraints = [one_hot(q[:, i]) for i in range(ncity)]

    # 制約の重み
    weight = kp * np.amax(dij)

    model = cost + weight * (sum(row_constraints) + sum(col_constraints))
    return q, model
