# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx
import numpy as np
import tsplib95
from amplify import BinaryQuadraticModel, BinarySymbolGenerator, SolverSolution, einsum  # type: ignore
from amplify.constraint import one_hot  # type: ignore
from tsplib95.utils import RadianGeo

from ..downloader import download_instance_file
from ..timer import print_log, timer
from .base import Problem


class Tsp(Problem):
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
        self._symbols = None

        ncity, distances, locations, best_known = self.__load(self._instance, seed, path)
        self._ncity = ncity
        self._distances = distances
        self._locations = locations  # not used
        self._best_known = best_known

    def make_model(self):
        print_log(f"make model of {self._instance}")
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

        if instance.startswith("random"):
            ncity, distance_matrix, locations = gen_random_tsp_instance(instance, seed)
        elif path is not None:
            ncity, distance_matrix, locations = load_tsp_file(path)
        else:
            instance_file = get_instance_file(instance)
            ncity, distance_matrix, locations = load_tsp_file(instance_file)
            best_known = load_tsp_opt_distance(instance)

        return ncity, distance_matrix, locations, best_known


def get_instance_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    tsp_dir = cur_dir / "data" / "TSPLIB"
    instance_file = tsp_dir / (instance + ".tsp")
    if not instance_file.exists():
        download_instance_file("Tsp", instance, dest=str(instance_file))
        assert instance_file.exists()
    return str(instance_file)


def gen_random_tsp_instance(instance: str, seed: int = 0) -> Tuple[int, np.ndarray, np.ndarray]:
    if not instance.startswith("random"):
        raise RuntimeError(f"instance:{instance} is not random instance name.")

    ncity = int(instance[6:])  # instance = random{ncity}
    np.random.seed(seed)

    locations = np.random.uniform(size=(ncity, 2)).tolist()
    all_diffs = np.expand_dims(locations, axis=1) - np.expand_dims(locations, axis=0)
    distances: np.ndarray = np.sqrt(np.sum(all_diffs**2, axis=-1))
    return ncity, distances.astype(float), locations


def load_tsp_file(problem_file: str) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
    problem = tsplib95.load(problem_file)
    ncity: int = problem.dimension  # type: ignore

    if ncity > 1000:
        raise RuntimeError(f"{problem.name} number of cities too large: {ncity}")

    # convert into a networkx.Graph
    graph = problem.get_graph()
    distance_matrix = np.array(networkx.to_numpy_matrix(graph))

    if problem.is_depictable():
        locations_dict: dict = dict()
        if len(problem.display_data) != 0:  # type: ignore
            locations_dict = problem.display_data  # type: ignore
        else:

            def get_location_dict(node_coords, map_func) -> dict:
                return {k: map_func(v) for k, v in node_coords.items()}

            if problem.edge_weight_type == "GEO":
                locations_dict = get_location_dict(problem.node_coords, lambda v: (RadianGeo(v).lat, RadianGeo(v).lng))
            else:
                locations_dict = get_location_dict(problem.node_coords, lambda v: v)

        locations = np.array([locations_dict[i + 1] for i in range(ncity)])
    else:
        locations = None

    return ncity, distance_matrix, locations


def load_tsp_opt_distance(instance: str) -> Optional[int]:
    cur_dir = Path(__file__).parent
    tsp_dir = cur_dir / "data" / "TSPLIB"

    best_dict: dict[str, int] = dict()
    with open(tsp_dir / "best_solutions.csv", "r") as f:
        for line in f.readlines():
            ln = line.split(",")
            best_dict[ln[0]] = int(ln[1])

    best_known: Optional[int] = best_dict.get(instance, None)
    return best_known


# tourの総距離を返す
def calc_tour_dist(tour: list, distances: np.ndarray) -> int:
    """Return the sum of weights for the given tour. (city index start from 0)"""
    ncity = len(tour)
    best_dist = sum([distances[tour[i]][tour[(i + 1) % ncity]] for i in range(ncity)])

    return int(best_dist)


# tourの総距離を返す
def calc_tour_dist_from_problem(tour: list, problem_file: str) -> int:
    """Return the sum of weights for the given tour. (city index start from 1)"""
    problem = tsplib95.load(problem_file)
    best_dist = problem.trace_tours([tour])[0]
    return int(best_dist)


# 辿るルートを地図画像として出力
# TODO: use networkx
def show_tour(tour: list, distances: np.ndarray, locations, file_name: str = "result.png"):
    ncity = len(tour)
    best_dist = calc_tour_dist(tour, distances)

    x = [i[0] for i in locations]
    y = [i[1] for i in locations]
    plt.figure(figsize=(7, 7))
    plt.title(f"Route distance={best_dist}")
    plt.xlabel("x")
    plt.ylabel("y")

    for i in range(ncity):
        r = tour[i]
        n = tour[(i + 1) % ncity]
        plt.plot([x[r], x[n]], [y[r], y[n]], "b-")
    plt.plot(x, y, "ro")
    plt.savefig(file_name)


@timer
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
