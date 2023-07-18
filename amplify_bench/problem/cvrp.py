# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import networkx
import numpy as np
import tsplib95
import vrplib
from amplify import BinaryQuadraticModel  # type: ignore
from amplify import BinarySymbolGenerator  # type: ignore
from amplify import IntegerEncodingMethod  # type: ignore
from amplify import SolverSolution  # type: ignore
from amplify import einsum  # type: ignore
from amplify.constraint import less_equal, one_hot, penalty  # type: ignore

from ..timer import print_log, timer
from .base import Problem

method_dict = {
    "relaxation": "relaxation",
    "default": IntegerEncodingMethod.Default,
    "unary": IntegerEncodingMethod.Unary,
    "binary": IntegerEncodingMethod.Binary,
    "inverse_binary": IntegerEncodingMethod.InverseBinary,
    "linear": IntegerEncodingMethod.Linear,
    "inverse_linear": IntegerEncodingMethod.InverseLinear,
}


class Cvrp(Problem):
    def __init__(
        self,
        instance: str,
        constraint_weight: float = 1.0,
        method: str = "default",
        scale: int = 2,
        path: Optional[str] = None,
    ):
        super().__init__()
        self._instance: str = instance
        self._problem_parameters["constraint_weight"] = constraint_weight
        self._problem_parameters["method"] = method
        self._problem_parameters["scale"] = scale
        self._symbols = None

        (
            capacity,
            dimension,
            distances,
            demand,
            coord,
            depot,
            nvehicle,
            best_known,
        ) = self.__load(instance, path)

        self._capacity = capacity
        self._demand = demand
        self._dimension = dimension
        self._nvehicle = nvehicle
        self._distance_matrix = distances

        self._longest_possible_length = self.__ubOfTour(capacity, demand)
        self._best_known = best_known

    def make_model(self):
        print_log(f"make model of {self._instance}")
        symbols, model = make_cvrp_model(
            self._longest_possible_length,
            self._dimension,
            self._nvehicle,
            self._distance_matrix,
            self._demand,
            self._capacity,
            method_dict[self._problem_parameters["method"]],
            self._problem_parameters["constraint_weight"],
            self._problem_parameters["scale"],
        )
        self._symbols = symbols
        self._model = model

    def evaluate(self, solution: SolverSolution) -> Dict[str, Union[None, int, str]]:
        value: Optional[int] = None
        path: str = ""

        if solution.is_feasible:
            spins = solution.values
            variables = np.array(self._symbols.decode(spins))  # type: ignore
            sequence = onehot2sequence(variables)  # one hotな変数テーブルを辞書に変換。key：車両インデックス, value：各車両が訪問した需要地の順番が入ったlist
            best_tour = processSequence(sequence)  # 上の辞書からデポへの余計な訪問を取り除く
            value = int(calcTourLen(best_tour, self._distance_matrix))
            for k, v in best_tour.items():
                for i in v:
                    path += f" {i}"
            path = path.replace("0 0", "0")
        else:
            pass

        return {"label": "total distances", "value": value, "path": path}

    @staticmethod
    def __load(
        instance: str, path: Optional[str] = None
    ) -> Tuple[int, int, np.ndarray, list, dict, int, Optional[int], Optional[int]]:
        best_known: Optional[int] = None
        nvehicle: Optional[int] = None

        if path is not None:
            instance_file = path
        else:
            instance_file = get_instance_file(instance)

        capacity, dimension, distances, demand, coord, depot = load_cvrp_file(instance_file)

        if path is not None:  # TODO: solution fileのパス指定に対応する
            pass
        else:
            instance_opt_file = get_sol_file(instance)
            best_known, nvehicle = load_cvrp_opt_distance_and_nvehicle(instance_opt_file)
        return capacity, dimension, distances, demand, coord, depot, nvehicle, best_known

    @staticmethod
    def __ubOfTour(capacity, demand):
        """
        ありうる最長のツアー長を計算する関数

        Parameters
        ----------
        capacity : int
            各車両の積載可能量
        demand : list
            各需要地の需要

        Returns
        -------
        longest_possible_length : int
            一台の車両が訪問できる需要地数の最大値+2
        """
        longest_possible_length = 2
        tmp = 0
        for s in sorted(demand):
            tmp += s
            if tmp <= capacity:
                longest_possible_length += 1
            else:
                return longest_possible_length
        return longest_possible_length


def get_instance_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    cvrp_dir = cur_dir / "data" / "CVRPLIB"
    if not cvrp_dir.exists():
        cvrp_dir.mkdir(parents=True)

    instance_file = cvrp_dir / (instance + ".vrp")
    if not instance_file.exists():
        vrplib.download_instance(instance, str(instance_file))

    return str(instance_file)


def get_sol_file(instance: str) -> str:
    cur_dir = Path(__file__).parent
    cvrp_dir = cur_dir / "data" / "CVRPLIB"

    sol_file = cvrp_dir / (instance + ".sol")
    if not sol_file.exists():
        vrplib.download_solution(instance, str(sol_file))
    return str(sol_file)


class NumCitiesError(Exception):
    pass


class WeightTypeError(Exception):
    pass


def load_cvrp_file(filepath) -> Tuple[int, int, np.ndarray, list, dict, int]:
    problem = tsplib95.load(filepath)

    capacity: int = problem.capacity  # type: ignore
    dimension: int = problem.dimension  # type: ignore
    demand: dict = problem.demands  # type: ignore
    depot: int = problem.depots  # type: ignore
    coord: dict = problem.node_coords  # type: ignore

    if dimension > 1000:
        raise NumCitiesError(f"{problem.name} number of cities too large: {dimension}")

    edge_weight_type = problem.edge_weight_type
    if edge_weight_type != "EUC_2D":
        raise WeightTypeError(f"Cannot use {edge_weight_type} type of weights.")

    # convert into a networkx.Graph
    graph = problem.get_graph()
    distances = np.array(networkx.to_numpy_matrix(graph))

    return capacity, dimension, distances, list(demand.values()), coord, depot


def load_cvrp_opt_routes(opt_file_path) -> Tuple[int, dict]:
    routes = dict()
    k = 0
    cost = 0
    with open(opt_file_path, "r") as f:
        for _line in f:
            line = _line.split()
            if len(line) == 0:
                break
            if line[0] == "Cost":
                cost = int(line[1])
                continue
            if line[0] != "Route":
                continue
            routes[k] = [0] + list(map(int, line[2:])) + [0]
            k += 1
    return cost, routes


def calcTourLen(sequence, distances):
    objective = 0
    for i in sequence:
        seq = sequence[i]
        for s, t in zip(seq, seq[1:]):
            objective += distances[s][t]
    return objective


def load_cvrp_opt_distance_and_nvehicle(opt_file_path) -> Tuple[int, int]:
    opt_tour_len, opt_sol = load_cvrp_opt_routes(opt_file_path)
    nvehicle_opt = len(opt_sol)
    return opt_tour_len, nvehicle_opt


def onehot2sequence(x_values) -> Dict[int, list]:
    """
    Solverから返ってきたワンホットベクトルを訪問順に番号が入ったリストに変換する関数

    Parameters
    ----------
    x_values : list
        最適化問題の解を格納した、サイズが[L x dimension x nvehicle]のリスト

    Returns
    -------
    sequence : dict
        key : int
            車両インデックス
        value : list
            需要地インデックスが訪問順に格納されたリスト
    """
    tour_len, ncity, nvehicle = x_values.shape
    sequence: dict = {k: list() for k in range(nvehicle)}
    for i in range(tour_len):
        for k in range(nvehicle):
            for j in range(ncity):
                if x_values[i][j][k] == 1:
                    sequence[k].append(j)
    return sequence


def processSequence(sequence) -> Dict[int, list]:
    """
    onehot2sequenceで作ったリストからデポへの余計な訪問を取り除いたリストを返す関数

    Parameters
    ----------
    sequence : dict
        key : int
            車両インデックス
        value : list
            需要地インデックスが訪問順に格納されたリスト

    Returns
    -------
    new_seq : dict
        sequenceから、余分なデポへの訪問を取り除いたもの
        key : int
            車両インデックス
        value : list
            需要地インデックスが訪問順に格納されたリスト
    """
    new_seq: dict = {k: list() for k in sequence}
    for k in sequence:
        new_seq[k].append(0)  # 始点はデポ(index 0)
        for place in sequence[k]:
            if place != 0:
                new_seq[k].append(place)
        new_seq[k].append(0)  # 終点はデポ(index 0)
    return new_seq


def setObjective(x, D):
    """
    目的関数を定義する

    Parameters
    ----------
    x : amplify.BinaryPolyArray
        変数を格納したサイズが[L x dimension x nvehicle]のリスト
    D : list
        需要地(+デポ)間の距離行列

    Returns
    -------
    objective : amplify.BinartPoly
        車両の総移動距離の合計を表す多項式
    """
    # 初期地点と最終地点の固定化
    x[0, 1:, :] = 0
    x[-1, 1:, :] = 0
    x[0, 0, :] = 1
    x[-1, 0, :] = 1

    # 経路の総距離
    objective = einsum("jl,ijk,ilk->", D, x[:-1, :, :], x[1:, :, :])

    return objective


def setConstraints(x: np.ndarray, demand: list, capacity: int, D: list, method=IntegerEncodingMethod.Default, scale=2):
    """
    制約条件を定義する

    Parameters
    ----------
    x : list[amplify.BinaryPoly]
        変数を格納した、サイズが[L x dimension x nvehicle]のリスト
        (np.ndarray互換のメソッドを持つ)
    demand : list
        各需要地の需要
    capacity : int
        各車両の積載可能量
    D : list
        需要地(+デポ)間の距離行列

    Returens
    --------
    constraints : amplify.BinaryConstraint
        CVRPの制約項
    """
    tour_len, ncity, nvehicle = x.shape

    distance_max = np.max(D)

    # (1) 同時に一箇所にしか訪問できない
    constraint1 = [distance_max * one_hot(x[i, :, k]) for i in range(tour_len) for k in range(nvehicle)]

    # (2) デポ以外は一回だけ訪問する
    constraint2 = [distance_max * one_hot(x[:, j, :]) for j in range(1, ncity)]

    # (3) 容量制約 TODO いろいろ試してみる
    if method == "relaxation":
        constraint3 = [
            scale
            * distance_max
            * penalty(
                ((demand * x[:, :, k]).sum() / capacity) ** 2,  # type: ignore
                le=1,
            )
            for k in range(nvehicle)
        ]
    else:
        constraint3 = [
            scale
            * distance_max
            * less_equal(
                demand * x[:, :, k],
                capacity,
                method=method,
            )
            / capacity
            / capacity
            for k in range(nvehicle)
        ]
    constraints = sum(constraint1) + sum(constraint2) + sum(constraint3)
    return constraints


@timer
def make_cvrp_model(
    longest_possible_length,
    ncity,
    nvehicle,
    distances,
    demand,
    capacity,
    method,
    kp,
    scale,
) -> Tuple[np.ndarray, BinaryQuadraticModel]:
    q = BinarySymbolGenerator().array(longest_possible_length, ncity, nvehicle)
    objective = setObjective(q, distances)
    constraints = setConstraints(q, demand, capacity, distances, method, scale)
    model = BinaryQuadraticModel(objective + kp * constraints)
    return q, model
