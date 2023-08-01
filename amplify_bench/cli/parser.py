# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path
from typing import Any, List, Tuple, Union

import yaml
from jsonschema import validate

from ..client_config.base import ClientConfig, get_client_config
from ..problem.base import Problem, gen_problem
from ..timer import timer


def _validation(input_json: dict[str, Any]) -> None:
    with open(Path(__file__).parent / "schemas" / "benchmark.json") as f:
        json_schema = json.load(f)
    validate(instance=input_json, schema=json_schema)


def replace_recursive(obj: Union[dict[str, Any], list[Any], str], variables: dict[str, Any]) -> Any:
    # objの中に$から始まる文字列があれば、variablesの中から置換する
    # 入力したobjとvariablesは書き換えられる
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_recursive(value, variables)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = replace_recursive(value, variables)
    elif isinstance(obj, str):
        if obj.startswith("$"):
            obj = replace_recursive(variables[obj[1:]], variables)
    return obj


@timer
def parse_input_data(filepath: Path) -> List[Tuple[Problem, ClientConfig, int]]:
    if filepath.suffix == ".json":
        j = json.load(filepath.open())
    elif filepath.suffix == ".yml" or filepath.suffix == ".yaml":
        j = yaml.safe_load(filepath.open())
    else:
        raise ValueError(f"invalid file extension: {filepath.suffix} must be .json or .yml or .yaml")
    _validation(j)
    global_variables = j["variables"] if "variables" in j.keys() else {}
    jobs = j["jobs"]
    ret = []
    for job in jobs:
        problem_list = []
        client_list = []
        num_samples_list = []
        matrix = job["matrix"] if "matrix" in job.keys() else {}
        matrix = [dict(zip(matrix.keys(), r)) for r in product(*matrix.values())]
        for m in matrix:
            # matrixキーを除いてコピー
            d = {k: v for k, v in job.items() if k != "matrix"}
            d = copy.deepcopy(d)
            variables = {**global_variables, **m}
            variables = copy.deepcopy(variables)
            try:
                variables = replace_recursive(variables, variables)
                d = replace_recursive(d, variables)
            except RecursionError:
                raise ValueError("detect circular reference in input data")
            _validation({"jobs": [d]})
            problem_parameters = d["problem"]["parameters"] if "parameters" in d["problem"].keys() else {}
            problem_list.append(gen_problem(d["problem"]["class"], d["problem"]["instance"], **problem_parameters))
            client_parameters = {}
            client_settings = copy.deepcopy(d["client"][1])
            if "parameters" in client_settings.keys():
                client_parameters = copy.deepcopy(client_settings["parameters"])
                del client_settings["parameters"]
            client_list.append(get_client_config(client_settings, client_parameters, d["client"][0]))
            num_samples_list.append(d["num_samples"])
        ret.extend(list(zip(problem_list, client_list, num_samples_list)))

    return ret
