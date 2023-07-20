# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Union

from amplify import SolverSolution  # type: ignore
from amplify.client import ABSClient  # type: ignore
from amplify.client import FixstarsClient  # type: ignore
from amplify.client import FujitsuDA3SolverClient  # type: ignore
from amplify.client import FujitsuDA4SolverClient  # type: ignore
from amplify.client import GurobiClient  # type: ignore
from amplify.client import HitachiClient  # type: ignore
from amplify.client import NECClient  # type: ignore
from amplify.client import ToshibaClient  # type: ignore
from amplify.client import ToshibaSQBM2Client  # type: ignore
from amplify.client.ocean import DWaveSamplerClient, LeapHybridSamplerClient  # type: ignore

from ..util import dict_to_hash

AmplifyClient = Union[
    FixstarsClient,
    FujitsuDA3SolverClient,
    ToshibaClient,
    HitachiClient,
    ABSClient,
    GurobiClient,
    DWaveSamplerClient,
    LeapHybridSamplerClient,
    ToshibaSQBM2Client,
    FujitsuDA4SolverClient,
    NECClient,
]


@dataclass(frozen=True)
class ClientConfig:
    settings: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    timeout_like: dict = field(init=False)
    version: str = field(init=False)

    name: str = ""
    _timeout_like_name: str = ""
    _extract_keys: tuple = field(default_factory=tuple)  # settings の内でログに含めるべき属性のリスト

    def __post_init__(self) -> None:
        if self.name == "":
            cls = type(self)
            object.__setattr__(self, "name", cls.__name__[:-6])

        try:
            client = self.get_configured_client()
            object.__setattr__(self, "version", client.version)
        except Exception:
            object.__setattr__(self, "version", "unknown")

        object.__setattr__(
            self,
            "timeout_like",
            {
                "name": self._timeout_like_name,
                "value": self.parameters.get(self._timeout_like_name, None),
            },
        )

    def get_execution_parameters(self, client_result) -> dict:
        """Return dictionary of execution parameters.

        Args:
            client_result : Solver.client_result

        Returns:
            dict: Parameters used to obtain a given sample.
        """
        execution_parameters = {
            key: getattr(client_result, key)
            for key in dir(client_result)
            if key[0] != "_" and type(getattr(client_result, key)) in [str, float, int]
        }
        return execution_parameters

    def get_sampling_time(self, client_result, solutions: list[SolverSolution]) -> list:
        """Return 1 solution sampling time [ms].

        Args:
            client_result: Solver.client_result
            solutions: list of SolverSolution

        Returns:
            list: Sampling time values in 1 job.
        """
        timing = getattr(client_result, "timing")()
        return [timing] * len(solutions)

    def get_input_parameter(self) -> dict:
        """Return client configuration data as dict.
        In doing so, remove timeout_like parameter and unnecessary settings configuration such as URLs.

        Returns:
            dict: client settings/parameters configuration
        """

        client_config_dict = asdict(self)
        del client_config_dict["_extract_keys"]
        del client_config_dict["_timeout_like_name"]

        client_config_dict["parameters"].pop(self._timeout_like_name, None)  # type: ignore
        for key in client_config_dict["settings"].keys() ^ self._extract_keys:  # キーの差集合を削除
            client_config_dict["settings"].pop(key, None)

        return client_config_dict

    def get_client_id(self):
        dict_ = self.get_input_parameter()
        del dict_["timeout_like"]
        return dict_to_hash(dict_)

    def get_configured_client(self) -> AmplifyClient:
        """Generate and return a client object.
        Set only the information listed in 'settings', such as token, url, etc.
        For the convenience of calling from within a parallel process,
        the client object is not placed in a member variable.
        """
        m = importlib.import_module("amplify.client")
        client_class = getattr(m, self.name)

        client = client_class()
        for k, v in self.settings.items():
            _set_client_attr(client, k, v)
        for k, v in self.parameters.items():
            _set_client_attr(client.parameters, k, v)

        return client


def get_client_config(settings: dict, parameters: dict, name: str, _timeout_like_name: str = "") -> ClientConfig:
    module = importlib.import_module("..", __name__)
    if hasattr(module, name) and hasattr(module, name + "Config"):
        client_config_class = getattr(module, name + "Config")
        return client_config_class(settings, parameters)
    else:
        return ClientConfig(settings, parameters, name, _timeout_like_name)


def _set_client_attr(obj, k, v):
    def _setattr_recur(obj, k: str, v):
        # k = "a.b.c" のとき、obj.a.b.c に v を代入する
        sub_obj = obj
        attrs = k.split(".")
        for attr in attrs[:-1]:
            sub_obj = getattr(sub_obj, attr)
        setattr(sub_obj, attrs[-1], v)

    def _setattr_recur_for_dict(obj, k: str, v):
        if type(v) is dict:
            sub_obj = getattr(obj, k)
            for k_, v_ in v.items():
                _setattr_recur_for_dict(sub_obj, k_, v_)
        else:
            _setattr_recur(obj, k, v)

    _setattr_recur_for_dict(obj, k, v)
