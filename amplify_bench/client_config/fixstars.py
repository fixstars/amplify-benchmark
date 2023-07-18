# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from amplify import SolverSolution  # type: ignore
from amplify.client import FixstarsClientResult  # type: ignore

from .base import ClientConfig


@dataclass(frozen=True)
class FixstarsClientConfig(ClientConfig):
    _timeout_like_name: str = "timeout"

    def get_execution_parameters(self, client_result) -> dict:
        if not hasattr(client_result, "execution_parameters"):
            raise RuntimeError()

        execution_parameters = {
            key: getattr(client_result.execution_parameters, key)
            for key in dir(client_result.execution_parameters)
            if key[0] != "_"
        }
        if "penalty_multipliers" in execution_parameters:
            del execution_parameters["penalty_multipliers"]
        return execution_parameters

    def get_sampling_time(self, client_result: FixstarsClientResult, solutions: list[SolverSolution]) -> list:
        sampling_time_list = list(client_result.timing.time_stamps)
        assert len(sampling_time_list) == len(solutions)
        return sampling_time_list
