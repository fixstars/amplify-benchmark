# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from .base import ClientConfig


@dataclass(frozen=True)
class ToshibaClientConfig(ClientConfig):
    _timeout_like_name: str = "timeout"
    _extract_keys: tuple = ("solver",)  # settings の内でログに含めるべき属性のリスト

    def get_execution_parameters(self, client_result) -> dict:
        execution_parameters = {
            key: getattr(client_result, key)
            for key in ["C", "dt", "deeper", "algo", "runs", "steps", "stats", "NAME"]
            if hasattr(client_result, key)
        }
        return execution_parameters
