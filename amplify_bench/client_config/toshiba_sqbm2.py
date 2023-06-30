# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from .base import ClientConfig


@dataclass(frozen=True)
class ToshibaSQBM2ClientConfig(ClientConfig):
    _timeout_like_name: str = "timeout"

    def get_execution_parameters(self, client_result) -> dict:
        if not hasattr(client_result, "parameters"):
            raise RuntimeError()

        execution_parameters = {
            key: getattr(client_result.parameters, key) for key in dir(client_result.parameters) if key[0] != "_"
        }
        return execution_parameters
