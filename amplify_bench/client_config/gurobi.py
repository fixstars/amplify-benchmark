# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from .base import ClientConfig


@dataclass(frozen=True)
class GurobiClientConfig(ClientConfig):
    _timeout_like_name: str = "time_limit"
