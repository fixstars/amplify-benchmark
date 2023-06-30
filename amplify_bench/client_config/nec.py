# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from .base import ClientConfig


@dataclass(frozen=True)
class NECClientConfig(ClientConfig):
    _timeout_like_name: str = "num_sweeps"
    _extract_keys: tuple = (
        "set_andzero",
        "set_maxone",
        "set_minmaxone",
        "set_onehot",
        "set_orone",
    )  # settings の内でログに含めるべき属性のリスト
