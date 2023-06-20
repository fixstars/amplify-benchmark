# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from .client_config.base import ClientConfig
from .problem.base import Problem
from .util import dict_to_hash


@dataclass
class Job:
    job_id: str
    problem: Problem
    client_config: ClientConfig
    label: str = field(default="")
    problem_id: str = field(init=False)
    client_id: str = field(init=False)
    group_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.problem_id = self.problem.get_id()
        self.client_id = self.client_config.get_client_id()
        self.group_id = dict_to_hash({"client_id": self.client_id, "problem_id": self.problem_id})
