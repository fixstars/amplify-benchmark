# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import amplify
from amplify import SolverSolution  # type: ignore

from .job import Job
from .problem.base import AmplifyModel


def _get_amplify_version() -> str:
    try:
        return amplify.__version__
    except AttributeError as e:
        print(e)
        return ""


def _make_basic_summary(model: AmplifyModel, solution: SolverSolution) -> dict:  # type: ignore
    target_energy = solution.energy
    checks = model.check_constraints(solution.values)
    satisfied = len(list(filter(None, map(lambda x: x[1], checks))))
    num_constraints = len(checks)
    return {
        "target_energy": target_energy,
        "frequency": solution.frequency,
        "is_feasible": solution.is_feasible,
        "constraints": {"satisfied": satisfied, "broken": num_constraints - satisfied},
    }


def _make_result(job, solutions, client_result) -> dict:
    result = dict()
    result["annealing_time"] = client_result.timing()
    result["solutions"] = list()
    solution_summary_list = list()
    for solution in solutions:
        solution_summary = _make_basic_summary(job.problem.model, solution)
        solution_summary["objective_value"] = job.problem.evaluate(solution)
        solution_summary_list.append(solution_summary)

    result["solutions"] = solution_summary_list
    result["execution_parameters"] = job.client_config.get_execution_parameters(client_result)

    # time stamp を追加
    sampling_time_list = job.client_config.get_sampling_time(client_result, solutions)
    for i, t in enumerate(sampling_time_list):
        result["solutions"][i]["sampling_time"] = t

    return result


class NumTrialError(Exception):
    def __init__(self, num_trial, job_result):
        self.num_trial = num_trial
        self.job_result = job_result

    def __str__(self):
        return f"NumTrialError (num_trial: {self.num_trial})"


class JobFailedError(Exception):
    def __init__(self, job_id, message):
        self.job_id = job_id
        self.message = message

    def __str__(self):
        return f"JobFailedError (job id: {self.job_id}): {self.message}"


@dataclass(frozen=True)
class JobResult:
    job_id: str
    created_at: str
    amplify_version: str
    problem: dict
    client: dict
    result: dict
    group_id: str
    label: str
    error: str = field(default="")

    @classmethod
    def from_result(
        cls,
        job: Job,
        solutions: List[SolverSolution],
        client_result: Any,
        logical_result: Any,
        job_error: Optional[JobFailedError],
    ):
        result_dict = dict()
        error = ""
        if job_error is None:
            result_dict = _make_result(job, solutions, client_result)
        else:
            error = str(job_error)

        problem_dict: Dict[str, Any] = {"id": job.problem_id}
        problem_dict.update(job.problem.get_input_parameter())
        if len(solutions) == 0:
            problem_dict["num_vars"] = {
                "input": None,
                "logical": None,
                "physical": None,
            }
        else:
            problem_dict["num_vars"] = {
                "input": len(solutions[0].values),
                "logical": len(logical_result[0].values),
                "physical": len(client_result[0].values),
            }
        problem_dict["id"] = job.problem_id

        client_dict: Dict[str, Any] = {"id": job.client_id}
        client_dict.update(job.client_config.get_input_parameter())

        return cls(
            job_id=job.job_id,
            created_at=datetime.datetime.now().isoformat(),
            amplify_version=_get_amplify_version(),
            problem=problem_dict,
            client=client_dict,
            result=result_dict,
            group_id=job.group_id,
            label=job.label,
            error=error,
        )

    def get_summary(self) -> dict:
        return asdict(self)

    @property
    def is_ok(self):
        return self.error == ""


@dataclass
class BenchmarkResult:
    job_results: List[JobResult]

    def __post_init__(self):
        if self.__len__() == 0:
            return
        elif isinstance(self.job_results[0], JobResult):
            pass
        elif isinstance(self.job_results[0], dict):
            self.job_results = [JobResult(**asdict(job_result)) for job_result in self.job_results]
        else:
            raise RuntimeError(f"type of job_results error. type:{type(self.job_results[0])}")

    def __len__(self):
        return len(self.job_results)

    def __iter__(self):
        yield from self.job_results

    def get_summaries(self) -> List[dict]:
        return [j.get_summary() for j in self.job_results]

    def get_valid_summaries(self) -> List[dict]:
        return [j.get_summary() for j in self.job_results if j.is_ok]

    def get_errors(self) -> List[dict]:
        return [j.get_summary() for j in self.job_results if not j.is_ok]
