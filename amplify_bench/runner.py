# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import traceback
import warnings
from multiprocessing import Pool
from typing import List
from uuid import uuid4

from amplify import Solver  # type: ignore

from .client_config.base import ClientConfig
from .job import Job
from .problem.base import Problem
from .result import BenchmarkResult, JobFailedError, JobResult


class Runner:
    def __init__(self):
        self._jobs: List[Job] = list()

    @property
    def jobs(self) -> List[Job]:
        return self._jobs

    def register(
        self,
        problem: Problem,
        client_config: ClientConfig,
        num_samples: int = 1,
        label: str = "",
    ):
        job_list: List[Job] = [Job(str(uuid4()), problem, client_config, label) for _ in range(num_samples)]
        self._jobs.extend(job_list)

    def run(self) -> BenchmarkResult:
        results: List[JobResult] = list()
        try:

            def handler(signum, frame) -> None:
                raise OSError(f"signal.SIGTERM. PID={os.getpid()}")

            signal.signal(signal.SIGTERM, handler)
            for job in self._jobs:
                results.append(_run_job_impl(job))
        except Exception as err:
            print(f"{type(err).__name__}: {err}")
        finally:
            self._jobs.clear()
            return BenchmarkResult(results)


class ParallelRunner(Runner):
    def __init__(self, nproc: int):
        super().__init__()
        self._nproc = nproc

    def run(self) -> BenchmarkResult:
        results: List[JobResult] = []
        try:
            with Pool(processes=self._nproc) as processes:

                def handler(signum, frame) -> None:
                    processes.terminate()
                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    raise OSError(f"signal.SIGTERM. PID={os.getpid()}")

                signal.signal(signal.SIGTERM, handler)
                results_pool = []
                for job in self._jobs:
                    r = processes.apply_async(
                        run_job_wrapper,
                        args=(job,),
                    )
                    results_pool.append(r)
                for r in results_pool:
                    results.append(r.get())

        except Exception as err:
            print(f"{type(err).__name__}: {err}")
        finally:
            self._jobs.clear()
            return BenchmarkResult(results)


def run_job_wrapper(job: Job) -> JobResult:
    return _run_job_impl(job)


def _run_job_impl(job: Job) -> JobResult:
    client = job.client_config.get_configured_client()
    solver = Solver(client)
    solver.filter_solution = False
    solver.sort_solution = False
    if hasattr(solver.client.parameters, "outputs"):
        solver.client.parameters.outputs.sort = False
        solver.client.parameters.outputs.num_outputs = 0

    try:
        result = solver.solve(job.problem.model)
        job_result = JobResult.from_result(
            job,
            result.solutions,
            solver.client_result,
            solver.logical_result,
            job_error=None,
        )
    except RuntimeError as err:
        # KeyboardInterruptではなく RuntimeError('KeyboardInterrupt') をcatch
        if err.args[0] == "KeyboardInterrupt":
            raise err
        warnings.warn(err.args[0])
        job_result = JobResult.from_result(
            job,
            [],
            solver.client_result,
            solver.logical_result,
            job_error=JobFailedError(job.job_id, "Failed to send job to the solver\n" + traceback.format_exc()),
        )
        return job_result
    except OSError as err:
        raise err
    else:
        return job_result
