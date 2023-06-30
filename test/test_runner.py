import pytest

from amplify_bench.client_config.base import get_client_config
from amplify_bench.problem.base import gen_problem
from amplify_bench.runner import ParallelRunner, Runner


@pytest.mark.parametrize(
    "runner",
    [
        Runner(),
        ParallelRunner(nproc=4),
    ],
)
def test_runner_register(runner):
    problem = gen_problem("Tsp", "random50", seed=0, constraint_weight=0.5)

    client_name = "FixstarsClient"
    client_config0 = get_client_config(
        settings={},
        parameters={},
        name=client_name,
    )
    client_config1 = get_client_config(
        settings={"url": "http://localhost:8080/", "token": "XXXX", "proxy": "YYYY"},
        parameters={},
        name=client_name,
    )

    client_config2 = get_client_config(
        settings := {"url": "http://localhost:8080/", "token": "XXXX", "proxy": "YYYY"},
        parameters := {"outputs.num_outputs": 1, "penalty_calibration": False},
        name=client_name,
    )

    runner.register(problem, client_config0, label="test 0")
    runner.register(problem, client_config1, num_samples=1, label="test 1")
    runner.register(problem, client_config2, num_samples=10, label="test 2")

    assert runner.jobs[0].client_config.name == client_name
    assert runner.jobs[0].label == "test 0"

    assert runner.jobs[1].client_config.name == client_name
    assert runner.jobs[1].label == "test 1"

    assert runner.jobs[2].client_config.name == client_name
    assert runner.jobs[2].label == "test 2"
    assert runner.jobs[2].client_config.settings == settings
    assert runner.jobs[2].client_config.parameters == parameters

    _group_id = runner.jobs[2].group_id
    for job in runner.jobs[2:]:
        assert job.label == "test 2"
        assert job.group_id == _group_id

    # Checks if each element of job list has a unique id
    job_id_list = [job.job_id for job in runner.jobs]
    assert len(job_id_list) == len(list(set(job_id_list)))


@pytest.mark.parametrize(
    "runner",
    [
        Runner(),
        ParallelRunner(nproc=4),
    ],
)
def test_group_id(runner):
    problem = gen_problem("Tsp", "random50", seed=0, constraint_weight=0.5)
    client_name = "FixstarsClient"

    client_config3 = get_client_config({}, {"timeout": 1000}, name=client_name)
    client_config4 = get_client_config({}, {"timeout": 3000}, name=client_name)

    runner.register(problem, client_config3, num_samples=1, label="test 3")
    runner.register(problem, client_config4, num_samples=1, label="test 3")
    assert runner.jobs[0].group_id == runner.jobs[1].group_id
