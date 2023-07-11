from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from amplify_bench.client_config import (  # type: ignore
    FixstarsClientConfig,
    FujitsuDA4SolverClientConfig,
    GurobiClientConfig,
    ToshibaClientConfig,
    ToshibaSQBM2ClientConfig,
    get_client_config,
)

FIXSTARS_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TOSHIBA_TOKEN = "SQBM+/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


def test_get_client():
    client_config_data = FixstarsClientConfig(
        settings := {"token": FIXSTARS_TOKEN},
        parameters := {"timeout": 1000, "outputs.feasibilities": True},
    )
    client = client_config_data.get_configured_client()

    assert type(client).__name__ == client_config_data.name
    assert client.token == settings["token"]
    assert client.parameters.timeout == parameters["timeout"]


def test_fixstars_client_config(simple_problem):
    client_config_data = FixstarsClientConfig(
        settings := {"token": FIXSTARS_TOKEN},
        parameters := {"timeout": 1000, "outputs.feasibilities": True},
    )

    assert isinstance(client_config_data.version, str)
    assert client_config_data.settings == settings
    assert client_config_data.parameters == parameters
    assert client_config_data.timeout_like == {"name": "timeout", "value": 1000}

    config_dict = client_config_data.get_input_parameter()
    assert config_dict["name"] == "FixstarsClient"
    assert config_dict["parameters"]["outputs.feasibilities"] is True

    @dataclass
    class FixstarsClientExecParams:
        num_gpus: int = 1
        num_iterations: int = 10
        penalty_calibration: bool = False
        penalty_multipliers: list[float] = field(default_factory=list)
        timeout: float = 1000
        version: str = "v0.7.0"

    FixstarsClientResult = MagicMock()
    FixstarsClientResult.return_value.execution_parameters = FixstarsClientExecParams()
    client_config_data = FixstarsClientConfig()
    client_result = FixstarsClientResult()

    exec_params = client_config_data.get_execution_parameters(client_result)
    for k, v in exec_params.items():
        assert v == getattr(client_result.execution_parameters, k)


def test_toshiba_client_config(simple_problem):
    client_config_data = ToshibaClientConfig(
        settings := {"token": TOSHIBA_TOKEN, "solver": "autoising"},
        parameters := {"timeout": 1, "loops": 0, "maxout": 5},
    )

    assert isinstance(client_config_data.version, str)
    assert client_config_data.settings == settings
    assert client_config_data.parameters == parameters
    assert client_config_data.timeout_like == {"name": "timeout", "value": 1}

    config_dict = client_config_data.get_input_parameter()
    assert config_dict["name"] == "ToshibaClient"
    del parameters["timeout"]
    assert config_dict["parameters"] == parameters

    @dataclass
    class ToshibaClientResultMock:
        C: float = 0.1
        dt: float = 0.5
        runs: int = 320
        steps: int = 100

    client_result = ToshibaClientResultMock()
    exec_params = client_config_data.get_execution_parameters(client_result)

    for k, v in exec_params.items():
        assert v == getattr(client_result, k)


def test_toshiba_sqbm2(simple_problem):
    client_config_data = ToshibaSQBM2ClientConfig(
        settings := {"url": "http://xxx.xxx.xxx.xxx:8000"},
        parameters := {"timeout": 1, "loops": 0, "maxout": 5},
    )

    assert isinstance(client_config_data.version, str)
    assert client_config_data.settings == settings
    assert client_config_data.parameters == parameters
    assert client_config_data.timeout_like == {"name": "timeout", "value": 1}

    config_dict = client_config_data.get_input_parameter()
    assert config_dict["name"] == "ToshibaSQBM2Client"
    del parameters["timeout"]
    assert config_dict["parameters"] == parameters

    @dataclass
    class ToshibaSQBM2ClientParamsMock:
        C: float = 0.1
        dt: float = 0.5
        runs: int = 320
        steps: int = 100

    @dataclass
    class ToshibaSQBM2ClientResultMock:
        parameters: ToshibaSQBM2ClientParamsMock = field(default_factory=ToshibaSQBM2ClientParamsMock)

    client_result = ToshibaSQBM2ClientResultMock()
    exec_params = client_config_data.get_execution_parameters(client_result)

    for k, v in exec_params.items():
        assert v == getattr(client_result.parameters, k)


def test_gurobi_client_config(simple_problem):
    client_config_data = GurobiClientConfig({}, {"time_limit": 10})

    assert isinstance(client_config_data.version, str)
    assert client_config_data.timeout_like == {"name": "time_limit", "value": 10}


def test_fujitsu_da4(simple_problem):
    client_config_data = FujitsuDA4SolverClientConfig(
        settings := {"token": "xxxxx", "set_inequalities": True, "set_penalty_binary_polynomial": True},
        parameters := {"time_limit_sec": 10},
    )

    assert isinstance(client_config_data.version, str)
    assert client_config_data.settings == settings
    assert client_config_data.parameters == parameters
    assert client_config_data.timeout_like == {"name": "time_limit_sec", "value": 10}

    config_dict = client_config_data.get_input_parameter()
    assert config_dict["name"] == "FujitsuDA4SolverClient"
    del parameters["time_limit_sec"]
    assert config_dict["parameters"] == parameters


def test_nec_client_config():
    client_config_data = get_client_config(
        settings := {"token": "NEC/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"},
        parameters := {
            "timeout": 10,
            "num_reads": 1,
            "num_sweeps": 50,
        },
        name := "NECClient",
        _timeout_like_name := "num_sweeps",
    )

    assert isinstance(client_config_data.version, str)
    assert client_config_data.version == "2.0"
    assert client_config_data.settings == settings
    assert client_config_data.parameters == parameters
    assert client_config_data.timeout_like == {"name": _timeout_like_name, "value": parameters[_timeout_like_name]}

    config_dict = client_config_data.get_input_parameter()
    assert config_dict["name"] == name
    del parameters[_timeout_like_name]
    assert config_dict["parameters"] == parameters


def test_client_config():
    client_config_data = get_client_config({"token": "xxx"}, {"timeout": 1000}, name := "FixstarsClient")
    assert type(client_config_data) == FixstarsClientConfig

    client_config_data = get_client_config(
        settings := {"token": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"},
        parameters := {
            "timeout": 10,
            "num_reads": 1,
        },
        name := "HogeClient",
    )

    assert client_config_data.name == name
    assert client_config_data.version == "unknown"
    assert client_config_data.settings == settings
    assert client_config_data.parameters == parameters
    assert client_config_data.timeout_like == {"name": "", "value": None}

    client_config_data2 = get_client_config(
        settings := {"token": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"},
        parameters := {
            "timeout": 10,
            "num_reads": 1,
        },
        name := "HogeClient",
        _timeout_like_name="timeout",
    )
    assert client_config_data2.timeout_like == {"name": "timeout", "value": 10}
