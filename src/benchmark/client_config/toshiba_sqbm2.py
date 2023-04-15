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
