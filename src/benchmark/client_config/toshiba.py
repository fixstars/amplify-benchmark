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
