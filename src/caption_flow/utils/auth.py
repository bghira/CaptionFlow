"""Authentication management."""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class WorkerAuthenticationDetails:
    """Details for worker authentication."""

    name: str
    token: str
    role: str


class AuthManager:
    """Manages authentication tokens."""

    def __init__(self, config: Dict[str, Any]):
        self.worker_tokens = {}
        self.admin_tokens = set()

        # Load worker tokens
        for worker in config.get("worker_tokens", []):
            worker_name = worker.get("name", None)
            assert worker_name is not None, "Worker token must have a name"
            self.worker_tokens[worker["token"]] = worker_name

        # Load admin tokens
        self.admin_tokens = set(config.get("admin_tokens", []))

    def authenticate(self, token: str) -> Optional[str]:
        """Authenticate token and return role."""
        role = None
        if token in self.worker_tokens:
            role = "worker"
        elif token in self.admin_tokens:
            role = "monitor"
        worker_auth_details = WorkerAuthenticationDetails(
            role=role, name=self.worker_tokens.get(token, "Unknown Worker"), token=token
        )
        return worker_auth_details
