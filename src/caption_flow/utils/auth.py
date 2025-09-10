"""Authentication management."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


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
        self.admin_tokens = {}
        self.monitor_tokens = {}
        if "orchestrator" in config:
            # compatibility with nested config as well.
            config = config.get("orchestrator").get("auth")

        # Load worker tokens
        for worker in config.get("worker_tokens", []):
            worker_name = worker.get("name", None)
            assert worker_name is not None, "Worker token must have a name"
            self.worker_tokens[worker["token"]] = worker_name

        # Load admin tokens
        for admin in config.get("admin_tokens", []):
            admin_name = admin.get("name", None)
            assert admin_name is not None, "Admin token must have a name"
            self.admin_tokens[admin["token"]] = admin_name

        # Load monitor tokens
        for monitor in config.get("monitor_tokens", []):
            monitor_name = monitor.get("name", None)
            assert monitor_name is not None, "Monitor token must have a name"
            self.monitor_tokens[monitor["token"]] = monitor_name

    def authenticate(self, token: str) -> Optional[str]:
        """Authenticate token and return role."""
        role = None
        for worker_token in self.worker_tokens:
            if token == worker_token:
                role = "worker"
                break
        if role is None:
            for admin_token in self.admin_tokens:
                if token == admin_token:
                    role = "admin"
                    break
        if role is None:
            for monitor_token in self.monitor_tokens:
                if token == monitor_token:
                    role = "monitor"
                    break

        worker_auth_details = WorkerAuthenticationDetails(
            role=role, name=self.worker_tokens.get(token, f"Anonymous {role}"), token=token
        )
        return worker_auth_details
