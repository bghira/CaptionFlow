"""Authentication management."""

from typing import Dict, Any, Optional

class AuthManager:
    """Manages authentication tokens."""
    
    def __init__(self, config: Dict[str, Any]):
        self.worker_tokens = {}
        self.admin_tokens = set()
        
        # Load worker tokens
        for worker in config.get("worker_tokens", []):
            self.worker_tokens[worker["token"]] = worker.get("name", "worker")
        
        # Load admin tokens
        self.admin_tokens = set(config.get("admin_tokens", []))
    
    def authenticate(self, token: str) -> Optional[str]:
        """Authenticate token and return role."""
        if token in self.worker_tokens:
            return "worker"
        elif token in self.admin_tokens:
            return "monitor"
        return None