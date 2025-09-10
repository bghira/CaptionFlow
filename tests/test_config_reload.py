import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from websockets.asyncio.server import ServerConnection

# Import pytest-asyncio
pytest_plugins = ("pytest_asyncio",)
import pytest_asyncio
from caption_flow.orchestrator import Orchestrator


class TestConfigReload:
    """Test suite for configuration reload functionality."""

    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Create an Orchestrator instance with auth configuration."""
        config = {
            "dataset": {"processor_type": "webdataset"},
            "chunks_per_request": 10,
            "auth": {
                "worker_tokens": [
                    {"name": "test_worker_1", "token": "worker_token_1"},
                    {"name": "test_worker_2", "token": "worker_token_2"},
                ],
                "admin_tokens": [{"name": "test_admin", "token": "admin_token_1"}],
                "monitor_tokens": [{"name": "test_monitor", "token": "monitor_token_1"}],
            },
            "storage": {"caption_buffer_size": 100},
        }

        # Use patch to avoid actual storage/processor initialization
        with patch("caption_flow.orchestrator.StorageManager") as mock_storage_class:
            with patch(
                "caption_flow.orchestrator.WebDatasetOrchestratorProcessor"
            ) as mock_processor_class:
                mock_storage = Mock()
                mock_storage_class.return_value = mock_storage
                mock_storage.initialize = AsyncMock()

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_processor.initialize = Mock()
                mock_processor.release_assignments = Mock()

                orchestrator = Orchestrator(config)

                # Mock the activity sending
                orchestrator._send_activity = AsyncMock()

        return orchestrator

    @pytest.mark.asyncio
    async def test_config_reload_preserves_existing_auth_object_when_unchanged(self, orchestrator):
        """Test that config reload doesn't recreate AuthManager when auth config is unchanged."""
        # Get reference to initial auth manager
        initial_auth_manager = orchestrator.auth

        # Simulate a config reload without auth section (should not recreate auth)
        new_config = {
            "orchestrator": {
                "chunks_per_request": 15,  # Changed, but no auth section
            }
        }

        # Mock websocket
        websocket = AsyncMock(spec=ServerConnection)

        # Perform config reload
        await orchestrator._handle_config_reload(websocket, new_config)

        # Check that AuthManager object is preserved when no auth in config
        assert (
            orchestrator.auth is initial_auth_manager
        ), "AuthManager should be preserved when auth section is not in reload config"

        # Verify chunks_per_request was updated
        assert orchestrator.chunks_per_request == 15

    @pytest.mark.asyncio
    async def test_config_reload_preserves_auth_when_identical(self, orchestrator):
        """Test that config reload preserves AuthManager when auth config is identical."""
        # Get reference to initial auth manager
        initial_auth_manager = orchestrator.auth

        # Simulate a config reload with identical auth section
        new_config = {
            "orchestrator": {
                "chunks_per_request": 15,
                "auth": {
                    "worker_tokens": [
                        {"name": "test_worker_1", "token": "worker_token_1"},
                        {"name": "test_worker_2", "token": "worker_token_2"},
                    ],
                    "admin_tokens": [{"name": "test_admin", "token": "admin_token_1"}],
                    "monitor_tokens": [{"name": "test_monitor", "token": "monitor_token_1"}],
                },
            }
        }

        # Mock websocket
        websocket = AsyncMock(spec=ServerConnection)

        # Perform config reload
        await orchestrator._handle_config_reload(websocket, new_config)

        # Auth manager should be preserved when config is identical
        assert (
            orchestrator.auth is initial_auth_manager
        ), "AuthManager should be preserved when auth config is identical"

        # Tokens should still work
        assert orchestrator.auth.worker_tokens["worker_token_1"] == "test_worker_1"

    @pytest.mark.asyncio
    async def test_config_reload_preserves_auth_runtime_state(self, orchestrator):
        """Test that config reload preserves auth runtime state when config is identical."""
        # Get reference to initial auth manager and add runtime state
        initial_auth_manager = orchestrator.auth

        # Add some hypothetical runtime state (simulate active connections, etc.)
        initial_auth_manager._runtime_state = {"active_connections": ["conn1", "conn2"]}

        # Get initial config that matches current auth
        new_config = {
            "orchestrator": {
                "chunks_per_request": 15,
                "auth": {
                    "worker_tokens": [
                        {"name": "test_worker_1", "token": "worker_token_1"},
                        {"name": "test_worker_2", "token": "worker_token_2"},
                    ],
                    "admin_tokens": [{"name": "test_admin", "token": "admin_token_1"}],
                    "monitor_tokens": [{"name": "test_monitor", "token": "monitor_token_1"}],
                },
            }
        }

        # Mock websocket
        websocket = AsyncMock(spec=ServerConnection)

        # Perform config reload
        await orchestrator._handle_config_reload(websocket, new_config)

        # Auth manager should be preserved when config is identical
        assert (
            orchestrator.auth is initial_auth_manager
        ), "AuthManager should be preserved when auth config is identical to current config"

        # Runtime state should be preserved
        assert hasattr(orchestrator.auth, "_runtime_state"), "Runtime state should be preserved"
        assert orchestrator.auth._runtime_state["active_connections"] == [
            "conn1",
            "conn2",
        ], "Runtime state should be preserved"

    @pytest.mark.asyncio
    async def test_config_reload_updates_auth_when_changed(self, orchestrator):
        """Test that config reload properly updates auth when auth config changes."""
        # Simulate a config reload with changed auth configuration
        new_config = {
            "orchestrator": {
                "auth": {
                    "worker_tokens": [
                        {"name": "test_worker_1", "token": "worker_token_1"},
                        {"name": "new_worker", "token": "new_worker_token"},  # Added new worker
                    ],
                    "admin_tokens": [
                        {"name": "test_admin", "token": "admin_token_1"},
                        {"name": "new_admin", "token": "new_admin_token"},  # Added new admin
                    ],
                    "monitor_tokens": [{"name": "test_monitor", "token": "monitor_token_1"}],
                }
            }
        }

        # Mock websocket
        websocket = AsyncMock(spec=ServerConnection)

        # Perform config reload
        await orchestrator._handle_config_reload(websocket, new_config)

        # Check that new auth tokens are present
        assert "new_worker_token" in orchestrator.auth.worker_tokens
        assert "new_admin_token" in orchestrator.auth.admin_tokens
        assert orchestrator.auth.worker_tokens["new_worker_token"] == "new_worker"
        assert orchestrator.auth.admin_tokens["new_admin_token"] == "new_admin"

    @pytest.mark.asyncio
    async def test_config_reload_handles_auth_failure_gracefully(self, orchestrator):
        """Test that config reload handles auth update failures gracefully."""
        # Simulate a config reload with invalid auth configuration
        new_config = {
            "orchestrator": {
                "auth": {
                    "worker_tokens": [{"token": "missing_name_token"}]  # Invalid: missing name
                }
            }
        }

        # Mock websocket
        websocket = AsyncMock(spec=ServerConnection)

        # Perform config reload - should not crash
        await orchestrator._handle_config_reload(websocket, new_config)

        # Verify websocket.send was called with warnings about auth failure
        websocket.send.assert_called()
        call_args = websocket.send.call_args[0][0]
        response = json.loads(call_args)

        assert response["type"] == "reload_complete"
        assert any("Auth update failed" in warning for warning in response["warnings"])

    @pytest.mark.asyncio
    async def test_authenticate_tokens_work_after_reload(self, orchestrator):
        """Test that authentication still works after config reload."""
        # Test authentication before reload
        auth_result = orchestrator.auth.authenticate("worker_token_1")
        assert auth_result.role == "worker"
        assert auth_result.name == "test_worker_1"

        # Reload config with same auth
        new_config = {
            "orchestrator": {
                "auth": {
                    "worker_tokens": [
                        {"name": "test_worker_1", "token": "worker_token_1"},
                        {"name": "test_worker_2", "token": "worker_token_2"},
                    ],
                    "admin_tokens": [{"name": "test_admin", "token": "admin_token_1"}],
                    "monitor_tokens": [{"name": "test_monitor", "token": "monitor_token_1"}],
                }
            }
        }

        websocket = AsyncMock(spec=ServerConnection)
        await orchestrator._handle_config_reload(websocket, new_config)

        # Test authentication after reload - should still work
        auth_result_after = orchestrator.auth.authenticate("worker_token_1")
        assert auth_result_after.role == "worker"
        assert auth_result_after.name == "test_worker_1"
