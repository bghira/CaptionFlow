"""Comprehensive tests for the Monitor module."""

import ssl
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

from caption_flow.monitor import Monitor


@pytest.fixture
def monitor_config():
    """Sample monitor configuration."""
    return {"server": "wss://localhost:8765/monitor", "token": "test-token", "verify_ssl": True}


@pytest.fixture
def monitor_config_no_ssl():
    """Monitor configuration with SSL disabled."""
    return {"server": "wss://localhost:8765/monitor", "token": "test-token", "verify_ssl": False}


@pytest.fixture
def sample_stats():
    """Sample statistics data."""
    return {
        "total_items": 1000,
        "completed_items": 750,
        "active_workers": 3,
        "current_rate": 15.5,
        "average_rate": 12.3,
        "expected_rate": 20.0,
        "queue_size": 250,
    }


@pytest.fixture
def sample_leaderboard():
    """Sample leaderboard data."""
    return [
        {"name": "Worker-1", "completed": 300, "rate": 5.2},
        {"name": "Worker-2", "completed": 250, "rate": 4.8},
        {"name": "Worker-3", "completed": 200, "rate": 4.3},
    ]


@pytest.fixture
def sample_activity():
    """Sample activity data."""
    return [
        {
            "timestamp": datetime.now().isoformat(),
            "worker": "Worker-1",
            "action": "completed",
            "item": "image_001.jpg",
        },
        {
            "timestamp": datetime.now().isoformat(),
            "worker": "Worker-2",
            "action": "started",
            "item": "image_002.jpg",
        },
    ]


class TestMonitorInit:
    """Test Monitor initialization."""

    def test_init_basic(self, monitor_config):
        """Test basic monitor initialization."""
        monitor = Monitor(monitor_config)

        assert monitor.config == monitor_config
        assert monitor.server_url == monitor_config["server"]
        assert monitor.token == monitor_config["token"]
        assert monitor.stats == {}
        assert monitor.leaderboard == []
        assert monitor.recent_activity == []
        assert not monitor.running
        assert monitor.ssl_context is not None

    def test_init_ssl_verification_disabled(self, monitor_config_no_ssl):
        """Test monitor initialization with SSL verification disabled."""
        monitor = Monitor(monitor_config_no_ssl)

        assert monitor.ssl_context is not None
        assert not monitor.ssl_context.check_hostname
        assert monitor.ssl_context.verify_mode == ssl.CERT_NONE

    @patch("ssl.create_default_context")
    def test_setup_ssl_with_verification(self, mock_ssl_context, monitor_config):
        """Test SSL setup with verification enabled."""
        mock_context = Mock()
        mock_ssl_context.return_value = mock_context

        Monitor(monitor_config)

        mock_ssl_context.assert_called_once()
        # Should not modify context when verification is enabled
        assert not hasattr(mock_context, "check_hostname") or mock_context.check_hostname

    @patch("ssl.create_default_context")
    def test_setup_ssl_without_verification(self, mock_ssl_context, monitor_config_no_ssl):
        """Test SSL setup with verification disabled."""
        mock_context = Mock()
        mock_ssl_context.return_value = mock_context

        Monitor(monitor_config_no_ssl)

        mock_ssl_context.assert_called_once()
        assert not mock_context.check_hostname
        assert mock_context.verify_mode == ssl.CERT_NONE


class TestMonitorHandleUpdate:
    """Test Monitor update handling."""

    @pytest.mark.asyncio
    async def test_handle_stats_update(self, monitor_config, sample_stats):
        """Test handling stats updates."""
        monitor = Monitor(monitor_config)

        update_data = {"type": "stats", "data": sample_stats}

        await monitor._handle_update(update_data)

        assert monitor.stats == sample_stats
        assert monitor.rate_info["current_rate"] == sample_stats["current_rate"]

    @pytest.mark.asyncio
    async def test_handle_leaderboard_update(self, monitor_config, sample_leaderboard):
        """Test handling leaderboard updates."""
        monitor = Monitor(monitor_config)

        update_data = {"type": "leaderboard", "data": sample_leaderboard}

        await monitor._handle_update(update_data)

        assert monitor.leaderboard == sample_leaderboard

    @pytest.mark.asyncio
    async def test_handle_activity_update(self, monitor_config, sample_activity):
        """Test handling activity updates."""
        monitor = Monitor(monitor_config)

        update_data = {"type": "activity", "data": sample_activity}

        await monitor._handle_update(update_data)

        # The monitor appends the data as a single item to recent_activity
        assert len(monitor.recent_activity) == 1
        assert monitor.recent_activity[0] == sample_activity

    @pytest.mark.asyncio
    async def test_handle_unknown_update_type(self, monitor_config):
        """Test handling unknown update types."""
        monitor = Monitor(monitor_config)
        original_stats = monitor.stats.copy()

        update_data = {"type": "unknown_type", "data": {"some": "data"}}

        # Should not raise exception and not modify stats
        await monitor._handle_update(update_data)
        assert monitor.stats == original_stats

    @pytest.mark.asyncio
    async def test_handle_update_missing_type(self, monitor_config):
        """Test handling updates without type field."""
        monitor = Monitor(monitor_config)
        original_stats = monitor.stats.copy()

        update_data = {"data": {"some": "data"}}

        # Should not raise exception and not modify stats
        await monitor._handle_update(update_data)
        assert monitor.stats == original_stats


class TestMonitorConnection:
    """Test Monitor WebSocket connection handling."""

    @pytest.mark.asyncio
    async def test_connect_to_orchestrator_connection_error(self, monitor_config):
        """Test connection error handling."""
        monitor = Monitor(monitor_config)
        monitor.running = True

        call_count = 0

        def connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Stop after first error to avoid infinite loop
            monitor.running = False
            raise ConnectionError("Connection failed")

        with patch("websockets.connect", side_effect=connect_side_effect):
            with patch("asyncio.sleep"):
                try:
                    await monitor._connect_to_orchestrator()
                except ConnectionError:
                    pass

                # Should have attempted connection
                assert call_count == 1

    @pytest.mark.asyncio
    async def test_connect_with_ssl_context(self, monitor_config):
        """Test connection uses SSL context for wss URLs."""
        monitor = Monitor(monitor_config)
        monitor.running = True

        mock_websocket = AsyncMock()
        mock_websocket.__aenter__ = AsyncMock(return_value=mock_websocket)
        mock_websocket.__aexit__ = AsyncMock(return_value=None)
        mock_websocket.__aiter__.return_value = iter([])

        call_count = 0

        def connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Stop after first attempt
            monitor.running = False
            return mock_websocket

        with patch(
            "caption_flow.monitor.websockets.connect", side_effect=connect_side_effect
        ) as mock_connect:
            try:
                await monitor._connect_to_orchestrator()
            except:
                pass

            # Should have called connect with SSL context
            mock_connect.assert_called()
            args, kwargs = mock_connect.call_args
            assert "ssl" in kwargs
            assert kwargs["ssl"] is not None

    @pytest.mark.asyncio
    async def test_connect_without_ssl_for_ws_urls(self, monitor_config):
        """Test connection without SSL for ws:// URLs."""
        # Change to ws:// URL
        monitor_config["server"] = "ws://localhost:8765/monitor"
        monitor = Monitor(monitor_config)
        monitor.running = True

        mock_websocket = AsyncMock()
        mock_websocket.__aenter__ = AsyncMock(return_value=mock_websocket)
        mock_websocket.__aexit__ = AsyncMock(return_value=None)
        mock_websocket.__aiter__.return_value = iter([])

        call_count = 0

        def connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Stop after first attempt
            monitor.running = False
            return mock_websocket

        with patch(
            "caption_flow.monitor.websockets.connect", side_effect=connect_side_effect
        ) as mock_connect:
            try:
                await monitor._connect_to_orchestrator()
            except:
                pass

            # Should have called connect without SSL context
            mock_connect.assert_called()
            args, kwargs = mock_connect.call_args
            assert kwargs.get("ssl") is None


class TestMonitorStart:
    """Test Monitor start functionality."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, monitor_config):
        """Test that start sets the running flag."""
        monitor = Monitor(monitor_config)

        with patch.object(monitor, "_connect_to_orchestrator"):
            with patch.object(monitor, "_display_loop") as mock_display:
                with patch("asyncio.create_task") as mock_create_task:
                    mock_display.return_value = AsyncMock()

                    await monitor.start()

                    assert monitor.running
                    mock_create_task.assert_called_once()  # Connection task created
                    mock_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_creates_connection_task(self, monitor_config):
        """Test that start creates connection task."""
        monitor = Monitor(monitor_config)

        with patch.object(monitor, "_connect_to_orchestrator"):
            with patch.object(monitor, "_display_loop") as mock_display:
                with patch("asyncio.create_task") as mock_create_task:
                    mock_display.return_value = AsyncMock()

                    await monitor.start()

                    mock_create_task.assert_called_once()


class TestMonitorDisplayLoop:
    """Test Monitor display loop functionality."""

    @pytest.mark.asyncio
    async def test_display_loop_basic(self, monitor_config):
        """Test basic display loop functionality."""
        monitor = Monitor(monitor_config)
        monitor.running = True

        # Mock Rich components
        with patch("caption_flow.monitor.Live") as mock_live_class:
            with patch("caption_flow.monitor.Layout"):
                with patch.object(monitor, "_create_layout") as mock_create_layout:
                    with patch.object(monitor, "_update_layout"):
                        mock_live = Mock()
                        mock_live.__enter__ = Mock(return_value=mock_live)
                        mock_live.__exit__ = Mock(return_value=None)
                        mock_live_class.return_value = mock_live

                        # Stop immediately to avoid infinite loop
                        monitor.running = False

                        try:
                            await monitor._display_loop()
                        except Exception:
                            pass

                        # Should have created layout
                        mock_create_layout.assert_called()


class TestMonitorIntegration:
    """Integration tests for Monitor functionality."""

    @pytest.mark.asyncio
    async def test_full_update_cycle(self, monitor_config, sample_stats, sample_leaderboard):
        """Test a full update cycle with different message types."""
        monitor = Monitor(monitor_config)

        # Test stats update
        stats_update = {"type": "stats", "data": sample_stats}
        await monitor._handle_update(stats_update)

        assert monitor.stats == sample_stats
        assert monitor.rate_info["current_rate"] == sample_stats["current_rate"]

        # Test leaderboard update
        leaderboard_update = {"type": "leaderboard", "data": sample_leaderboard}
        await monitor._handle_update(leaderboard_update)

        assert monitor.leaderboard == sample_leaderboard

    def test_rate_info_initialization(self, monitor_config):
        """Test that rate info is properly initialized."""
        monitor = Monitor(monitor_config)

        expected_keys = ["current_rate", "average_rate", "expected_rate"]
        for key in expected_keys:
            assert key in monitor.rate_info
            assert monitor.rate_info[key] == 0.0

    def test_console_initialization(self, monitor_config):
        """Test that Rich console is properly initialized."""
        monitor = Monitor(monitor_config)

        assert monitor.console is not None
        assert hasattr(monitor.console, "print")


if __name__ == "__main__":
    pytest.main([__file__])
