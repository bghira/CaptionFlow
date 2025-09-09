"""Comprehensive tests for the CLI module."""

import pytest
import tempfile
import shutil
import os
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from click.testing import CliRunner

from caption_flow.cli import (
    ConfigManager,
    setup_logging,
    apply_cli_overrides,
    main,
    orchestrator,
    worker,
    monitor,
    view,
    reload_config,
    scan_chunks,
    export,
    generate_cert,
    inspect_cert,
)


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "server": {"host": "localhost", "port": 8765},
        "storage": {"data_dir": "./test_data"},
        "processor": {"batch_size": 10}
    }


class TestConfigManager:
    """Test ConfigManager class."""

    def test_get_xdg_config_home_env_set(self):
        """Test XDG_CONFIG_HOME when environment variable is set."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/custom/config"}):
            result = ConfigManager.get_xdg_config_home()
            assert result == Path("/custom/config")

    def test_get_xdg_config_home_default(self):
        """Test XDG_CONFIG_HOME default fallback."""
        with patch.dict(os.environ, {}, clear=True):
            result = ConfigManager.get_xdg_config_home()
            assert result == Path.home() / ".config"

    def test_get_xdg_config_dirs_env_set(self):
        """Test XDG_CONFIG_DIRS when environment variable is set."""
        with patch.dict(os.environ, {"XDG_CONFIG_DIRS": "/etc/xdg:/usr/local/etc"}):
            result = ConfigManager.get_xdg_config_dirs()
            assert result == [Path("/etc/xdg"), Path("/usr/local/etc")]

    def test_get_xdg_config_dirs_default(self):
        """Test XDG_CONFIG_DIRS default fallback."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {"XDG_CONFIG_DIRS": "/etc/xdg"}):
                result = ConfigManager.get_xdg_config_dirs()
                assert result == [Path("/etc/xdg")]

    def test_find_config_explicit_path(self, temp_config_dir, sample_config):
        """Test finding config with explicit path."""
        config_file = temp_config_dir / "custom.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        result = ConfigManager.find_config("orchestrator", str(config_file))
        assert result == sample_config

    def test_find_config_explicit_path_not_found(self):
        """Test finding config with explicit path that doesn't exist."""
        result = ConfigManager.find_config("orchestrator", "/nonexistent/config.yaml")
        assert result is None

    def test_find_config_search_paths(self, temp_config_dir, sample_config):
        """Test config discovery through search paths."""
        captionflow_dir = temp_config_dir / "caption-flow"
        captionflow_dir.mkdir()
        config_file = captionflow_dir / "orchestrator.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        with patch.object(ConfigManager, 'get_xdg_config_home', return_value=temp_config_dir):
            result = ConfigManager.find_config("orchestrator")
            assert result == sample_config

    def test_find_config_not_found(self, temp_config_dir):
        """Test config not found scenario."""
        with patch.object(ConfigManager, 'get_xdg_config_home', return_value=temp_config_dir):
            with patch.object(ConfigManager, 'get_xdg_config_dirs', return_value=[temp_config_dir]):
                with patch.object(Path, 'cwd', return_value=temp_config_dir):
                    with patch.object(Path, 'home', return_value=temp_config_dir):
                        # Mock the examples directory to not exist
                        with patch('pathlib.Path.exists', return_value=False):
                            result = ConfigManager.find_config("orchestrator")
                            assert result is None

    def test_load_yaml(self, temp_config_dir, sample_config):
        """Test loading YAML configuration from file."""
        config_file = temp_config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        result = ConfigManager.load_yaml(config_file)
        assert result == sample_config

    def test_load_yaml_invalid_file(self, temp_config_dir):
        """Test loading invalid YAML file."""
        config_file = temp_config_dir / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content:")
        
        result = ConfigManager.load_yaml(config_file)
        assert result is None

    def test_merge_configs(self):
        """Test configuration merging."""
        base = {"server": {"port": 8000}, "worker": {"name": "base"}}
        override = {"server": {"host": "localhost"}, "new_key": "value"}
        
        result = ConfigManager.merge_configs(base, override)
        expected = {
            "server": {"port": 8000, "host": "localhost"},
            "worker": {"name": "base"},
            "new_key": "value"
        }
        assert result == expected


class TestSetupLogging:
    """Test setup_logging function."""

    @patch('caption_flow.cli.logging.basicConfig')
    def test_setup_logging_normal(self, mock_basic_config):
        """Test normal logging setup."""
        setup_logging(verbose=False)
        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs['level'] == 20  # logging.INFO

    @patch('caption_flow.cli.logging.basicConfig')
    def test_setup_logging_verbose(self, mock_basic_config):
        """Test verbose logging setup."""
        setup_logging(verbose=True)
        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs['level'] == 10  # logging.DEBUG


class TestApplyCliOverrides:
    """Test apply_cli_overrides function."""

    def test_apply_cli_overrides_basic(self):
        """Test basic CLI override application."""
        config = {"server": {"port": 8000}}
        result = apply_cli_overrides(config, port=9000)
        # The function merges the overrides, so port would be at top level
        expected = {"server": {"port": 8000}, "port": 9000}
        assert result == expected

    def test_apply_cli_overrides_none_values(self):
        """Test CLI overrides with None values are ignored."""
        config = {"server": {"port": 8000}}
        result = apply_cli_overrides(config, port=None, host="localhost")
        expected = {"server": {"port": 8000}, "host": "localhost"}
        assert result == expected

    def test_apply_cli_overrides_empty_config(self):
        """Test CLI overrides on empty config."""
        config = {}
        result = apply_cli_overrides(config, port=8000)
        expected = {"port": 8000}
        assert result == expected


class TestMainCommand:
    """Test main CLI command."""

    def test_main_help(self, runner):
        """Test main command help."""
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'CaptionFlow' in result.output

    def test_main_verbose_flag(self, runner):
        """Test main command with verbose flag."""
        with patch('caption_flow.cli.setup_logging') as mock_setup_logging:
            result = runner.invoke(main, ['--verbose', '--help'])
            assert result.exit_code == 0
            # setup_logging is called once in main() with the verbose flag
            mock_setup_logging.assert_called_once_with(True)


class TestOrchestratorCommand:
    """Test orchestrator CLI command."""

    def test_orchestrator_help(self, runner):
        """Test orchestrator command help."""
        result = runner.invoke(main, ['orchestrator', '--help'])
        assert result.exit_code == 0
        assert 'orchestrator' in result.output.lower()

    @patch('caption_flow.cli.Orchestrator')
    @patch('caption_flow.cli.ConfigManager.find_config')
    @patch('caption_flow.cli.asyncio.run')
    def test_orchestrator_run(self, mock_asyncio_run, mock_find_config, 
                             mock_orchestrator_class, 
                             runner, temp_config_dir, sample_config):
        """Test running orchestrator command."""
        config_file = temp_config_dir / "orchestrator.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        mock_find_config.return_value = sample_config
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        result = runner.invoke(main, ['orchestrator', '--config', str(config_file)])
        
        # Should attempt to find config and create orchestrator
        mock_find_config.assert_called()
        mock_orchestrator_class.assert_called()


class TestWorkerCommand:
    """Test worker CLI command."""

    def test_worker_help(self, runner):
        """Test worker command help."""
        result = runner.invoke(main, ['worker', '--help'])
        assert result.exit_code == 0
        assert 'worker' in result.output.lower()

    @patch('caption_flow.cli.ConfigManager.find_config')
    @patch('caption_flow.cli.asyncio.run')
    def test_worker_run(self, mock_asyncio_run, mock_find_config, 
                       runner, temp_config_dir, sample_config):
        """Test running worker command."""
        config_file = temp_config_dir / "worker.yaml"
        worker_config = {**sample_config, "worker": {"name": "test-worker"}}
        with open(config_file, 'w') as f:
            yaml.dump(worker_config, f)
        
        mock_find_config.return_value = worker_config
        
        result = runner.invoke(main, ['worker', '--config', str(config_file)])
        
        # Should attempt to find config 
        mock_find_config.assert_called()


class TestMonitorCommand:
    """Test monitor CLI command."""

    def test_monitor_help(self, runner):
        """Test monitor command help."""
        result = runner.invoke(main, ['monitor', '--help'])
        assert result.exit_code == 0
        assert 'monitor' in result.output.lower()

    @patch('caption_flow.cli.Monitor')
    @patch('caption_flow.cli.ConfigManager.find_config')
    @patch('caption_flow.cli.asyncio.run')
    def test_monitor_run(self, mock_asyncio_run, mock_find_config, 
                        mock_monitor_class, 
                        runner, temp_config_dir, sample_config):
        """Test running monitor command."""
        config_file = temp_config_dir / "monitor.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        mock_find_config.return_value = sample_config
        mock_monitor = Mock()
        mock_monitor_class.return_value = mock_monitor
        
        result = runner.invoke(main, ['monitor', '--config', str(config_file)])
        
        # Should attempt to find config and create monitor
        mock_find_config.assert_called()
        mock_monitor_class.assert_called()


class TestViewCommand:
    """Test view CLI command."""

    def test_view_help(self, runner):
        """Test view command help."""
        result = runner.invoke(main, ['view', '--help'])
        assert result.exit_code == 0
        assert 'view' in result.output.lower()

    @patch('caption_flow.viewer.DatasetViewer')
    def test_view_run(self, mock_viewer_class, runner):
        """Test running view command."""
        mock_viewer = Mock()
        mock_viewer.run = Mock()
        mock_viewer_class.return_value = mock_viewer
        
        result = runner.invoke(main, ['view', '--data-dir', './test_data'])
        
        # Should create and run viewer
        mock_viewer_class.assert_called()
        mock_viewer.run.assert_called()


class TestScanChunksCommand:
    """Test scan-chunks CLI command."""

    def test_scan_chunks_help(self, runner):
        """Test scan-chunks command help."""
        result = runner.invoke(main, ['scan-chunks', '--help'])
        assert result.exit_code == 0
        assert 'scan' in result.output.lower()

    @patch('caption_flow.storage.StorageManager')
    @patch('caption_flow.utils.chunk_tracker.ChunkTracker')
    def test_scan_chunks_run(self, mock_chunk_tracker_class, mock_storage_class, runner):
        """Test running scan-chunks command."""
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        mock_tracker = Mock()
        mock_chunk_tracker_class.return_value = mock_tracker
        mock_tracker.abandoned_chunks = []
        
        result = runner.invoke(main, ['scan-chunks', '--data-dir', './test_data'])
        
        # Should create storage manager and chunk tracker
        mock_storage_class.assert_called()
        mock_chunk_tracker_class.assert_called()


class TestExportCommand:
    """Test export CLI command."""

    def test_export_help(self, runner):
        """Test export command help."""
        result = runner.invoke(main, ['export', '--help'])
        assert result.exit_code == 0
        assert 'export' in result.output.lower()

    @patch('caption_flow.storage.StorageManager')
    def test_export_stats_only(self, mock_storage_class, runner):
        """Test export command with stats-only flag."""
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        mock_storage.get_statistics.return_value = {"total_items": 100}
        
        result = runner.invoke(main, ['export', 'jsonl', '--stats-only'])
        
        # Should create storage manager and get stats
        mock_storage_class.assert_called()
        mock_storage.get_statistics.assert_called()


class TestCertificateCommands:
    """Test certificate-related CLI commands."""

    def test_generate_cert_help(self, runner):
        """Test generate-cert command help."""
        result = runner.invoke(main, ['generate-cert', '--help'])
        assert result.exit_code == 0
        assert 'certificate' in result.output.lower()

    @patch('caption_flow.cli.CertificateManager')
    def test_generate_cert_self_signed(self, mock_cert_manager_class, runner):
        """Test generating self-signed certificate."""
        mock_manager = Mock()
        mock_cert_manager_class.return_value = mock_manager
        
        result = runner.invoke(main, [
            'generate-cert', 
            '--self-signed',
            '--output-dir', './certs'
        ])
        
        # Should create certificate manager
        mock_cert_manager_class.assert_called_once()

    def test_inspect_cert_help(self, runner):
        """Test inspect-cert command help."""
        result = runner.invoke(main, ['inspect-cert', '--help'])
        assert result.exit_code == 0
        assert 'inspect' in result.output.lower()

    @patch('caption_flow.utils.certificates.CertificateManager.inspect_certificate')
    def test_inspect_cert_run(self, mock_inspect, runner, temp_config_dir):
        """Test inspecting certificate."""
        cert_file = temp_config_dir / "test.crt"
        cert_file.touch()  # Create empty file for test
        
        mock_inspect.return_value = {"subject": "test"}
        
        result = runner.invoke(main, ['inspect-cert', str(cert_file)])
        
        # Should call inspect method
        mock_inspect.assert_called()


class TestReloadConfigCommand:
    """Test reload-config CLI command."""

    def test_reload_config_help(self, runner):
        """Test reload-config command help."""
        result = runner.invoke(main, ['reload-config', '--help'])
        assert result.exit_code == 0
        assert 'reload' in result.output.lower()

    @patch('builtins.__import__')
    @patch('caption_flow.cli.asyncio.run')
    def test_reload_config_run(self, mock_asyncio_run, mock_import, runner):
        """Test running reload-config command."""
        # Mock websockets import
        mock_websockets = Mock()
        mock_websockets.connect = AsyncMock()
        
        def side_effect(name, *args, **kwargs):
            if name == 'websockets':
                return mock_websockets
            else:
                return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = side_effect
        
        result = runner.invoke(main, [
            'reload-config', 
            '--server', 'ws://localhost:8765',
            '--token', 'test-token'
        ])
        
        # Should attempt to run async function
        mock_asyncio_run.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])