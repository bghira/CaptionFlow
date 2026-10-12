"""Check the published worker dependencies without installing a GPU runtime."""

import runpy
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest
from packaging.requirements import Requirement


@pytest.fixture(scope="module")
def package_config():
    setuptools = ModuleType("setuptools")
    setuptools.find_packages = Mock(return_value=["caption_flow"])
    setuptools.setup = Mock()
    with patch.dict("sys.modules", {"setuptools": setuptools}):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    return setuptools.setup.call_args.kwargs


def requirements_by_name(requirements):
    parsed = [Requirement(value) for value in requirements]
    return {requirement.name: requirement for requirement in parsed}


@pytest.mark.parametrize("extra", ["vllm", "captioning", "all"])
def test_vllm_extras_select_matching_runtime(package_config, extra):
    dependencies = requirements_by_name(package_config["extras_require"][extra])
    assert dependencies["vllm"].specifier.contains("0.31.0")
    assert not dependencies["vllm"].specifier.contains("0.30.0")
    assert not dependencies["vllm"].specifier.contains("0.32.0")
    for name, version in (
        ("torch", "2.13.0"),
        ("torchvision", "0.28.0"),
        ("torchaudio", "2.11.0"),
    ):
        assert str(dependencies[name].specifier) == f"=={version}"


@pytest.mark.parametrize("extra", ["vllm", "captioning", "all", "dev"])
def test_transformers_matches_vllm_supported_range(package_config, extra):
    dependencies = requirements_by_name(package_config["extras_require"][extra])
    versions = dependencies["transformers"].specifier
    assert versions.contains("5.10.4")
    assert versions.contains("5.17.0")
    assert not versions.contains("5.10.3")
    assert not versions.contains("5.18.0")


@pytest.mark.parametrize("extra", ["cpu", "cuda", "cuda13", "rocm", "apple"])
def test_pytorch_extras_require_updated_family(package_config, extra):
    dependencies = requirements_by_name(package_config["extras_require"][extra])
    assert dependencies["torch"].specifier.contains("2.13.0")
    assert not dependencies["torch"].specifier.contains("2.12.0")
    assert dependencies["torchvision"].specifier.contains("0.28.0")
    assert not dependencies["torchvision"].specifier.contains("0.27.0")
    assert dependencies["torchaudio"].specifier.contains("2.11.0")
    assert "vllm" not in dependencies


def test_base_and_api_workers_do_not_require_gpu_stack(package_config):
    dependencies = requirements_by_name(
        package_config["install_requires"] + package_config["extras_require"]["openai"]
    )
    assert not {"vllm", "vllm-metal", "torch", "torchvision", "torchaudio"} & dependencies.keys()


def test_captioning_alias_matches_vllm_extra(package_config):
    extras = package_config["extras_require"]
    assert extras["captioning"] == extras["vllm"] == extras["all"]
