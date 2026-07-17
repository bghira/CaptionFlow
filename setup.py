#!/usr/bin/env python3

"""Setuptools configuration for CaptionFlow.

The project keeps dependency selection here so that the GPU worker stack can
be installed separately from the orchestrator and monitor components.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


def get_version() -> str:
    """Read the package version without importing CaptionFlow's dependencies."""
    version_file = ROOT / "src" / "caption_flow" / "__init__.py"
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("Unable to determine CaptionFlow version")


base_deps = [
    "websockets>=16.0,<17.0",
    "pyarrow>=21.0.0,<26.0.0",
    "click>=8.2.0,<9.0.0",
    "pydantic>=2.12.0,<3.0.0",
    "aiofiles>=24.1.0,<26.0.0",
    "rich>=14.0.0,<16.0.0",
    "cryptography>=45.0.0,<50.0.0",
    "PyYAML>=6.0.2,<7.0.0",
    "certbot>=5.0.0,<6.0.0",
    "numpy>=2.2.0,<3.0.0",
    "Pillow>=11.3.0,<13.0.0",
    "pandas>=3.0.0,<4.0.0",
    "datasets>=5.0.0,<6.0.0",
    "boto3>=1.43.0,<2.0.0",
    "webshart>=0.4.6,<0.5.0",
    "pylance>=8.0.0,<9.0.0",
    "duckdb>=1.5.0,<2.0.0",
    "aiohttp>=3.13.3,<4.0.0",
    "fastapi>=0.133.0,<0.137.0",
    "uvicorn>=0.35.0,<1.0.0",
    "huggingface-hub>=1.5.0,<2.0.0",
    "opencv-python-headless>=4.13.0,<6.0.0",
    "psutil>=7.0.0,<8.0.0",
    "requests>=2.32.0,<3.0.0",
    "tqdm>=4.67.0,<5.0.0",
    "urwid>=3.0.2,<5.0.0",
]

# vLLM 0.25.1 currently selects the PyTorch 2.11.0 family. Keep these
# explicit so a worker install cannot silently combine incompatible torch
# packages when another dependency broadens its requirements.
vllm_deps = [
    "vllm>=0.25.1,<0.26.0",
    "torch==2.11.0",
    "torchvision==0.26.0",
    "torchaudio==2.11.0",
    "transformers>=5.5.3,<6.0.0",
    "qwen-vl-utils>=0.0.14,<0.1.0",
]

PYTORCH_DEPENDENCIES = [
    "torch>=2.11.0",
    "torchvision>=0.26.0",
    "torchaudio>=2.11.0",
]

APPLE_PYTORCH_DEPENDENCIES = [
    "torch>=2.13.0",
    "torchvision>=0.28.0",
    "torchaudio>=2.11.0",
    # vLLM Metal publishes Apple Silicon wheels for Python 3.12+.
    "vllm-metal==0.1.0; platform_system == 'Darwin' and platform_machine == 'arm64' and python_version >= '3.12'",
]

extras_require = {
    "vllm": vllm_deps,
    "captioning": vllm_deps,
    "cpu": PYTORCH_DEPENDENCIES,
    "cuda": PYTORCH_DEPENDENCIES,
    "cuda13": PYTORCH_DEPENDENCIES,
    "rocm": PYTORCH_DEPENDENCIES,
    "apple": APPLE_PYTORCH_DEPENDENCIES,
    "dev": [
        "pytest>=8.0.0,<9.0.0",
        "pytest-asyncio>=1.1.0,<2.0.0",
        "pytest-cov>=6.0.0,<7.0.0",
        "black>=25.0.0,<26.0.0",
        "ruff>=0.12.0,<1.0.0",
        "mypy>=1.17.0,<2.0.0",
    ],
    "all": [*vllm_deps],
}


setup(
    name="caption-flow",
    version=get_version(),
    description="Self-contained distributed community captioning system",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="bghira",
    author_email="bghira@users.github.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.11,<3.13",
    install_requires=base_deps,
    extras_require=extras_require,
    entry_points={"console_scripts": ["caption-flow=caption_flow.cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=["captioning", "distributed", "vllm", "dataset", "community"],
    project_urls={
        "Homepage": "https://github.com/bghira/CaptionFlow",
        "Repository": "https://github.com/bghira/CaptionFlow",
    },
)
