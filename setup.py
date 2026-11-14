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
    "websockets>=17.4,<18.0",
    "pyarrow>=26.0.1,<27.0.0",
    "click>=8.5.1,<9.0.0",
    "pydantic>=2.14.3,<3.0.0",
    "aiofiles>=25.1.0,<26.0.0",
    "rich>=15.0.0,<16.0.0",
    "cryptography>=50.0.2,<51.0.0",
    "PyYAML>=6.0.3,<7.0.0",
    "certbot>=5.10.0,<6.0.0",
    # vLLM's mistral-common requires NumPy <2.4 on Python 3.12;
    # its pinned numba 0.65 requires NumPy <2.5 on Python 3.13.
    "numpy>=2.3.5,<2.4.0; python_version < '3.13'",
    "numpy>=2.4.6,<2.5.0; python_version >= '3.13'",
    "Pillow>=13.0.0,<14.0.0",
    "trainingsample>=0.3.2,<0.4.0",
    "pandas>=3.1.1,<4.0.0",
    "datasets>=5.2.0,<6.0.0",
    "boto3>=1.44.3,<2.0.0",
    "webshart>=0.7.0,<0.8.0",
    "pylance>=8.0.1,<9.0.0",
    "duckdb>=1.5.6,<2.0.0",
    "aiohttp>=3.14.5,<4.0.0",
    # vLLM 0.33 requires FastAPI below 0.137.
    "fastapi>=0.136.3,<0.137.0",
    "uvicorn>=0.54.0,<1.0.0",
    "huggingface-hub>=1.33.0,<2.0.0",
    "psutil>=7.2.2,<8.0.0",
    "requests>=2.34.2,<3.0.0",
    "tqdm>=4.70.1,<5.0.0",
    "urwid>=5.1.3,<6.0.0",
]

# Match vLLM 0.33's published requirements, including torchaudio 2.11.0.
# Keep these explicit so a worker install cannot silently combine incompatible
# torch packages when another dependency broadens its requirements.
vllm_deps = [
    "vllm>=0.33.0,<0.34.0",
    "torch==2.14.0",
    "torchvision==0.29.0",
    "torchaudio==2.11.0",
    "transformers>=5.16.1,<5.21.0",
    "qwen-vl-utils>=0.0.14,<0.1.0",
]

PYTORCH_DEPENDENCIES = [
    "torch>=2.14.0",
    "torchvision>=0.29.0",
    "torchaudio>=2.11.0",
]

APPLE_PYTORCH_DEPENDENCIES = [
    *PYTORCH_DEPENDENCIES,
    "qwen-vl-utils>=0.0.14,<0.1.0",
    # Install the matching vLLM core and Metal wheels from upstream GitHub
    # releases separately; PyPI's vllm-metal 0.1.0 is obsolete.
]

extras_require = {
    "vllm": vllm_deps,
    "captioning": vllm_deps,
    # The OpenAI-compatible worker uses aiohttp from the base install and does
    # not pull in the GPU stack. This named extra makes deployment intent clear.
    "openai": [],
    "cpu": PYTORCH_DEPENDENCIES,
    "cuda": PYTORCH_DEPENDENCIES,
    "cuda13": PYTORCH_DEPENDENCIES,
    "rocm": PYTORCH_DEPENDENCIES,
    "apple": APPLE_PYTORCH_DEPENDENCIES,
    "dev": [
        "pytest>=9.2.0,<10.0.0",
        "pytest-asyncio>=1.5.0,<2.0.0",
        "pytest-cov>=7.1.0,<8.0.0",
        "transformers>=5.16.1,<5.21.0",
        "black>=26.10.1,<27.0.0",
        "ruff>=0.17.5,<1.0.0",
        "mypy>=2.4.1,<3.0.0",
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
    python_requires=">=3.12,<3.14",
    install_requires=base_deps,
    extras_require=extras_require,
    entry_points={"console_scripts": ["caption-flow=caption_flow.cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords=["captioning", "distributed", "vllm", "dataset", "community"],
    project_urls={
        "Homepage": "https://github.com/bghira/CaptionFlow",
        "Repository": "https://github.com/bghira/CaptionFlow",
    },
)
