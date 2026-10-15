"""The published dependency must support native structured caption round trips."""

import ast
from pathlib import Path

from packaging.requirements import Requirement


def test_base_dependency_requires_structured_caption_support():
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    dependencies = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "base_deps" for target in node.targets
        )
    )
    requirement = next(
        requirement
        for dependency in dependencies
        if (requirement := Requirement(dependency)).name == "webshart"
    )
    assert "0.6.0" in requirement.specifier
    assert "0.5.4" not in requirement.specifier
    assert "0.7.0" not in requirement.specifier
