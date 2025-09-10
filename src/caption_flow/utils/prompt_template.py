"""Prompt template system for dynamic column substitution."""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Handles prompt templates with column substitution."""

    # Pattern to match {column:column_name} or {col:column_name}
    COLUMN_PATTERN = re.compile(r"\{(?:column|col):([\w-]+)\}")

    def __init__(self, template: str):
        """Initialize with a prompt template.

        Args:
        ----
            template: Prompt template string, e.g.
                     "describe this image. tags: {column:user_tags}"

        """
        self.template = template
        self.required_columns = self._extract_columns()

    def _extract_columns(self) -> List[str]:
        """Extract required column names from template."""
        matches = self.COLUMN_PATTERN.findall(self.template)
        return list(set(matches))  # Remove duplicates

    def format(self, item_data: Dict[str, Any]) -> str:
        """Format the template with actual column values.

        Args:
        ----
            item_data: Dictionary containing column values from dataset

        Returns:
        -------
            Formatted prompt string

        """
        prompt = self.template

        # Replace all column references
        for match in self.COLUMN_PATTERN.finditer(self.template):
            full_match = match.group(0)  # e.g., {column:user_tags}
            column_name = match.group(1)  # e.g., user_tags

            # Get column value with fallback
            value = item_data.get(column_name, "")

            # Handle different value types
            if value is None:
                value = ""
            elif isinstance(value, list):
                # Join list items with commas
                value = ", ".join(str(v) for v in value if v)
            elif not isinstance(value, str):
                value = str(value)

            # Replace in prompt
            prompt = prompt.replace(full_match, value)

        return prompt.strip()

    def validate_columns(self, available_columns: List[str]) -> List[str]:
        """Validate that required columns are available.

        Returns
        -------
            List of missing column names

        """
        missing = []
        for col in self.required_columns:
            if col not in available_columns:
                missing.append(col)
        return missing


class PromptTemplateManager:
    """Manages multiple prompt templates."""

    def __init__(self, prompts: List[str]):
        """Initialize with list of prompt strings (which may contain templates).

        Args:
        ----
            prompts: List of prompt strings

        """
        self.templates = [PromptTemplate(p) for p in prompts]
        self._all_required_columns = None

    @property
    def required_columns(self) -> List[str]:
        """Get all required columns across all templates."""
        if self._all_required_columns is None:
            cols = set()
            for template in self.templates:
                cols.update(template.required_columns)
            self._all_required_columns = list(cols)
        return self._all_required_columns

    def format_all(self, item_data: Dict[str, Any]) -> List[str]:
        """Format all templates with item data.

        Args:
        ----
            item_data: Dictionary containing column values

        Returns:
        -------
            List of formatted prompts

        """
        formatted = []
        for template in self.templates:
            try:
                prompt = template.format(item_data)
                formatted.append(prompt)
            except Exception as e:
                logger.error(f"Error formatting prompt template '{template.template}': {e}")
                # Fall back to raw template
                formatted.append(template.template)

        return formatted

    def validate_all(self, available_columns: List[str]) -> Dict[str, List[str]]:
        """Validate all templates against available columns.

        Returns
        -------
            Dict mapping template string to list of missing columns

        """
        issues = {}
        for template in self.templates:
            missing = template.validate_columns(available_columns)
            if missing:
                issues[template.template] = missing
        return issues
