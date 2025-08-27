"""Storage exporter for converting Parquet data to various formats."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import pandas as pd
import numpy as np
from ..models import StorageContents, ExportError

logger = logging.getLogger(__name__)


class StorageExporter:
    """Exports StorageContents to various formats."""

    def __init__(self, contents: StorageContents):
        """Initialize exporter with storage contents.

        Args:
            contents: StorageContents instance to export
        """
        self.contents = contents
        self._validate_contents()

    def _validate_contents(self):
        """Validate that contents are suitable for export."""
        if not self.contents.rows:
            logger.warning("No rows to export")
        if not self.contents.columns:
            raise ExportError("No columns defined for export")

    def _flatten_lists(self, value: Any) -> str:
        """Convert list values to newline-separated strings."""
        if isinstance(value, list):
            # Strip newlines from each element and join
            return "\n".join(str(item).replace("\n", " ") for item in value)
        return str(value) if value is not None else ""

    def _serialize_value(self, value: Any) -> Any:
        """Convert values to JSON-serializable format."""
        if pd.api.types.is_datetime64_any_dtype(type(value)) or isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        return value

    def to_jsonl(self, output_path: Union[str, Path]) -> int:
        """Export to JSONL (JSON Lines) format.

        Args:
            output_path: Path to output JSONL file

        Returns:
            Number of rows exported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows_written = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for row in self.contents.rows:
                # Convert non-serializable values
                serializable_row = {k: self._serialize_value(v) for k, v in row.items()}
                # Write each row as a JSON object on its own line
                json_line = json.dumps(serializable_row, ensure_ascii=False)
                f.write(json_line + "\n")
                rows_written += 1

        logger.info(f"Exported {rows_written} rows to JSONL: {output_path}")
        return rows_written

    def _get_filename_from_row(self, row: Dict[str, Any], filename_column: str) -> Optional[str]:
        """Extract filename from row, falling back to URL if needed."""
        # Try the specified filename column first
        filename = row.get(filename_column)
        if filename:
            return filename

        # Fall back to URL if available
        url = row.get("url")
        if url:
            # Extract filename from URL path
            from urllib.parse import urlparse

            parsed = urlparse(str(url))
            path_parts = parsed.path.rstrip("/").split("/")
            if path_parts and path_parts[-1]:
                return path_parts[-1]

        return None

    def to_json(self, output_dir: Union[str, Path], filename_column: str = "filename") -> int:
        """Export to individual JSON files based on filename column.

        Args:
            output_dir: Directory to write JSON files
            filename_column: Column containing the base filename

        Returns:
            Number of files created
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if we need to fall back to URL
        using_url_fallback = False
        if filename_column not in self.contents.columns and "url" in self.contents.columns:
            logger.warning(f"Column '{filename_column}' not found, falling back to 'url' column")
            using_url_fallback = True
        elif filename_column not in self.contents.columns:
            raise ExportError(f"Column '{filename_column}' not found and no 'url' column available")

        files_created = 0
        skipped_count = 0

        for row in self.contents.rows:
            filename = self._get_filename_from_row(row, filename_column)
            if not filename:
                skipped_count += 1
                logger.warning(f"Skipping row with no extractable filename")
                continue

            # Create JSON filename from original filename
            base_name = Path(filename).stem
            json_path = output_dir / f"{base_name}.json"

            # Convert non-serializable values
            serializable_row = {k: self._serialize_value(v) for k, v in row.items()}

            # Write row data as JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_row, f, ensure_ascii=False, indent=2)

            files_created += 1

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} rows with no extractable filename")

        logger.info(f"Created {files_created} JSON files in: {output_dir}")
        return files_created

    def to_csv(self, output_path: Union[str, Path]) -> int:
        """Export to CSV format, skipping complex columns.

        Args:
            output_path: Path to output CSV file

        Returns:
            Number of rows exported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Identify complex columns to skip
        complex_columns = set()
        csv_safe_columns = []

        # Check column types by sampling data
        sample_size = min(10, len(self.contents.rows))
        for row in self.contents.rows[:sample_size]:
            for col, value in row.items():
                if col not in complex_columns and value is not None:
                    # Skip dictionaries and non-output field lists
                    if isinstance(value, dict):
                        complex_columns.add(col)
                        logger.warning(
                            f"Column '{col}' contains dict type and will be skipped. "
                            "Consider using JSONL format for complete data export."
                        )
                    elif isinstance(value, list) and col not in self.contents.output_fields:
                        complex_columns.add(col)
                        logger.warning(
                            f"Column '{col}' contains list type and will be skipped. "
                            "Consider using JSONL format for complete data export."
                        )

        # Build list of CSV-safe columns
        csv_safe_columns = [col for col in self.contents.columns if col not in complex_columns]

        if not csv_safe_columns:
            raise ExportError("No columns suitable for CSV export. Use JSONL format instead.")

        # Prepare rows for CSV export with safe columns only
        csv_rows = []
        for row in self.contents.rows:
            csv_row = {}
            for col in csv_safe_columns:
                value = row.get(col)
                # Handle list values (like captions) by joining with newlines
                if isinstance(value, list):
                    csv_row[col] = self._flatten_lists(value)
                elif pd.api.types.is_datetime64_any_dtype(type(value)) or isinstance(
                    value, pd.Timestamp
                ):
                    csv_row[col] = self._serialize_value(value)
                else:
                    csv_row[col] = value
            csv_rows.append(csv_row)

        # Write to CSV
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_safe_columns)
            writer.writeheader()
            writer.writerows(csv_rows)

        # Log results
        if complex_columns:
            skipped_msg = f"Skipped {len(complex_columns)} complex columns: {', '.join(sorted(complex_columns))}"
            logger.warning(skipped_msg)

        logger.info(
            f"Exported {len(csv_rows)} rows to CSV: {output_path} "
            f"(with {len(csv_safe_columns)}/{len(self.contents.columns)} columns)"
        )

        return len(csv_rows)

    def to_txt(
        self,
        output_dir: Union[str, Path],
        filename_column: str = "filename",
        export_column: str = "captions",
    ) -> int:
        """Export specific column to individual text files.

        Args:
            output_dir: Directory to write text files
            filename_column: Column containing the base filename
            export_column: Column to export to text files

        Returns:
            Number of files created
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if we need to fall back to URL
        using_url_fallback = False
        if filename_column not in self.contents.columns and "url" in self.contents.columns:
            logger.warning(f"Column '{filename_column}' not found, falling back to 'url' column")
            using_url_fallback = True
        elif filename_column not in self.contents.columns:
            raise ExportError(f"Column '{filename_column}' not found and no 'url' column available")

        if export_column not in self.contents.columns:
            # Check if it's an output field
            if export_column not in self.contents.output_fields:
                raise ExportError(f"Column '{export_column}' not found in data")

        files_created = 0
        skipped_no_filename = 0
        skipped_no_content = 0

        for row in self.contents.rows:
            filename = self._get_filename_from_row(row, filename_column)
            if not filename:
                skipped_no_filename += 1
                logger.warning(f"Skipping row with no extractable filename")
                continue

            content = row.get(export_column)
            if content is None:
                skipped_no_content += 1
                logger.warning(f"No {export_column} for {filename}")
                continue

            # Create text filename from original filename
            base_name = Path(filename).stem
            txt_path = output_dir / f"{base_name}.txt"

            # Write content
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(self._flatten_lists(content))

            files_created += 1

        if skipped_no_filename > 0:
            logger.warning(f"Skipped {skipped_no_filename} rows with no extractable filename")
        if skipped_no_content > 0:
            logger.warning(f"Skipped {skipped_no_content} rows with no {export_column} content")

        logger.info(f"Created {files_created} text files in: {output_dir}")
        return files_created
