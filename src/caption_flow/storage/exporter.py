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

    def to_huggingface_hub(
        self,
        dataset_name: str,
        token: Optional[str] = None,
        license: Optional[str] = None,
        private: bool = False,
        nsfw: bool = False,
        tags: Optional[List[str]] = None,
        language: str = "en",
        task_categories: Optional[List[str]] = None,
    ) -> str:
        """Export to Hugging Face Hub as a dataset.

        Args:
            dataset_name: Name for the dataset (e.g., "username/dataset-name")
            token: Hugging Face API token
            license: License for the dataset (required for new repos)
            private: Whether to make the dataset private
            nsfw: Whether to add not-for-all-audiences tag
            tags: Additional tags for the dataset
            language: Language code (default: "en")
            task_categories: Task categories (default: ["text-to-image", "image-to-image"])

        Returns:
            URL of the uploaded dataset
        """
        try:
            from huggingface_hub import HfApi, DatasetCard, create_repo
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ExportError(
                "huggingface_hub and pyarrow are required for HF export. "
                "Install with: pip install huggingface_hub pyarrow"
            )

        # Initialize HF API
        api = HfApi(token=token)

        # Check if repo exists
        repo_exists = False
        try:
            api.dataset_info(dataset_name)
            repo_exists = True
            logger.info(f"Dataset {dataset_name} already exists, will update it")
        except:
            logger.info(f"Creating new dataset: {dataset_name}")
            if not license:
                raise ExportError("License is required when creating a new dataset")

        # Create repo if it doesn't exist
        if not repo_exists:
            create_repo(repo_id=dataset_name, repo_type="dataset", private=private, token=token)

        # Prepare data for parquet
        df = pd.DataFrame(self.contents.rows)

        # Convert any remaining non-serializable types
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: self._serialize_value(x) if x is not None else None
                )

        # Determine size category
        num_rows = len(df)
        if num_rows < 1000:
            size_category = "n<1K"
        elif num_rows < 10000:
            size_category = "1K<n<10K"
        elif num_rows < 100000:
            size_category = "10K<n<100K"
        elif num_rows < 1000000:
            size_category = "100K<n<1M"
        elif num_rows < 10000000:
            size_category = "1M<n<10M"
        else:
            size_category = "n>10M"

        # Prepare tags
        all_tags = tags or []
        if nsfw:
            all_tags.append("not-for-all-audiences")

        # Default task categories
        if task_categories is None:
            task_categories = ["text-to-image", "image-to-image"]

        # Create dataset card
        card_content = f"""---
license: {license or 'unknown'}
language:
- {language}
size_categories:
- {size_category}
task_categories:
{self._yaml_list(task_categories)}"""

        if all_tags:
            card_content += f"\ntags:\n{self._yaml_list(all_tags)}"

        card_content += f"""
---

# Caption Dataset

This dataset contains {num_rows:,} captioned items exported from CaptionFlow.

## Dataset Structure

### Data Fields

"""

        # Add field descriptions
        for col in df.columns:
            dtype = str(df[col].dtype)
            if col in self.contents.output_fields:
                card_content += f"- `{col}`: List of captions/outputs\n"
            else:
                card_content += f"- `{col}`: {dtype}\n"

        if self.contents.metadata:
            card_content += "\n## Export Information\n\n"
            if "export_timestamp" in self.contents.metadata:
                card_content += (
                    f"- Export timestamp: {self.contents.metadata['export_timestamp']}\n"
                )
            if "field_stats" in self.contents.metadata:
                card_content += "\n### Field Statistics\n\n"
                for field, stats in self.contents.metadata["field_stats"].items():
                    card_content += f"- `{field}`: {stats['total_items']:,} items across {stats['rows_with_data']:,} rows\n"

        # Create temporary parquet file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            # Write parquet file
            table = pa.Table.from_pandas(df)
            pq.write_table(table, temp_path, compression="snappy")

            # Upload parquet file
            api.upload_file(
                path_or_fileobj=str(temp_path),
                path_in_repo="data.parquet",
                repo_id=dataset_name,
                repo_type="dataset",
                token=token,
            )

            # Create and upload dataset card
            card = DatasetCard(card_content)
            card.push_to_hub(dataset_name, token=token)

            dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
            logger.info(f"Successfully uploaded dataset to: {dataset_url}")

            return dataset_url

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def _yaml_list(self, items: List[str]) -> str:
        """Format a list for YAML."""
        return "\n".join(f"- {item}" for item in items)


# Addition to StorageManager class
async def get_storage_contents(
    self,
    limit: Optional[int] = None,
    columns: Optional[List[str]] = None,
    include_metadata: bool = True,
) -> StorageContents:
    """Retrieve storage contents for export.

    Args:
        limit: Maximum number of rows to retrieve
        columns: Specific columns to include (None for all)
        include_metadata: Whether to include metadata in the result

    Returns:
        StorageContents instance with the requested data
    """
    if not self.captions_path.exists():
        return StorageContents(
            rows=[],
            columns=[],
            output_fields=list(self.known_output_fields),
            total_rows=0,
            metadata={"message": "No captions file found"},
        )

    # Flush buffers first to ensure all data is on disk
    await self.checkpoint()

    # Determine columns to read
    if columns:
        # Validate requested columns exist
        table_metadata = pq.read_metadata(self.captions_path)
        available_columns = set(table_metadata.schema.names)
        invalid_columns = set(columns) - available_columns
        if invalid_columns:
            raise ValueError(f"Columns not found: {invalid_columns}")
        columns_to_read = columns
    else:
        # Read all columns
        columns_to_read = None

    # Read the table
    table = pq.read_table(self.captions_path, columns=columns_to_read)
    df = table.to_pandas()

    # Apply limit if specified
    if limit:
        df = df.head(limit)

    # Convert to list of dicts
    rows = df.to_dict("records")

    # Parse metadata JSON strings back to dicts if present
    if "metadata" in df.columns:
        for row in rows:
            if row.get("metadata"):
                try:
                    row["metadata"] = json.loads(row["metadata"])
                except:
                    pass  # Keep as string if parsing fails

    # Prepare metadata
    metadata = {}
    if include_metadata:
        stats = await self.get_caption_stats()
        metadata.update(
            {
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "total_available_rows": stats.get("total_rows", 0),
                "rows_exported": len(rows),
                "storage_path": str(self.captions_path),
                "field_stats": stats.get("field_stats", {}),
            }
        )

    return StorageContents(
        rows=rows,
        columns=list(df.columns),
        output_fields=list(self.known_output_fields),
        total_rows=len(df),
        metadata=metadata,
    )
