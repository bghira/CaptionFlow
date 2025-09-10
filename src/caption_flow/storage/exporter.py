"""Storage exporter for Lance datasets to various formats."""

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import lance
import numpy as np
import pandas as pd

from ..models import ExportError, StorageContents
from .manager import StorageManager

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("CAPTIONFLOW_LOG_LEVEL", "INFO").upper())


class LanceStorageExporter:
    """Exports Lance storage contents to various formats."""

    def __init__(self, storage_manager: StorageManager):
        """Initialize exporter with storage manager.

        Args:
        ----
            storage_manager: StorageManager instance

        """
        self.storage_manager = storage_manager

    async def export_shard(
        self,
        shard_name: str,
        format: str,
        output_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> int:
        """Export a single shard to specified format.

        Args:
        ----
            shard_name: Name of the shard to export
            format: Export format ('jsonl', 'json', 'csv', 'parquet', 'txt')
            output_path: Output file or directory path
            columns: Specific columns to export
            limit: Maximum number of rows to export
            **kwargs: Format-specific options

        Returns:
        -------
            Number of items exported

        """
        logger.debug(f"Getting shard contents for {shard_name}")
        await self.storage_manager.initialize()
        contents = await self.storage_manager.get_shard_contents(
            shard_name, limit=limit, columns=columns
        )

        if not contents.rows:
            logger.warning(f"No data to export for shard {shard_name}")
            return 0

        exporter = StorageExporter(contents)

        # Add shard suffix to output path
        output_path = Path(output_path)
        if format in ["jsonl", "csv", "parquet"]:
            # Single file formats - add shard name to filename
            if output_path.suffix:
                output_file = (
                    output_path.parent / f"{output_path.stem}_{shard_name}{output_path.suffix}"
                )
            else:
                output_file = output_path / f"{shard_name}.{format}"
        else:
            # Directory-based formats
            output_file = output_path / shard_name

        # Export based on format
        if format == "jsonl":
            return exporter.to_jsonl(output_file)
        elif format == "json":
            return exporter.to_json(output_file, kwargs.get("filename_column", "filename"))
        elif format == "csv":
            return exporter.to_csv(output_file)
        elif format == "parquet":
            return await self.export_shard_to_parquet(shard_name, output_file, columns, limit)
        elif format == "txt":
            return exporter.to_txt(
                output_file,
                kwargs.get("filename_column", "filename"),
                kwargs.get("export_column", "captions"),
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def export_all_shards(
        self,
        format: str,
        output_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        limit_per_shard: Optional[int] = None,
        shard_filter: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, int]:
        """Export all shards (or filtered shards) to specified format.

        Args:
        ----
            format: Export format
            output_path: Base output path
            columns: Columns to export
            limit_per_shard: Max rows per shard
            shard_filter: List of specific shards to export
            **kwargs: Format-specific options

        Returns:
        -------
            Dictionary mapping shard names to export counts

        """
        results = {}

        # Get shards to export
        await self.storage_manager.initialize()
        if shard_filter:
            shards = [s for s in shard_filter if s in self.storage_manager.shard_datasets]
        else:
            shards = list(self.storage_manager.shard_datasets.keys())

        logger.info(f"Exporting {len(shards)} shards to {format} format")

        for shard_name in shards:
            try:
                count = await self.export_shard(
                    shard_name,
                    format,
                    output_path,
                    columns=columns,
                    limit=limit_per_shard,
                    **kwargs,
                )
                results[shard_name] = count
                logger.info(f"Exported {count} items from shard {shard_name}")
            except Exception as e:
                logger.error(f"Failed to export shard {shard_name}: {e}")
                results[shard_name] = 0

        return results

    async def export_shard_to_parquet(
        self,
        shard_name: str,
        output_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> int:
        """Export a shard directly to Parquet format.

        This is efficient as Lance is already columnar.
        """
        if shard_name not in self.storage_manager.shard_datasets:
            raise ValueError(f"Shard {shard_name} not found")

        dataset = self.storage_manager.shard_datasets[shard_name]

        # Build scanner
        scanner = dataset.scanner(columns=columns)
        if limit:
            scanner = scanner.limit(limit)

        # Get table and write to parquet
        table = scanner.to_table()

        import pyarrow.parquet as pq

        pq.write_table(table, str(output_path), compression="snappy")

        return table.num_rows

    async def export_to_lance(
        self,
        output_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        shard_filter: Optional[List[str]] = None,
    ) -> int:
        """Export to a new Lance dataset, optionally filtering shards.

        Args:
        ----
            output_path: Path for the output Lance dataset
            columns: Specific columns to include
            shard_filter: List of shard names to include

        Returns:
        -------
            Total number of rows exported

        """
        output_path = Path(output_path)
        if output_path.exists():
            raise ValueError(f"Output path already exists: {output_path}")

        # Get shards to export
        if shard_filter:
            shards = [s for s in shard_filter if s in self.storage_manager.shard_datasets]
        else:
            shards = list(self.storage_manager.shard_datasets.keys())

        if not shards:
            raise ValueError("No shards to export")

        total_rows = 0
        first_shard = True

        for shard_name in shards:
            dataset = self.storage_manager.shard_datasets[shard_name]

            # Build scanner
            scanner = dataset.scanner(columns=columns)
            table = scanner.to_table()

            if first_shard:
                # Create new dataset
                lance.write_dataset(table, str(output_path), mode="create")
                first_shard = False
            else:
                # Append to existing
                lance.write_dataset(table, str(output_path), mode="append")

            total_rows += table.num_rows
            logger.info(f"Exported {table.num_rows} rows from shard {shard_name}")

        logger.info(f"Created Lance dataset at {output_path} with {total_rows} rows")
        return total_rows

    async def export_to_huggingface_hub(
        self,
        dataset_name: str,
        token: Optional[str] = None,
        license: str = "apache-2.0",
        private: bool = False,
        nsfw: bool = False,
        tags: Optional[List[str]] = None,
        language: str = "en",
        task_categories: Optional[List[str]] = None,
        shard_filter: Optional[List[str]] = None,
        max_shard_size_gb: float = 2.0,
    ) -> str:
        """Export to Hugging Face Hub with per-shard parquet files.

        Args:
        ----
            dataset_name: Name for the dataset (e.g., "username/dataset-name")
            token: Hugging Face API token
            license: License for the dataset
            private: Whether to make the dataset private
            nsfw: Whether to add not-for-all-audiences tag
            tags: Additional tags
            language: Language code
            task_categories: Task categories
            shard_filter: Specific shards to export
            max_shard_size_gb: Max size per parquet file in GB

        Returns:
        -------
            URL of the uploaded dataset

        """
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import DatasetCard, HfApi, create_repo
        except ImportError:
            raise ExportError(
                "huggingface_hub is required for HF export. "
                "Install with: pip install huggingface_hub"
            )

        api = HfApi(token=token)

        # Check/create repo
        try:
            api.dataset_info(dataset_name)
            logger.info(f"Dataset {dataset_name} already exists, will update it")
        except:
            logger.info(f"Creating new dataset: {dataset_name}")
            create_repo(repo_id=dataset_name, repo_type="dataset", private=private, token=token)

        # Get shards to export
        if shard_filter:
            shards = [s for s in shard_filter if s in self.storage_manager.shard_datasets]
        else:
            shards = sorted(self.storage_manager.shard_datasets.keys())

        # Export each shard as a separate parquet file
        total_rows = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "data"
            data_dir.mkdir(exist_ok=True)

            # Export all shards to the data directory
            for shard_name in shards:
                # Export shard to parquet
                parquet_path = data_dir / f"{shard_name}.parquet"
                rows = await self.export_shard_to_parquet(shard_name, parquet_path)

                if rows > 0:
                    # Check file size
                    file_size_gb = parquet_path.stat().st_size / (1024**3)
                    if file_size_gb > max_shard_size_gb:
                        logger.warning(
                            f"Shard {shard_name} is {file_size_gb:.2f}GB, "
                            f"exceeds limit of {max_shard_size_gb}GB"
                        )

                    total_rows += rows
                    logger.info(f"Prepared {shard_name}: {rows} rows, {file_size_gb:.2f}GB")

            # Create dataset card
            stats = await self.storage_manager.get_caption_stats()

            # Size category
            if total_rows < 1000:
                size_category = "n<1K"
            elif total_rows < 10000:
                size_category = "1K<n<10K"
            elif total_rows < 100000:
                size_category = "10K<n<100K"
            elif total_rows < 1000000:
                size_category = "100K<n<1M"
            elif total_rows < 1000000:
                size_category = "1M<n<10M"
            else:
                size_category = "n>10M"

            # Prepare tags
            default_tags = ["lance"]
            all_tags = default_tags + (tags or [])
            if nsfw:
                all_tags.append("not-for-all-audiences")

            # Default task categories
            if task_categories is None:
                task_categories = ["text-to-image", "image-to-image"]

            # Create card content
            card_content = f"""---
license: {license}
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

This dataset contains {total_rows:,} captioned items exported from CaptionFlow.

## Dataset Structure

"""

            card_content += "\n\n### Data Fields\n\n"

            # Add field descriptions
            all_fields = set()
            for field, _ in self.storage_manager.base_caption_fields:
                all_fields.add(field)
            for fields in self.storage_manager.shard_output_fields.values():
                all_fields.update(fields)

            for field in sorted(all_fields):
                if field in stats.get("output_fields", []):
                    card_content += f"- `{field}`: List of captions/outputs\n"
                else:
                    card_content += f"- `{field}`\n"

            if stats.get("field_stats"):
                card_content += "\n### Output Field Statistics\n\n"
                for field, count in stats["field_stats"].items():
                    card_content += f"- `{field}`: {count:,} total items\n"

            # Save README.md
            readme_path = tmpdir / "README.md"
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(card_content)

            # Upload the entire folder at once
            logger.info(f"Uploading dataset to {dataset_name}...")
            api.upload_large_folder(
                repo_id=dataset_name,
                folder_path=str(tmpdir),
                repo_type="dataset",
            )

            dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
            logger.info(f"Successfully uploaded dataset to: {dataset_url}")

            return dataset_url

    def _yaml_list(self, items: List[str]) -> str:
        """Format a list for YAML."""
        return "\n".join(f"- {item}" for item in items)


class StorageExporter:
    """Legacy exporter for StorageContents objects."""

    def __init__(self, contents: StorageContents):
        self.contents = contents
        self._validate_contents()

    def _validate_contents(self):
        if not self.contents.rows:
            logger.warning("No rows to export")
        if not self.contents.columns:
            raise ExportError("No columns defined for export")

    def _flatten_lists(self, value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(str(item).replace("\n", " ") for item in value)
        return str(value) if value is not None else ""

    def _serialize_value(self, value: Any) -> Any:
        import datetime as dt

        if pd.api.types.is_datetime64_any_dtype(type(value)) or isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, (dt.datetime, dt.date)):
            return value.isoformat()
        elif isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        return value

    def to_jsonl(self, output_path: Union[str, Path]) -> int:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows_written = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for row in self.contents.rows:
                serializable_row = {k: self._serialize_value(v) for k, v in row.items()}
                json_line = json.dumps(serializable_row, ensure_ascii=False)
                f.write(json_line + "\n")
                rows_written += 1

        logger.info(f"Exported {rows_written} rows to JSONL: {output_path}")
        return rows_written

    def _get_filename_from_row(self, row: Dict[str, Any], filename_column: str) -> Optional[str]:
        filename = row.get(filename_column)
        if filename:
            return filename

        url = row.get("url")
        if url:
            parsed = urlparse(str(url))
            path_parts = parsed.path.rstrip("/").split("/")
            if path_parts and path_parts[-1]:
                return path_parts[-1]

        return None

    def to_json(self, output_dir: Union[str, Path], filename_column: str = "filename") -> int:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename_column not in self.contents.columns and "url" in self.contents.columns:
            logger.warning(f"Column '{filename_column}' not found, falling back to 'url' column")
        elif filename_column not in self.contents.columns:
            raise ExportError(f"Column '{filename_column}' not found and no 'url' column available")

        files_created = 0
        skipped_count = 0

        for row in self.contents.rows:
            filename = self._get_filename_from_row(row, filename_column)
            if not filename:
                skipped_count += 1
                continue

            base_name = Path(filename).stem
            json_path = output_dir / f"{base_name}.json"

            serializable_row = {k: self._serialize_value(v) for k, v in row.items()}

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_row, f, ensure_ascii=False, indent=2)

            files_created += 1

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} rows with no extractable filename")

        logger.info(f"Created {files_created} JSON files in: {output_dir}")
        return files_created

    def to_csv(self, output_path: Union[str, Path]) -> int:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Identify complex columns to skip
        complex_columns = set()
        csv_safe_columns = []

        sample_size = min(10, len(self.contents.rows))
        for row in self.contents.rows[:sample_size]:
            for col, value in row.items():
                if col not in complex_columns and value is not None:
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

        csv_safe_columns = [col for col in self.contents.columns if col not in complex_columns]

        if not csv_safe_columns:
            raise ExportError("No columns suitable for CSV export. Use JSONL format instead.")

        csv_rows = []
        for row in self.contents.rows:
            csv_row = {}
            for col in csv_safe_columns:
                value = row.get(col)
                if isinstance(value, list):
                    csv_row[col] = self._flatten_lists(value)
                elif pd.api.types.is_datetime64_any_dtype(type(value)) or isinstance(
                    value, pd.Timestamp
                ):
                    csv_row[col] = self._serialize_value(value)
                else:
                    csv_row[col] = value
            csv_rows.append(csv_row)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_safe_columns)
            writer.writeheader()
            writer.writerows(csv_rows)

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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename_column not in self.contents.columns and "url" in self.contents.columns:
            logger.warning(f"Column '{filename_column}' not found, falling back to 'url' column")
        elif filename_column not in self.contents.columns:
            raise ExportError(f"Column '{filename_column}' not found and no 'url' column available")

        if export_column not in self.contents.columns:
            if export_column not in self.contents.output_fields:
                raise ExportError(f"Column '{export_column}' not found in data")

        files_created = 0
        skipped_no_filename = 0
        skipped_no_content = 0

        for row in self.contents.rows:
            filename = self._get_filename_from_row(row, filename_column)
            if not filename:
                skipped_no_filename += 1
                continue

            content = row.get(export_column)
            if content is None:
                skipped_no_content += 1
                continue

            base_name = Path(filename).stem
            txt_path = output_dir / f"{base_name}.txt"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(self._flatten_lists(content))

            files_created += 1

        if skipped_no_filename > 0:
            logger.warning(f"Skipped {skipped_no_filename} rows with no extractable filename")
        if skipped_no_content > 0:
            logger.warning(f"Skipped {skipped_no_content} rows with no {export_column} content")

        logger.info(f"Created {files_created} text files in: {output_dir}")
        return files_created
