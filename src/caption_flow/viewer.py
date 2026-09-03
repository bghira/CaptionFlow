"""TUI viewer for browsing CaptionFlow datasets with image preview using Urwid."""

import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import urwid

try:
    from term_image.image import BaseImage, from_file, from_url
    from term_image.widget import UrwidImage, UrwidImageScreen

    TERM_IMAGE_AVAILABLE = True
except ImportError:
    TERM_IMAGE_AVAILABLE = False
    logging.warning("term-image not available. Install with: pip install term-image")

logger = logging.getLogger(__name__)


class SelectableListItem(urwid.WidgetWrap):
    """A selectable list item that can be highlighted."""

    def __init__(self, content, on_select=None):
        self.content = content
        self.on_select = on_select
        self._w = urwid.AttrMap(urwid.Text(content), "normal", focus_map="selected")
        super().__init__(self._w)

    def selectable(self):
        return True

    def keypress(self, size, key):
        if key in ("enter", " ") and self.on_select:
            self.on_select()
            return None
        return key


class DatasetViewer:
    """Interactive dataset viewer with image preview using Urwid."""

    palette = [
        ("normal", "white", "black"),
        ("selected", "black", "light gray"),
        ("header", "white,bold", "dark blue"),
        ("footer", "white", "dark gray"),
        ("title", "yellow,bold", "black"),
        ("error", "light red", "black"),
        ("dim", "dark gray", "black"),
        ("caption_title", "light cyan,bold", "black"),
        ("caption_text", "white", "black"),
    ]

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        lance_path = self.data_dir / "captions.lance"
        parquet_path = self.data_dir / "captions.parquet"

        # Lance is CaptionFlow's live storage format. Keep Parquet support for
        # older exported datasets, preferring the live store when both exist.
        if lance_path.exists():
            self.captions_path = lance_path
        elif parquet_path.exists():
            self.captions_path = parquet_path
        else:
            raise FileNotFoundError(
                f"No captions dataset found in {self.data_dir}; expected "
                "captions.lance or captions.parquet"
            )

        # Data
        self.df = None
        self.shards = []
        self.current_shard_idx = 0
        self.current_item_idx = 0
        self.current_shard_items = []

        # UI components
        self.loop = None
        self.screen = None
        self.disable_images = False

        # Widgets
        self.shards_list = None
        self.items_list = None
        self.caption_box = None
        self.image_box = None
        self.image_widget = None
        self.image_container = None
        self.shards_box = None  # Store LineBox reference
        self.items_box = None  # Store LineBox reference

        # Image handling
        self.current_image_url = None
        self.session = None
        self.temp_files = []
        self.webshart_datasets = {}
        self.webshart_layouts = {}

    def load_data(self):
        """Load dataset synchronously."""
        logger.info("Loading dataset...")

        if self.captions_path.suffix == ".lance":
            import lance

            self.df = lance.dataset(str(self.captions_path)).to_table().to_pandas()
        else:
            self.df = pd.read_parquet(self.captions_path)

        # Get unique shards
        if "shard" in self.df.columns:
            self.shards = sorted(self.df["shard"].unique())
        else:
            # If no shard column, create a dummy one
            self.shards = ["all"]
            self.df["shard"] = "all"

        # Load first shard
        self._load_shard(0)

        logger.info(f"Loaded {len(self.df)} items across {len(self.shards)} shards")

    def _load_shard(self, shard_idx: int):
        """Load items from a specific shard."""
        if 0 <= shard_idx < len(self.shards):
            self.current_shard_idx = shard_idx
            shard_name = self.shards[shard_idx]

            # Get items in this shard
            shard_df = self.df[self.df["shard"] == shard_name]

            # Sort by item_index if available, otherwise by job_id
            if "item_index" in shard_df.columns:
                shard_df = shard_df.sort_values("item_index")
            elif "job_id" in shard_df.columns:
                shard_df = shard_df.sort_values("job_id")

            self.current_shard_items = shard_df.to_dict("records")
            self.current_item_idx = 0

            # Reset image
            self.current_image_url = None

    def create_ui(self):
        """Create the Urwid UI."""
        # Header
        header = urwid.AttrMap(urwid.Text("CaptionFlow Dataset Viewer", align="center"), "header")

        # Shards list
        self._create_shards_list()
        self.shards_box = urwid.LineBox(self.shards_list, title="Shards", title_attr="title")

        # Items list
        self._create_items_list()
        self.items_box = urwid.LineBox(
            self.items_list, title=f"Items ({len(self.current_shard_items)})", title_attr="title"
        )

        # Caption display
        self.caption_text = urwid.Text("")
        self.caption_box = urwid.LineBox(
            urwid.Filler(self.caption_text, valign="top"), title="Captions", title_attr="title"
        )

        # Image display
        if TERM_IMAGE_AVAILABLE and not self.disable_images:
            image_content = self._image_message_widget("No image loaded")
        else:
            msg = "Image preview disabled" if self.disable_images else "term-image not installed"
            image_content = self._image_message_widget(msg)

        self.image_container = urwid.WidgetPlaceholder(image_content)
        self.image_box = urwid.LineBox(
            self.image_container, title="Image Preview", title_attr="title"
        )

        # Preview area (captions + image) - ADJUSTED WEIGHTS
        preview = urwid.Pile(
            [
                ("weight", 1, self.caption_box),  # Reduced from 2 to 1
                ("weight", 1, self.image_box),  # Increased from 1 to 1 (equal space)
            ]
        )

        # Main body columns - ADJUSTED WEIGHTS AND FIXED WIDTH
        body = urwid.Columns(
            [
                (12, self.shards_box),  # Reduced from 15 to 12
                (30, self.items_box),  # Reduced from 35 to 30
                ("weight", 1, preview),  # More space for preview
            ]
        )

        # Footer
        footer = urwid.AttrMap(
            urwid.Text(
                "↑/↓/j/k: Navigate | ←/→/h/l: Shards | Space/b: Page | q: Quit", align="center"
            ),
            "footer",
        )

        # Main layout
        self.main_widget = urwid.Frame(body=body, header=header, footer=footer)

    def _create_shards_list(self):
        """Create the shards list widget."""
        shard_items = []
        for idx, shard in enumerate(self.shards):
            count = len(self.df[self.df["shard"] == shard])
            # Truncate shard name if too long for narrower column
            shard_display = shard if len(shard) <= 8 else shard[:5] + "..."
            text = f"{shard_display} ({count})"
            if idx == self.current_shard_idx:
                text = f"▶ {text}"
            item = SelectableListItem(text, on_select=lambda i=idx: self._select_shard(i))
            shard_items.append(item)

        self.shards_walker = urwid.SimpleFocusListWalker(shard_items)
        self.shards_list = urwid.ListBox(self.shards_walker)

        # Set focus to current shard
        if self.shards_walker:
            self.shards_walker.set_focus(self.current_shard_idx)

    def _create_items_list(self):
        """Create the items list widget."""
        item_widgets = []
        for idx, item in enumerate(self.current_shard_items):
            # Extract filename
            filename = item.get("filename", "")
            if not isinstance(filename, str):
                filename = ""

            url = item.get("url", "")
            if not isinstance(url, str):
                url = ""

            if not filename and url:
                filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"item_{idx}"

            # Truncate more aggressively for narrower fixed width
            if len(filename) > 20:
                filename = filename[:17] + "..."

            # Count captions
            self._count_captions(item)

            text = f"{idx:3d}. {filename}"
            if idx == self.current_item_idx:
                text = f"▶ {text}"

            item_widget = SelectableListItem(text, on_select=lambda i=idx: self._select_item(i))
            item_widgets.append(item_widget)

        self.items_walker = urwid.SimpleFocusListWalker(item_widgets)
        self.items_list = urwid.ListBox(self.items_walker)

        # Set focus to current item
        if self.items_walker and self.current_item_idx < len(self.items_walker):
            self.items_walker.set_focus(self.current_item_idx)

    def _count_captions(self, item):
        """Count the number of captions in an item."""
        caption_count = 0
        for field in ["captions", "descriptions", "alt_text", "long_caption", "short_caption"]:
            if field in item and item[field] is not None:
                try:
                    value = item[field]
                    # Handle numpy arrays
                    if hasattr(value, "__array__"):
                        value = value.tolist()

                    if isinstance(value, list):
                        # Filter out None/empty values
                        non_empty = [v for v in value if v is not None and str(v).strip()]
                        caption_count += len(non_empty)
                    elif value and str(value).strip():
                        caption_count += 1
                except:
                    pass
        return caption_count

    def _select_shard(self, idx):
        """Select a shard."""
        if idx != self.current_shard_idx:
            self._load_shard(idx)
            self._update_ui()

    def _select_item(self, idx):
        """Select an item."""
        if idx != self.current_item_idx:
            self.current_item_idx = idx
            self._update_preview()

    def _update_ui(self):
        """Update the entire UI."""
        # Update shards list
        self._create_shards_list()
        self.shards_box.original_widget = self.shards_list

        # Update items list
        self._create_items_list()
        self.items_box.original_widget = self.items_list
        self.items_box.set_title(f"Items ({len(self.current_shard_items)})")

        # Update preview
        self._update_preview()

    def _update_preview(self):
        """Update the preview area (captions and image)."""
        if not self.current_shard_items:
            return

        item = self.current_shard_items[self.current_item_idx]

        # Update captions
        caption_text = self._format_captions(item)
        self.caption_text.set_text(caption_text)

        # Update image
        if TERM_IMAGE_AVAILABLE and not self.disable_images:
            self._update_image(item)

    def _extract_caption_values(self, value):
        """Extract caption values from various formats."""
        results = []

        # Handle numpy arrays
        if hasattr(value, "__array__"):
            value = value.tolist()

        # Handle string representation of lists/arrays
        if isinstance(value, str):
            # Try to parse string representations like "['caption1', 'caption2']"
            if value.startswith("[") and value.endswith("]"):
                try:
                    import ast

                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        results.extend([str(v) for v in parsed if v])
                    else:
                        results.append(str(parsed))
                except:
                    # If parsing fails, treat as regular string
                    results.append(value)
            else:
                results.append(value)
        elif isinstance(value, list):
            # Process each item in the list
            for item in value:
                if item is not None and str(item).strip():
                    # Recursively extract in case of nested lists
                    sub_values = self._extract_caption_values(item)
                    results.extend(sub_values)
        elif value is not None and str(value).strip():
            results.append(str(value))

        return results

    def _format_captions(self, item):
        """Format captions for display."""
        parts = []

        # Standard caption fields to check
        caption_fields = [
            ("captions", "Captions"),
            ("descriptions", "Descriptions"),
            ("alt_text", "Alt Text"),
            ("long_caption", "Long Caption"),
            ("short_caption", "Short Caption"),
        ]

        for field_name, display_name in caption_fields:
            if field_name in item and item[field_name] is not None:
                # Extract all caption values
                captions = self._extract_caption_values(item[field_name])

                if captions:
                    parts.append(f"\n{display_name}:")
                    for i, caption in enumerate(captions, 1):
                        # Wrap text for readability - adjust for wider preview area
                        wrapped = self._wrap_text(caption, 100)
                        parts.append(f"  {i}. {wrapped}")

        # Add metadata
        metadata_parts = []
        if "job_id" in item:
            metadata_parts.append(f"Job: {item['job_id']}")
        if "contributor_id" in item:
            metadata_parts.append(f"By: {item['contributor_id']}")

        if metadata_parts:
            parts.insert(0, " | ".join(metadata_parts))

        return "\n".join(parts) if parts else "No captions available"

    def _wrap_text(self, text, width):
        """Simple text wrapping."""
        import textwrap

        lines = textwrap.wrap(text, width=width)
        return "\n      ".join(lines)

    def _image_message_widget(self, message):
        """Create a centered box widget for image-preview status text."""
        return urwid.Filler(urwid.Text(message, align="center"))

    def _show_image_message(self, message):
        """Replace the image preview with centered status text."""
        self.image_container.original_widget = self._image_message_widget(message)

    def _update_image(self, item):
        """Update the image display."""
        url = item.get("url", "")
        if not isinstance(url, str) or not url.strip():
            url = ""

        webshart_ref = self._webshart_reference(item) if not url else None
        image_ref = url or webshart_ref or ""

        # Skip if the same image is already displayed.
        if image_ref == self.current_image_url:
            return

        self.current_image_url = image_ref

        if not image_ref:
            self._show_image_message("No image source available")
            return

        # Show loading message
        self._show_image_message("Loading image...")
        self.loop.draw_screen()

        try:
            # Download image
            import urllib.error
            import urllib.request

            if url:
                # Create request with user agent to avoid 403 errors.
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        )
                    },
                )

                with urllib.request.urlopen(request, timeout=10) as response:
                    image_data = response.read()
                suffix = Path(urlparse(url).path).suffix or ".jpg"
            else:
                image_data, suffix = self._load_webshart_image(item)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
                self.temp_files.append(tmp_path)

            # Create term_image from file
            image = from_file(tmp_path)

            # Put the image widget directly in the box. UrwidImage then receives
            # the pane's exact (width, height), fits to both axes while preserving
            # aspect ratio, and recalculates when the terminal is resized.
            self.image_widget = UrwidImage(image, upscale=False)
            self.image_container.original_widget = self.image_widget

        except urllib.error.HTTPError as e:
            self._show_image_message(f"HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            self._show_image_message(f"URL Error: {str(e)}")
        except Exception as e:
            self._show_image_message(f"Error: {str(e)}")

    def _webshart_reference(self, item):
        """Return a stable reference for a WebShart-backed caption row."""
        dataset = item.get("dataset")
        shard = item.get("shard")
        filename = item.get("filename")
        if not all(isinstance(value, str) and value for value in (dataset, shard, filename)):
            return None
        if not re.fullmatch(r"shard-\d+", shard):
            return None
        return f"webshart://{dataset}/{shard}/{filename}"

    def _load_webshart_image(self, item):
        """Fetch one source image directly from its indexed tar byte range."""
        import json
        import urllib.request

        import webshart

        dataset_name = item["dataset"]
        shard_name = item["shard"]
        filename = item["filename"]
        shard_idx = int(shard_name.rsplit("-", 1)[1])

        dataset = self.webshart_datasets.get(dataset_name)
        if dataset is None:
            dataset = webshart.discover_dataset(source=dataset_name)
            self.webshart_datasets[dataset_name] = dataset

        layout_key = (dataset_name, shard_idx)
        layout = self.webshart_layouts.get(layout_key)
        if layout is None:
            shard_info = dataset.get_shard_info(shard_idx)
            metadata_url = shard_info.get("json_path")
            tar_url = shard_info.get("tar_path") or shard_info.get("path")
            if not metadata_url or not tar_url:
                raise ValueError(f"Missing WebShart URLs for {shard_name}")

            with urllib.request.urlopen(metadata_url, timeout=30) as response:
                metadata = json.loads(response.read())
            layout = {"tar_url": tar_url, "files": metadata.get("files", {})}
            self.webshart_layouts[layout_key] = layout

        entry = layout["files"].get(filename)
        if not isinstance(entry, dict):
            raise ValueError(f"Image {filename} is missing from {shard_name} metadata")

        offset = int(entry["offset"])
        length = int(entry.get("length", entry.get("size", 0)))
        if length <= 0:
            raise ValueError(f"Invalid WebShart image length for {filename}")

        request = urllib.request.Request(
            layout["tar_url"],
            headers={"Range": f"bytes={offset}-{offset + length - 1}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            image_data = response.read()
        if len(image_data) != length:
            raise ValueError(
                f"WebShart returned {len(image_data)} bytes for {filename}; expected {length}"
            )

        return image_data, Path(filename).suffix or ".img"

    def handle_input(self, key):
        """Handle keyboard input."""
        if key in ("q", "Q"):
            self.cleanup()
            raise urwid.ExitMainLoop()
        elif key in ("down", "j"):
            self._navigate_items(1)
        elif key in ("up", "k"):
            self._navigate_items(-1)
        elif key in ("left", "h"):
            self._navigate_shards(-1)
        elif key in ("right", "l"):
            self._navigate_shards(1)
        elif key == " ":  # Space - page down
            self._navigate_items(10)
        elif key == "b":  # b - page up
            self._navigate_items(-10)
        elif key == "page down":
            self._navigate_items(10)
        elif key == "page up":
            self._navigate_items(-10)

    def _navigate_items(self, delta):
        """Navigate through items."""
        if not self.current_shard_items:
            return

        new_idx = self.current_item_idx + delta
        new_idx = max(0, min(new_idx, len(self.current_shard_items) - 1))

        if new_idx != self.current_item_idx:
            self.current_item_idx = new_idx

            # Update items list focus
            if self.items_walker and new_idx < len(self.items_walker):
                self.items_walker.set_focus(new_idx)

            # Update the list display
            self._create_items_list()
            self.items_box.original_widget = self.items_list

            self._update_preview()

    def _navigate_shards(self, delta):
        """Navigate through shards."""
        new_idx = self.current_shard_idx + delta
        new_idx = max(0, min(new_idx, len(self.shards) - 1))

        if new_idx != self.current_shard_idx:
            self._select_shard(new_idx)

    def cleanup(self):
        """Clean up resources."""
        # Clean up temp files
        for tmp_file in self.temp_files:
            try:
                os.unlink(tmp_file)
            except:
                pass

    def run(self):
        """Run the viewer."""
        # Load data first
        self.load_data()

        # Create UI
        self.create_ui()

        # Create event loop and screen
        if TERM_IMAGE_AVAILABLE:
            self.screen = UrwidImageScreen()
        else:
            self.screen = urwid.raw_display.Screen()

        # Main loop
        self.loop = urwid.MainLoop(
            self.main_widget,
            palette=self.palette,
            screen=self.screen,
            unhandled_input=self.handle_input,
        )

        try:
            self.loop.run()
        finally:
            self.cleanup()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="View CaptionFlow dataset")
    parser.add_argument("data_dir", type=Path, help="Dataset directory")
    parser.add_argument("--no-images", action="store_true", help="Disable image preview")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create and run viewer
    try:
        viewer = DatasetViewer(args.data_dir)
        if args.no_images:
            viewer.disable_images = True
        viewer.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
