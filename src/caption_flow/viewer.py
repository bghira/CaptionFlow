"""TUI viewer for browsing CaptionFlow datasets with image preview using Urwid."""

import logging
import os
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
        self.captions_path = self.data_dir / "captions.parquet"

        if not self.captions_path.exists():
            raise FileNotFoundError(f"No captions file found at {self.captions_path}")

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
        self.shards_box = None  # Store LineBox reference
        self.items_box = None  # Store LineBox reference

        # Image handling
        self.current_image_url = None
        self.session = None
        self.temp_files = []

    def load_data(self):
        """Load dataset synchronously."""
        logger.info("Loading dataset...")

        # Read parquet file
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
            self.image_placeholder = urwid.Text("No image loaded", align="center")
            self.image_filler = urwid.Filler(self.image_placeholder)
        else:
            msg = "Image preview disabled" if self.disable_images else "term-image not installed"
            self.image_filler = urwid.Filler(urwid.Text(msg, align="center"))

        self.image_box = urwid.LineBox(self.image_filler, title="Image Preview", title_attr="title")

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
            if not filename and "url" in item:
                filename = os.path.basename(urlparse(item["url"]).path)
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

    def _update_image(self, item):
        """Update the image display."""
        url = item.get("url", "")

        # Skip if same URL
        if url == self.current_image_url:
            return

        self.current_image_url = url

        if not url:
            self.image_filler.body = urwid.Text("No URL available", align="center")
            return

        # Show loading message
        self.image_filler.body = urwid.Text("Loading image...", align="center")
        self.loop.draw_screen()

        try:
            # Download image
            import urllib.error
            import urllib.request

            # Create request with user agent to avoid 403 errors
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                image_data = response.read()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
                self.temp_files.append(tmp_path)

            # Create term_image from file
            image = from_file(tmp_path)

            # Dynamic sizing based on terminal size and actual space
            if self.loop and self.screen:
                cols, rows = self.screen.get_cols_rows()

                # Calculate available space more accurately
                # Left columns take 42 chars (12 + 30) + borders/padding
                available_width = max(30, cols - 46)

                # Image box has weight 1 out of 2 total in the preview pile (equal space)
                # But account for borders, header, footer (about 6 rows total)
                preview_height = rows - 6
                image_height = preview_height // 2  # 1/2 of preview area now
                available_height = max(20, image_height - 2)  # Account for box borders

                # Try to preserve aspect ratio while maximizing use of space
                # Prioritize HEIGHT to avoid vertical cropping
                try:
                    # First try setting only height, let width auto-adjust
                    image.set_size(height=available_height)
                except:
                    # If that fails, try setting both but prioritize height
                    try:
                        image.set_size(width=available_width, height=available_height)
                    except:
                        # Last resort - use fixed size
                        image.set_size(60, 30)
            else:
                # Fallback size
                image.set_size(60, 30)

            # Create UrwidImage widget without upscaling to maintain proper bounds
            self.image_widget = UrwidImage(image, upscale=False)
            # Center the image in the available space
            self.image_filler.body = urwid.Padding(self.image_widget, align="center")

        except urllib.error.HTTPError as e:
            self.image_filler.body = urwid.Text(f"HTTP Error {e.code}: {e.reason}", align="center")
        except urllib.error.URLError as e:
            self.image_filler.body = urwid.Text(f"URL Error: {str(e)}", align="center")
        except Exception as e:
            self.image_filler.body = urwid.Text(f"Error: {str(e)}", align="center")

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
