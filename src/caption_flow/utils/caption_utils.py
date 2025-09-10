"""Caption processing utilities from the original vLLM script."""

from typing import Dict, List


class CaptionUtils:
    """Utilities for cleaning and combining captions."""

    @staticmethod
    def clean_caption(c: str) -> str:
        """Clean a single caption by removing generic phrases and formatting."""
        if not c:
            return ""

        generic = [
            "in this image we can see ",
            "this image shows ",
            "the image depicts ",
            "the image features ",
            "this is an image of ",
            "the image contains ",
            "the picture shows ",
            "we can see ",
            "there is ",
            "there are ",
        ]

        low = c.lower()
        for p in generic:
            if low.startswith(p):
                c = c[len(p) :]
                if c:
                    c = c[0].upper() + c[1:]
                break

        # Remove leading articles if the rest isn't capitalized
        if c.lower().startswith(("a ", "an ")):
            parts = c.split(maxsplit=1)
            if len(parts) > 1 and not parts[1][0].isupper():
                c = parts[1]
                c = c[0].upper() + c[1:]

        # Clean whitespace
        c = " ".join(c.split())

        # Add period if missing
        if c and c[-1] not in ".!?":
            c += "."

        return c

    @classmethod
    def combine(cls, descs: List[str]) -> str:
        """Combine multiple descriptions into a rich, multi-line caption."""
        if not descs:
            return ""

        filtered = []
        heads = [
            "in this image we can see",
            "this image shows",
            "the image depicts",
            "a cartoon",
            "a drawing",
            "an illustration",
        ]

        # Filter out short generic descriptions
        for d in descs:
            if not d:
                continue
            dl = d.lower().strip()
            if any(dl.startswith(h) and len(dl.split()) < 8 for h in heads):
                continue
            if len(d) > 10:
                filtered.append(d)

        if not filtered:
            filtered = [max(descs, key=len, default="")]

        # Use the longest as the main description
        main = cls.clean_caption(max(filtered, key=len))
        parts = [main]
        seen = set(main.lower().split())

        # Categorize additional descriptions
        buckets = {
            "characters": [
                "character",
                "person",
                "animal",
                "anthro",
                "wearing",
                "dressed",
            ],
            "actions": ["doing", "action", "playing", "running", "sitting", "standing"],
            "settings": [
                "room",
                "outdoor",
                "indoor",
                "setting",
                "background",
                "environment",
            ],
            "styles": ["style", "art", "drawn", "sketch", "painted", "digital"],
            "moods": [
                "mood",
                "emotion",
                "feeling",
                "atmosphere",
                "happy",
                "sad",
                "angry",
            ],
        }

        def categorize(text: str) -> str:
            """Categorize a description based on keywords."""
            text_lower = text.lower()
            for category, keywords in buckets.items():
                if any(keyword in text_lower for keyword in keywords):
                    return category
            return "details"

        # Group descriptions by category
        by_bucket: Dict[str, List[str]] = {}
        for desc in filtered:
            category = categorize(desc)
            by_bucket.setdefault(category, []).append(desc)

        # Add descriptions from each category
        for category in ["characters", "actions", "settings", "moods", "styles", "details"]:
            if category in by_bucket and by_bucket[category]:
                desc = by_bucket[category][0]
                words = desc.lower().split()

                # Check if this adds enough new information
                new_words = [w for w in words if w not in seen and len(w) > 3]
                if len(new_words) > 3:
                    clean = cls.clean_caption(desc)
                    if clean and clean not in parts:
                        parts.append(clean)
                        seen.update(words)

        # Return each part as a separate line for rich captions
        return "\n".join(parts)

    @staticmethod
    def validate_caption(caption: str, min_length: int = 20) -> bool:
        """Validate if a caption meets quality standards."""
        if not caption or len(caption) < min_length:
            return False

        # Check for refusal patterns
        refusal_patterns = [
            "i'm sorry",
            "i cannot",
            "i apologize",
            "inappropriate",
            "unable to",
            "refuse to",
        ]

        caption_lower = caption.lower()
        if any(pattern in caption_lower for pattern in refusal_patterns):
            return False

        # Check for too generic
        if caption_lower in ["image", "picture", "photo", "illustration"]:
            return False

        return True
