"""Tests for the caption_utils module."""

import pytest
from caption_flow.utils.caption_utils import CaptionUtils


class TestCaptionUtilsCleanCaption:
    """Test the clean_caption static method."""

    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        result = CaptionUtils.clean_caption("")
        assert result == ""

    def test_clean_none(self):
        """Test cleaning None."""
        result = CaptionUtils.clean_caption(None)
        assert result == ""

    def test_clean_generic_phrases_start(self):
        """Test removal of generic starting phrases."""
        test_cases = [
            ("in this image we can see a cat", "Cat."),
            ("this image shows a dog running", "Dog running."),
            ("the image depicts a sunset", "Sunset."),
            ("the image features two birds", "Two birds."),
            ("this is an image of flowers", "Flowers."),
            ("the image contains many objects", "Many objects."),
            ("the picture shows a house", "House."),
            ("we can see mountains", "Mountains."),
            ("there is a tree", "Tree."),
            ("there are clouds", "Clouds."),
        ]

        for input_text, expected in test_cases:
            result = CaptionUtils.clean_caption(input_text)
            assert (
                result == expected
            ), f"Failed for '{input_text}': got '{result}', expected '{expected}'"

    def test_clean_generic_phrases_case_insensitive(self):
        """Test that generic phrase removal is case insensitive."""
        result = CaptionUtils.clean_caption("IN THIS IMAGE WE CAN SEE a beautiful scene")
        assert result == "Beautiful scene."

    def test_clean_leading_articles(self):
        """Test removal of leading articles when appropriate."""
        # Should remove leading articles if the rest isn't capitalized
        result = CaptionUtils.clean_caption("a small dog")
        assert result == "Small dog."

        result = CaptionUtils.clean_caption("an old tree")
        assert result == "Old tree."

        # Should NOT remove if the rest is capitalized (proper noun)
        result = CaptionUtils.clean_caption("a German Shepherd")
        assert result == "a German Shepherd."

        result = CaptionUtils.clean_caption("an American flag")
        assert result == "an American flag."

    def test_clean_whitespace_normalization(self):
        """Test that multiple whitespaces are cleaned up."""
        result = CaptionUtils.clean_caption("a   dog    running     fast")
        assert result == "Dog running fast."

    def test_add_period_if_missing(self):
        """Test that period is added if missing."""
        result = CaptionUtils.clean_caption("a beautiful sunset")
        assert result == "Beautiful sunset."

        # Should not add period if other punctuation exists
        result = CaptionUtils.clean_caption("a question?")
        assert result == "Question?"

        result = CaptionUtils.clean_caption("an exclamation!")
        assert result == "Exclamation!"

    def test_combined_cleaning(self):
        """Test that all cleaning operations work together."""
        result = CaptionUtils.clean_caption("in this image we can see  a   small  dog   running")
        assert result == "a small dog running."

    def test_preserve_proper_punctuation(self):
        """Test that existing proper punctuation is preserved."""
        result = CaptionUtils.clean_caption("this image shows a cat!")
        assert result == "Cat!"

        result = CaptionUtils.clean_caption("the image depicts what is happening?")
        assert result == "What is happening?"


class TestCaptionUtilsCombine:
    """Test the combine class method."""

    def test_combine_empty_list(self):
        """Test combining empty list."""
        result = CaptionUtils.combine([])
        assert result == ""

    def test_combine_single_description(self):
        """Test combining single description."""
        result = CaptionUtils.combine(["a beautiful landscape"])
        assert result == "Beautiful landscape."

    def test_filter_short_generic_descriptions(self):
        """Test that short generic descriptions are filtered out."""
        descriptions = [
            "in this image we can see dog",  # Short generic, should be filtered
            "this image shows cat",  # Short generic, should be filtered
            "a cartoon animal",  # Short generic, should be filtered
            "a detailed drawing of a magnificent forest with many trees and wildlife",
        ]

        result = CaptionUtils.combine(descriptions)
        # Should only use the long detailed description
        assert "forest" in result
        assert "magnificent" in result

    def test_filter_short_descriptions(self):
        """Test that very short descriptions are filtered out."""
        descriptions = [
            "dog",  # Too short (<=10 chars)
            "cat runs",  # Too short
            "a beautiful landscape scene with mountains and rivers",
        ]

        result = CaptionUtils.combine(descriptions)
        assert "landscape" in result
        assert "mountains" in result
        assert "dog" not in result
        assert "cat" not in result

    def test_fallback_to_longest_when_all_filtered(self):
        """Test fallback when all descriptions are filtered out."""
        descriptions = ["a cartoon", "this image shows cat", "dog"]

        result = CaptionUtils.combine(descriptions)
        # Should fallback to the longest (even if normally filtered)
        assert "this image shows cat" in result.lower() or "cat" in result.lower()

    def test_use_longest_as_main(self):
        """Test that longest description is used as main."""
        descriptions = [
            "a cat sitting",
            "a magnificent tabby cat sitting gracefully on a wooden chair in sunlight",
            "a chair",
        ]

        result = CaptionUtils.combine(descriptions)
        lines = result.split("\n")
        main_line = lines[0]

        assert "magnificent" in main_line.lower()
        assert "tabby" in main_line.lower()
        assert "gracefully" in main_line.lower()

    def test_categorization_characters(self):
        """Test categorization of character descriptions."""
        descriptions = [
            "a landscape scene with mountains",
            "a person wearing a red coat",
            "an animal running in the field",
        ]

        result = CaptionUtils.combine(descriptions)

        # Should have some meaningful content from the descriptions
        full_result = result.lower()
        assert any(
            keyword in full_result
            for keyword in ["landscape", "mountains", "person", "wearing", "animal", "running"]
        )

    def test_categorization_actions(self):
        """Test categorization of action descriptions."""
        descriptions = [
            "a beautiful park setting",
            "someone running quickly through the area",
            "a person sitting on a bench",
        ]

        result = CaptionUtils.combine(descriptions)
        full_result = result.lower()
        assert any(keyword in full_result for keyword in ["running", "sitting"])

    def test_avoid_duplicate_information(self):
        """Test that duplicate information is avoided."""
        descriptions = [
            "a red car parked on street",
            "a red vehicle on the road",  # Similar information
            "a sunny day with clear skies",
        ]

        result = CaptionUtils.combine(descriptions)
        result.split("\n")

        # Should not repeat very similar information
        full_text = result.lower()
        red_count = full_text.count("red")
        assert red_count <= 2  # Allow some repetition but not excessive

    def test_preserve_new_information(self):
        """Test that genuinely new information is preserved."""
        descriptions = [
            "a simple house",
            "an outdoor mountainous environment with pine trees",
            "digital art style with vibrant colors",
        ]

        result = CaptionUtils.combine(descriptions)
        lines = result.split("\n")

        assert len(lines) >= 2
        full_text = result.lower()
        assert "mountainous" in full_text
        assert "pine" in full_text or "trees" in full_text
        assert "digital" in full_text or "vibrant" in full_text

    def test_category_order(self):
        """Test that categories are processed and content is preserved."""
        descriptions = [
            "a basic scene",
            "a character with blue hair",  # characters
            "running through the forest",  # actions
            "in a mystical forest setting",  # settings
            "with a happy mood",  # moods
            "drawn in anime style",  # styles
            "with intricate details",  # details
        ]

        result = CaptionUtils.combine(descriptions)

        # Should preserve meaningful content from descriptions
        full_text = result.lower()
        # At least some of the key descriptive words should be present
        assert any(
            word in full_text for word in ["mystical", "forest", "character", "blue", "hair"]
        )

    def test_multiline_output(self):
        """Test that output is properly formatted with multiple lines."""
        descriptions = [
            "a detailed fantasy landscape with mountains and rivers",
            "a brave warrior character wielding a sword",
            "standing heroically in the sunlight",
        ]

        result = CaptionUtils.combine(descriptions)
        lines = result.split("\n")

        assert len(lines) >= 2
        # Each line should end with proper punctuation
        for line in lines:
            if line.strip():
                assert line.strip()[-1] in ".!?"


class TestCaptionUtilsValidateCaption:
    """Test the validate_caption static method."""

    def test_validate_empty_caption(self):
        """Test validation of empty caption."""
        assert not CaptionUtils.validate_caption("")
        assert not CaptionUtils.validate_caption(None)

    def test_validate_too_short(self):
        """Test validation of too short captions."""
        assert not CaptionUtils.validate_caption("short")
        assert not CaptionUtils.validate_caption("a cat", min_length=10)

    def test_validate_minimum_length(self):
        """Test validation with custom minimum length."""
        caption = "a beautiful sunset scene"
        assert not CaptionUtils.validate_caption(caption, min_length=30)
        assert CaptionUtils.validate_caption(caption, min_length=20)

    def test_validate_refusal_patterns(self):
        """Test rejection of captions with refusal patterns."""
        refusal_captions = [
            "I'm sorry, I cannot describe this image",
            "I cannot provide a caption for this",
            "I apologize, but this content is inappropriate",
            "This image is inappropriate for description",
            "I'm unable to caption this image",
            "I refuse to describe this content",
        ]

        for caption in refusal_captions:
            assert not CaptionUtils.validate_caption(caption)

    def test_validate_refusal_patterns_case_insensitive(self):
        """Test that refusal pattern detection is case insensitive."""
        assert not CaptionUtils.validate_caption("I'M SORRY, I CANNOT HELP WITH THIS IMAGE")
        assert not CaptionUtils.validate_caption("i'm sorry, i cannot describe this")

    def test_validate_too_generic(self):
        """Test rejection of overly generic captions."""
        generic_captions = ["image", "picture", "photo", "illustration"]

        for caption in generic_captions:
            assert not CaptionUtils.validate_caption(caption)

    def test_validate_good_captions(self):
        """Test acceptance of good quality captions."""
        good_captions = [
            "a beautiful landscape with mountains and rivers flowing through green valleys",
            "a tabby cat sitting gracefully on a wooden chair in warm sunlight",
            "an ancient castle perched on a cliff overlooking the stormy ocean below",
            "a bustling marketplace filled with colorful fruits and vegetables being sold by vendors",
        ]

        for caption in good_captions:
            assert CaptionUtils.validate_caption(caption)

    def test_validate_borderline_cases(self):
        """Test validation of borderline cases."""
        # Should pass - contains refusal word but in valid context
        assert CaptionUtils.validate_caption(
            "the warrior refuses to surrender in this epic battle scene"
        )

        # Should pass - contains generic word but in longer context
        assert CaptionUtils.validate_caption(
            "this beautiful picture shows a magnificent sunset over mountains"
        )

        # Should fail - too short even though contains valid words
        assert not CaptionUtils.validate_caption("great image!")


class TestCaptionUtilsIntegration:
    """Integration tests for CaptionUtils methods working together."""

    def test_clean_then_validate(self):
        """Test cleaning a caption then validating it."""
        raw_caption = "in this image we can see a magnificent sunset over the ocean"

        cleaned = CaptionUtils.clean_caption(raw_caption)
        is_valid = CaptionUtils.validate_caption(cleaned)

        assert cleaned == "Magnificent sunset over the ocean."
        assert is_valid

    def test_combine_then_validate(self):
        """Test combining descriptions then validating the result."""
        descriptions = [
            "a beautiful landscape scene with rolling hills",
            "a peaceful countryside setting with farmhouses",
            "golden sunlight streaming through clouds",
        ]

        combined = CaptionUtils.combine(descriptions)
        is_valid = CaptionUtils.validate_caption(combined)

        assert is_valid
        assert len(combined.split("\n")) >= 2

    def test_full_pipeline(self):
        """Test full pipeline: combine, clean individual parts, and validate."""
        raw_descriptions = [
            "this image shows a detailed fantasy castle",
            "there are dragons flying around the towers",
            "the image depicts a mystical atmosphere",
        ]

        # Test the combine method handles the cleaning internally
        result = CaptionUtils.combine(raw_descriptions)
        is_valid = CaptionUtils.validate_caption(result)

        assert is_valid
        # Should preserve some meaningful content
        result_lower = result.lower()
        assert any(
            word in result_lower
            for word in [
                "castle",
                "dragons",
                "flying",
                "fantasy",
                "mystical",
                "towers",
                "atmosphere",
            ]
        )

        # Result should not contain the generic phrases at the start
        assert not result_lower.startswith("this image shows")
        assert not result_lower.startswith("there are")
        assert not result_lower.startswith("the image depicts")


if __name__ == "__main__":
    pytest.main([__file__])
