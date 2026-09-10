"""Caption acceptance and JSON transformations, independent of inference backends."""

import json
import logging
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_REFUSAL_MARKERS = (
    "i'm sorry",
    "i’m sorry",
    "i cannot",
    "i can't",
    "i can’t",
    "unable to provide",
    "unable to describe",
    "cannot provide",
    "can't provide",
    "can’t provide",
    "cannot assist",
    "can't assist",
    "can’t assist",
)


def validate_response_format(value: Any) -> None:
    """Validate the shared constrained-decoding envelope before either backend runs."""
    if value is None:
        return
    if not isinstance(value, dict) or value.get("type") not in {
        "text",
        "json_object",
        "json_schema",
    }:
        raise ValueError(
            "response_format must be a mapping with type text, json_object or json_schema"
        )
    if value["type"] == "json_schema":
        schema = value.get("json_schema")
        if not isinstance(schema, dict) or not isinstance(schema.get("schema"), dict):
            raise ValueError("response_format.json_schema.schema must be a mapping")


def normalize_refusal_markers(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(marker, str) or not marker.strip() for marker in value
    ):
        raise ValueError("refusal_markers must be a list of non-empty strings")
    return tuple(dict.fromkeys(marker.strip().casefold() for marker in value))


@dataclass(frozen=True)
class OutputPolicy:
    """Shared interpretation of a model's caption output."""

    validate_json_output: bool = False
    repair_invalid_json_escapes: bool = False
    canonicalize_json_output: bool = False
    normalize_yxyx_bboxes: bool = False
    deduplicate_json_elements: bool = False
    refusal_markers: tuple[str, ...] = DEFAULT_REFUSAL_MARKERS

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, partial: bool = False) -> "OutputPolicy":
        options = dict(config)
        known = {field.name for field in fields(cls)}
        if unknown := options.keys() - known:
            raise ValueError(f"Unknown output_processing option(s): {', '.join(sorted(unknown))}")
        for name, value in options.items():
            if name != "refusal_markers" and not isinstance(value, bool):
                raise ValueError(f"output_processing.{name} must be a boolean")
        if "refusal_markers" in options:
            options["refusal_markers"] = normalize_refusal_markers(options["refusal_markers"])
        policy = cls(**options)
        check_dependencies = not partial or "validate_json_output" in options
        if (
            check_dependencies
            and not policy.validate_json_output
            and any(
                getattr(policy, name)
                for name in known - {"validate_json_output", "refusal_markers"}
            )
        ):
            raise ValueError(
                "output_processing.validate_json_output must be enabled for JSON transforms"
            )
        return policy

    def is_refusal(self, text: str) -> bool:
        return any(marker in text.strip().casefold() for marker in self.refusal_markers)

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        for token in ("<|end|>", "<|endoftext|>", "<|im_end|>"):
            text = text.split(token, 1)[0]
        return text.strip()

    def process(self, text: str) -> Optional[str]:
        text = self.clean(text)
        if not text or self.is_refusal(text):
            return None
        if not self.validate_json_output:
            return text
        try:
            parsed = self._loads(text)
        except ValueError as error:
            if not self.repair_invalid_json_escapes:
                logger.warning("Rejecting invalid JSON output: %s", error)
                return None
            repaired = self._repair_json_escapes(text)
            try:
                parsed = self._loads(repaired)
            except ValueError:
                logger.warning("Rejecting invalid JSON output: %s", error)
                return None
            text = repaired
        if self.normalize_yxyx_bboxes:
            self._normalize_yxyx_bbox_values(parsed)
        if self.deduplicate_json_elements:
            self._deduplicate_element_arrays(parsed)
        if (
            self.canonicalize_json_output
            or self.normalize_yxyx_bboxes
            or self.deduplicate_json_elements
        ):
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return text

    @staticmethod
    def _loads(text: str) -> Any:
        def reject_constant(value: str) -> None:
            raise ValueError(f"Invalid JSON numeric constant: {value}")

        def finite_float(value: str) -> float:
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("JSON number exceeds finite float range")
            return result

        return json.loads(text, parse_constant=reject_constant, parse_float=finite_float)

    @classmethod
    def _deduplicate_element_arrays(cls, value: Any) -> None:
        if isinstance(value, dict):
            # Normalize children first so recursively equivalent elements deduplicate.
            for child in value.values():
                cls._deduplicate_element_arrays(child)
            elements = value.get("elements")
            if isinstance(elements, list):
                unique, seen = [], set()
                for element in elements:
                    fingerprint = json.dumps(element, sort_keys=True, ensure_ascii=False)
                    if fingerprint not in seen:
                        seen.add(fingerprint)
                        unique.append(element)
                value["elements"] = unique
        elif isinstance(value, list):
            for child in value:
                cls._deduplicate_element_arrays(child)

    @classmethod
    def _normalize_yxyx_bbox_values(cls, value: Any) -> None:
        if isinstance(value, dict):
            bbox = value.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4 and all(type(v) is int for v in bbox):
                ymin, xmin, ymax, xmax = bbox
                value["bbox"] = [min(ymin, ymax), min(xmin, xmax), max(ymin, ymax), max(xmin, xmax)]
            for child in value.values():
                cls._normalize_yxyx_bbox_values(child)
        elif isinstance(value, list):
            for child in value:
                cls._normalize_yxyx_bbox_values(child)

    @staticmethod
    def _repair_json_escapes(text: str) -> str:
        """Preserve valid escape pairs, escaping only invalid backslashes."""
        repaired = []
        index = 0
        while index < len(text):
            char = text[index]
            if char != "\\":
                repaired.append(char)
                index += 1
                continue
            following = text[index + 1 : index + 2]
            digits = text[index + 2 : index + 6]
            valid_unicode = (
                following == "u"
                and len(digits) == 4
                and all(digit in "0123456789abcdefABCDEF" for digit in digits)
            )
            if following and (following in '"\\/bfnrt' or valid_unicode):
                size = 6 if valid_unicode else 2
                repaired.append(text[index : index + size])
                index += size
            else:
                repaired.append("\\\\")
                index += 1
        return "".join(repaired)
