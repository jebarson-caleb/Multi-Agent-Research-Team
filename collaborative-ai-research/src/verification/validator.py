"""Output validator for checking agent outputs against specifications."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ValidationResult:
    """Result of an output validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0  # 0.0-1.0 quality score
    metadata: dict[str, Any] = field(default_factory=dict)


class OutputValidator:
    """Validates agent outputs against format and content specifications.

    Checks:
    - JSON format validity
    - Required fields presence
    - Value type correctness
    - Content length limits
    - Confidence score ranges
    - No empty/null required values
    """

    # Schema definitions for different output types
    SCHEMAS: dict[str, dict[str, Any]] = {
        "research": {
            "required_fields": ["findings", "summary"],
            "optional_fields": ["gaps", "suggested_follow_up", "sources"],
            "findings_item_fields": ["fact", "confidence"],
        },
        "analysis": {
            "required_fields": ["summary"],
            "optional_fields": ["patterns", "insights", "trends", "limitations"],
        },
        "synthesis": {
            "required_fields": ["summary"],
            "optional_fields": [
                "key_findings", "conclusions", "recommendations",
                "sources", "confidence", "metadata",
            ],
        },
        "verification": {
            "required_fields": ["overall_score"],
            "optional_fields": ["verified_findings", "inconsistencies", "recommendations"],
        },
    }

    def __init__(
        self,
        max_output_length: int = 50000,
        max_findings: int = 100,
    ) -> None:
        """Initialize the validator.

        Args:
            max_output_length: Maximum output string length.
            max_findings: Maximum number of findings items.
        """
        self._max_output_length = max_output_length
        self._max_findings = max_findings
        self._validation_history: list[ValidationResult] = []

    def validate(
        self,
        output: Any,
        output_type: str = "research",
    ) -> ValidationResult:
        """Validate an agent output.

        Args:
            output: Output to validate (dict or string).
            output_type: Type of output (research, analysis, synthesis, verification).

        Returns:
            ValidationResult with errors and warnings.
        """
        errors = []
        warnings = []

        # Parse if string
        if isinstance(output, str):
            if len(output) > self._max_output_length:
                errors.append(
                    f"Output exceeds maximum length ({len(output)} > {self._max_output_length})"
                )
            try:
                json_start = output.find("{")
                json_end = output.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    output = json.loads(output[json_start:json_end])
                else:
                    errors.append("Output is not valid JSON")
                    return self._make_result(False, errors, warnings, 0.0)
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error: {str(e)}")
                return self._make_result(False, errors, warnings, 0.0)

        if not isinstance(output, dict):
            errors.append(f"Output must be a dict, got {type(output).__name__}")
            return self._make_result(False, errors, warnings, 0.0)

        # Schema validation
        schema = self.SCHEMAS.get(output_type)
        if schema:
            schema_errors, schema_warnings = self._validate_schema(output, schema)
            errors.extend(schema_errors)
            warnings.extend(schema_warnings)

        # Content validation
        content_errors, content_warnings = self._validate_content(output)
        errors.extend(content_errors)
        warnings.extend(content_warnings)

        # Calculate quality score
        total_checks = max(1, len(errors) + len(warnings) + 5)
        score = max(0.0, 1.0 - (len(errors) * 0.2 + len(warnings) * 0.05))

        result = self._make_result(len(errors) == 0, errors, warnings, score)
        self._validation_history.append(result)
        return result

    def _validate_schema(
        self,
        output: dict[str, Any],
        schema: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Validate output against schema."""
        errors = []
        warnings = []

        # Check required fields
        for field_name in schema.get("required_fields", []):
            if field_name not in output:
                errors.append(f"Missing required field: '{field_name}'")
            elif output[field_name] is None or output[field_name] == "":
                errors.append(f"Required field '{field_name}' is empty")

        # Check optional fields
        for field_name in schema.get("optional_fields", []):
            if field_name not in output:
                warnings.append(f"Optional field '{field_name}' not present")

        # Validate findings items if applicable
        findings = output.get("findings", [])
        if isinstance(findings, list):
            if len(findings) > self._max_findings:
                warnings.append(
                    f"Large number of findings ({len(findings)} > {self._max_findings})"
                )
            for i, item in enumerate(findings):
                if isinstance(item, dict):
                    for item_field in schema.get("findings_item_fields", []):
                        if item_field not in item:
                            warnings.append(
                                f"Finding [{i}] missing field '{item_field}'"
                            )

        return errors, warnings

    def _validate_content(
        self,
        output: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Validate content values."""
        errors = []
        warnings = []

        # Check confidence scores are in range
        for key in ["confidence", "overall_score"]:
            if key in output:
                value = output[key]
                if isinstance(value, (int, float)):
                    if not 0.0 <= value <= 1.0:
                        errors.append(
                            f"'{key}' score {value} out of range [0.0, 1.0]"
                        )

        # Check findings confidence scores
        for item in output.get("findings", []):
            if isinstance(item, dict) and "confidence" in item:
                conf = item["confidence"]
                if isinstance(conf, (int, float)) and not 0.0 <= conf <= 1.0:
                    warnings.append(f"Finding confidence {conf} out of range")

        # Check for empty lists that should have content
        for key in ["findings", "conclusions", "key_findings"]:
            if key in output and isinstance(output[key], list) and len(output[key]) == 0:
                warnings.append(f"'{key}' is empty")

        # Check summary length
        summary = output.get("summary", "")
        if isinstance(summary, str):
            if len(summary) < 10:
                warnings.append("Summary is very short")
            if len(summary) > 10000:
                warnings.append("Summary is unusually long")

        return errors, warnings

    def _make_result(
        self,
        valid: bool,
        errors: list[str],
        warnings: list[str],
        score: float,
    ) -> ValidationResult:
        """Create a ValidationResult."""
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            score=min(max(score, 0.0), 1.0),
        )

    def validate_json_format(self, text: str) -> bool:
        """Check if text contains valid JSON."""
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json.loads(text[json_start:json_end])
                return True
        except json.JSONDecodeError:
            pass
        return False

    def get_metrics(self) -> dict[str, Any]:
        """Get validation metrics."""
        if not self._validation_history:
            return {"total_validations": 0}

        valid_count = sum(1 for r in self._validation_history if r.valid)
        scores = [r.score for r in self._validation_history]

        return {
            "total_validations": len(self._validation_history),
            "valid_rate": valid_count / len(self._validation_history),
            "avg_score": sum(scores) / len(scores),
            "total_errors": sum(len(r.errors) for r in self._validation_history),
            "total_warnings": sum(len(r.warnings) for r in self._validation_history),
        }
