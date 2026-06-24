"""
ClinIQ ML — Deviation NLP Classifier
Rule-based spaCy Matcher classifying free-text deviations into five categories:
consent, dosing, eligibility, documentation, safety.

Design rationale: keyword-pattern rules are auditable (required for GCP contexts),
require no model download, and are easily extended by clinical ops staff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import spacy
from spacy.matcher import Matcher

CATEGORIES = ["consent", "dosing", "eligibility", "documentation", "safety"]

# ---------------------------------------------------------------------------
# Keyword pattern definitions
# Each pattern is a list of token dicts (spaCy Matcher format).
# LOWER makes matching case-insensitive.
# ---------------------------------------------------------------------------
_PATTERNS: dict[str, list[list[dict]]] = {
    "consent": [
        [{"LOWER": "consent"}],
        [{"LOWER": "icf"}],
        [{"LOWER": "informed"}, {"LOWER": "consent"}],
        [{"LOWER": "reconsent"}],
        [{"LOWER": "re-consent"}],
        [{"LOWER": "re"}, {"LOWER": "-"}, {"LOWER": "consented"}],
        [{"LOWER": "re-consented"}],
        [{"LOWER": "assent"}],
    ],
    "dosing": [
        [{"LOWER": "dose"}],
        [{"LOWER": "dosing"}],
        [{"LOWER": "drug"}, {"LOWER": "administered"}],
        [{"LOWER": "imp"}],
        [{"LOWER": "investigational"}, {"LOWER": "medicinal"}],
        [{"LOWER": "administration"}],
        [{"LOWER": "window"}],          # "outside the permitted window" context
    ],
    "eligibility": [
        [{"LOWER": "eligibility"}],
        [{"LOWER": "eligible"}],
        [{"LOWER": "inclusion"}],
        [{"LOWER": "exclusion"}],
        [{"LOWER": "criterion"}],
        [{"LOWER": "criteria"}],
        [{"LOWER": "enrolled"}, {"LOWER": "with"}],
        [{"LOWER": "washout"}],
        [{"LOWER": "egfr"}],
        [{"LOWER": "ecg"}],
        [{"LOWER": "laboratory"}, {"LOWER": "result"}],
        [{"LOWER": "lab"}, {"LOWER": "result"}],
    ],
    "documentation": [
        [{"LOWER": "documentation"}],
        [{"LOWER": "document"}],
        [{"LOWER": "source"}, {"LOWER": "data"}],
        [{"LOWER": "crf"}],
        [{"LOWER": "ecrf"}],
        [{"LOWER": "signature"}],
        [{"LOWER": "reported"}],
        [{"LOWER": "not"}, {"LOWER": "documented"}],
        [{"LOWER": "missing"}],
        [{"LOWER": "narrative"}],
        [{"LOWER": "isf"}],             # investigator site file
    ],
    "safety": [
        [{"LOWER": "adverse"}, {"LOWER": "event"}],
        [{"LOWER": "ae"}],
        [{"LOWER": "sae"}],
        [{"LOWER": "serious"}, {"LOWER": "adverse"}],
        [{"LOWER": "safety"}],
        [{"LOWER": "24-hour"}, {"LOWER": "window"}],
        [{"LOWER": "regulatory"}, {"LOWER": "deadline"}],
        [{"LOWER": "reported"}, {"LOWER": "within"}],
    ],
}


class DeviationClassifier:
    """
    Classifies a free-text deviation description into one of five categories.
    Falls back to 'documentation' when no patterns match (most common miscellaneous type).
    """

    def __init__(self):
        self.nlp = spacy.blank("en")
        self.matcher = Matcher(self.nlp.vocab)
        for category, patterns in _PATTERNS.items():
            self.matcher.add(category, patterns)

    def classify(self, text: str) -> str:
        """Return the best-matching category for a free-text deviation description."""
        if not text or not text.strip():
            return "documentation"

        doc = self.nlp(text.lower())
        matches = self.matcher(doc)

        if not matches:
            return "documentation"

        # Count matches per category and return the highest
        counts: dict[str, int] = {}
        for match_id, _, _ in matches:
            label = self.nlp.vocab.strings[match_id]
            counts[label] = counts.get(label, 0) + 1

        return max(counts, key=counts.get)

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Classify a list of deviation descriptions."""
        return [self.classify(t) for t in texts]

    def confidence_scores(self, text: str) -> dict[str, int]:
        """Return match counts per category (useful for borderline cases)."""
        if not text or not text.strip():
            return {c: 0 for c in CATEGORIES}

        doc = self.nlp(text.lower())
        matches = self.matcher(doc)
        counts: dict[str, int] = {c: 0 for c in CATEGORIES}
        for match_id, _, _ in matches:
            label = self.nlp.vocab.strings[match_id]
            if label in counts:
                counts[label] += 1
        return counts


# Module-level singleton — avoids rebuilding the Matcher on every call
_classifier: Optional[DeviationClassifier] = None


def get_classifier() -> DeviationClassifier:
    global _classifier
    if _classifier is None:
        _classifier = DeviationClassifier()
    return _classifier


def classify_deviation(text: str) -> str:
    """Convenience function using the module singleton."""
    return get_classifier().classify(text)
