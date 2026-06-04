"""Unit tests for src/explain/attributions.py — pure functions only (no model)."""
from __future__ import annotations

import numpy as np

from src.explain.attributions import (
    _merge_wordpieces,
    to_html_highlight,
)


# ── WordPiece merging ─────────────────────────────────────────────────────────

def test_merge_wordpieces_basic():
    tokens = ["[CLS]", "play", "##ing", "well", "[SEP]"]
    scores = np.array([0.0, 0.4, 0.6, 0.5, 0.0])
    merged = _merge_wordpieces(tokens, scores)
    words = [w for w, _ in merged]
    assert words == ["playing", "well"]


def test_merge_wordpieces_averages_subtokens():
    tokens = ["un", "##think", "##able"]
    scores = np.array([0.3, 0.6, 0.9])
    merged = _merge_wordpieces(tokens, scores)
    assert merged[0][0] == "unthinkable"
    assert merged[0][1] == np.mean([0.3, 0.6, 0.9])


def test_merge_wordpieces_skips_specials():
    tokens = ["[CLS]", "hello", "[SEP]", "[PAD]"]
    scores = np.array([1.0, 0.5, 0.0, 0.0])
    merged = _merge_wordpieces(tokens, scores)
    assert [w for w, _ in merged] == ["hello"]


# ── HTML highlighting ─────────────────────────────────────────────────────────

def test_to_html_highlight_escapes_markup():
    """User text that looks like HTML must be escaped, not rendered."""
    pairs = [("<script>", 0.9), ("alert", 0.8), ("safe", 0.1)]
    out = to_html_highlight(pairs, top_k=3)
    assert "<script>" not in out          # raw tag must not survive
    assert "&lt;script&gt;" in out         # escaped form present


def test_to_html_highlight_empty():
    assert to_html_highlight([], top_k=5) == ""


def test_to_html_highlight_highlights_top_k():
    pairs = [("low", 0.1), ("high", 0.9), ("mid", 0.5)]
    out = to_html_highlight(pairs, top_k=1)
    # Only the top-1 ("high") should get a background span
    assert out.count("background-color") == 1
    assert "high" in out
