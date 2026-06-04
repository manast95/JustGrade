"""Token/phrase attribution + HTML highlight for explainability.

Uses attention rollout on the fine-tuned DistilBERT model — no extra dependencies
(captum has a scipy conflict on Python 3.14). Rollout propagates attention through
all layers so the CLS-token attribution reflects deep contextual importance, not
just the last layer.
"""
from __future__ import annotations

import html
import re

import numpy as np
import torch


# ── Attention rollout ─────────────────────────────────────────────────────────

def _rollout(attentions: tuple) -> np.ndarray:
    """Propagate attention through all layers (attention rollout).

    Fuses multi-head attention per layer by averaging, adds residual identity,
    then chains the matrices across layers. Returns a (seq_len,) array of
    importance scores relative to the CLS token at position 0.

    Reference: Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers".
    """
    # attentions: tuple of (batch=1, heads, seq, seq) tensors
    mat = torch.eye(attentions[0].shape[-1], dtype=torch.float32)
    for attn in attentions:
        # Average over heads → (seq, seq)
        fused = attn[0].mean(dim=0).cpu().float()
        # Add residual (identity) and row-normalise
        fused = fused + torch.eye(fused.shape[0])
        fused = fused / fused.sum(dim=-1, keepdim=True)
        mat = fused @ mat
    # Row 0 = how much CLS attends to every other token
    return mat[0].numpy()


# ── Word-level aggregation ────────────────────────────────────────────────────

def _merge_wordpieces(tokens: list[str], scores: np.ndarray) -> list[tuple[str, float]]:
    """Merge WordPiece sub-tokens into whole words, averaging their scores.

    Skips special tokens ([CLS], [SEP], [PAD]).
    """
    SKIP = {"[CLS]", "[SEP]", "[PAD]"}
    words: list[tuple[str, float]] = []
    current_word, current_scores = "", []

    for tok, sc in zip(tokens, scores):
        if tok in SKIP:
            if current_word:
                words.append((current_word, float(np.mean(current_scores))))
                current_word, current_scores = "", []
            continue
        if tok.startswith("##"):
            current_word += tok[2:]
            current_scores.append(sc)
        else:
            if current_word:
                words.append((current_word, float(np.mean(current_scores))))
            current_word = tok
            current_scores = [sc]

    if current_word:
        words.append((current_word, float(np.mean(current_scores))))

    return words


# ── Public API ────────────────────────────────────────────────────────────────

def _load_eager_model(model):
    """Return a version of the model configured for eager attention.

    Transformers 5.x defaults to SDPA which doesn't support output_attentions.
    We reload from the saved path with attn_implementation='eager'; this copy
    is used only for attribution and is discarded afterwards.
    """
    from transformers import AutoModelForSequenceClassification  # noqa: PLC0415

    saved_path = getattr(model, "_saved_path", None)
    if saved_path is None:
        raise RuntimeError(
            "TransformerScorer._saved_path is not set. "
            "Load the model via TransformerScorer.load(path) before computing attributions."
        )
    eager = AutoModelForSequenceClassification.from_pretrained(
        saved_path, attn_implementation="eager"
    )
    eager.eval()
    return eager


def token_attributions(
    model,          # TransformerScorer (must be loaded via .load())
    text: str,
) -> list[tuple[str, float]]:
    """Compute per-word importance via attention rollout.

    Args:
        model: a loaded TransformerScorer (TransformerScorer.load(path)).
        text: the full concatenated input (Q: … Rubric: … Response: …).

    Returns:
        List of (word, weight) pairs with weights normalised to [0, 1].
        Words are reconstructed from WordPiece sub-tokens.
    """
    tok = model._get_tokenizer()
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=model._max_length,
    )

    # Use an eager-attention copy so output_attentions=True works under Transformers 5.x
    eager_model = _load_eager_model(model)
    device = next(model._model.parameters()).device
    eager_model = eager_model.to(device)

    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = eager_model(**inputs, output_attentions=True)

    scores = _rollout(outputs.attentions)             # (seq_len,)
    token_ids = enc["input_ids"][0].tolist()
    tokens = tok.convert_ids_to_tokens(token_ids)    # includes [CLS] / [SEP]

    words = _merge_wordpieces(tokens, scores)

    if not words:
        return []
    max_w = max(w for _, w in words) or 1.0
    return [(word, w / max_w) for word, w in words]


def response_attributions(
    model,
    question: str,
    rubric: str,
    response: str,
) -> list[tuple[str, float]]:
    """Convenience wrapper: build the concatenated text and return attributions
    restricted to tokens that appear in the response portion only.

    The full context (Q + rubric) is still given to the model so attention is
    contextualised, but only response words are returned.

    The response boundary is found by counting how many merged words make up the
    fixed prefix ("Q: … Rubric: … Response: "). This is robust even when the
    question or rubric themselves contain the literal word "response" — a naive
    text scan for "response" would otherwise match inside the rubric.
    """
    import string as _string  # noqa: PLC0415

    full_text = f"Q: {question}\nRubric: {rubric}\nResponse: {response}"
    all_pairs = token_attributions(model, full_text)

    # Count merged words in the prefix (no model forward needed — just tokenise).
    # Because the prefix ends in "Response: " (with a trailing space), WordPiece
    # never merges across the boundary, so the prefix word-count matches the
    # leading words of the full-text attribution list exactly.
    prefix = f"Q: {question}\nRubric: {rubric}\nResponse: "
    tok = model._get_tokenizer()
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    prefix_tokens = tok.convert_ids_to_tokens(prefix_ids)
    n_prefix_words = len(_merge_wordpieces(prefix_tokens, np.zeros(len(prefix_tokens))))

    if n_prefix_words < len(all_pairs):
        pairs = all_pairs[n_prefix_words:]
    else:
        # Response was truncated away entirely — fall back to the full list.
        pairs = all_pairs

    # Drop pure-punctuation tokens — attention weights them heavily as structural
    # anchors but they're not meaningful for phrase highlighting.
    _punct = set(_string.punctuation)
    pairs = [(w, s) for w, s in pairs if not all(c in _punct for c in w)]

    # Re-normalise after filtering
    if not pairs:
        return []
    max_w = max(s for _, s in pairs) or 1.0
    return [(w, s / max_w) for w, s in pairs]


# ── HTML renderer ─────────────────────────────────────────────────────────────

_COLOURS = [
    # (r, g, b) gradient: pale yellow → orange → red  (low → high importance)
    (255, 253, 200),
    (255, 220, 130),
    (255, 180,  50),
    (255, 130,   0),
    (220,  60,   0),
]


def _importance_colour(weight: float) -> str:
    """Map a normalised weight [0,1] to an rgba colour string."""
    idx = min(int(weight * len(_COLOURS)), len(_COLOURS) - 1)
    r, g, b = _COLOURS[idx]
    alpha = 0.3 + 0.7 * weight     # min alpha 0.3 so highlights are always visible
    return f"rgba({r},{g},{b},{alpha:.2f})"


def to_html_highlight(
    word_weights: list[tuple[str, float]],
    top_k: int = 10,
) -> str:
    """Render word–weight pairs as an HTML string with highlighted phrases.

    Top-k words by importance are coloured; others are plain text.  Safe to
    embed directly in ``st.markdown(..., unsafe_allow_html=True)``.

    Args:
        word_weights: list of (word, weight) from token_attributions / response_attributions.
        top_k: number of top-importance words to highlight.

    Returns:
        HTML string.
    """
    if not word_weights:
        return ""

    sorted_idx = sorted(range(len(word_weights)),
                        key=lambda i: word_weights[i][1], reverse=True)
    highlight_set = set(sorted_idx[:top_k])

    parts: list[str] = []
    for i, (word, weight) in enumerate(word_weights):
        # Restore spacing: no leading space after punctuation that precedes text
        prefix = "" if (parts and re.search(r"[\-/]$", parts[-1])) else " "
        clean = word.strip()
        if not clean:
            continue
        # Escape so any HTML-like tokens render as text, not markup
        # (st.markdown is called with unsafe_allow_html=True downstream).
        safe = html.escape(clean)

        if i in highlight_set:
            colour = _importance_colour(weight)
            parts.append(
                f'{prefix}<span style="background-color:{colour};'
                f'border-radius:3px;padding:1px 3px;font-weight:500;">'
                f'{safe}</span>'
            )
        else:
            parts.append(f"{prefix}{safe}")

    rendered = "".join(parts).strip()
    # Wrap in a styled div
    return (
        '<div style="font-size:1rem;line-height:1.7;font-family:sans-serif;">'
        + rendered
        + "</div>"
    )
