"""Unit tests for src/utils/config.py dot-notation getter edge cases."""
from __future__ import annotations

import src.utils.config as config_mod
from src.utils.config import get, load_config


def test_get_top_level_key():
    assert get("seed") == 42


def test_get_nested_key():
    assert get("data.label_col") == "human_score"


def test_get_missing_key_returns_default():
    assert get("does.not.exist", "fallback") == "fallback"


def test_get_missing_key_default_none():
    assert get("nope") is None


def test_get_traverses_into_nonexistent_intermediate():
    # 'seed' is an int, not a dict — descending further must yield the default
    assert get("seed.deeper", "X") == "X"


def test_get_dict_default_not_threaded_into_walk(monkeypatch):
    """A dict-valued default must be returned as-is, never continue the traversal.

    The naive implementation passed `default` into each .get(), so a dict default
    could accidentally satisfy the next path segment. This guards that bug.
    """
    fake = {"a": {"b": 1}}
    monkeypatch.setattr(config_mod, "load_config", lambda: fake)
    # 'a.x' is missing; default is a dict that *contains* the next-looked-up key.
    # Correct behaviour: return the default object verbatim.
    sentinel = {"c": 99}
    assert get("a.x.c", sentinel) == sentinel


def test_get_falsy_values_preserved(monkeypatch):
    """Stored falsy values (0, False, '') must be returned, not replaced by default."""
    fake = {"flags": {"zero": 0, "off": False, "empty": ""}}
    monkeypatch.setattr(config_mod, "load_config", lambda: fake)
    assert get("flags.zero", 99) == 0
    assert get("flags.off", True) is False
    assert get("flags.empty", "x") == ""
