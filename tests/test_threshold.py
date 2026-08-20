"""Tests for how the serving threshold is chosen.

This is the logic most worth protecting: the project's central claim is that the
cutoff is picked on expected value rather than left at 0.5, and that claim is
only true if the value the API serves is the value training chose. A regression
here would be silent — predictions keep returning 200 with a plausible-looking
probability, just thresholded at the wrong place.
"""

import pytest

from src import config
from src.api import predict as engine


@pytest.fixture(autouse=True)
def restore_config():
    """Reloading a module mutates it in place for every importer.

    The tests below reload `config` to check how it parses the environment, so
    without this the last reload's values would leak into whatever runs next.
    """
    yield
    import importlib

    importlib.reload(config)


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.setattr(config, "DECISION_THRESHOLD_OVERRIDE", None)


@pytest.fixture
def with_override(monkeypatch):
    monkeypatch.setattr(config, "DECISION_THRESHOLD_OVERRIDE", 0.42)


def test_model_threshold_used_when_no_override(no_override):
    value, source = engine._resolve_threshold((0.31, "model v2 (run abc12345)"))
    assert value == 0.31
    assert "model v2" in source


def test_env_override_beats_the_model(with_override):
    value, source = engine._resolve_threshold((0.31, "model v2 (run abc12345)"))
    assert value == 0.42
    assert "override" in source


def test_override_applies_even_when_model_has_no_threshold(with_override):
    value, source = engine._resolve_threshold(None)
    assert value == 0.42
    assert "override" in source


def test_fallback_is_announced_not_silent(no_override):
    """A model with no logged threshold must not look like a deliberate 0.5."""
    value, source = engine._resolve_threshold(None)
    assert value == config.FALLBACK_THRESHOLD
    assert "fallback" in source


def test_unset_env_var_is_none_not_a_number(monkeypatch):
    """The original bug: an absent variable defaulted to 0.5 and won silently."""
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    import importlib

    reloaded = importlib.reload(config)
    assert reloaded.DECISION_THRESHOLD_OVERRIDE is None


def test_empty_env_var_is_treated_as_unset(monkeypatch):
    """docker-compose passes an empty string when the variable is not set."""
    monkeypatch.setenv("DECISION_THRESHOLD", "")
    import importlib

    reloaded = importlib.reload(config)
    assert reloaded.DECISION_THRESHOLD_OVERRIDE is None


def test_explicit_env_var_is_parsed(monkeypatch):
    monkeypatch.setenv("DECISION_THRESHOLD", "0.37")
    import importlib

    reloaded = importlib.reload(config)
    assert reloaded.DECISION_THRESHOLD_OVERRIDE == 0.37
