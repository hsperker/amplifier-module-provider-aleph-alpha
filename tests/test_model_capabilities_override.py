"""Tests for the model_capabilities config override.

The override lets bundle authors describe model capabilities for any model
served behind a custom base_url (e.g. Aleph Alpha's `kimi-k2.5`) without
needing to extend the static capabilities table in _capabilities.py.
"""

from amplifier_module_provider_openai_like import OpenAIProvider


def _make_provider(**overrides) -> OpenAIProvider:
    config = {**overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def test_unknown_model_falls_back_to_conservative_defaults():
    provider = _make_provider(default_model="kimi-k2.5")
    info = provider.get_info()
    # Without overrides, the unknown family advertises 200K context, the
    # ModelCapabilities default. Documenting current behavior to make
    # any regression visible.
    assert info.defaults["context_window"] == 200_000


def test_model_capabilities_override_is_applied():
    """get_info() picks up overrides for the configured default model."""
    provider = _make_provider(
        default_model="kimi-k2.5",
        model_capabilities={
            "kimi-k2.5": {
                "context_window": 131_072,
                "max_output_tokens": 131_072,
                "supports_vision": True,
                "supports_reasoning": False,
                "capability_tags": ["tools", "streaming", "vision"],
            }
        },
    )
    info = provider.get_info()
    assert info.defaults["context_window"] == 131_072
    assert info.defaults["max_output_tokens"] == 131_072


def test_override_only_targets_named_model():
    """Other models keep their static-table values when not in the override."""
    provider = _make_provider(
        default_model="gpt-5.5",
        model_capabilities={"kimi-k2.5": {"context_window": 131_072}},
    )
    info = provider.get_info()
    assert info.defaults["context_window"] == 1_000_000


def test_override_partial_fields_inherit_base():
    """Fields absent from the override fall through to the base capabilities."""
    provider = _make_provider(
        default_model="kimi-k2.5",
        model_capabilities={"kimi-k2.5": {"context_window": 131_072}},
    )
    info = provider.get_info()
    assert info.defaults["context_window"] == 131_072
    # max_output_tokens not overridden, so we keep the unknown-family default
    # (128_000 from ModelCapabilities).
    assert info.defaults["max_output_tokens"] == 128_000


def test_capability_tags_stored_as_tuple():
    """capability_tags accepts a list in config and stores as tuple internally."""
    provider = _make_provider(
        default_model="kimi-k2.5",
        model_capabilities={
            "kimi-k2.5": {"capability_tags": ["tools", "streaming"]}
        },
    )
    caps = provider._resolve_capabilities("kimi-k2.5")
    assert caps.capability_tags == ("tools", "streaming")
