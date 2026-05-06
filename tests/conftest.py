"""
Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances

We seed ALEPH_ALPHA_API_KEY with a placeholder if it is not already
set so structural and behavioral validators can mount the provider
without needing real credentials. Tests that hit the network are
skipped explicitly when no real key is present.
"""

import os

os.environ.setdefault("ALEPH_ALPHA_API_KEY", "test-key-for-pytest")
