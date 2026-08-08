"""Compatibility facade for model components.

New code should import from `prism.models`; this module remains for existing
training and analysis scripts.
"""

from prism.models import *  # noqa: F401,F403
