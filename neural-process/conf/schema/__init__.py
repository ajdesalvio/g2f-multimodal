"""ConfigStore registration for structured schemas.

Registration intentionally uses a bare ``name=`` with NO ``group=``.
Registering under ``group="misc"`` triggers Hydra's automatic
yaml-filename → schema binding, which under Hydra 1.3 binds the strict
dataclass to every yaml at ``conf/misc/*.yaml`` and rejects keys that
later additions intend to introduce. Loading the schema via an explicit
``defaults:`` entry from ``conf/misc/base.yaml`` instead keeps
validation opt-in and predictable.
"""
from __future__ import annotations

from hydra.core.config_store import ConfigStore

from .misc import MiscConfig

cs = ConfigStore.instance()
cs.store(name="base_misc", node=MiscConfig)  # NO group — see module docstring
