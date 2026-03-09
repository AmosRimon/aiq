# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AI-Q Blueprint core package.

This module uses lazy imports to avoid loading heavy dependencies (langgraph, etc.)
when only lightweight submodules like `aiq_agent.knowledge` are needed.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_logger = logging.getLogger(__name__)


def _find_project_root() -> Path | None:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


_project_root = _find_project_root()
if _project_root:
    load_dotenv(_project_root / "deploy" / ".env", override=False)


def _init_deepchecks() -> None:
    """Initialize Deepchecks LLM observability if configured."""
    dc_api_key = os.environ.get("DEEPCHECKS_API_KEY")
    if not dc_api_key:
        _logger.info("DEEPCHECKS_API_KEY not set, skipping Deepchecks integration")
        return

    try:
        import sys
        # Use local copy of deepchecks_llm_client instead of installed package
        _client_path = str(Path(__file__).resolve().parent.parent / "client")
        if _client_path not in sys.path:
            sys.path.insert(0, _client_path)

        from deepchecks_llm_client.data_types import EnvType
        from deepchecks_llm_client.otel import LanggraphIntegration

        dc_host = os.environ.get("DC_STAGING_HOST", "https://app.llm.deepchecks.com/")
        dc_app_name = os.environ.get("DC_APP_NAME", "rotem_nvidia1")
        dc_version_name = os.environ.get("DC_VERSION_NAME", "v1")

        LanggraphIntegration().register_dc_exporter(
            host=dc_host,
            api_key=dc_api_key,
            app_name=dc_app_name,
            version_name=dc_version_name,
            env_type=EnvType.EVAL,
            log_to_console=True,
            isolated=True,
        )
        _logger.info("Deepchecks integration initialized (app=%s, version=%s)", dc_app_name, dc_version_name)
    except Exception:
        _logger.warning("Deepchecks integration failed to initialize, continuing without it", exc_info=True)


_init_deepchecks()

__all__ = [
    "chat_deepresearcher_agent",
    "shallow_research_agent",
    "deep_research_agent",
]

from typing import Any

# Cache for lazy-loaded modules to avoid repeated imports
_lazy_imports: dict[str, Any] = {}


def __getattr__(name: str):
    """Lazy import agents to avoid loading langgraph/ray dependencies unnecessarily.

    This allows `from aiq_agent.knowledge import ...` to work without pulling in
    the full agent stack and its heavy dependencies (langgraph, etc.).
    """
    if name in _lazy_imports:
        return _lazy_imports[name]

    if name == "chat_deepresearcher_agent":
        from .agents import chat_deepresearcher_agent

        _lazy_imports[name] = chat_deepresearcher_agent
        return chat_deepresearcher_agent

    if name == "shallow_research_agent":
        from .agents import shallow_research_agent

        _lazy_imports[name] = shallow_research_agent
        return shallow_research_agent

    if name == "deep_research_agent":
        from .agents import deep_research_agent

        _lazy_imports[name] = deep_research_agent
        return deep_research_agent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
