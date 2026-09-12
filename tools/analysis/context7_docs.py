"""Context7-backed documentation lookup: fresh, version-correct library docs.

Static Layer 3 skills (`.agents/skills/remotion-best-practices`, `gsap-*`, …) cover the
libraries OpenMontage uses every day, but atelier-mode compositions routinely pull in a
one-off npm package (a GSAP plugin, a charting lib, an animation utility) that no skill
file documents. Context7 resolves a package name to a canonical library id and returns
current, version-pinned documentation snippets pulled from the library's own source —
so the agent can verify an API before writing code against it instead of guessing from
training-data memory of a possibly-stale version.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

API_BASE_URL = "https://context7.com/api/v1"


class Context7Docs(BaseTool):
    name = "context7_docs"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "documentation"
    provider = "context7"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    # Works unauthenticated at low rate limits; a free key raises the ceiling.
    dependencies = []
    install_instructions = (
        "Works without a key at low rate limits. For higher limits, set the "
        "CONTEXT7_API_KEY environment variable:\n"
        "  export CONTEXT7_API_KEY=your_key_here\n"
        "Get a free key at https://context7.com/dashboard"
    )
    fallback = None
    fallback_tools = []

    capabilities = [
        "resolve_library_id",
        "fetch_library_docs",
    ]
    supports = {
        "javascript_rendering": False,
        "offline": False,
    }
    best_for = [
        "verifying a third-party npm package's current API before writing bespoke "
        "(atelier-mode) Remotion or HyperFrames code that imports it",
        "checking a library version-specific signature the static Layer 3 skills "
        "don't cover",
        "grounding generated code in the library's own docs instead of "
        "training-data memory of a possibly-stale API",
    ]
    not_good_for = [
        "libraries already covered by a static Layer 3 skill (read the skill first — "
        "it has OpenMontage-specific usage guidance Context7 won't)",
        "general web research or news (use firecrawl_research)",
    ]

    input_schema = {
        "type": "object",
        "required": ["mode"],
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["resolve", "docs"],
                "description": (
                    "'resolve' finds the Context7 library id for a package name; "
                    "'docs' fetches documentation for a known library id"
                ),
            },
            "library_name": {
                "type": "string",
                "description": "Required for mode='resolve', e.g. 'gsap' or 'zod'",
            },
            "library_id": {
                "type": "string",
                "description": (
                    "Required for mode='docs'. A Context7 id like '/vercel/next.js' "
                    "(obtain via mode='resolve' first)"
                ),
            },
            "topic": {
                "type": "string",
                "description": "Optional: narrow the docs to a topic, e.g. 'hooks' or 'timeline'",
            },
            "tokens": {
                "type": "integer",
                "default": 5000,
                "minimum": 500,
                "maximum": 20000,
                "description": "Approximate max tokens of documentation to return",
            },
            "output_path": {
                "type": "string",
                "description": "Optional path to save the raw JSON response",
            },
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string"},
            "results": {"type": "array"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=64, vram_mb=0, disk_mb=5, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["mode", "library_name", "library_id", "topic", "tokens"]
    side_effects = ["calls Context7 API"]
    user_visible_verification = [
        "Confirm the resolved library_id matches the intended package (not a same-named fork)",
        "Spot-check that a cited API signature matches the version actually installed in package.json",
    ]

    def get_status(self) -> ToolStatus:
        # Public endpoint works unauthenticated at low rate limits; a key only
        # raises the ceiling, so the tool is available either way.
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        mode = inputs.get("mode")
        if mode not in ("resolve", "docs"):
            return ToolResult(success=False, error="mode must be 'resolve' or 'docs'")

        start = time.time()
        try:
            if mode == "resolve":
                result = self._resolve(inputs)
            else:
                result = self._docs(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Context7 request failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = 0.0

        output_path = inputs.get("output_path")
        if result.success and output_path:
            import json

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result.data, indent=2), encoding="utf-8")
            result.artifacts = [str(path)]

        return result

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        api_key = os.environ.get("CONTEXT7_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _resolve(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        library_name = inputs.get("library_name")
        if not library_name:
            return ToolResult(success=False, error="mode='resolve' requires 'library_name'")

        response = requests.get(
            f"{API_BASE_URL}/search",
            headers=self._headers(),
            params={"query": library_name},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        candidates = []
        for item in payload.get("results", []):
            candidates.append({
                "library_id": item.get("id"),
                "title": item.get("title"),
                "description": item.get("description"),
                "trust_score": item.get("trustScore"),
                "snippet_count": item.get("totalSnippets"),
            })

        return ToolResult(
            success=True,
            data={
                "mode": "resolve",
                "library_name": library_name,
                "results": candidates,
            },
        )

    def _docs(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        library_id = inputs.get("library_id")
        if not library_id:
            return ToolResult(success=False, error="mode='docs' requires 'library_id'")

        params: dict[str, Any] = {
            "type": "txt",
            "tokens": inputs.get("tokens", 5000),
        }
        topic = inputs.get("topic")
        if topic:
            params["topic"] = topic

        response = requests.get(
            f"{API_BASE_URL}{library_id}",
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        return ToolResult(
            success=True,
            data={
                "mode": "docs",
                "library_id": library_id,
                "topic": topic,
                "documentation": response.text,
            },
        )
