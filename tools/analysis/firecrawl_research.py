"""Firecrawl-backed research tool: URL scraping and web search with content extraction.

Complements the research stage's built-in web search: where a plain search only
returns titles/snippets, Firecrawl fetches the full rendered page and returns
clean markdown, so data points and quotes can be grounded in complete source
text rather than a two-line snippet.
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

API_BASE_URL = "https://api.firecrawl.dev/v1"


class FirecrawlResearch(BaseTool):
    name = "firecrawl_research"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "research"
    provider = "firecrawl"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:FIRECRAWL_API_KEY"]
    install_instructions = (
        "Set the FIRECRAWL_API_KEY environment variable:\n"
        "  export FIRECRAWL_API_KEY=your_key_here\n"
        "Get a key at https://www.firecrawl.dev"
    )
    fallback = None
    fallback_tools = []

    capabilities = [
        "scrape_url",
        "web_search",
        "markdown_extraction",
    ]
    supports = {
        "javascript_rendering": True,
        "offline": False,
    }
    best_for = [
        "pulling full source text for a data point instead of a search snippet",
        "verifying a claim from a VideoAnalysisBrief against the live page",
        "content landscape scans where snippet text is too thin to judge a gap",
    ]
    not_good_for = [
        "large multi-page site crawls (use Firecrawl's crawl endpoint directly for that)",
        "real-time/streaming content",
    ]

    input_schema = {
        "type": "object",
        "required": ["mode"],
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["scrape", "search"],
                "description": "'scrape' fetches one URL; 'search' runs a web search and can return full page content per result",
            },
            "url": {"type": "string", "description": "Required for mode='scrape'"},
            "query": {"type": "string", "description": "Required for mode='search'"},
            "limit": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Max results for mode='search'",
            },
            "scrape_results": {
                "type": "boolean",
                "default": True,
                "description": "For mode='search': also fetch full markdown content for each result, not just title/snippet",
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
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=10, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["mode", "url", "query", "limit", "scrape_results"]
    side_effects = ["calls Firecrawl API"]
    user_visible_verification = [
        "Spot-check that the returned markdown matches the live page",
        "Confirm cited data points trace back to a specific source_url in the results",
    ]

    def get_status(self) -> ToolStatus:
        if os.environ.get("FIRECRAWL_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Firecrawl bills in credits, not USD directly. Approximate at the
        # Standard-plan rate of ~$0.005/page fetched (1 credit/page scraped;
        # search pages that are also scraped cost one credit each).
        if inputs.get("mode") == "scrape":
            return 0.005
        limit = inputs.get("limit", 5)
        scrape_results = inputs.get("scrape_results", True)
        return round(limit * 0.005, 4) if scrape_results else 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return ToolResult(success=False, error="No Firecrawl API key. " + self.install_instructions)

        mode = inputs.get("mode")
        if mode not in ("scrape", "search"):
            return ToolResult(success=False, error="mode must be 'scrape' or 'search'")

        start = time.time()
        try:
            if mode == "scrape":
                result = self._scrape(inputs, api_key)
            else:
                result = self._search(inputs, api_key)
        except Exception as exc:
            return ToolResult(success=False, error=f"Firecrawl request failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)

        output_path = inputs.get("output_path")
        if result.success and output_path:
            import json

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result.data, indent=2), encoding="utf-8")
            result.artifacts = [str(path)]

        return result

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _scrape(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        url = inputs.get("url")
        if not url:
            return ToolResult(success=False, error="mode='scrape' requires 'url'")

        response = requests.post(
            f"{API_BASE_URL}/scrape",
            headers=self._headers(api_key),
            json={"url": url, "formats": ["markdown"]},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success", True):
            return ToolResult(success=False, error=payload.get("error", "Firecrawl scrape failed"))

        page = payload.get("data", {})
        metadata = page.get("metadata", {})
        return ToolResult(
            success=True,
            data={
                "mode": "scrape",
                "url": url,
                "title": metadata.get("title"),
                "description": metadata.get("description"),
                "markdown": page.get("markdown", ""),
                "source_url": metadata.get("sourceURL", url),
            },
        )

    def _search(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        query = inputs.get("query")
        if not query:
            return ToolResult(success=False, error="mode='search' requires 'query'")

        limit = inputs.get("limit", 5)
        scrape_results = inputs.get("scrape_results", True)

        body: dict[str, Any] = {"query": query, "limit": limit}
        if scrape_results:
            body["scrapeOptions"] = {"formats": ["markdown"]}

        response = requests.post(
            f"{API_BASE_URL}/search",
            headers=self._headers(api_key),
            json=body,
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success", True):
            return ToolResult(success=False, error=payload.get("error", "Firecrawl search failed"))

        results = []
        for item in payload.get("data", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description"),
                "markdown": item.get("markdown", "") if scrape_results else None,
            })

        return ToolResult(
            success=True,
            data={
                "mode": "search",
                "query": query,
                "results": results,
            },
        )
