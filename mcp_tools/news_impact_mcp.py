# news_impact_mcp.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Context, Tool
from news_impact_extractor import extract_impacts_from_json

app = FastMCP("news-impact")


@app.tool()
def process_reuters_json(
    ctx: Context,
    json_path: str,
    out_csv: str | None = None,
    max_items: int | None = None,
) -> str:
    """
    MCP tool: process a Reuters MRN_STORY JSON file into a CSV
    of structured shocks.

    Parameters
    ----------
    json_path: path to input JSON
    out_csv: optional output CSV; if omitted, we write next to the JSON
    max_items: optional cap on number of articles
    """
    json_path = os.path.abspath(json_path)
    if out_csv is None:
        stem = Path(json_path).stem
        out_csv = os.path.join(os.path.dirname(json_path), f"{stem}_shocks.csv")
    out_csv = os.path.abspath(out_csv)

    asyncio.run(extract_impacts_from_json(json_path, out_csv, max_items))
    return out_csv


if __name__ == "__main__":
    # run as MCP server
    app.run()