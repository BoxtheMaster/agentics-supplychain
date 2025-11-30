"""
factset_supply_chain_mcp.py

MCP server exposing tools to:
  - list available NAICS sectors in the FactSet supply-chain graph
  - build sector/date filtered subgraphs and persist them as GraphML/GEXF/CSV/HTML

Run with e.g.:

    uv run factset_supply_chain_mcp.py

and configure your MCP client / Agentics environment to talk to it via stdio.
"""

from __future__ import annotations

import os
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import networkx as nx  # type: ignore

from factset_supply_chain_core import (
    DATA_DIR_DEFAULT,
    ENTITY_META_CSV_DEFAULT,
    OUT_DIR_DEFAULT,
    build_full_graph,
    list_available_sectors,
    build_sector_subgraph,
    save_graph_variants,
)


# ----------------------------------------------------------------------
# Initialization: load full graph once
# ----------------------------------------------------------------------
DATA_DIR = os.environ.get("FACTSET_DATA_DIR", DATA_DIR_DEFAULT)
ENTITY_META_CSV = os.environ.get("FACTSET_ENTITY_META_CSV", ENTITY_META_CSV_DEFAULT)
OUT_DIR = os.environ.get("FACTSET_GRAPH_OUT", OUT_DIR_DEFAULT)

import os
import networkx as nx
from factset_supply_chain_core import (
    DATA_DIR_DEFAULT,
    ENTITY_META_CSV_DEFAULT,
    OUT_DIR_DEFAULT,
    load_or_build_full_graph,
    list_available_sectors,
    build_sector_subgraph,
    save_graph_variants,
)

DATA_DIR = os.environ.get("FACTSET_DATA_DIR", DATA_DIR_DEFAULT)
ENTITY_META_CSV = os.environ.get("FACTSET_ENTITY_META_CSV", ENTITY_META_CSV_DEFAULT)
OUT_DIR = os.environ.get("FACTSET_GRAPH_OUT", OUT_DIR_DEFAULT)

GRAPHML_PATH = os.environ.get(
    "FACTSET_FULL_GRAPHML",
    os.path.join(OUT_DIR, "supply_chain_ALL.graphml"),
)

print("[INIT] Preparing full FactSet supply-chain graph...")
G_FULL: nx.MultiDiGraph = load_or_build_full_graph(
    graphml_path=GRAPHML_PATH,
    data_dir=DATA_DIR,
    entity_meta_csv=ENTITY_META_CSV,
)
print("[INIT] Full graph ready.\n")



# ----------------------------------------------------------------------
# MCP server & tool definitions
# ----------------------------------------------------------------------
mcp = FastMCP("factset_supply_chain")


class SupplyChainGraphResult(BaseModel):
    graph_id: str
    sectors: List[str]
    as_of_date: str
    n_nodes: int
    n_edges: int
    graphml_path: str
    gexf_path: str
    csv_path: str
    html_path: str


@mcp.tool()
def list_sectors() -> List[str]:
    """
    Return the sorted list of NAICS sectors that appear in the current graph.
    """
    return list_available_sectors(G_FULL)


@mcp.tool()
def build_supply_chain_graph(
    sectors: Optional[List[str]] = None,
    as_of_date: Optional[str] = None,
    graph_id_prefix: str = "factset_sc",
) -> SupplyChainGraphResult:
    """
    Build a sector/date-filtered subgraph and save GraphML/GEXF/CSV/HTML.

    Parameters
    ----------
    sectors : list[str] or None
        Sector names (must match node attr 'naics_sector'). If None or empty,
        all sectors are included.
    as_of_date : str or None
        If provided, filter edges active at this date (YYYY-MM-DD).
    graph_id_prefix : str
        Prefix for the saved graph filenames.

    Returns
    -------
    SupplyChainGraphResult
        Paths and metadata for the saved graph.
    """
    sub = build_sector_subgraph(G_FULL, sectors or [], as_of_date)
    meta = save_graph_variants(
        sub,
        sectors=sectors or [],
        as_of_date=as_of_date,
        out_dir=OUT_DIR,
        graph_id_prefix=graph_id_prefix,
    )
    return SupplyChainGraphResult(**meta)


if __name__ == "__main__":
    # FastMCP will expose this server over stdio by default
    mcp.run()
