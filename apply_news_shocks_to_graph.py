
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _parse_list_cell(cell) -> List[str]:
    """
    Robustly parse a list-like value coming from CSV/Parquet.

    Handles:
      * real Python lists
      * stringified lists like "['A', 'B']"
      * comma-separated strings "A, B"
      * NaNs / None -> []
    """
    import ast

    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if cell is None:
        return []
    try:
        if isinstance(cell, float) and math.isnan(cell):
            return []
    except Exception:
        pass
    s = str(cell).strip()
    if not s:
        return []

    # Try literal_eval first
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            v = val.strip()
            return [v] if v else []
    except Exception:
        # Fall back to comma-split
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"[^A-Za-z0-9 ]+", " ", s).upper().strip()


def _norm_tokens(label: str) -> set[str]:
    """Normalize a sector/region label into a set of tokens."""
    if not isinstance(label, str):
        label = str(label)
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", label).upper()
    toks = [
        t
        for t in cleaned.split()
        if t and t not in {"AND", "SECTOR", "SECTORS", "INDUSTRY", "INDUSTRIES", "REGION", "REGIONS"}
    ]
    return set(toks)


def _tokens_fuzzy_match(ts1: set[str], ts2: set[str]) -> bool:
    """
    Fuzzy overlap between token sets.

    True if:
      * exact token intersection, OR
      * substring overlap for tokens length >= 4, OR
      * common prefix of length >= 4.
    """
    # Exact token overlap
    if ts1.intersection(ts2):
        return True

    for a in ts1:
        la = a.lower()
        for b in ts2:
            lb = b.lower()

            # substring
            if len(la) >= 4 and la in lb:
                return True
            if len(lb) >= 4 and lb in la:
                return True

            # common prefix of length >= 4
            m = min(len(la), len(lb))
            for k in range(m, 3, -1):
                if la[:k] == lb[:k]:
                    return True
    return False


def _region_norm(s: str) -> str:
    """Very light normalization for regions."""
    if not isinstance(s, str):
        s = str(s)
    return s.strip().upper()


# ----------------------------------------------------------------------
# Core mapping
# ----------------------------------------------------------------------
def apply_shocks_to_graph(
    graphml_path: str,
    shocks_csv: str,
    firm_master_csv: str,
    out_graphml: str,
    out_nodes_csv: str,
    enable_firm: bool = True,
    enable_sector: bool = True,
    enable_region: bool = True,
) -> None:
    """
    Load a GraphML supply-chain graph and a `news_shocks.csv`, then
    map shocks to nodes via:

      1) firm-name → firm master → node (FACTSET_ENTITY_ID / gvkey)
      2) sector labels → node['naics_sector'] (fuzzy)
      3) region labels → node['loc'] (coarse match)

    Writes:
      * `out_graphml` with node attributes aggregated from shocks
      * `out_nodes_csv` with a node table + aggregated shock columns
    """
    graphml_path = os.path.abspath(graphml_path)
    shocks_csv = os.path.abspath(shocks_csv)
    firm_master_csv = os.path.abspath(firm_master_csv)
    out_graphml = os.path.abspath(out_graphml)
    out_nodes_csv = os.path.abspath(out_nodes_csv)

    print(f"[SHOCKMAP] Loading base graph: {graphml_path}")
    G = nx.read_graphml(graphml_path)
    print(f"[SHOCKMAP] Graph loaded: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    print(f"[SHOCKMAP] Loading shocks from: {shocks_csv}")
    df_shocks = pd.read_csv(shocks_csv)
    print(f"[SHOCKMAP] Shocks: {df_shocks.shape}")

    print(f"[SHOCKMAP] Loading firm master: {firm_master_csv}")
    df_firm = pd.read_csv(firm_master_csv)
    print(f"[SHOCKMAP] Firm master: {df_firm.shape}")

    # Ensure some canonical columns exist
    # EXPECTED in firm master: FACTSET_ENTITY_ID, gvkey, conm, naics_sector, loc
    for col in ["FACTSET_ENTITY_ID", "gvkey", "conm"]:
        if col not in df_firm.columns:
            raise ValueError(f"Firm master missing required column: {col}")

    # Build firm lookup by normalized company name
    df_firm["conm_norm"] = df_firm["conm"].astype(str).map(_norm_name)
    firm_name_map: Dict[str, List[Dict]] = {}
    for _, row in df_firm.iterrows():
        key = row["conm_norm"]
        firm_name_map.setdefault(key, []).append(row.to_dict())

    # Precompute node metadata
    # We assume nodes have some combination of:
    #   - factset_entity_id
    #   - gvkey
    #   - conm
    #   - naics_sector
    #   - loc
    node_meta: Dict[str, Dict] = {}
    for node_id, attrs in G.nodes(data=True):
        meta = dict(attrs)
        meta.setdefault("node_id", node_id)
        # Normalize name field if present
        if "conm" in meta:
            meta["conm_norm"] = _norm_name(meta["conm"])
        else:
            meta["conm_norm"] = ""
        # Extract identifiers
        meta["factset_entity_id"] = str(meta.get("factset_entity_id", "") or "")
        meta["gvkey"] = str(meta.get("gvkey", "") or "")
        meta["naics_sector"] = meta.get("naics_sector", "")
        meta["loc"] = meta.get("loc", "")
        node_meta[node_id] = meta

    # Build quick lookup from entity/gvkey/name to node IDs
    ent_to_nodes: Dict[str, List[str]] = {}
    gvkey_to_nodes: Dict[str, List[str]] = {}
    name_to_nodes: Dict[str, List[str]] = {}

    for nid, meta in node_meta.items():
        ent = meta.get("factset_entity_id", "")
        if ent:
            ent_to_nodes.setdefault(ent, []).append(nid)

        gv = meta.get("gvkey", "")
        if gv:
            gvkey_to_nodes.setdefault(gv, []).append(nid)

        cn = meta.get("conm_norm", "")
        if cn:
            name_to_nodes.setdefault(cn, []).append(nid)

    # Precompute node sector / region tokens
    for meta in node_meta.values():
        sec = meta.get("naics_sector", "")
        meta["sector_tokens"] = _norm_tokens(sec) if sec else set()
        meta["region_norm"] = _region_norm(meta.get("loc", "")) if meta.get("loc") else ""

    # Aggregation containers
    # Per node we collect:
    #   - set of event_ids
    #   - counts of shocks
    #   - sum of magnitudes (if numeric)
    node_events: Dict[str, set] = {nid: set() for nid in G.nodes()}
    node_shock_count: Dict[str, int] = {nid: 0 for nid in G.nodes()}
    node_shock_up: Dict[str, int] = {nid: 0 for nid in G.nodes()}     # shock_sign > 0
    node_shock_down: Dict[str, int] = {nid: 0 for nid in G.nodes()}   # shock_sign < 0
    node_shock_neutral: Dict[str, int] = {nid: 0 for nid in G.nodes()}

    def _add_shock_to_nodes(node_ids: Iterable[str], ev_id: str, sign: str):
        for nid in node_ids:
            node_events[nid].add(ev_id)
            node_shock_count[nid] += 1
            s = (sign or "").lower()
            if s in {"+", "pos", "positive"}:
                node_shock_up[nid] += 1
            elif s in {"-", "neg", "negative"}:
                node_shock_down[nid] += 1
            else:
                node_shock_neutral[nid] += 1

    # ------------------------------------------------------------------
    # Iterate over shocks and map them
    # ------------------------------------------------------------------
    n_rows = len(df_shocks)
    firm_matches = 0
    sector_matches = 0
    region_matches = 0

    for idx, row in df_shocks.iterrows():
        ev_id = str(row.get("event_id", idx))
        # sign may be "+", "-", "neutral", etc.
        sign = str(row.get("shock_sign", "") or "")

        # 1) Firm-level mapping
        mapped_nodes: set[str] = set()

        if enable_firm and "firms" in df_shocks.columns:
            firms = _parse_list_cell(row.get("firms"))
            for fname in firms:
                fnorm = _norm_name(fname)
                if not fnorm:
                    continue

                # Exact normalized name match
                # e.g. "JPMORGAN CHASE & CO" -> firm row(s)
                for frow in firm_name_map.get(fnorm, []):
                    ent_id = str(frow.get("FACTSET_ENTITY_ID", "") or "")
                    gv = str(frow.get("gvkey", "") or "")
                    # Map via entity and gvkey maps
                    if ent_id in ent_to_nodes:
                        mapped_nodes.update(ent_to_nodes[ent_id])
                    if gv in gvkey_to_nodes:
                        mapped_nodes.update(gvkey_to_nodes[gv])

                # Fuzzy fallback on node names (substring / prefix)
                if not mapped_nodes and enable_firm:
                    for node_norm, nids in name_to_nodes.items():
                        # simple fuzzy check
                        la, lb = fnorm.lower(), node_norm.lower()
                        if la in lb or lb in la:
                            mapped_nodes.update(nids)

        if mapped_nodes:
            firm_matches += 1
            _add_shock_to_nodes(mapped_nodes, ev_id, sign)
            continue  # firm-level wins; we don't fall back

        # 2) Sector-level mapping
        if enable_sector and "sectors" in df_shocks.columns:
            secs = _parse_list_cell(row.get("sectors"))
            sec_token_sets = [_norm_tokens(s) for s in secs if str(s).strip()]

            if sec_token_sets:
                target_nodes: set[str] = set()
                for nid, meta in node_meta.items():
                    ntoks = meta["sector_tokens"]
                    if not ntoks:
                        continue
                    for stoks in sec_token_sets:
                        if _tokens_fuzzy_match(ntoks, stoks):
                            target_nodes.add(nid)
                            break
                if target_nodes:
                    sector_matches += 1
                    _add_shock_to_nodes(target_nodes, ev_id, sign)
                    continue

        # 3) Region-level mapping
        if enable_region and "regions" in df_shocks.columns:
            regs = [_region_norm(r) for r in _parse_list_cell(row.get("regions")) if r]
            norm_regs = {r for r in regs if r}
            if norm_regs:
                target_nodes: set[str] = set()
                for nid, meta in node_meta.items():
                    rn = meta.get("region_norm", "")
                    if not rn:
                        continue
                    # coarse check: exact country code match or substring
                    for rr in norm_regs:
                        la, lb = rr.lower(), rn.lower()
                        if la == lb or la in lb or lb in la:
                            target_nodes.add(nid)
                            break
                if target_nodes:
                    region_matches += 1
                    _add_shock_to_nodes(target_nodes, ev_id, sign)
                    continue

    print(f"[SHOCKMAP] Firm-level matches:   {firm_matches}")
    print(f"[SHOCKMAP] Sector-level matches: {sector_matches}")
    print(f"[SHOCKMAP] Region-level matches: {region_matches}")

    # ------------------------------------------------------------------
    # Attach aggregates back to graph + write outputs
    # ------------------------------------------------------------------
    for nid in G.nodes():
        G.nodes[nid]["shock_event_ids"] = ",".join(sorted(node_events[nid])) if node_events[nid] else ""
        G.nodes[nid]["shock_count"] = int(node_shock_count[nid])
        G.nodes[nid]["shock_up"] = int(node_shock_up[nid])
        G.nodes[nid]["shock_down"] = int(node_shock_down[nid])
        G.nodes[nid]["shock_neutral"] = int(node_shock_neutral[nid])

    out_graphml_path = Path(out_graphml)
    out_graphml_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SHOCKMAP] Writing shocked graph to: {out_graphml_path}")
    nx.write_graphml(G, out_graphml_path)

    # Build node-level DataFrame
    node_rows = []
    for nid, attrs in G.nodes(data=True):
        row = dict(attrs)
        row.setdefault("node_id", nid)
        node_rows.append(row)

    df_nodes = pd.DataFrame(node_rows)
    out_nodes_path = Path(out_nodes_csv)
    out_nodes_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SHOCKMAP] Writing shocked nodes table to: {out_nodes_path}")
    df_nodes.to_csv(out_nodes_path, index=False)

    print("[SHOCKMAP] Done.")


def main(
    graphml_path: str,
    shocks_csv: str,
    firm_master_csv: str,
    out_graphml: str,
    out_nodes_csv: str,
    enable_firm: bool = True,
    enable_sector: bool = True,
    enable_region: bool = True,
) -> None:
    apply_shocks_to_graph(
        graphml_path=graphml_path,
        shocks_csv=shocks_csv,
        firm_master_csv=firm_master_csv,
        out_graphml=out_graphml,
        out_nodes_csv=out_nodes_csv,
        enable_firm=enable_firm,
        enable_sector=enable_sector,
        enable_region=enable_region,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply news shocks to a supply-chain graph.")
    parser.add_argument("--graphml", required=True, help="Input GraphML path (base graph).")
    parser.add_argument("--shocks", required=True, help="news_shocks.csv (or subset) path.")
    parser.add_argument("--firms", required=True, help="Firm master CSV path.")
    parser.add_argument("--out-graphml", required=True, help="Output GraphML path (with shocks).")
    parser.add_argument("--out-nodes", required=True, help="Output nodes CSV path.")
    parser.add_argument("--no-firm", action="store_true", help="Disable firm-level mapping.")
    parser.add_argument("--no-sector", action="store_true", help="Disable sector-level mapping.")
    parser.add_argument("--no-region", action="store_true", help="Disable region-level mapping.")

    args = parser.parse_args()
    main(
        graphml_path=args.graphml,
        shocks_csv=args.shocks,
        firm_master_csv=args.firms,
        out_graphml=args.out_graphml,
        out_nodes_csv=args.out_nodes,
        enable_firm=not args.no_firm,
        enable_sector=not args.no_sector,
        enable_region=not args.no_region,
    )
