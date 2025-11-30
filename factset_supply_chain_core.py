"""
factset_supply_chain_core.py

Shared utilities for building FactSet supply-chain graphs and sector-level subgraphs.

You will typically:
1) Set DATA_DIR, ENTITY_META_CSV, and OUT_DIR to match your environment.
2) Call build_full_graph(...) once to construct an in-memory MultiDiGraph with firm names and NAICS sectors.
3) Use list_available_sectors(G), build_sector_subgraph(...), and save_graph_variants(...) from
   your MCP server or Streamlit / Agentics app.

This module does NOT depend on any LLMs or Agentics; it's pure data+graph logic.
"""

from __future__ import annotations

import os
import math
import datetime as dt
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd
import networkx as nx
from pyvis.network import Network



# ----------------------------------------------------------------------
# NAICS → sector helper
# ----------------------------------------------------------------------
def naics_to_sector(naics_val: Any) -> str:
    """Map a NAICS code (int/float/str) to a broad sector name.

    We only use the first two digits of the NAICS 2017 classification.
    If the code is missing or not recognized, return an empty string.
    """
    if naics_val in (None, ""):
        return ""
    try:
        # Cast like '522110', 522110.0, etc. → 52
        code_str = str(int(float(naics_val)))
        two = int(code_str[:2])
    except Exception:
        return ""

    # Mapping by 2‑digit NAICS prefix
    if 11 <= two <= 11:
        return "Agriculture, Forestry, Fishing and Hunting"
    if 21 <= two <= 21:
        return "Mining, Quarrying, and Oil and Gas Extraction"
    if 22 <= two <= 22:
        return "Utilities"
    if 23 <= two <= 23:
        return "Construction"
    if 31 <= two <= 33:
        return "Manufacturing"
    if 42 <= two <= 42:
        return "Wholesale Trade"
    if 44 <= two <= 45:
        return "Retail Trade"
    if 48 <= two <= 49:
        return "Transportation and Warehousing"
    if 51 <= two <= 51:
        return "Information"
    if 52 <= two <= 52:
        return "Finance and Insurance"
    if 53 <= two <= 53:
        return "Real Estate and Rental and Leasing"
    if 54 <= two <= 54:
        return "Professional, Scientific, and Technical Services"
    if 55 <= two <= 55:
        return "Management of Companies and Enterprises"
    if 56 <= two <= 56:
        return "Administrative and Support and Waste Management and Remediation Services"
    if 61 <= two <= 61:
        return "Educational Services"
    if 62 <= two <= 62:
        return "Health Care and Social Assistance"
    if 71 <= two <= 71:
        return "Arts, Entertainment, and Recreation"
    if 72 <= two <= 72:
        return "Accommodation and Food Services"
    if 81 <= two <= 81:
        return "Other Services (except Public Administration)"
    if 92 <= two <= 92:
        return "Public Administration"
    return ""
# === You should customize these paths in your environment ===
DATA_DIR_DEFAULT = "/Users/boxuanli/Desktop/FactSetRevere/"  # folder with ent_scr_*.txt etc.
ENTITY_META_CSV_DEFAULT = "/Users/boxuanli/Code/Agentics/SupplyChain/factset_entityid_to_gvkey_with_company_name.csv"
OUT_DIR_DEFAULT = "/Users/boxuanli/Code/Agentics/SupplyChain/Outputs/"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Robust loader for FactSet TXT (pipe-separated, ugly quoting)
# ----------------------------------------------------------------------
def load_factset_table(path: str, sep: str = "|") -> Optional[pd.DataFrame]:
    """
    Extremely robust loader for FactSet TXT files.

    It does NOT trust pandas' CSV parser; instead, it:
      - Reads raw lines.
      - Splits on the separator.
      - Strips quotes.
      - Pads/truncates each row to the header length.

    Returns
    -------
    pd.DataFrame or None
    """
    if not os.path.exists(path):
        print(f"[ERROR] File does not exist: {path}")
        return None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    if not lines:
        print(f"[WARN] File is empty: {path}")
        return None

    header_raw = lines[0]
    header = [h.strip().strip('"') for h in header_raw.split(sep)]
    print(f"[INFO] Loading {os.path.basename(path)}")
    print(f"       Header columns ({len(header)}): {header}")

    rows: list[list[str]] = []
    for idx, line in enumerate(lines[1:], start=2):
        parts = line.split(sep)
        # Pad / trim to header length
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        elif len(parts) > len(header):
            parts = parts[:len(header)]
        cleaned = [p.strip().strip('"') for p in parts]
        rows.append(cleaned)

    df = pd.DataFrame(rows, columns=header)
    print(f"       Loaded shape: {df.shape}")
    print()
    return df


# ----------------------------------------------------------------------
# 2. Entity metadata: FACTSET_ENTITY_ID -> {name, gvkey, naics, naics_sector, country}
# ----------------------------------------------------------------------
def load_entity_metadata(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load a CSV with at least:
        FACTSET_ENTITY_ID, gvkey, conm, naics, loc, naics_sector

    Returns a dict keyed by FACTSET_ENTITY_ID with clean string attrs.
    """
    if not os.path.exists(path):
        print(f"[WARN] Entity metadata CSV not found: {path}")
        return {}

    df = pd.read_csv(path, dtype={"FACTSET_ENTITY_ID": str}, keep_default_na=False)
    print(f"[INFO] Loaded entity metadata: {path} shape={df.shape}")

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    eid_col = cols.get("factset_entity_id") or cols.get("factset_id") or "FACTSET_ENTITY_ID"
    name_col = cols.get("conm") or cols.get("name") or cols.get("company_name")
    gvkey_col = cols.get("gvkey")
    naics_col = cols.get("naics")
    sector_col = cols.get("naics_sector") or cols.get("sector") or cols.get("sector_name")
    country_col = cols.get("loc") or cols.get("country")

    meta: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        eid_raw = row.get(eid_col)
        if not isinstance(eid_raw, str):
            eid = str(eid_raw) if pd.notna(eid_raw) else ""
        else:
            eid = eid_raw.strip()
        if not eid:
            continue

        name = (str(row.get(name_col)) if name_col else "").strip()
        gvkey_val = row.get(gvkey_col) if gvkey_col else ""
        if pd.isna(gvkey_val):
            gvkey = ""
        else:
            gvkey = str(gvkey_val).strip()
            # Strip trailing .0 if it came from float
            if gvkey.endswith(".0"):
                gvkey = gvkey[:-2]

        naics_val = row.get(naics_col) if naics_col else ""
        # Normalize NAICS to a compact string like '522110'
        if naics_val not in ("", None) and not pd.isna(naics_val):
            try:
                naics = str(int(float(naics_val)))
            except Exception:
                naics = str(naics_val).strip()
        else:
            naics = ""
        # Prefer an explicit sector column if present; otherwise derive from NAICS
        sector_raw = (str(row.get(sector_col)) if sector_col else "").strip()
        if sector_raw:
            sector = sector_raw
        else:
            sector = naics_to_sector(naics) if naics else ""
        country = (str(row.get(country_col)) if country_col else "").strip()

        meta[eid] = {
            "name": name,
            "gvkey": gvkey,
            "naics": naics,
            "naics_sector": sector,
            "country": country,
        }

    print(f"[INFO] Entity metadata entries: {len(meta)}")
    return meta


# ----------------------------------------------------------------------
# 3. Coverage map: ent_scr_coverage.txt
# ----------------------------------------------------------------------
def build_coverage_map(df_cov: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if df_cov is None or df_cov.empty:
        print("[INFO] No coverage data loaded.")
        return {}

    cov_map: Dict[str, Dict[str, Any]] = {}
    for _, row in df_cov.iterrows():
        eid = row.get("FACTSET_ENTITY_ID")
        if not isinstance(eid, str) or not eid:
            continue
        cov_map[eid] = {
            "has_direct_rel": row.get("HAS_DIRECT_REL", "0"),
            "has_reverse_rel": row.get("HAS_REVERSE_REL", "0"),
            "actively_covered": row.get("ACTIVELY_COVERED", "0"),
        }
    print(f"[INFO] Coverage entries: {len(cov_map)}")
    return cov_map


# ----------------------------------------------------------------------
# 4. Relationship keywords: scr_relationships_keyword.txt
# ----------------------------------------------------------------------
def build_relationship_keywords(df_kw: Optional[pd.DataFrame]) -> Dict[str, str]:
    if df_kw is None or df_kw.empty:
        print("[INFO] No keyword data loaded.")
        return {}

    kw_cols = [c for c in df_kw.columns if c.upper().startswith("RELATIONSHIP_KEYWORD")]
    rel2kw: Dict[str, set[str]] = {}
    for _, row in df_kw.iterrows():
        rel_id = row.get("ID")
        if not isinstance(rel_id, str) or not rel_id:
            continue
        kws: list[str] = []
        for c in kw_cols:
            val = row.get(c)
            if isinstance(val, str) and val.strip():
                kws.append(val.strip())
        if kws:
            rel2kw.setdefault(rel_id, set()).update(kws)
    rel2kw_flat = {k: "; ".join(sorted(list(v))) for k, v in rel2kw.items()}
    print(f"[INFO] Built keyword map for {len(rel2kw_flat)} relationships.")
    return rel2kw_flat


# ----------------------------------------------------------------------
# 5. Overlap metrics from scr_relationships_summary.txt
# ----------------------------------------------------------------------
def build_overlap_lookup(df_summary: Optional[pd.DataFrame]) -> Dict[tuple[str, str], Dict[str, Any]]:
    if df_summary is None or df_summary.empty:
        print("[INFO] No summary data for overlaps.")
        return {}

    cols = {c.upper(): c for c in df_summary.columns}
    src_col = cols.get("SOURCE_FACTSET_ENTITY_ID")
    tgt_col = None
    for c in df_summary.columns:
        if "TARGET_FACTSET_ENTITY_ID" in c.upper():
            tgt_col = c
            break

    if not src_col or not tgt_col:
        print("[WARN] Could not find SOURCE/TARGET in summary; columns:", df_summary.columns)
        return {}

    overlap_cols = [c for c in df_summary.columns if "OVERLAP" in c.upper()]
    lookup: Dict[tuple[str, str], Dict[str, Any]] = {}

    for _, row in df_summary.iterrows():
        s = row.get(src_col)
        t = row.get(tgt_col)
        if not s or not t:
            continue
        metrics: Dict[str, Any] = {}
        for oc in overlap_cols:
            val = row.get(oc)
            if val is None or val == "":
                metrics[oc] = ""
            else:
                try:
                    metrics[oc] = float(val)
                except (ValueError, TypeError):
                    metrics[oc] = val
        lookup[(s, t)] = metrics

    print(f"[INFO] Overlap pairs in summary: {len(lookup)}")
    return lookup


# ----------------------------------------------------------------------
# 6. Build the full graph
# ----------------------------------------------------------------------
def build_full_graph(
    data_dir: str = DATA_DIR_DEFAULT,
    entity_meta_csv: str = ENTITY_META_CSV_DEFAULT,
) -> nx.MultiDiGraph:
    """
    Load all FactSet supply-chain relationship tables and construct a MultiDiGraph.

    Node IDs: FACTSET_ENTITY_ID (string)
    Node attributes:
        - name, gvkey, naics, naics_sector, country
        - has_direct_rel, has_reverse_rel, actively_covered

    Edge attributes:
        - edge_type: "supply_chain" or "competitor"
        - revenue_pct (for supply-chain edges)
        - relationship_id (for competitor edges)
        - overlap_weight (for competitor edges, from summary)
        - keywords (for competitor edges, from keyword table)
        - start_date, end_date
        - source_reporting_entity (for supply-chain edges)
    """
    # File paths
    path_rel = os.path.join(data_dir, "ent_scr_relationships.txt")
    path_supply = os.path.join(data_dir, "ent_scr_supply_chain.txt")
    path_summary = os.path.join(data_dir, "scr_relationships_summary.txt")
    path_keywords = os.path.join(data_dir, "scr_relationships_keyword.txt")
    path_coverage = os.path.join(data_dir, "ent_scr_coverage.txt")

    # Load tables
    df_rel = load_factset_table(path_rel)
    df_supply = load_factset_table(path_supply)
    df_summary = load_factset_table(path_summary)
    df_kw = load_factset_table(path_keywords)
    df_cov = load_factset_table(path_coverage)

    # Support structures
    coverage_map = build_coverage_map(df_cov)
    rel_keywords = build_relationship_keywords(df_kw)
    overlap_lookup = build_overlap_lookup(df_summary)
    entity_meta = load_entity_metadata(entity_meta_csv)

    G = nx.MultiDiGraph()

    def ensure_node(node_id: str) -> None:
        if not isinstance(node_id, str) or not node_id:
            return
        if G.has_node(node_id):
            return
        attrs: Dict[str, Any] = {}
        # Coverage flags
        attrs.update(coverage_map.get(node_id, {}))
        # Entity metadata
        attrs.update(entity_meta.get(node_id, {}))
        G.add_node(node_id, **attrs)

    # -- Supply-chain edges: ent_scr_supply_chain --
    def add_supply_chain_edges() -> int:
        if df_supply is None or df_supply.empty:
            print("[WARN] No supply-chain data, skipping edges.")
            return 0
        cols = {c.upper(): c for c in df_supply.columns}
        sup_col = cols.get("SUPPLIER_FACTSET_ENTITY_ID")
        cus_col = cols.get("CUSTOMER_FACTSET_ENTITY_ID")
        rev_col = cols.get("REVENUE_PCT")
        start_col = cols.get("START_DATE")
        end_col = cols.get("END_DATE")
        src_col = cols.get("SOURCE_FACTSET_ENTITY_ID")

        if not sup_col or not cus_col:
            print("[ERROR] SUPPLIER/CUSTOMER columns not found in ent_scr_supply_chain.")
            print("        Columns are:", df_supply.columns)
            return 0

        count = 0
        for _, row in df_supply.iterrows():
            supplier = row.get(sup_col)
            customer = row.get(cus_col)
            if not supplier or not customer:
                continue
            ensure_node(supplier)
            ensure_node(customer)

            rev_raw = row.get(rev_col)
            try:
                rev_pct = float(rev_raw) if rev_raw not in (None, "", "NaN") else ""
            except (ValueError, TypeError):
                rev_pct = ""

            attrs = {
                "edge_type": "supply_chain",
                "revenue_pct": rev_pct,
                "start_date": row.get(start_col, ""),
                "end_date": row.get(end_col, ""),
                "source_reporting_entity": row.get(src_col, ""),
            }
            G.add_edge(supplier, customer, **attrs)
            count += 1

        print(f"[INFO] Supply-chain edges added: {count}")
        return count

    # -- Competitor edges: ent_scr_relationships --
    def add_competitor_edges() -> int:
        if df_rel is None or df_rel.empty:
            print("[WARN] No relationships data, skipping competitor edges.")
            return 0

        cols = {c.upper(): c for c in df_rel.columns}
        reltype_col = cols.get("REL_TYPE")
        src_col = cols.get("SOURCE_FACTSET_ENTITY_ID")
        tgt_col = cols.get("TARGET_FACTSET_ENTITY_ID")
        id_col = cols.get("ID")
        start_col = cols.get("START_DATE")
        end_col = cols.get("END_DATE")

        if not reltype_col or not src_col or not tgt_col:
            print("[ERROR] Missing REL_TYPE/SOURCE/TARGET columns in ent_scr_relationships.")
            print("        Columns:", df_rel.columns)
            return 0

        comp_df = df_rel[df_rel[reltype_col] == "COMPETITOR"].copy()
        print(f"[INFO] Competitor relationships rows: {len(comp_df)}")

        count = 0
        for _, row in comp_df.iterrows():
            s = row.get(src_col)
            t = row.get(tgt_col)
            if not s or not t:
                continue
            ensure_node(s)
            ensure_node(t)

            rel_id = row.get(id_col, "")
            kw_str = rel_keywords.get(rel_id, "")

            metrics = overlap_lookup.get((s, t), {})
            # Try generic OVERLAP column first; fallback to any numeric key
            weight = metrics.get("OVERLAP", "")
            if weight == "":
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        weight = v
                        break

            attrs = {
                "edge_type": "competitor",
                "relationship_id": rel_id,
                "keywords": kw_str,
                "start_date": row.get(start_col, ""),
                "end_date": row.get(end_col, ""),
                "overlap_weight": weight,
            }
            # Add both directions
            G.add_edge(s, t, **attrs)
            G.add_edge(t, s, **attrs)
            count += 2

        print(f"[INFO] Competitor edges added (both directions): {count}")
        return count

    sc_edges = add_supply_chain_edges()
    comp_edges = add_competitor_edges()

    print("\n========== FULL GRAPH SUMMARY ==========")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()} "
          f"(supply-chain ~{sc_edges}, competitor ~{comp_edges})")
    print("========================================\n")

    return G

import os
import networkx as nx

def load_or_build_full_graph(
    graphml_path: str,
    data_dir: str = DATA_DIR_DEFAULT,
    entity_meta_csv: str = ENTITY_META_CSV_DEFAULT,
):
    """
    Robust loader for the full FactSet supply-chain graph.

    - If graphml_path exists: load and return that graph.
    - Else: build from raw FactSet text files + metadata, save to graphml_path, return it.
    """
    # Try loading existing GraphML
    if graphml_path and os.path.exists(graphml_path):
        print(f"[INIT] Loading full graph from GraphML: {graphml_path}")
        G = nx.read_graphml(graphml_path)
        print(f"[INIT] Loaded graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        return G

    # Otherwise build from raw tables
    print(f"[INIT] GraphML not found at {graphml_path}. Building from raw FactSet tables...")
    G = build_full_graph(data_dir=data_dir, entity_meta_csv=entity_meta_csv)

    # Save for future runs
    if graphml_path:
        os.makedirs(os.path.dirname(graphml_path), exist_ok=True)
        nx.write_graphml(G, graphml_path)
        print(f"[INIT] Saved full graph to GraphML: {graphml_path}")

    return G

# ----------------------------------------------------------------------
# 7. Sector list and sector/date subgraphs
# ----------------------------------------------------------------------
def list_available_sectors(G: nx.MultiDiGraph) -> List[str]:
    sectors: set[str] = set()
    for _, data in G.nodes(data=True):
        sec = (data.get("naics_sector") or "").strip()
        if sec:
            sectors.add(sec)
    return sorted(sectors)


def _parse_date_safe(s: Any) -> Optional[dt.date]:
    if s is None:
        return None
    if isinstance(s, dt.date):
        return s
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return None
    # Handle typical FactSet YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
    try:
        return dt.datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def build_sector_subgraph(
    G: nx.MultiDiGraph,
    sectors: Optional[Iterable[str]] = None,
    as_of_date: Optional[str] = None,
) -> nx.MultiDiGraph:
    """
    Build a subgraph filtered by NAICS sector(s) and an optional as-of date.

    Parameters
    ----------
    G : MultiDiGraph
        Full FactSet graph.
    sectors : iterable of str or None
        Sector names (must match node attr 'naics_sector'). If None or empty,
        all sectors are included.
    as_of_date : str or None
        If provided, filter edges active at that date (YYYY-MM-DD).
    """
    sectors = list(sectors or [])
    sectors_clean = [s.strip() for s in sectors if str(s).strip()]

    # Determine allowed node set
    if not sectors_clean:
        allowed_nodes = set(G.nodes())
    else:
        sector_set = set(sectors_clean)
        allowed_nodes = {
            n for n, data in G.nodes(data=True)
            if (data.get("naics_sector") or "").strip() in sector_set
        }

    if not allowed_nodes:
        print("[WARN] No nodes found for selected sectors; returning empty graph.")
        return nx.MultiDiGraph()

    # Parse as_of_date
    as_of: Optional[dt.date] = None
    if as_of_date:
        as_of = _parse_date_safe(as_of_date)
        if as_of is None:
            print(f"[WARN] Could not parse as_of_date='{as_of_date}', ignoring date filter.")

    edges_to_keep: list[tuple[str, str, int]] = []
    for u, v, k, data in G.edges(keys=True, data=True):
        # Sector filter: keep edges only if BOTH endpoints in allowed_nodes
        if u not in allowed_nodes and v not in allowed_nodes:
            continue

        if as_of is not None:
            sd = _parse_date_safe(data.get("start_date"))
            ed = _parse_date_safe(data.get("end_date"))
            if sd and as_of < sd:
                continue
            if ed and as_of > ed:
                continue

        edges_to_keep.append((u, v, k))

    H = G.edge_subgraph(edges_to_keep).copy()

    # Ensure all allowed nodes are present, even if isolated
    for n in allowed_nodes:
        if not H.has_node(n) and G.has_node(n):
            H.add_node(n, **G.nodes[n])

    print("\n========== SUBGRAPH SUMMARY ==========")
    print(f"Sectors: {', '.join(sectors_clean) if sectors_clean else '(ALL)'}")
    print(f"As-of date: {as_of_date or '(no filter)'}")
    print(f"Nodes: {H.number_of_nodes()}")
    print(f"Edges: {H.number_of_edges()}")
    print("======================================\n")

    return H


# ----------------------------------------------------------------------
# 8. Save GraphML / GEXF / edges CSV / PyVis HTML
# ----------------------------------------------------------------------
def _slugify(value: str) -> str:
    v = "".join(ch if ch.isalnum() else "_" for ch in value)
    while "__" in v:
        v = v.replace("__", "_")
    return v.strip("_") or "all"


def save_graph_variants(
    H: nx.MultiDiGraph,
    sectors: Iterable[str],
    as_of_date: Optional[str],
    out_dir: str = OUT_DIR_DEFAULT,
    graph_id_prefix: str = "factset_sc",
) -> Dict[str, Any]:
    """
    Persist a subgraph to multiple formats and append an entry to graph_index.csv.

    Returns a metadata dict with:
        graph_id, sectors, as_of_date, n_nodes, n_edges,
        graphml_path, gexf_path, csv_path, html_path
    """
    _ensure_dir(out_dir)

    sectors_clean = [s.strip() for s in sectors if str(s).strip()]
    if sectors_clean:
        sector_slug = "_".join(sorted(_slugify(s) for s in sectors_clean))
    else:
        sector_slug = "ALL"

    date_slug = _slugify(as_of_date) if as_of_date else "all_dates"
    graph_id = f"{graph_id_prefix}_{sector_slug}_{date_slug}"

    graphml_path = os.path.join(out_dir, f"{graph_id}.graphml")
    gexf_path = os.path.join(out_dir, f"{graph_id}.gexf")
    csv_path = os.path.join(out_dir, f"{graph_id}_edges.csv")
    html_path = os.path.join(out_dir, f"{graph_id}.html")
    index_csv = os.path.join(out_dir, "graph_index.csv")

    # --- NetworkX formats ---
    nx.write_graphml(H, graphml_path)
    nx.write_gexf(H, gexf_path)

    # --- Edges CSV (with node metadata) ---
    rows: list[Dict[str, Any]] = []
    for u, v, data in H.edges(data=True):
        src = H.nodes[u]
        tgt = H.nodes[v]
        rows.append({
            "source_entity": u,
            "target_entity": v,
            "source_name": src.get("name", ""),
            "target_name": tgt.get("name", ""),
            "source_gvkey": src.get("gvkey", ""),
            "target_gvkey": tgt.get("gvkey", ""),
            "source_naics": src.get("naics", ""),
            "target_naics": tgt.get("naics", ""),
            "source_naics_sector": src.get("naics_sector", ""),
            "target_naics_sector": tgt.get("naics_sector", ""),
            "edge_type": data.get("edge_type", ""),
            "revenue_pct": data.get("revenue_pct", ""),
            "overlap_weight": data.get("overlap_weight", ""),
            "keywords": data.get("keywords", ""),
            "start_date": data.get("start_date", ""),
            "end_date": data.get("end_date", ""),
        })
    df_edges = pd.DataFrame(rows)
    df_edges.to_csv(csv_path, index=False)

    # --- PyVis interactive HTML (for Streamlit / browser) ---
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # Add nodes
    for n, data in H.nodes(data=True):
        name = data.get("name") or n
        sector = data.get("naics_sector", "")
        title_lines = [
            f"<b>{name}</b>",
            f"Entity ID: {n}",
        ]
        if sector:
            title_lines.append(f"Sector: {sector}")
        if data.get("naics"):
            title_lines.append(f"NAICS: {data.get('naics')}")
        if data.get("country"):
            title_lines.append(f"Country: {data.get('country')}")
        tooltip = "<br>".join(title_lines)
        group = sector or "Unknown"
        net.add_node(n, label=name, title=tooltip, group=group)

    # Add edges
    for u, v, data in H.edges(data=True):
        e_type = data.get("edge_type", "")
        # Use different dashes for supply-chain vs competitor
        dashed = e_type == "competitor"
        title_parts = [e_type or "edge"]
        if data.get("revenue_pct") not in ("", None):
            title_parts.append(f"Rev%: {data.get('revenue_pct')}")
        if data.get("overlap_weight") not in ("", None):
            title_parts.append(f"Overlap: {data.get('overlap_weight')}")
        if data.get("keywords"):
            title_parts.append(f"KW: {data.get('keywords')}")
        tooltip = " | ".join(str(x) for x in title_parts if x)

        net.add_edge(u, v, title=tooltip, physics=True, smooth=False, dashes=dashed)

    # IMPORTANT: write_html does NOT try to open a browser (unlike show())
    net.write_html(html_path)

    meta = {
        "graph_id": graph_id,
        "sectors": sectors_clean,
        "as_of_date": as_of_date or "",
        "n_nodes": int(H.number_of_nodes()),
        "n_edges": int(H.number_of_edges()),
        "graphml_path": graphml_path,
        "gexf_path": gexf_path,
        "csv_path": csv_path,
        "html_path": html_path,
    }

    # --- Append to index CSV ---
    idx_df = pd.DataFrame([meta])
    if os.path.exists(index_csv):
        idx_df.to_csv(index_csv, mode="a", header=False, index=False)
    else:
        idx_df.to_csv(index_csv, mode="w", header=True, index=False)

    print(f"[OK] Saved graph '{graph_id}':")
    print(f"     Nodes={meta['n_nodes']}  Edges={meta['n_edges']}")
    print(f"     GraphML: {graphml_path}")
    print(f"     GEXF:    {gexf_path}")
    print(f"     CSV:     {csv_path}")
    print(f"     HTML:    {html_path}")
    print(f"     Index:   {index_csv}")

    return meta
