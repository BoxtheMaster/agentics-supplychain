from __future__ import annotations

import asyncio
import ast
import os
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
import re
import streamlit as st

import apply_news_shocks_to_graph as shockmap


# Optional: for inline HTML rendering of PyVis graphs
try:
    import streamlit.components.v1 as components

    HAS_COMPONENTS = True
except Exception:
    HAS_COMPONENTS = False

# ----------------------------------------------------------------------
# Try to import the FactSet supply-chain core
# ----------------------------------------------------------------------
try:
    from factset_supply_chain_core import (
        load_or_build_full_graph,
        list_available_sectors,
        build_sector_subgraph,
        save_graph_variants,
    )

    HAS_FACTSET_CORE = True
    CORE_IMPORT_ERROR = None
except Exception as e:
    HAS_FACTSET_CORE = False
    CORE_IMPORT_ERROR = e

# ----------------------------------------------------------------------
# Try to import the news impact extractor
# ----------------------------------------------------------------------
try:
    import news_impact_extractor as nie

    HAS_NEWS_EXTRACTOR = True
    NEWS_EXTRACTOR_ERROR = None
except Exception as e:
    HAS_NEWS_EXTRACTOR = False
    NEWS_EXTRACTOR_ERROR = e

# ----------------------------------------------------------------------
# Local defaults (paths & model)
#   Adapt these if your environment changes.
# ----------------------------------------------------------------------
FACTSET_DATA_DIR = Path("/Users/boxuanli/Desktop/FactSetRevere")
ENTITY_META_CSV = Path(
    "/Users/boxuanli/Code/Agentics/SupplyChain/"
    "factset_entityid_to_gvkey_with_company_name.csv"
)
DEFAULT_GRAPH_OUT_DIR = Path("/Users/boxuanli/Code/Agentics/SupplyChain/Outputs")

DEFAULT_NEWS_BASE_DIR = Path("/Users/boxuanli/Desktop/2008")
DEFAULT_NEWS_OUT_DIR = Path("/Users/boxuanli/Code/Agentics/SupplyChain/news_output")

# Hard-coded NAICS sector names (used for selection UI).
# These should be consistent with how you labelled `naics_sector` in the graph.
NAICS_SECTORS: List[str] = [
    "Agriculture, Forestry, Fishing and Hunting",
    "Mining, Quarrying, and Oil and Gas Extraction",
    "Utilities",
    "Construction",
    "Manufacturing",
    "Wholesale Trade",
    "Retail Trade",
    "Transportation and Warehousing",
    "Information",
    "Finance and Insurance",
    "Real Estate and Rental and Leasing",
    "Professional, Scientific, and Technical Services",
    "Management of Companies and Enterprises",
    "Administrative and Support and Waste Management and Remediation Services",
    "Educational Services",
    "Health Care and Social Assistance",
    "Arts, Entertainment, and Recreation",
    "Accommodation and Food Services",
    "Other Services (except Public Administration)",
    "Public Administration",
]
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


# ======================================================================
# Shared helpers
# ======================================================================
@st.cache_resource(show_spinner=True)
def get_full_graph(graphml_path: str):
    """Load (or build) the full FactSet graph from `graphml_path`.

    We *only* return the graph here. The sector list for the UI comes from
    the static NAICS_SECTORS constant instead of dynamically inspecting
    the graph, to avoid any issues inside list_available_sectors.
    """
    if not HAS_FACTSET_CORE:
        raise RuntimeError(
            f"factset_supply_chain_core is not importable: {CORE_IMPORT_ERROR}"
        )

    gpath = Path(graphml_path)
    gpath.parent.mkdir(parents=True, exist_ok=True)

    G_full = load_or_build_full_graph(
        graphml_path=gpath,
        data_dir=str(FACTSET_DATA_DIR),
        entity_meta_csv=str(ENTITY_META_CSV),
    )
    return G_full


def slugify_sector_list(sectors: List[str]) -> str:
    """Turn a list of sector names into a filesystem-friendly slug."""
    if not sectors:
        return "ALL"
    cleaned: List[str] = []
    for s in sectors:
        s2 = s.replace("&", "and")
        s2 = "".join(ch if ch.isalnum() else "_" for ch in s2)
        s2 = "_".join(filter(None, s2.split("_")))
        cleaned.append(s2)
    return "_".join(cleaned)


def asof_suffix(as_of: Optional[str]) -> str:
    """YYYY-MM-DD → YYYY_MM_DD; None → all_dates."""
    return (as_of or "all_dates").replace("-", "_")


# ======================================================================
# TAB 1: Supply-chain Graph Builder & Visualizer (Stage 1)
# ======================================================================
def tab_graph_builder(graph_out_dir: Path):
    st.header("1. Supply-Chain Graph Builder")

    if not HAS_FACTSET_CORE:
        st.error(
            "factset_supply_chain_core could not be imported.\n\n"
            f"Python error: {CORE_IMPORT_ERROR}\n\n"
            "Make sure it is installed and on PYTHONPATH."
        )
        return

    st.markdown(
        """
        This tab lets you build **FactSet supply-chain graphs** by sector and date:

        1. Load or build the **full** FactSet graph (from TXT files + metadata).
        2. Choose one or more **sectors** and an optional **as-of date**.
        3. Build the corresponding **subgraph**.
        4. Save GraphML / GEXF / edges CSV / PyVis HTML and view the PyVis graph inline.
        """
    )

    # Output folder (for full graph + sector subgraphs)
    out_dir_str = st.text_input(
        "Graph outputs folder",
        value=str(graph_out_dir),
        help="Full graph, sector subgraphs, and graph_index.csv will live here.",
    )
    out_dir = Path(out_dir_str).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_graphml_path = out_dir / "full_graph.graphml"

    # Build or load the full graph once. Sectors for the UI come from NAICS_SECTORS.
    with st.spinner(f"Loading or building full graph at {full_graphml_path} ..."):
        G_full = get_full_graph(str(full_graphml_path))

    st.success(
        f"Full graph ready: **{G_full.number_of_nodes():,}** nodes, "
        f"**{G_full.number_of_edges():,}** edges."
    )

    # ---------------- Controls ----------------
    st.markdown("### Sector and date selection")

    col_sel, col_opts = st.columns([2, 1])

    with col_sel:
        # Dynamically list sectors from the graph so we stay consistent
        try:
            sectors_available = list_available_sectors(G_full)
            if not sectors_available:
                sectors_available = NAICS_SECTORS
        except Exception as e:
            st.warning(
                f"Could not list sectors from graph; using default NAICS sectors. Details: {e}"
            )
            sectors_available = NAICS_SECTORS

        selected_sectors = st.multiselect(
            "NAICS sectors (multiple allowed)",
            options=sectors_available,
            default=[],
            help="Leave empty and/or tick the box below to use ALL sectors.",
        )

        use_all = st.checkbox(
            "Ignore sector filter (use ALL sectors)",
            value=not selected_sectors,
        )

        filter_by_date = st.checkbox(
            "Filter by as-of date", value=False, help="If off, use all dates."
        )
        as_of_str: Optional[str] = None
        if filter_by_date:
            d = st.date_input(
                "As-of date",
                value=date(2020, 1, 1),
                format="YYYY-MM-DD",
            )
            as_of_str = d.isoformat()

    with col_opts:
        graph_id_prefix = st.text_input(
            "Graph ID prefix",
            value="factset_sc",
            help="Prefix used in saved filenames.",
        )
        st.markdown(
            f"""
            **Effective output directory**

            ```text
            {out_dir}
            ```
            """
        )
        run_button = st.button("Generate sector graph", type="primary")

    if not run_button:
        return

    # Validate sector selection
    if use_all:
        sector_filter: List[str] = []
    else:
        sector_filter = selected_sectors

    if not sector_filter and not use_all:
        st.warning(
            "You turned OFF 'ignore sector filter' but did not select any sectors.\n\n"
            "Either choose sectors in the multiselect or tick 'Ignore sector filter'."
        )
        return

    # ---------------- Build and save subgraph ----------------
    with st.spinner("Building sector subgraph and saving outputs..."):
        G_sub = build_sector_subgraph(G_full, sector_filter, as_of_str)

        meta = save_graph_variants(
            G_sub,
            sectors=sector_filter,
            as_of_date=as_of_str,
            out_dir=str(out_dir),
            graph_id_prefix=graph_id_prefix,
        )

    # Normalize meta → dict
    if hasattr(meta, "model_dump"):
        meta_dict = meta.model_dump()
    elif hasattr(meta, "dict"):
        meta_dict = meta.dict()
    elif isinstance(meta, dict):
        meta_dict = meta
    else:
        meta_dict = dict(meta)

    graph_id = meta_dict.get("graph_id") or (
        f"{graph_id_prefix}_{slugify_sector_list(sector_filter)}_"
        f"{asof_suffix(as_of_str)}"
    )
    n_nodes = meta_dict.get("n_nodes", G_sub.number_of_nodes())
    n_edges = meta_dict.get("n_edges", G_sub.number_of_edges())
    graphml_out = meta_dict.get("graphml_path")
    gexf_out = meta_dict.get("gexf_path")
    csv_out = meta_dict.get("csv_path")
    html_out = meta_dict.get("html_path")

    st.success(
        f"Graph **{graph_id}** generated: **{n_nodes:,}** nodes, **{n_edges:,}** edges."
    )

    with st.expander("Saved files", expanded=True):
        if graphml_out:
            st.write(f"**GraphML:** `{graphml_out}`")
        if gexf_out:
            st.write(f"**GEXF:** `{gexf_out}`")
        if csv_out:
            st.write(f"**Edges CSV:** `{csv_out}`")
        if html_out:
            st.write(f"**PyVis HTML:** `{html_out}`")

    # Inline PyVis visualization
    if html_out and HAS_COMPONENTS and os.path.exists(html_out):
        try:
            with open(html_out, "r", encoding="utf-8") as f:
                html_str = f.read()
            st.markdown("### Interactive graph view")
            components.html(html_str, height=800, scrolling=True)
        except Exception as e:
            st.warning(f"Could not render PyVis HTML: {e}")
    else:
        st.info(
            "PyVis HTML not found or Streamlit components unavailable.\n\n"
            "You can still open the saved `.html` file directly in a browser."
        )


# ======================================================================
# TAB 2: News Processor (Reuters JSON → news_shocks CSV)
# ======================================================================
def find_json_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("*.json"))


def tab_news_processor(news_base_dir: Path, news_output_dir: Path):
    st.header("2. News Impact Extraction")

    if not HAS_NEWS_EXTRACTOR:
        st.error(
            "Module `news_impact_extractor` could not be imported.\n\n"
            f"Python error: {NEWS_EXTRACTOR_ERROR}\n\n"
            "Make sure it is on PYTHONPATH and up to date."
        )
        return

    st.markdown(
        """
        This tab runs the **Agentics-based news impact extractor** on Reuters JSON:

        1. Pick a JSON file (e.g. a month from 2008).
        2. Choose the OpenAI model and a cap on number of items.
        3. Run the extractor to produce `news_shocks.csv` plus an optional debug JSONL.
        """
    )

    # Base directory for Reuters JSON
    base_dir_str = st.text_input(
        "Base folder for Reuters JSON files",
        value=str(news_base_dir),
    )
    base_dir = Path(base_dir_str).expanduser()

    json_files = find_json_files(base_dir)
    if not json_files:
        st.warning(f"No `.json` files found under `{base_dir}`.")
        return

    json_labels = [str(p.relative_to(base_dir)) for p in json_files]
    idx = st.selectbox(
        "Select a JSON file",
        options=list(range(len(json_files))),
        format_func=lambda i: json_labels[i],
    )
    selected_path = json_files[idx]

    # Output directory + file names
    out_dir_str = st.text_input(
        "News output folder",
        value=str(news_output_dir),
    )
    out_dir = Path(out_dir_str).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    default_out_csv = out_dir / f"{selected_path.stem}_shocks.csv"
    default_debug_jsonl = out_dir / f"{selected_path.stem}_debug.jsonl"

    col1, col2 = st.columns(2)
    with col1:
        out_csv_str = st.text_input(
            "Output CSV (news_shocks)",
            value=str(default_out_csv),
        )
        model_name = st.text_input(
            "OpenAI model name",
            value=DEFAULT_OPENAI_MODEL,
            help="Passed to the extractor via `nie.DEFAULT_OPENAI_MODEL`.",
        )
    with col2:
        debug_jsonl_str = st.text_input(
            "Debug JSONL (for failed / trivial rows)",
            value=str(default_debug_jsonl),
        )
        max_items = st.number_input(
            "Max items to process (0 = all)",
            min_value=0,
            value=0,
            step=10,
        )

    run_btn = st.button("Run extraction", type="primary")

    if not run_btn:
        return

    out_csv = Path(out_csv_str).expanduser()
    debug_jsonl = Path(debug_jsonl_str).expanduser()
    debug_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Wire model into the extractor if it exposes DEFAULT_OPENAI_MODEL
    if hasattr(nie, "DEFAULT_OPENAI_MODEL"):
        setattr(nie, "DEFAULT_OPENAI_MODEL", model_name)

    max_items_arg = int(max_items) if max_items > 0 else None

    with st.spinner("Running Agentics transduction over news items..."):
        asyncio.run(
            nie.extract_impacts_from_json(
                json_path=str(selected_path),
                out_csv=str(out_csv),
                max_items=max_items_arg,
                debug_failed_jsonl=str(debug_jsonl),
            )
        )

    st.success(f"Extraction finished. CSV written to `{out_csv}`.")

    try:
        df = pd.read_csv(out_csv)
        st.write(f"**Result shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(30))
    except Exception as e:
        st.error(f"Could not read output CSV: {e}")


# ======================================================================
# TAB 3: Shock Overlay on Graph
# ======================================================================

def tab_shock_overlay(graph_out_dir: Path, news_output_dir: Path):
    """
    Tab 3: take an existing sector graph from graph_index.csv,
    map news shocks (from news_shocks.csv) onto its nodes using
    apply_news_shocks_to_graph, and write:

      * *_with_shocks.graphml
      * *_with_shocks_nodes.csv

    We also preview the shocked nodes table in the UI.
    """
    st.header("3. Shock Overlay on Supply-Chain Graph")

    index_path = graph_out_dir / "graph_index.csv"
    shocks_csv_default = news_output_dir / "news_shocks.csv"
    firm_master_path = ENTITY_META_CSV

    # Basic existence checks
    if not index_path.exists():
        st.warning(
            f"`graph_index.csv` not found in `{graph_out_dir}`.\n\n"
            "Generate at least one sector graph in Tab 1 first."
        )
        return

    if not firm_master_path.exists():
        st.warning(
            f"Firm master CSV not found at `{firm_master_path}`.\n\n"
            "Make sure `ENTITY_META_CSV` points to your factset_entityid_to_gvkey_with_company_name.csv."
        )
        return

    # Load index of available graphs
    try:
        df_index = pd.read_csv(index_path)
    except Exception as e:
        st.error(f"Failed to read graph index at `{index_path}`: {e}")
        return

    if df_index.empty:
        st.info("`graph_index.csv` is empty. Build a sector graph in Tab 1 first.")
        return

    # Nicely formatted labels for each graph entry
    def _row_label(row: pd.Series) -> str:
        sectors = str(row.get("sectors", "[]"))
        as_of = row.get("as_of_date", "")
        if pd.isna(as_of) or as_of == "":
            as_of_str = "all dates"
        else:
            as_of_str = str(as_of)
        gid = row.get("graph_id", "graph")
        return f"{gid} | sectors={sectors} | as_of={as_of_str}"

    labels = [_row_label(r) for _, r in df_index.iterrows()]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    st.subheader("Select a sector graph")
    selected_label = st.selectbox(
        "Available graphs (from graph_index.csv)",
        options=labels,
        index=0,
        key="shock_graph_select",
    )
    row_idx = label_to_idx[selected_label]
    row = df_index.iloc[row_idx]

    graphml_path = Path(str(row["graphml_path"])).expanduser()
    st.markdown(f"**Base graph:** `{graphml_path}`")

    # Shocks CSV selection
    shocks_path_str = st.text_input(
        "News shocks CSV (output of Tab 2)",
        value=str(shocks_csv_default),
        key="shocks_csv_tab3",
    )
    shocks_path = Path(shocks_path_str).expanduser()

    # Default output paths
    default_out_graphml = graphml_path.with_name(graphml_path.stem + "_with_shocks.graphml")
    default_out_nodes = graphml_path.with_name(graphml_path.stem + "_with_shocks_nodes.csv")

    out_graphml_str = st.text_input(
        "Output shocked GraphML path",
        value=str(default_out_graphml),
        key="out_graphml_tab3",
    )
    out_nodes_str = st.text_input(
        "Output shocked nodes CSV path",
        value=str(default_out_nodes),
        key="out_nodes_tab3",
    )

    if st.button("Map shocks onto graph and save outputs", type="primary", key="run_shock_mapping"):
        if not graphml_path.exists():
            st.error(f"Base GraphML file does not exist: `{graphml_path}`")
            return
        if not shocks_path.exists():
            st.error(f"News shocks CSV does not exist: `{shocks_path}`")
            return

        out_graphml_path = Path(out_graphml_str).expanduser()
        out_nodes_path = Path(out_nodes_str).expanduser()
        out_graphml_path.parent.mkdir(parents=True, exist_ok=True)
        out_nodes_path.parent.mkdir(parents=True, exist_ok=True)

        with st.spinner("Mapping news shocks onto graph via apply_news_shocks_to_graph ..."):
            try:
                # Delegate core logic to the shared module so CLI & Streamlit stay consistent.
                shockmap.main(
                    graphml_path=str(graphml_path),
                    shocks_csv=str(shocks_path),
                    firm_master_csv=str(firm_master_path),
                    out_graphml=str(out_graphml_path),
                    out_nodes_csv=str(out_nodes_path),
                )
            except Exception as e:
                st.error(f"Failed to apply shocks to graph: {e}")
                return

        st.success("Shocks mapped and saved!")

        # Preview shocked nodes table
        try:
            df_nodes = pd.read_csv(out_nodes_path)
        except Exception as e:
            st.warning(f"Shocked nodes CSV was written but could not be loaded for preview: {e}")
            return

        st.markdown(
            f"**Shocked nodes table** — {df_nodes.shape[0]} rows "
            f"(file: `{out_nodes_path.name}`)"
        )
        st.dataframe(df_nodes.head(100))

        # Optional: simple download button for the CSV
        st.download_button(
            "Download shocked nodes CSV",
            data=df_nodes.to_csv(index=False),
            file_name=out_nodes_path.name,
            mime="text/csv",
            key="download_shocked_nodes_csv",
        )
def main():
    st.set_page_config(
        page_title="FactSet Supply-Chain + News Shocks",
        layout="wide",
    )

    st.sidebar.title("Configuration")

    graph_out_dir_str = st.sidebar.text_input(
        "Graph outputs folder (Tabs 1 & 3)",
        value=str(DEFAULT_GRAPH_OUT_DIR),
    )
    graph_out_dir = Path(graph_out_dir_str).expanduser()
    graph_out_dir.mkdir(parents=True, exist_ok=True)

    news_base_dir_str = st.sidebar.text_input(
        "Reuters JSON base folder (Tab 2)",
        value=str(DEFAULT_NEWS_BASE_DIR),
    )
    news_base_dir = Path(news_base_dir_str).expanduser()

    news_out_dir_str = st.sidebar.text_input(
        "News output folder (Tabs 2 & 3)",
        value=str(DEFAULT_NEWS_OUT_DIR),
    )
    news_out_dir = Path(news_out_dir_str).expanduser()
    news_out_dir.mkdir(parents=True, exist_ok=True)

    tab1, tab2, tab3 = st.tabs(
        ["1. Build Graph", "2. Process News", "3. Overlay Shocks"]
    )

    with tab1:
        tab_graph_builder(graph_out_dir)

    with tab2:
        tab_news_processor(news_base_dir, news_out_dir)

    with tab3:
        tab_shock_overlay(graph_out_dir, news_out_dir)


# ----------------------------------------------------------------------
# Tab 3 helpers: parse list-like cells and filter shocks by sectors
# ----------------------------------------------------------------------
def _parse_list_cell(cell):
    """Robustly parse a list-like cell from CSV.

    Handles:
      * real Python lists
      * stringified lists like "['A', 'B']"
      * comma-separated strings "A, B"
      * NaNs / None -> []
    """
    import math
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if cell is None:
        return []
    # Handle pandas NaN
    try:
        if isinstance(cell, float) and math.isnan(cell):
            return []
    except Exception:
        pass
    s = str(cell).strip()
    if not s:
        return []
    # Try to literal-eval a Python-style list
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            v = val.strip()
            return [v] if v else []
    except Exception:
        # Fall back to comma split
        return [x.strip() for x in s.split(",") if x.strip()]
    return []



def _filter_shocks_for_graph(
    df_shocks: pd.DataFrame,
    graph_sectors,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """Filter shocks by optional date range AND by sector overlap with the graph.

    Sector overlap is fuzzy:
      * we parse the shocks' `sectors` column as a list
      * we normalize both shock sectors and graph sectors (uppercased, stripped)
      * we consider them overlapping if:
          - there is an exact token match (e.g., "UTILITIES" == "UTILITIES"), OR
          - any token from one side is a substring of a token on the other side
            with length >= 4 (e.g., "FINANCIALS" vs "FINANCE" -> match).
    """
    df = df_shocks.copy()

    # --- Optional date filtering if a 'date' column is present ---
    if "date" in df.columns:
        try:
            dcol = pd.to_datetime(df["date"], errors="coerce").dt.date
            df = df.assign(date=dcol)
        except Exception:
            pass

        from datetime import datetime

        if date_from:
            try:
                dt_from = datetime.fromisoformat(date_from).date()
                df = df[df["date"] >= dt_from]
            except Exception:
                pass
        if date_to:
            try:
                dt_to = datetime.fromisoformat(date_to).date()
                df = df[df["date"] <= dt_to]
            except Exception:
                pass

    # --- Sector-based filtering ---
    if "sectors" not in df.columns or not graph_sectors:
        return df

    # Normalize graph sectors into token sets
    def norm_tokens(label: str) -> set[str]:
        if not isinstance(label, str):
            label = str(label)
        cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", label).upper()
        toks = [
            t
            for t in cleaned.split()
            if t and t not in {"AND", "SECTOR", "SECTORS", "INDUSTRY", "INDUSTRIES"}
        ]
        return set(toks)

    graph_token_sets = [norm_tokens(s) for s in graph_sectors if str(s).strip()]
    if not graph_token_sets:
        return df

    def tokens_fuzzy_match(ts1: set[str], ts2: set[str]) -> bool:
        # exact token overlap
        if ts1.intersection(ts2):
            return True
        # substring / prefix / common-prefix overlap
        for a in ts1:
            la = a.lower()
            for b in ts2:
                lb = b.lower()
                # substring check
                if len(la) >= 4 and la in lb:
                    return True
                if len(lb) >= 4 and lb in la:
                    return True
                # common prefix of length >= 4 (e.g., FINANCE vs FINANCIALS)
                m = min(len(la), len(lb))
                for k in range(m, 3, -1):
                    if la[:k] == lb[:k]:
                        return True
        return False

    def has_overlap(cell) -> bool:
        secs = _parse_list_cell(cell)
        if not secs:
            return False
        shock_tokens_list = [norm_tokens(s) for s in secs if str(s).strip()]
        for g_tokens in graph_token_sets:
            for s_tokens in shock_tokens_list:
                if tokens_fuzzy_match(g_tokens, s_tokens):
                    return True
        return False

    return df[df["sectors"].apply(has_overlap)].copy()



# ----------------------------------------------------------------------
# Tab 3: overlay shocks onto an existing graph, with preview
# ----------------------------------------------------------------------
def tab_shock_overlay(graph_out_dir: Path, news_output_dir: Path):
    """Tab 3: pick a saved sector graph, preview relevant shocks, then overlay.

    Workflow:
      1. User selects a graph from graph_index.csv.
      2. We load `news_shocks.csv` (from Tab 2).
      3. We *preview* only those shocks whose sectors overlap the graph sectors.
      4. On confirmation, we write a *filtered* shocks CSV and call
         apply_news_shocks_to_graph.main on that CSV, producing:

           * *_with_shocks.graphml
           * *_with_shocks_nodes.csv
    """
    st.header("3. Shock Overlay on Supply-Chain Graph")

    index_path = graph_out_dir / "graph_index.csv"
    shocks_csv_default = news_output_dir / "news_shocks.csv"
    firm_master_path = ENTITY_META_CSV

    # Basic existence checks
    if not index_path.exists():
        st.warning(
            f"Graph index not found at `{index_path}`.\n\nBuild graphs first in Tab 1."
        )
        return

    if not firm_master_path.exists():
        st.warning(
            f"Firm master CSV not found at `{firm_master_path}`.\n\n"
            "Make sure ENTITY_META_CSV points to your factset_entityid_to_gvkey_with_company_name.csv."
        )
        return

    # Load graph index
    try:
        df_index = pd.read_csv(index_path)
    except Exception as e:
        st.error(f"Failed to load graph index: {e}")
        return

    if df_index.empty:
        st.warning("Graph index is empty. Build at least one graph in Tab 1.")
        return

    # Helper for display labels
    def _row_label(row):
        gid = row.get("graph_id", "graph")
        sectors = str(row.get("sectors", ""))
        as_of = row.get("as_of_date", "all_dates")
        return f"{gid} | sectors={sectors} | as_of={as_of}"

    options = df_index.index.tolist()

    selected_idx = st.selectbox(
        "Available graphs (from graph_index.csv)",
        options=options,
        format_func=lambda i: _row_label(df_index.loc[i]),
        key="shock_graph_select_v2",
    )

    row = df_index.loc[selected_idx]

    graphml_path = Path(str(row["graphml_path"])).expanduser()
    st.markdown(f"**Base graph:** `{graphml_path}`")

    # Parse graph sectors as a Python list
    sectors_raw = row.get("sectors", "")
    graph_sectors = []
    if isinstance(sectors_raw, str):
        try:
            val = ast.literal_eval(sectors_raw)
            if isinstance(val, list):
                graph_sectors = [str(x).strip() for x in val if str(x).strip()]
            elif isinstance(val, str) and val.strip():
                graph_sectors = [val.strip()]
        except Exception:
            if sectors_raw.strip():
                graph_sectors = [sectors_raw.strip()]
    elif isinstance(sectors_raw, (list, tuple)):
        graph_sectors = [str(x).strip() for x in sectors_raw if str(x).strip()]

    st.markdown(
        "**Graph sectors:** "
        + (", ".join(graph_sectors) if graph_sectors else "(ALL)")
    )

    # Shocks CSV selection
    shocks_path_str = st.text_input(
        "News shocks CSV (output of Tab 2)",
        value=str(shocks_csv_default),
        key="shocks_csv_tab3_v2",
    )
    shocks_path = Path(shocks_path_str).expanduser()

    if not shocks_path.exists():
        st.warning(f"Shocks CSV not found at `{shocks_path}`. Run Tab 2 first.")
        return

    try:
        df_shocks = pd.read_csv(shocks_path)
    except Exception as e:
        st.error(f"Failed to load shocks CSV: {e}")
        return

    # --------------------------------------------------------------
    # Preview ALL shocks with optional filters (independent of graph)
    # --------------------------------------------------------------
    st.markdown("### Preview all shocks (with filters)")

    df_all = df_shocks.copy()

    # Collect filter options
    shock_types = sorted(
        [x for x in df_all.get("shock_type", pd.Series()).dropna().unique().tolist()]
    )
    shock_signs = sorted(
        [x for x in df_all.get("shock_sign", pd.Series()).dropna().unique().tolist()]
    )
    mag_levels = sorted(
        [x for x in df_all.get("shock_mag_level", pd.Series()).dropna().unique().tolist()]
    )

    # Extract all sector labels from the shocks file
    all_secs = set()
    if "sectors" in df_all.columns:
        for cell in df_all["sectors"]:
            for s in _parse_list_cell(cell):
                all_secs.add(s)
    sector_options = sorted(all_secs)

    colA, colB, colC = st.columns(3)
    with colA:
        selected_types = st.multiselect(
            "Shock types",
            options=shock_types,
            default=shock_types,
            key="all_shocks_types",
        )
    with colB:
        selected_signs = st.multiselect(
            "Shock sign",
            options=shock_signs,
            default=shock_signs,
            key="all_shocks_signs",
        )
    with colC:
        selected_mags = st.multiselect(
            "Magnitude level",
            options=mag_levels,
            default=mag_levels,
            key="all_shocks_mag",
        )

    selected_secs = st.multiselect(
        "Filter by shocks' own sectors",
        options=sector_options,
        key="all_shocks_secs",
    )

    # Apply filters
    if selected_types:
        df_all = df_all[df_all.get("shock_type").isin(selected_types)]
    if selected_signs:
        df_all = df_all[df_all.get("shock_sign").isin(selected_signs)]
    if selected_mags:
        df_all = df_all[df_all.get("shock_mag_level").isin(selected_mags)]
    if selected_secs and "sectors" in df_all.columns:
        norm_selected = {str(s).strip().upper() for s in selected_secs}

        def _has_sec(cell):
            secs = [str(x).strip().upper() for x in _parse_list_cell(cell)]
            return any(s in norm_selected for s in secs)

        df_all = df_all[df_all["sectors"].apply(_has_sec)]

    preview_cols = [
        "event_id",
        "headline",
        "summary",
        "firms",
        "sectors",
        "regions",
        "shock_type",
        "shock_mag_level",
        "shock_sign",
        "shock_confidence",
    ]
    preview_cols = [c for c in preview_cols if c in df_all.columns]

    st.write(f"**Filtered shocks (all shocks view):** {len(df_all)} rows")
    if not df_all.empty:
        st.dataframe(df_all[preview_cols].head(200))
    else:
        st.write("No shocks match the selected filters.")

    st.markdown("---")

    st.markdown("### Preview shocks relevant to this graph")

    if st.button("Preview relevant shocks", key="preview_shocks_button"):
        df_filtered = _filter_shocks_for_graph(df_shocks, graph_sectors)
        st.info(f"Found {len(df_filtered)} shocks overlapping the graph sectors.")

        if not df_filtered.empty:
            cols_show = [
                "event_id",
                "headline",
                "summary",
                "firms",
                "sectors",
                "regions",
                "shock_type",
                "shock_mag_level",
                "shock_sign",
                "shock_confidence",
            ]
            cols_show = [c for c in cols_show if c in df_filtered.columns]
            st.dataframe(df_filtered[cols_show].head(200))
        else:
            st.write("No shocks match these sectors.")

    st.markdown("---")
    st.markdown("### Map sector-relevant shocks onto graph and save")

    # Default output paths
    default_out_graphml = graphml_path.with_name(graphml_path.stem + "_with_shocks.graphml")
    default_out_nodes = graphml_path.with_name(graphml_path.stem + "_with_shocks_nodes.csv")

    col1, col2 = st.columns(2)
    with col1:
        out_graphml_str = st.text_input(
            "Output GraphML (with shocks)",
            value=str(default_out_graphml),
            key="out_graphml_tab3_v2",
        )
    with col2:
        out_nodes_str = st.text_input(
            "Output nodes CSV",
            value=str(default_out_nodes),
            key="out_nodes_tab3_v2",
        )

    if st.button(
        "Map sector-relevant shocks onto graph and save",
        type="primary",
        key="run_shock_mapping_v2",
    ):
        if not graphml_path.exists():
            st.error(f"Base GraphML file does not exist: `{graphml_path}`")
            return

        df_filtered = _filter_shocks_for_graph(df_shocks, graph_sectors)
        if df_filtered.empty:
            st.warning("No shocks remain after sector filtering; nothing to map.")
            return

        out_graphml_path = Path(out_graphml_str).expanduser()
        out_nodes_path = Path(out_nodes_str).expanduser()

        out_graphml_path.parent.mkdir(parents=True, exist_ok=True)
        out_nodes_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a temporary filtered shocks CSV to feed into apply_news_shocks_to_graph
        tmp_filtered = news_output_dir / f"{graphml_path.stem}_shocks_for_overlay.csv"
        try:
            df_filtered.to_csv(tmp_filtered, index=False)
        except Exception as e:
            st.error(f"Failed to write temporary filtered shocks CSV: {e}")
            return

        with st.spinner(
            "Mapping sector-relevant news shocks onto graph via apply_news_shocks_to_graph ..."
        ):
            try:
                shockmap.main(
                    graphml_path=str(graphml_path),
                    shocks_csv=str(tmp_filtered),
                    firm_master_csv=str(firm_master_path),
                    out_graphml=str(out_graphml_path),
                    out_nodes_csv=str(out_nodes_path),
                )
            except Exception as e:
                st.error(f"Failed to apply shocks to graph: {e}")
                return

        st.success("Shocks mapped and saved!")

        # Preview shocked nodes table
        try:
            df_nodes = pd.read_csv(out_nodes_path)
        except Exception as e:
            st.warning(
                "Shocked nodes CSV was written but could not be loaded for preview: "
                f"{e}"
            )
            return

        st.markdown(
            f"**Shocked nodes table** — {df_nodes.shape[0]} rows "
            f"(file: `{out_nodes_path.name}`)"
        )
        st.dataframe(df_nodes.head(100))

        # Optional: simple download button for the CSV
        st.download_button(
            label="Download shocked nodes CSV",
            data=df_nodes.to_csv(index=False),
            file_name=out_nodes_path.name,
            mime="text/csv",
            key="download_shocked_nodes_tab3_v2",
        )



if __name__ == "__main__":
    main()
