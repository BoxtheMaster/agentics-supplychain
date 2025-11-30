# Supply-Chain Graph × News Shock Pipeline

This repo builds a **firm-level supply-chain network** from FactSet data and
links it to **macro / financial news shocks** extracted by an LLM (via IBM
Agentics + OpenAI). The goal is to end up with a dynamic graph where each firm
(node) carries **shock indicators** derived from Reuters news.

The Streamlit app (`streamlit_final.py`) lets you:

1. Build / visualize **sector-level supply-chain graphs** from FactSet.
2. Process raw **Reuters JSON** news into a structured `news_shocks.csv`.
3. Map shocks onto the supply-chain graph (by **firm name**, **sector**, and
   **region**) and preview which nodes are hit.

---

## 1. High-Level Overview

### Stage 1: Build the Supply-Chain Graph (FactSet)

Inputs:

- FactSet supply-chain relationship tables (TXT or CSV):
  - `ent_scr_relationships.txt`
  - `ent_scr_supply_chain.txt`
  - `scr_relationships_summary.txt`
  - `scr_relationships_keyword.txt`
  - `ent_scr_coverage.txt`
- A **firm master** table linking FactSet entities to Compustat:
  - `factset_entityid_to_gvkey_with_company_name.csv`  
    with at least:
    - `FACTSET_ENTITY_ID`
    - `gvkey`
    - `conm` (company name)
    - `naics` (or equivalent)
    - `loc` (country code or region)

What we do:

- Build a large directed graph `G_FULL` where nodes are firm-level entities and
  edges represent **supply-chain** and **competitor** relationships.
- Enrich each node with metadata (`conm`, `gvkey`, `FACTSET_ENTITY_ID`,
  `naics_sector`, `loc`, etc.).
- Save sector subgraphs (e.g., **Finance and Insurance**, **Manufacturing**)
  as:

  - `factset_sc_<Sector>_all_dates.graphml`
  - `factset_sc_<Sector>_all_dates.gexf`
  - `factset_sc_<Sector>_all_dates_edges.csv`
  - PyVis HTML for quick visualization.

### Stage 2: Extract News Shocks (Reuters + LLM)

Inputs:

- Raw **Reuters JSON** files (e.g. monthly files from 2008):
  - Example: `200801.json`
- Each file contains a top-level object, with:
  - `Items`: list of articles
  - Each article has `data.headline`, `data.body`, timestamps, subjects, etc.

What we do:

- Use **Agentics** to run an LLM over each article and extract a structured
  **shock record**:

  - `event_id`
  - `headline`
  - `summary`
  - `firms` (list of firm names, only if explicitly mentioned in title/body)
  - `sectors` (list of affected sectors, e.g. `"Financials"`, `"Energy"`)
  - `regions` (list of affected regions/countries)
  - `shock_type` (e.g. `"policy"`, `"credit"`, `"macro"`)
  - `shock_magnitude` (free text)
  - `shock_direction` (e.g. `"tightening"`, `"loosening"`, `"negative"`)
  - `shock_sign` (e.g. `"+"`, `"-"`, `"neutral"`)
  - `shock_mag_level` (e.g. `"small"`, `"medium"`, `"large"`)
  - `shock_confidence` (0–1 or categorical)
  - `shock_scope` (local vs global, sector-specific, etc.)

- Output a single CSV:

  - `news_output/news_shocks.csv`

### Stage 3: Map Shocks to the Supply-Chain Graph

Inputs:

- A **sector-level GraphML** from Stage 1, e.g.  
  `Outputs/factset_sc_Finance_and_Insurance_all_dates.graphml`
- The **firm master** CSV from Stage 1.
- `news_output/news_shocks.csv` from Stage 2.

What we do:

- Map each shock row to graph nodes via three channels:

  1. **Firm-level mapping** (highest priority)
     - Parse `firms` column (handles real lists, stringified lists, or
       comma-separated strings).
     - Normalize firm names (uppercase + strip punctuation).
     - Match against firm master (`conm`) to get `FACTSET_ENTITY_ID` and
       `gvkey`.
     - Then map to nodes where:
       - `node.factset_entity_id == FACTSET_ENTITY_ID`, or
       - `node.gvkey == gvkey`.
     - If that fails, fuzzy-match against node `conm` directly.

  2. **Sector-level mapping**
     - Parse `sectors` column.
     - Convert each sector string and each node’s `naics_sector` into
       normalized token sets (e.g. `"Finance and Insurance"` →
       `{FINANCE, INSURANCE}`).
     - Fuzzy token matching (exact intersection, substring overlap, or long
       common prefix ≥ 4 chars) so `"Financials"` matches
       `"Finance and Insurance"`.

  3. **Region-level mapping**
     - Parse `regions`.
     - Compare with node `loc` (country/region), allowing equality or
       substring matches (e.g. `"US"` vs `"USA"` vs `"United States"`).

- For each node, aggregate:

  - `shock_event_ids` (comma-separated)
  - `shock_count`
  - `shock_up` (positive shocks)
  - `shock_down` (negative shocks)
  - `shock_neutral`

- Save:

  - `factset_sc_<Sector>_all_dates_with_shocks.graphml`
  - `factset_sc_<Sector>_all_dates_with_shocks_nodes.csv`

These are then usable for:

- Visualizing shocked networks in Gephi / Cytoscape.
- Downstream econometric or ML models (e.g., shock propagation, stress tests).

At a high level:

```text
FactSet SCR TXT/CSV    Firm Master (FactSet → gvkey)     Reuters JSON
        │                         │                          │
        │                         └─────────────┐            │
        │                                       │            │
        ▼                                       ▼            ▼
factset_supply_chain_core.py          factset_entityid_to_gvkey...   news_impact_extractor.py
        │                                       │            │
        │                                       └────────────┘
        ▼
Outputs/full_graph.graphml + sector GraphMLs       news_output/news_shocks.csv
        │                                                       │
        │                                                       │
        └──────────►  apply_news_shocks_to_graph.py  ◄──────────┘
                             │
                             ▼
      sector_with_shocks.graphml + sector_with_shocks_nodes.csv
                             │
                             ▼
                      streamlit_final.py
          (Graph builder, news processor, shock overlay)

---


## 2. Repository Layout (key files)

**Core graph building**

- `factset_supply_chain_core.py`  
  Functions to:
  - Load raw FactSet tables
  - Build `G_FULL` and sector subgraphs
  - Save GraphML/GEXF/CSV/PyVis for each sector

**News processing / shocks**

- `news_impact_extractor.py`  
  Uses Agentics + OpenAI to convert Reuters JSON into `news_shocks.csv`.

- `news_output/news_shocks.csv`  
  Structured shocks file generated from Reuters JSON.

**Shock → graph mapping**

- `apply_news_shocks_to_graph.py`  
  CLI + Python function `apply_shocks_to_graph(...)` that:
  - Loads a GraphML
  - Loads firm master + `news_shocks.csv`
  - Annotates nodes with shock aggregates
  - Writes `*_with_shocks.graphml` and `*_with_shocks_nodes.csv`

**Streamlit app**

- `streamlit_final.py`  
  Main UI with three tabs:

  1. **Supply-Chain Graph Builder**
     - Build and visualize sector-level graphs
     - Save GraphML / GEXF / CSV / HTML
  2. **News → Shock Extraction**
     - Select Reuters month
     - Run Agentics over articles
     - Produce `news_output/news_shocks.csv`
  3. **Shock Overlay on Graph**
     - Select a sector graph
     - Preview all shocks (with filters)
     - Map shocks onto graph nodes
     - Preview shocked nodes

---

## 3. Environment & Installation

### Python & venv

- Python 3.10+ recommended.
- Create and activate a virtual environment:

```bash
cd /path/to/Agentics/SupplyChain

python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# or .venv\Scripts\activate    # Windows



