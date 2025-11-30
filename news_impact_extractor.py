# news_impact_extractor.py
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from agentics import AG
from news_impact_models import NewsImpactRecord


# ================================
# HARD-WIRED DEFAULT PATHS / MODEL
# ================================
DEFAULT_JSON_PATH = "/Users/boxuanli/Desktop/2008/200801_sample.json"
DEFAULT_OUT_CSV = "/Users/boxuanli/Code/Agentics/SupplyChain/news_output/news_shocks.csv"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

# Number of “failed / trivial” states to debug (re-run LLM, log raw output)
DEBUG_FAILED_LIMIT = 20  # set to 0 to disable
# ================================


@dataclass
class RawNewsItem:
    event_id: str
    headline: str
    body: str
    timestamp: Optional[str] = None  # firstCreated / versionCreated
    ric: Optional[str] = None        # e.g. MRN_STORY


# ---------- Reuters JSON loader (robust to your sample format) ----------

def _load_reuters_json(path: str) -> dict:
    """
    Load Reuters monthly JSON dump.

    Your sample file is almost valid JSON but missing the final `]}`.
    This function repairs that case by trimming the trailing comma.
    """
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    text = raw.strip()
    if text.endswith("]}"):
        # already a valid object -> try direct load
        return json.loads(text)

    # Try a simple repair: trim the last ",\n" after the last `}}` and close `]}"
    last = text.rfind("}},")
    if last == -1:
        # fallback: just try json.loads and let it fail loudly
        return json.loads(text)

    repaired = text[: last + 2] + "]}"
    return json.loads(repaired)


def _iter_reuters_items(doc: dict) -> List[RawNewsItem]:
    items = doc.get("Items", []) or []
    out: List[RawNewsItem] = []

    ric = doc.get("RIC")

    for item in items:
        data = item.get("data", {}) or {}
        body = str(data.get("body", "") or "")
        headline = str(data.get("headline", "") or "")
        first_created = data.get("firstCreated") or data.get("versionCreated")
        event_id = data.get("id") or item.get("guid") or ""

        if not headline and not body:
            continue

        out.append(
            RawNewsItem(
                event_id=str(event_id),
                headline=headline.strip(),
                body=body.strip(),
                timestamp=first_created,
                ric=ric,
            )
        )

    return out


# ---------- Prompt construction ----------

def _build_prompt(item: RawNewsItem) -> str:
    """
    User prompt for Agentics typed transduction.

    We focus on *semantics*; Agentics will handle the schema / JSON mapping.
    """
    body_snip = item.body

    return f"""
You are a financial and supply-chain news analyst.

Your task is to analyze the following news article and fill an internal
schema with fields like:
- firms: directly mentioned firms/issuers (headline or body)
- sectors: broad economic sectors affected
- regions: regions/countries affected
- summary: 1–3 sentences summarizing the economically relevant content
- shock_type: short label for the type of economic shock
- shock_magnitude: overall severity in [0,1]
- shock_direction: 'supply', 'demand', 'cost_of_capital', 'regulation',
                   'sentiment', 'other', or 'unknown'
- shock_sign: 'positive', 'negative', 'mixed', or 'unclear'
- shock_scope: 'firm', 'sector', 'multi-sector', 'macro', or 'idiosyncratic'
- shock_confidence: confidence in [0,1]
- shock_mag_level: 'low', 'medium', or 'high'

Guidelines:
- If the article is mostly a schedule / diary / generic calendar with no
  new information (e.g., a list of upcoming auctions), treat it as
  very low impact: shock_magnitude ≈ 0.0, shock_mag_level = 'low',
  shock_confidence can be modest, and leave firms/sectors/regions empty
  unless something is clearly affected.
- If there is a clear macro or sector shock, reflect it in sectors,
  regions, shock_type, shock_scope, and shock_sign.
- If the article is ambiguous or neutral, you can keep shock_magnitude small
  and shock_sign = 'unclear'.

Now analyze this article and internally fill the schema for it:

EVENT_ID: {item.event_id}
HEADLINE: {item.headline}

BODY:
{body_snip}
""".strip()


def _build_debug_prompt(item: RawNewsItem) -> str:
    """
    Debug prompt: explicitly ask for JSON, used only for the failed cases
    we want to inspect. This does NOT go through Agentics typed transduction.
    """
    body_snip = item.body[:2000]  # truncate for safety

    return f"""
You are a financial and supply-chain risk analyst.

For the news article below, produce a single JSON object with the following keys:

{{
  "summary": "short summary of the economically relevant content",
  "firms": ["list", "of", "firm names"],
  "sectors": ["list", "of", "broad economic sectors"],
  "regions": ["list", "of", "regions or countries"],
  "shock_type": "short label for the type of shock",
  "shock_magnitude": 0.0,
  "shock_direction": "supply | demand | cost_of_capital | regulation | sentiment | other | unknown",
  "shock_sign": "positive | negative | mixed | unclear",
  "shock_scope": "firm | sector | multi-sector | macro | idiosyncratic",
  "shock_confidence": 0.0,
  "shock_mag_level": "low | medium | high"
}}

Rules:
- Respond with JSON ONLY. No commentary, no markdown, no extra text.
- If the article is mainly a diary/schedule with no clear impact,
  use shock_magnitude ≈ 0.0 and shock_mag_level = "low", and leave firms,
  sectors, and regions as empty lists.

Article to analyze:

EVENT_ID: {item.event_id}
HEADLINE: {item.headline}

BODY:
{body_snip}
""".strip()


# ---------- Helper: trivial vs nontrivial records ----------

def _is_trivial_record(rec: NewsImpactRecord) -> bool:
    """
    Heuristic: consider a record "trivial" if nothing was really filled.
    """
    if rec.summary and rec.summary.strip():
        return False
    if rec.shock_magnitude not in (0.0, 0, None):
        return False
    if rec.shock_confidence not in (0.0, 0, None):
        return False
    if rec.firms or rec.sectors or rec.regions:
        return False
    # shock_type default is "none"
    if rec.shock_type and rec.shock_type != "none":
        return False
    return True


# ---------- Agentics-powered extraction + debug ----------

async def extract_impacts_from_json(
    json_path: str,
    out_csv: str,
    max_items: Optional[int] = None,
    debug_failed_jsonl: Optional[str] = None,
    debug_failed_limit: int = 0,
) -> None:
    """
    Main async entrypoint: read Reuters JSON, use Agentics to get structured shocks,
    and write a CSV. Optionally, debug a subset of 'trivial' records by logging
    raw LLM responses into a JSONL file.
    """
    json_path = os.path.abspath(json_path)
    out_csv = os.path.abspath(out_csv)
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)

    print(f"[NEWS-IMPACT] Loading Reuters JSON from {json_path} ...")
    doc = _load_reuters_json(json_path)
    raw_items = _iter_reuters_items(doc)
    print(f"[NEWS-IMPACT] Parsed {len(raw_items)} items from JSON.")

    if max_items is not None:
        raw_items = raw_items[:max_items]
        print(f"[NEWS-IMPACT] Truncated to first {len(raw_items)} items for this run.")

    prompts = [_build_prompt(it) for it in raw_items]

    if not prompts:
        print("[NEWS-IMPACT] No non-empty articles found; writing empty CSV.")
        pd.DataFrame(columns=list(NewsImpactRecord.model_fields.keys())).to_csv(out_csv, index=False)
        return

    print("[NEWS-IMPACT] Creating Agentics AG with NewsImpactRecord schema ...")

    ag = AG(atype=NewsImpactRecord, llm=AG.get_llm_provider("openai"))

    print(f"[NEWS-IMPACT] Running logical transduction over {len(prompts)} articles ...")
    ag = await (ag << prompts)

    # Post-process states: inject event_id and headline from raw_items
    records: List[NewsImpactRecord] = []
    for state, item in zip(ag.states, raw_items):
        rec = state.model_copy()
        rec.event_id = item.event_id
        rec.headline = item.headline
        records.append(rec)

    df = pd.DataFrame([r.model_dump() for r in records])
    print(f"[NEWS-IMPACT] Got {df.shape[0]} structured rows (including trivial ones).")

    # Optional debug: re-run LLM in a simpler JSON mode for failed/trivial states
    if debug_failed_jsonl and debug_failed_limit > 0:
        debug_failed_jsonl = os.path.abspath(debug_failed_jsonl)
        Path(os.path.dirname(debug_failed_jsonl)).mkdir(parents=True, exist_ok=True)

        print(
            f"[DEBUG] Inspecting up to {debug_failed_limit} trivial records; "
            f"raw responses -> {debug_failed_jsonl}"
        )

        with open(debug_failed_jsonl, "w", encoding="utf-8") as f_out:
            debug_count = 0
            for rec, item in zip(records, raw_items):
                if not _is_trivial_record(rec):
                    continue
                if debug_count >= debug_failed_limit:
                    break

                dbg_prompt = _build_debug_prompt(item)
                try:
                    dbg_response = llm.run(dbg_prompt, temperature=0)
                except Exception as e:
                    dbg_response = f"ERROR: {repr(e)}"

                obj = {
                    "event_id": item.event_id,
                    "headline": item.headline,
                    "body_snippet": item.body[:500],
                    "debug_prompt": dbg_prompt,
                    "llm_raw_response": dbg_response,
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                debug_count += 1

        print(f"[DEBUG] Wrote {debug_count} debug entries to {debug_failed_jsonl}")

    print(f"[NEWS-IMPACT] Saving CSV to {out_csv}")
    df.to_csv(out_csv, index=False)
    print("[NEWS-IMPACT] Done.")


if __name__ == "__main__":
    print("[CONFIG] Using hard-wired paths:")
    print(f"         Input JSON: {DEFAULT_JSON_PATH}")
    print(f"         Output CSV: {DEFAULT_OUT_CSV}")
    print(f"         Model:      {DEFAULT_OPENAI_MODEL}")
    print(f"         Debug limit: {DEBUG_FAILED_LIMIT}")

    # Derive debug jsonl path from output CSV
    debug_jsonl = DEFAULT_OUT_CSV.replace(".csv", "_failed_debug.jsonl")

    asyncio.run(
        extract_impacts_from_json(
            json_path=DEFAULT_JSON_PATH,
            out_csv=DEFAULT_OUT_CSV,
            max_items=None,
            debug_failed_jsonl=debug_jsonl,
            debug_failed_limit=DEBUG_FAILED_LIMIT,
        )
    )

    print(f"[DONE] Wrote structured shocks to {DEFAULT_OUT_CSV}")
    if DEBUG_FAILED_LIMIT > 0:
        print(f"[DONE] Wrote debug LLM outputs (trivial states) to {debug_jsonl}")