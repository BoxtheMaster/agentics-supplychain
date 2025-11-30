# news_impact_models.py
from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field


class NewsImpactRecord(BaseModel):
    """
    Structured impact record for one news article.
    Must be default-constructible so Agentics can do target_type().
    """

    # basic metadata – we'll overwrite event_id/headline from raw JSON
    event_id: str = ""
    headline: str = ""
    summary: str = Field(
        default="",
        description="1–3 sentence summary of the news focused on economic / supply-chain implications.",
    )

    # entities / scope
    firms: List[str] = Field(
        default_factory=list,
        description="Firm names explicitly mentioned in headline or body.",
    )
    sectors: List[str] = Field(
        default_factory=list,
        description="Broad economic sectors affected by this news.",
    )
    regions: List[str] = Field(
        default_factory=list,
        description="Countries/regions affected (e.g. 'Canada', 'Eurozone', 'Global').",
    )

    # shock characterization
    shock_type: str = Field(
        default="none",
        description="Short label for the shock (e.g. 'interest-rate policy', 'trade tariffs', 'liquidity facility').",
    )
    shock_magnitude: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall severity in [0,1]. 0=negligible, 1=very large.",
    )
    shock_direction: Literal[
        "supply", "demand", "cost_of_capital", "regulation", "sentiment", "other", "unknown"
    ] = "unknown"
    shock_sign: Literal["positive", "negative", "mixed", "unclear"] = "unclear"
    shock_scope: Literal["firm", "sector", "multi-sector", "macro", "idiosyncratic"] = "macro"

    shock_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in the assessment in [0,1].",
    )
    shock_mag_level: Literal["low", "medium", "high"] = "low"
