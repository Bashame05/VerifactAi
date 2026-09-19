"""
api.py
──────
FastAPI backend wrapper for the VerifactAI misinformation-detection pipeline.

Exposes a clean HTTP interface:
  - GET  /         -> Serves the index.html test frontend
  - POST /analyze  -> Runs analyze_article(article_text) from pipeline.py
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import analyze_article

app = FastAPI(
    title="VerifactAI API",
    description="Misinformation Detection and Fact-Checking Pipeline API",
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────
# CORS Setup
# ─────────────────────────────────────────────────────────────
# Allow all origins for local testing and ad-hoc frontend callers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    article_text: str = Field(
        ...,
        description="Raw article or paragraph text to analyze.",
        min_length=1,
    )
    claim_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Threshold probability for check-worthiness.",
    )


@app.get("/", summary="Serve test interface")
def serve_index() -> FileResponse:
    """Serve the local index.html test interface."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_path)


@app.post("/analyze", summary="Analyze article text")
def analyze_endpoint(payload: AnalyzeRequest) -> Dict[str, Any]:
    """Run the full VerifactAI pipeline on input article text.

    Returns the complete analysis dictionary containing:
      - article_length
      - num_claims_found
      - claims (list with p_claim, p_ai, verdict, contradiction, risk, evidence)
      - overall_risk_score
      - overall_risk_tier
      - highest_risk_claim
      - article_metrics (readability + coherence)
    """
    cleaned_text = payload.article_text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="article_text cannot be empty or whitespace-only.",
        )

    try:
        result = analyze_article(
            article_text=cleaned_text,
            claim_threshold=payload.claim_threshold,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error during execution: {type(exc).__name__}: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    print("Starting VerifactAI API server on http://127.0.0.1:8000 ...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
