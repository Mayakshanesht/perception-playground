"""
Cloudbee Research Copilot — Research API Router

Modular FastAPI router for paper analysis, hypothesis generation,
notebook creation, and model registry.

Deploy independently with your own LLM provider and GPU cluster.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger("research-copilot")
router = APIRouter(prefix="/api/research", tags=["research"])

# ─── Configuration ───────────────────────────────────────────────────
# Switch LLM provider by changing these env vars
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://ai.gateway.lovable.dev/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")


# ─── Schemas ─────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    max_papers: int = Field(10, ge=3, le=20)


class NotebookRequest(BaseModel):
    query: str
    hypothesis_name: str
    hypothesis_architecture: str
    hypothesis_dataset: str
    hypothesis_accuracy: str


class ModelRegistration(BaseModel):
    name: str
    huggingface_link: str
    dataset_used: Optional[str] = ""
    metrics: Optional[Dict[str, Any]] = {}


class Paper(BaseModel):
    title: str
    authors: str
    year: int
    problem: str
    method: str
    results: str
    github: Optional[str] = None
    dataset: str = ""


class Hypothesis(BaseModel):
    id: str
    name: str
    architecture: str
    expected_accuracy: str
    compute: str
    dataset: str
    reasoning: str


class Proposal(BaseModel):
    title: str
    summary: str
    pipeline: List[str]
    key_insight: str


class AnalysisResult(BaseModel):
    papers: List[Paper]
    hypotheses: List[Hypothesis]
    proposal: Proposal
    datasets: List[Dict[str, str]]


# ─── LLM Client ─────────────────────────────────────────────────────
async def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Call any OpenAI-compatible LLM endpoint.
    Swap LLM_BASE_URL and LLM_API_KEY to use your own models.
    """
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="LLM_API_KEY not configured")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
        )

    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if resp.status_code == 402:
        raise HTTPException(status_code=402, detail="Credits exhausted")
    if resp.status_code != 200:
        logger.error("LLM error: %d %s", resp.status_code, resp.text[:500])
        raise HTTPException(status_code=502, detail="LLM provider error")

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content


def clean_json_response(text: str) -> str:
    """Strip markdown fences from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


# ─── System Prompts ──────────────────────────────────────────────────
ANALYZE_PROMPT = """You are the Cloudbee Research Copilot, an expert AI research assistant for computer vision.

When a user asks a research question, respond with EXACTLY this JSON structure (no markdown, pure JSON):

{
  "papers": [
    {
      "title": "Paper Title",
      "authors": "Author1, Author2",
      "year": 2024,
      "problem": "What problem it solves",
      "method": "Core methodology",
      "results": "Key results",
      "github": "https://github.com/...",
      "dataset": "Dataset used"
    }
  ],
  "hypotheses": [
    {
      "id": "A",
      "name": "Hypothesis name",
      "architecture": "Architecture description",
      "expected_accuracy": "Expected performance",
      "compute": "Compute requirements",
      "dataset": "Recommended dataset",
      "reasoning": "Why this approach"
    }
  ],
  "proposal": {
    "title": "Recommended approach title",
    "summary": "2-3 sentence summary",
    "pipeline": ["Step 1", "Step 2", "Step 3"],
    "key_insight": "The main insight"
  },
  "datasets": [
    {
      "name": "Dataset name",
      "size": "Size info",
      "source": "ultralytics or other",
      "url": "URL"
    }
  ]
}

RULES:
- Only include papers with REAL, VERIFIED GitHub repos. Set github to null if unsure.
- Include 5-10 relevant papers.
- Generate exactly 3 hypotheses labeled A, B, C.
- Prefer Ultralytics-compatible datasets.
- Return ONLY valid JSON."""

NOTEBOOK_PROMPT = """You are the Cloudbee Research Copilot notebook generator.
Generate a complete Jupyter notebook (.ipynb JSON format).

Structure:
1. Title and hypothesis description (markdown)
2. Install Dependencies (code) - pip install
3. Import Libraries (code)
4. Dataset Loading (code) - use Ultralytics datasets when possible
5. Model Architecture (code) - full PyTorch model
6. Training Loop (code) - with loss, optimizer, scheduler
7. Evaluation (code) - metrics
8. Save to HuggingFace (code) - huggingface_hub push_to_hub
9. Visualization (code) - plot results
10. Next Steps (markdown) - mention HuggingFace weights for inference

Include W&B logging (optional cell).
Return ONLY valid .ipynb JSON."""


# ─── Endpoints ───────────────────────────────────────────────────────
@router.post("/analyze", response_model=AnalysisResult)
async def analyze_research(req: AnalyzeRequest):
    """
    Analyze a research question: find papers, generate hypotheses, propose architecture.
    """
    raw = await call_llm(ANALYZE_PROMPT, req.query)
    cleaned = clean_json_response(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response: %s", str(e))
        raise HTTPException(status_code=502, detail="Invalid response from LLM")

    return AnalysisResult(**data)


@router.post("/notebook")
async def generate_notebook(req: NotebookRequest):
    """
    Generate a Jupyter notebook for a selected hypothesis.
    Returns .ipynb JSON.
    """
    user_msg = (
        f"Generate a training notebook for this hypothesis:\n\n"
        f"Hypothesis: {req.hypothesis_name}\n"
        f"Architecture: {req.hypothesis_architecture}\n"
        f"Dataset: {req.hypothesis_dataset}\n"
        f"Expected Accuracy: {req.hypothesis_accuracy}\n\n"
        f"Original research question: {req.query}"
    )

    raw = await call_llm(NOTEBOOK_PROMPT, user_msg)
    cleaned = clean_json_response(raw)

    try:
        notebook = json.loads(cleaned)
    except json.JSONDecodeError:
        # Return as raw text if not valid JSON
        return {"notebook": cleaned, "format": "text"}

    return {"notebook": notebook, "format": "ipynb"}


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "cloudbee-research-copilot",
        "llm_provider": LLM_BASE_URL,
        "model": LLM_MODEL,
    }
