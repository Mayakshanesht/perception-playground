"""
Cloudbee Research Copilot — Workspace API Router

Manages user workspaces, projects, experiments, and models.
Uses in-memory storage by default; swap with database adapter for production.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("workspace")
router = APIRouter(prefix="/api/workspace", tags=["workspace"])


# ─── In-Memory Store (swap with DB adapter in production) ────────────
_projects: Dict[str, Dict] = {}
_experiments: Dict[str, Dict] = {}
_models: Dict[str, Dict] = {}


# ─── Schemas ─────────────────────────────────────────────────────────
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    research_question: Optional[str] = ""


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    research_question: Optional[str] = None
    papers: Optional[List[Dict]] = None
    architecture: Optional[Dict] = None
    notebooks: Optional[List[Dict]] = None
    model_links: Optional[List[Dict]] = None
    status: Optional[str] = None


class ExperimentCreate(BaseModel):
    project_id: str
    title: str
    hypothesis: Optional[str] = ""
    architecture: Optional[Dict] = {}
    notebook: Optional[str] = ""


class ExperimentUpdate(BaseModel):
    status: Optional[str] = None
    metrics: Optional[Dict] = None
    notebook: Optional[str] = None


class ModelCreate(BaseModel):
    name: str
    huggingface_link: str
    dataset_used: Optional[str] = ""
    project_id: Optional[str] = None
    metrics: Optional[Dict] = {}


# ─── Project Endpoints ───────────────────────────────────────────────
@router.post("/projects")
async def create_project(req: ProjectCreate):
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    project = {
        "id": pid,
        "name": req.name,
        "research_question": req.research_question or "",
        "papers": [],
        "architecture": {},
        "notebooks": [],
        "model_links": [],
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }
    _projects[pid] = project
    return project


@router.get("/projects")
async def list_projects():
    return sorted(_projects.values(), key=lambda p: p["updated_at"], reverse=True)


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return _projects[project_id]


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, req: ProjectUpdate):
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    p = _projects[project_id]
    updates = req.dict(exclude_unset=True)
    for k, v in updates.items():
        p[k] = v
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    return p


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    del _projects[project_id]
    # Cascade delete experiments and models
    for eid in list(_experiments.keys()):
        if _experiments[eid].get("project_id") == project_id:
            del _experiments[eid]
    return {"deleted": True}


# ─── Experiment Endpoints ────────────────────────────────────────────
@router.post("/experiments")
async def create_experiment(req: ExperimentCreate):
    eid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    exp = {
        "id": eid,
        "project_id": req.project_id,
        "title": req.title,
        "hypothesis": req.hypothesis or "",
        "architecture": req.architecture or {},
        "metrics": {},
        "status": "notebook_generated",
        "notebook": req.notebook or "",
        "created_at": now,
        "updated_at": now,
    }
    _experiments[eid] = exp
    return exp


@router.get("/experiments")
async def list_experiments(project_id: Optional[str] = None):
    exps = list(_experiments.values())
    if project_id:
        exps = [e for e in exps if e.get("project_id") == project_id]
    return sorted(exps, key=lambda e: e["created_at"], reverse=True)


@router.patch("/experiments/{experiment_id}")
async def update_experiment(experiment_id: str, req: ExperimentUpdate):
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    exp = _experiments[experiment_id]
    updates = req.dict(exclude_unset=True)
    for k, v in updates.items():
        exp[k] = v
    exp["updated_at"] = datetime.now(timezone.utc).isoformat()
    return exp


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    del _experiments[experiment_id]
    return {"deleted": True}


# ─── Model Endpoints ────────────────────────────────────────────────
@router.post("/models")
async def register_model(req: ModelCreate):
    mid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    model = {
        "id": mid,
        "name": req.name,
        "huggingface_link": req.huggingface_link,
        "dataset_used": req.dataset_used or "",
        "project_id": req.project_id,
        "metrics": req.metrics or {},
        "created_at": now,
    }
    _models[mid] = model
    return model


@router.get("/models")
async def list_models(project_id: Optional[str] = None):
    mdls = list(_models.values())
    if project_id:
        mdls = [m for m in mdls if m.get("project_id") == project_id]
    return sorted(mdls, key=lambda m: m["created_at"], reverse=True)


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    del _models[model_id]
    return {"deleted": True}
