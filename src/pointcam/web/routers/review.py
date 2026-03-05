"""Review endpoints: run listing, review queue, resolve, export."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


def _runs_dir(request: Request) -> Path:
    return request.app.state.manager._project_root / "runs" / "live"


@router.get("/runs")
async def list_runs(request: Request):
    """List completed pipeline runs with their output files."""
    runs_root = _runs_dir(request)
    runs = []
    if runs_root.is_dir():
        for d in sorted(runs_root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            crossings = list(d.glob("*_crossings.csv"))
            review = list(d.glob("*_review_queue.json"))
            runs.append({
                "name": d.name,
                "has_crossings": len(crossings) > 0,
                "has_review_queue": len(review) > 0,
                "crossings_file": crossings[0].name if crossings else None,
                "review_file": review[0].name if review else None,
            })
    return runs


@router.get("/queue")
async def get_review_queue(request: Request, run: str = Query(...)):
    """Load the review queue for a specific run."""
    runs_root = _runs_dir(request)
    run_dir = runs_root / run
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run}")

    review_files = list(run_dir.glob("*_review_queue.json"))
    if not review_files:
        return {"items": [], "total": 0}

    with open(review_files[0]) as f:
        items = json.load(f)

    return {"items": items, "total": len(items)}


class ResolveRequest(BaseModel):
    run: str
    index: int
    action: str  # "confirm" | "correct" | "reject"
    corrected_bib: Optional[str] = None


@router.post("/resolve")
async def resolve_item(body: ResolveRequest, request: Request):
    """Resolve a review queue item (confirm, correct, or reject)."""
    runs_root = _runs_dir(request)
    run_dir = runs_root / body.run
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {body.run}")

    review_files = list(run_dir.glob("*_review_queue.json"))
    if not review_files:
        raise HTTPException(status_code=404, detail="No review queue")

    review_path = review_files[0]
    with open(review_path) as f:
        items = json.load(f)

    if body.index < 0 or body.index >= len(items):
        raise HTTPException(status_code=400, detail="Invalid index")

    item = items[body.index]
    item["review_action"] = body.action
    if body.action == "correct" and body.corrected_bib:
        item["corrected_bib"] = body.corrected_bib
    elif body.action == "reject":
        item["corrected_bib"] = None

    with open(review_path, "w") as f:
        json.dump(items, f, indent=2)

    return {"ok": True, "index": body.index, "action": body.action}


@router.get("/export")
async def export_crossings(request: Request, run: str = Query(...)):
    """Export corrected crossings CSV for a run."""
    runs_root = _runs_dir(request)
    run_dir = runs_root / run
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run}")

    crossing_files = list(run_dir.glob("*_crossings.csv"))
    if not crossing_files:
        raise HTTPException(status_code=404, detail="No crossings file")

    # Read original crossings
    crossing_path = crossing_files[0]
    rows = []
    with open(crossing_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    # Apply review corrections if available
    review_files = list(run_dir.glob("*_review_queue.json"))
    if review_files:
        with open(review_files[0]) as f:
            review_items = json.load(f)

        corrections = {}
        for item in review_items:
            action = item.get("review_action")
            if action == "correct" and item.get("corrected_bib"):
                seq = item.get("sequence") or item.get("frame_number")
                if seq is not None:
                    corrections[str(seq)] = item["corrected_bib"]
            elif action == "reject":
                seq = item.get("sequence") or item.get("frame_number")
                if seq is not None:
                    corrections[str(seq)] = "__REJECT__"

        if corrections:
            filtered = []
            for row in rows:
                seq = row.get("sequence", "")
                if seq in corrections:
                    if corrections[seq] == "__REJECT__":
                        continue  # skip rejected
                    row["bib_number"] = corrections[seq]
                filtered.append(row)
            rows = filtered

    # Build CSV response
    output = StringIO()
    if rows and fieldnames:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={run}_crossings_corrected.csv"},
    )
