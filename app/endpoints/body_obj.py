# app/endpoints/body_obj.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import os.path as osp

from config.config        import config
from utils.mesh_utils     import compute_body_obj
from utils.streaming      import iterfile

router = APIRouter()

@router.post("/compute_obj")
async def compute_obj(gender: str, betas: List[float]):
    if len(betas) < 2:
        raise HTTPException(400, "At least 2 beta values are required.")

    out_dir = config.get("paths.output_dir")
    try:
        obj_path, _, _ = compute_body_obj(gender, betas, out_dir)
    except Exception as e:
        raise HTTPException(500, str(e))

    return StreamingResponse(
        iterfile(obj_path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{osp.basename(obj_path)}"',
            "Content-Encoding":   "identity", # ensure no gzip
        }
    )
