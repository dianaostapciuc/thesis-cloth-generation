from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import os.path as osp

from app.models      import GarmentRequest
from services        import garment_service
from utils.streaming import iterfile

router = APIRouter()

@router.post("/generate")
async def generate_garment(request: GarmentRequest):
    try:
        zip_path = garment_service.generate_garment(request)
    except Exception as e:
        raise HTTPException(500, str(e))

    return StreamingResponse(
        iterfile(zip_path),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{osp.basename(zip_path)}"',
            "Content-Encoding":   "identity",
        }
    )

@router.post("/export_json")
async def export_skinning_json(request: GarmentRequest):
    """
    Stream back a ZIP containing the two skinning JSONs
    (body_<gender>_skin.json and <garment>_<gender>_skin.json).
    """
    try:
        zip_path = garment_service.export_skinning_json(request)
    except Exception as e:
        raise HTTPException(500, str(e))

    return StreamingResponse(
        iterfile(zip_path),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{osp.basename(zip_path)}"',
            "Content-Encoding":   "identity",
        }
    )
