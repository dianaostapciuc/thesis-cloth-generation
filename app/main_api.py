from fastapi import FastAPI
from app.endpoints import garment, body_obj

app = FastAPI(title="Garment Fitting API")
app.include_router(garment.router, prefix="/garment")
app.include_router(body_obj.router, prefix="/body")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
