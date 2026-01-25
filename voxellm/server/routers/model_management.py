"""
Model Management Router for VoxelLM API
Handles model saving, loading, exporting, importing, and listing
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import json
import time
from datetime import datetime

from ...core.engine import VoxelPathProductionEngine
from ..models import (
    ModelListResponse, ModelInfo,
    ModelSaveRequest, ModelSaveResponse,
    ModelLoadedInfo
)

router = APIRouter(prefix="/models", tags=["models"])

# Global engine reference
engine: VoxelPathProductionEngine = None

def set_engine(eng: VoxelPathProductionEngine):
    global engine
    engine = eng

@router.get("/list", response_model=ModelListResponse)
async def list_models():
    """List all saved models"""
    if not engine or not os.path.exists(engine.model_dir):
        return ModelListResponse(models=[])
    
    models = []
    for item in os.listdir(engine.model_dir):
        path = os.path.join(engine.model_dir, item)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.json")):
            stat = os.stat(path)
            
            # Read metadata
            meta = {}
            try:
                with open(os.path.join(path, "metadata.json"), 'r') as f:
                    meta = json.load(f)
            except: 
                pass
            
            # Calculate directory size
            size_mb = sum(os.path.getsize(os.path.join(path, f)) 
                         for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) / (1024 * 1024)
            
            models.append(ModelInfo(
                name=item,
                size_mb=round(size_mb, 2),
                created=time.ctime(stat.st_ctime),
                metrics=meta
            ))
    
    return ModelListResponse(models=models)

@router.post("/save", response_model=ModelSaveResponse)
async def save_model(req: ModelSaveRequest):
    """Save current model to disk"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    model_path = os.path.join(engine.model_dir, req.name)
    
    # Check if already exists
    if os.path.exists(model_path):
        raise HTTPException(status_code=409, detail="Model name already exists")
    
    # Save with metadata
    metadata = req.metadata or {}
    metadata.update({
        "saved_at": datetime.now().isoformat(),
        "ngrams_total": sum(engine.ngrams_learned.values()),
        "tokens_processed": engine.tokens_processed
    })
    
    success = engine.save_model(model_path, metadata=metadata)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save model")
    
    return ModelSaveResponse(
        status="saved",
        path=model_path
    )

@router.post("/load/{model_name}")
async def load_model(model_name: str):
    """Load a saved model"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    model_path = os.path.join(engine.model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    success = engine.load_model(model_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")
    
    return {"status": "loaded", "model": model_name}

@router.delete("/{model_name}")
async def delete_model(model_name: str):
    """Delete a saved model"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    model_path = os.path.join(engine.model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        shutil.rmtree(model_path)
        return {"status": "deleted", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loaded", response_model=list[ModelLoadedInfo])
async def get_loaded_models():
    """Get currently loaded models in memory"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Return info about current in-memory model
    ngrams_count = sum(len(nodes) for nodes in engine.ngram_memory.values())
    
    return [ModelLoadedInfo(
        name="current_session",
        ngrams_count=ngrams_count,
        memory_mb=engine.memory_usage_mb,
        loaded_at=datetime.now().isoformat()
    )]

@router.get("/export/{model_name}")
async def export_model(model_name: str):
    """Export model as downloadable file"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    model_path = os.path.join(engine.model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create zip archive
    archive_path = f"{model_path}.zip"
    shutil.make_archive(model_path, 'zip', model_path)
    
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"{model_name}.zip"
    )

@router.post("/import")
async def import_model(file: UploadFile = File(...)):
    """Import model from uploaded file"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Save uploaded file temporarily
    temp_path = os.path.join(engine.model_dir, f"temp_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract if it's a zip
    if file.filename.endswith(".zip"):
        model_name = file.filename[:-4]  # Remove .zip
        extract_path = os.path.join(engine.model_dir, model_name)
        
        shutil.unpack_archive(temp_path, extract_path)
        os.remove(temp_path)
        
        return {"status": "imported", "model": model_name}
    
    raise HTTPException(status_code=400, detail="Invalid file format. Expected .zip")
