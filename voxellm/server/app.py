
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import sys

# Ensure voxellm is in path if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from voxellm.core.engine import VoxelPathProductionEngine
from voxellm.server.models import SystemStatus, HealthResponse

# Import routers
from voxellm.server.routers import training, generation, model_management, rag

app = FastAPI(
    title="VoxelLM API",
    version="2.0.0",
    description="Production-grade API for VoxelLM language model"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. In prod, use strict list.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine Instance
engine: VoxelPathProductionEngine = None

@app.on_event("startup")
async def startup_event():
    global engine
    # Initialize engine
    engine = VoxelPathProductionEngine(model_dir="voxel_models")
    
    # Set engine reference in routers
    training.set_engine(engine)
    generation.set_engine(engine)
    model_management.set_engine(engine)
    rag.set_engine(engine)
    
    print("[SERVER] VoxelLM Engine Initialized")
    print("[SERVER] API Documentation: http://localhost:8000/docs")

# Include routers
app.include_router(training.router)
app.include_router(generation.router)
app.include_router(model_management.router)
app.include_router(rag.router)

# ============================================================================
# Legacy/System Endpoints (kept for backward compatibility)
# ============================================================================

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    if not engine:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    total_ngrams = sum(engine.ngrams_learned.values())
    uptime = time.time() - engine.start_time
    
    return SystemStatus(
        is_training=training.training_state["is_training"],
        ngrams_learned=total_ngrams,
        memory_usage_mb=engine.memory_usage_mb,
        uptime_seconds=uptime
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {}
    
    # Check engine
    if engine:
        components["engine"] = "ok"
    else:
        components["engine"] = "not_initialized"
    
    # Check storage
    try:
        if engine and engine.storage:
            components["storage"] = "ok"
        else:
            components["storage"] = "not_initialized"
    except:
        components["storage"] = "error"
    
    # Check RAG
    if engine and hasattr(engine, 'rag') and engine.rag:
        components["rag"] = "ok"
    else:
        components["rag"] = "not_initialized"
    
    # Determine overall status
    if all(v == "ok" for v in [components.get("engine"), components.get("storage")]):
        status = "healthy"
    elif components.get("engine") == "ok":
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        components=components
    )

@app.get("/")
async def root():
    """Root endpoint for API info"""
    return {
        "name": "VoxelLM API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "training": "/training/*",
            "generation": "/generation/*",
            "models": "/models/*",
            "rag": "/rag/*",
            "system": "/status, /health"
        }
    }

