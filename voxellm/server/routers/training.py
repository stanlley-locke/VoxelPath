"""
Training Router for VoxelLM API
Handles training with configuration, progress tracking, and cancellation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import time

from ...core.engine import VoxelPathProductionEngine
from ..models import (
    TrainingRequest, TrainingResponse,
    TrainingStatus
)

router = APIRouter(prefix="/training", tags=["training"])

# Global references (will be set by main app)
engine: VoxelPathProductionEngine = None
training_state: Dict = {
    "is_training": False,
    "progress": 0.0,
    "current_batch": 0,
    "total_batches": 0,
    "metrics": {}
}

def set_engine(eng: VoxelPathProductionEngine):
    global engine
    engine = eng

@router.post("/start", response_model=TrainingResponse)
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training with optional configuration"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if training_state["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    # Extract config or use defaults
    config = req.config.dict() if req.config else {}
    batch_size = config.get("batch_size", 10000)
    
    # Mark as training
    training_state["is_training"] = True
    training_state["progress"] = 0.0
    
    # Start background training
    background_tasks.add_task(
        run_training_task,
        req.corpus,
        batch_size,
        req.validation_corpus
    )
    
    return TrainingResponse(
        status="Training started in background",
        metrics={}
    )

def run_training_task(corpus: str, batch_size: int, validation_corpus: str = None):
    """Background training task"""
    global training_state
    try:
        print(f"[SERVER] Starting training on {len(corpus)} chars")
        
        # Train
        metrics = engine.train_corpus(
            corpus,
            batch_size=batch_size,
            validation_text=validation_corpus
        )
        
        training_state["metrics"] = metrics.to_dict() if hasattr(metrics, 'to_dict') else {}
        training_state["progress"] = 100.0
        
        print("[SERVER] Training complete")
    except Exception as e:
        print(f"[SERVER] Training error: {e}")
        training_state["metrics"] = {"error": str(e)}
    finally:
        training_state["is_training"] = False

@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return TrainingStatus(
        is_training=training_state["is_training"],
        progress_percent=training_state["progress"],
        current_batch=training_state["current_batch"],
        total_batches=training_state["total_batches"],
        metrics=training_state["metrics"]
    )

@router.post("/cancel")
async def cancel_training():
    """Cancel ongoing training"""
    if not training_state["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    # Set flag (engine needs to check this during training)
    engine.is_training = False
    training_state["is_training"] = False
    
    return {"status": "Training cancelled"}
