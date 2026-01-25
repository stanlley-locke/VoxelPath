"""
Generation Router for VoxelLM API
Handles text generation, chat, batch generation, and text analysis
"""

from fastapi import APIRouter, HTTPException
from typing import List
import time

from ...core.engine import VoxelPathProductionEngine
from ..models import (
    GenerationRequest, GenerationResponse,
    ChatRequest, ChatResponse, ChatMessage,
    BatchGenerationRequest, BatchGenerationResponse,
    TextAnalysisRequest, TextAnalysisResponse
)

router = APIRouter(prefix="/generation", tags=["generation"])

# Global engine reference
engine: VoxelPathProductionEngine = None

def set_engine(eng: VoxelPathProductionEngine):
    global engine
    engine = eng

@router.post("/generate", response_model=GenerationResponse)
async def generate(req: GenerationRequest):
    """Generate text from a prompt"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    generated_text, trajectory, metrics = engine.generate_text(
        seed_text=req.prompt,
        max_length=req.max_length,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop_sequences=req.stop_sequences,
        use_momentum=req.use_momentum
    )
    
    return GenerationResponse(
        text=generated_text,
        metrics=metrics.to_dict(),
        trajectory=trajectory
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat mode generation with conversation history"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Build conversation context
    conversation = []
    for msg in req.messages:
        prefix = "User:" if msg.role == "user" else "AI:"
        conversation.append(f"{prefix} {msg.content}")
    
    # Add AI prefix for response
    prompt = "\\n".join(conversation) + "\\nAI:"
    
    # Generate response
    generated_text, trajectory, metrics = engine.generate_text(
        seed_text=prompt,
        max_length=req.max_length,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop_sequences=["\\nUser:", "\\nAI:"],
        use_momentum=req.use_momentum
    )
    
    # Extract AI response (remove prompt)
    response_text = generated_text[len(prompt):].strip()
    
    return ChatResponse(
        message=ChatMessage(role="assistant", content=response_text),
        metrics=metrics.to_dict()
    )

@router.post("/batch", response_model=BatchGenerationResponse)
async def batch_generate(req: BatchGenerationRequest):
    """Generate text for multiple prompts"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    results = []
    for prompt in req.prompts:
        try:
            generated_text, trajectory, metrics = engine.generate_text(
                seed_text=prompt,
                max_length=req.max_length,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty
            )
            
            results.append({
                "prompt": prompt,
                "text": generated_text,
                "metrics": metrics.to_dict(),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "prompt": prompt,
                "text": "",
                "metrics": {},
                "status": "error",
                "error": str(e)
            })
    
    return BatchGenerationResponse(results=results)

@router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(req: TextAnalysisRequest):
    """Analyze text quality metrics"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Tokenize
    tokens = engine.tokenizer.tokenize(req.text)
    
    # Get analysis
    analysis = engine.tokenizer.analyze_text(tokens)
    
    return TextAnalysisResponse(
        coherence_score=analysis.coherence_score,
        diversity_score=analysis.diversity_score,
        perplexity=analysis.perplexity,
        unique_tokens=analysis.unique_tokens,
        total_tokens=analysis.total_tokens,
        avg_token_length=analysis.avg_token_length
    )
