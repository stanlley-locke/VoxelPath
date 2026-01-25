"""
RAG (Retrieval-Augmented Generation) Router for VoxelLM API
Handles document ingestion, querying, and knowledge base management
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import os

from ...core.engine import VoxelPathProductionEngine
from ..models import (
    RAGIngestRequest, RAGResponse,
    RAGQueryRequest, RAGQueryResponse,
    RAGDocumentsResponse, RAGDocument,
    RAGStatsResponse
)

router = APIRouter(prefix="/rag", tags=["rag"])

# Global engine reference
engine: VoxelPathProductionEngine = None

def set_engine(eng: VoxelPathProductionEngine):
    global engine
    engine = eng

@router.post("/ingest", response_model=RAGResponse)
async def ingest_document(req: RAGIngestRequest):
    """Ingest a document into the RAG knowledge base"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Ingest content
        engine.learn_document(req.content, source_name=req.filename)
        
        # Get document count
        total_docs = 0
        if hasattr(engine, 'rag') and hasattr(engine.rag, 'vector_store'):
            if hasattr(engine.rag.vector_store, 'documents'):
                total_docs = len(engine.rag.vector_store.documents)
        
        return RAGResponse(
            status="success",
            chunks_added=1,
            total_documents=total_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(req: RAGQueryRequest):
    """Query the RAG knowledge  base for relevant context"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not hasattr(engine, 'rag') or not engine.rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Query vector store
        results = engine.rag.query(req.query, top_k=req.top_k)
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.get("content", ""),
                "score": doc.get("score", 0.0),
                "source": doc.get("source", "unknown")
            })
        
        return RAGQueryResponse(results=formatted_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=RAGDocumentsResponse)
async def list_documents():
    """List all documents in the knowledge base"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not hasattr(engine, 'rag') or not engine.rag:
        return RAGDocumentsResponse(documents=[])
    
    documents = []
    if hasattr(engine.rag.vector_store, 'documents'):
        for idx, doc in enumerate(engine.rag.vector_store.documents):
            documents.append(RAGDocument(
                id=str(idx),
                source=doc.get("source", "unknown"),
                content=doc.get("content", "")[:200] + "...",  # Truncate
                created=doc.get("created", datetime.now().isoformat())
            ))
    
    return RAGDocumentsResponse(documents=documents)

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not hasattr(engine, 'rag') or not engine.rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Try to delete (implementation depends on vector store)
        doc_idx = int(doc_id)
        if hasattr(engine.rag.vector_store, 'documents'):
            if 0 <= doc_idx < len(engine.rag.vector_store.documents):
                del engine.rag.vector_store.documents[doc_idx]
                return {"status": "deleted", "doc_id": doc_id}
        
        raise HTTPException(status_code=404, detail="Document not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Get RAG system statistics"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not hasattr(engine, 'rag') or not engine.rag:
        return RAGStatsResponse(
            total_documents=0,
            total_vectors=0,
            storage_type="none"
        )
    
    # Get stats
    total_docs = 0
    total_vectors = 0
    storage_type = "simple"
    
    if hasattr(engine.rag.vector_store, 'documents'):
        total_docs = len(engine.rag.vector_store.documents)
        total_vectors = total_docs  # Simplified
    
    if hasattr(engine.rag.vector_store, '__class__'):
        storage_type = engine.rag.vector_store.__class__.__name__.lower()
    
    return RAGStatsResponse(
        total_documents=total_docs,
        total_vectors=total_vectors,
        storage_type=storage_type
    )
