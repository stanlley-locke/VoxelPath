
import os
import uuid
import json
import numpy as np
from typing import List, Dict, Optional
import time

class SimpleVectorStore:
    """
    Lightweight, pure-Python vector store using Numpy and JSON.
    Fallback for when ChromaDB cannot be installed (missing C++ tools).
    """
    def __init__(self, persistence_dir: str):
        self.persistence_dir = persistence_dir
        os.makedirs(persistence_dir, exist_ok=True)
        self.data_path = os.path.join(persistence_dir, "data.json")
        self.vector_path = os.path.join(persistence_dir, "vectors.npy")
        
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.vectors = None
        self.model = None
        
        self._load()
        self._init_model()
        
    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[RAG] SimpleVectorStore: Embedding model loaded")
        except ImportError:
            print("[RAG] sentence-transformers not found. RAG disabled.")
            self.model = None

    def _load(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get('documents', [])
                self.metadatas = data.get('metadatas', [])
                self.ids = data.get('ids', [])
                
        if os.path.exists(self.vector_path):
            self.vectors = np.load(self.vector_path)

    def _save(self):
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids
            }, f)
        
        if self.vectors is not None:
            np.save(self.vector_path, self.vectors)

    def add(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        if not self.model:
            return

        # Encode new documents
        new_vectors = self.model.encode(documents)
        
        # Update lists
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Update vectors
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
            
        self._save()
        print(f"[RAG] SimpleVectorStore: Persisted {len(documents)} documents")

    def query(self, query_texts: List[str], n_results: int = 3) -> Dict:
        if not self.model or self.vectors is None or len(self.documents) == 0:
            return {'documents': []}

        # Encode query
        query_vec = self.model.encode(query_texts[0])
        
        # Calculate cosine similarity
        # A . B / (|A| * |B|)
        norm_q = np.linalg.norm(query_vec)
        norm_v = np.linalg.norm(self.vectors, axis=1)
        
        scores = np.dot(self.vectors, query_vec) / (norm_v * norm_q)
        
        # Get top N indices
        top_indices = np.argsort(scores)[-n_results:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(self.documents[idx])
            
        return {'documents': [results]}
        
    def count(self) -> int:
        return len(self.documents)

class VectorStore:
    """
    Wrapper for Vector Database functionalities (RAG).
    Defaults to ChromaDB if available, otherwise falls back to SimpleVectorStore.
    """
    
    def __init__(self, collection_name: str = "voxel_knowledge", persistence_dir: str = "voxel_rag_db"):
        self.collection_name = collection_name
        self.persistence_dir = persistence_dir
        self.client = None
        self.collection = None
        self.mode = "disabled" # disabled, chroma, simple
        
        self._init_db()
        
    def _init_db(self):
        # 1. Try ChromaDB
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persistence_dir)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self.mode = "chroma"
            print(f"[RAG] ChromaDB initialized at {self.persistence_dir}")
            return
        except (ImportError, Exception) as e:
            print(f"[RAG] ChromaDB failed ({e}). Trying fallback...")

        # 2. Try SimpleVectorStore
        try:
            self.collection = SimpleVectorStore(persistence_dir=os.path.join(self.persistence_dir, "simple_store"))
            if self.collection.model:
                self.mode = "simple"
                print(f"[RAG] SimpleVectorStore initialized (Numpy Fallback)")
                return
        except Exception as e:
            print(f"[RAG] SimpleVectorStore failed: {e}")

        # 3. Give up
        print("[RAG] All RAG backends failed. Features disabled.")
        print("Please install: pip install sentence-transformers numpy")
            
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add text documents to the vector store"""
        if self.mode == "disabled":
            return
            
        if not documents:
            return
            
        ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{"source": "manual"} for _ in documents]
            
        try:
            if self.mode == "chroma":
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            elif self.mode == "simple":
                self.collection.add(documents, metadatas, ids)
                
            print(f"[RAG] Added {len(documents)} documents to knowledge base")
        except Exception as e:
            print(f"[RAG] Add failed: {e}")
        
    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant documents for a query"""
        if self.mode == "disabled":
            return []
            
        try:
            if self.mode == "chroma":
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
            elif self.mode == "simple":
                results = self.collection.query([query_text], n_results)
            
            # Standardize output
            if results and results['documents']:
                return results['documents'][0]
        except Exception as e:
            print(f"[RAG] Query failed: {e}")
            
        return []
        
    def count(self) -> int:
        if self.mode == "disabled":
            return 0
        return self.collection.count()

class DocumentProcessor:
    """Utilities to chunk and prepare text for RAG"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        if not text:
            return chunks
            
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Try to find a sentence break near the end
            if end < text_len:
                # Look back for punctuation
                lookback = text[start:end].rfind('.')
                if lookback != -1 and lookback > chunk_size * 0.5:
                    end = start + lookback + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
