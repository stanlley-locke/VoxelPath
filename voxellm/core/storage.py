import sqlite3
import json
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List, Any
from collections import defaultdict
import threading

class NGramStorage(ABC):
    """Abstract base class for N-Gram storage backends"""
    
    @abstractmethod
    def add_ngram(self, context: Tuple[str, ...], level: int, vector_pos: Optional[Tuple[int, int, int]] = None):
        """Register an N-Gram context"""
        pass
        
    @abstractmethod
    def add_transition(self, context: Tuple[str, ...], next_word: str, weight: float = 1.0):
        """Add a transition observation"""
        pass
        
    @abstractmethod
    def get_most_common_words(self, limit: int = 10) -> Dict[str, float]:
        """Get globally most frequent words for fallback"""
        pass
        
    @abstractmethod
    def get_ngram(self, context: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """Retrieve N-Gram data including next_words"""
        pass
        
    @abstractmethod
    def get_all_ngrams(self, level: int) -> Dict[Tuple[str, ...], Any]:
        """Retrieve all N-Grams for a specific level (Use with caution on large DBs)"""
        pass
        
    @abstractmethod
    def commit(self):
        """Commit changes to persistence"""
        pass
        
    @abstractmethod
    def close(self):
        """Close connection"""
        pass

from .structures import NGramNode

class SQLiteStorage(NGramStorage):
    """Production-grade SQLite backend for N-Gram memory with LRU Caching"""
    
    def __init__(self, db_path: str = "voxel_brain.db", cache_size: int = 50000):
        self.db_path = db_path
        self.cache_size = cache_size
        self._cache = {} # Simple dict for now, or OrderedDict for true LRU
        self._cache_order = [] # Queue for LRU eviction
        self.local = threading.local()
        self._init_db()
        
    def _get_conn(self):
        """Get thread-local connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.execute("PRAGMA journal_mode=WAL")
            self.local.conn.execute("PRAGMA synchronous=NORMAL")
            self.local.conn.execute("PRAGMA foreign_keys=ON") # Enforce constraints
        return self.local.conn

    def _update_cache(self, context: Tuple[str, ...], node: NGramNode):
        """Update cache with LRU eviction"""
        if context in self._cache:
            self._cache_order.remove(context)
        elif len(self._cache) >= self.cache_size:
            # Evict oldest
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
            
        self._cache[context] = node
        self._cache_order.append(context)

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # N-Grams Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ngrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_hash TEXT UNIQUE NOT NULL, -- Faster lookups
                context_json TEXT NOT NULL,        -- Actual content
                level INTEGER NOT NULL,
                total_count REAL DEFAULT 0,
                vector_x INTEGER,
                vector_y INTEGER,
                vector_z INTEGER,
                last_updated TIMESTAMP
            )
        """)
        
        # Transitions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transitions (
                ngram_id INTEGER,
                next_word TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY(ngram_id) REFERENCES ngrams(id) ON DELETE CASCADE,
                PRIMARY KEY (ngram_id, next_word)
            )
        """)
        
        # Indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ngrams_hash ON ngrams(context_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ngrams_level ON ngrams(level)")
        
        conn.commit()
        conn.close()
        
    def _hash_context(self, context: Tuple[str, ...]) -> str:
        """Create a consistent hash for the context"""
        return "::".join(context)
        
    def add_ngram(self, context: Tuple[str, ...], level: int, vector_pos: Optional[Tuple[int, int, int]] = None):
        # Update cache if exists
        node = self.get_ngram(context, use_cache=True)
        if node:
            if vector_pos:
                node.vector_position = vector_pos
            return # Already tracking changes in object, will flush? 
            # Actually, this add_ngram is usually called for new nodes.
            # In SQLite, "adding" a node just ensures the row exists.
            
        conn = self._get_conn()
        cursor = conn.cursor()
        ctx_hash = self._hash_context(context)
        
        cursor.execute("""
            INSERT OR IGNORE INTO ngrams (context_hash, context_json, level, last_updated)
            VALUES (?, ?, ?, ?)
        """, (ctx_hash, json.dumps(context), level, datetime.datetime.now()))
        
        if vector_pos:
            cursor.execute("""
                UPDATE ngrams SET vector_x=?, vector_y=?, vector_z=? WHERE context_hash=?
            """, (vector_pos[0], vector_pos[1], vector_pos[2], ctx_hash))
            
    def add_transition(self, context: Tuple[str, ...], next_word: str, weight: float = 1.0):
        # Hot-path optimization: Update in-memory cache object first
        node = self.get_ngram(context, use_cache=True)
        if node:
            node.add_transition(next_word, weight)
            # We must eventually persist this. 
            # For now, let's write-through to DB to be safe, 
            # but in high-load we might want a "dirty" flag.
            # Production Grade: Write-through for data safety for now.
        
        conn = self._get_conn()
        cursor = conn.cursor()
        ctx_hash = self._hash_context(context)
        
        # Ensure N-Gram exists (Upsert logic roughly)
        cursor.execute("SELECT id FROM ngrams WHERE context_hash=?", (ctx_hash,))
        row = cursor.fetchone()
        
        if not row:
            self.add_ngram(context, len(context) + 1)
            cursor.execute("SELECT id FROM ngrams WHERE context_hash=?", (ctx_hash,))
            row = cursor.fetchone()
            
        ngram_id = row[0]
        
        # Update/Insert Transition
        cursor.execute("""
            INSERT INTO transitions (ngram_id, next_word, weight)
            VALUES (?, ?, ?)
            ON CONFLICT(ngram_id, next_word) 
            DO UPDATE SET weight = weight + ?
        """, (ngram_id, next_word, weight, weight))
        
        # Update total count in parent
        cursor.execute("""
            UPDATE ngrams 
            SET total_count = total_count + ?, last_updated = ?
            WHERE id = ?
        """, (weight, datetime.datetime.now(), ngram_id))
        
    def get_ngram(self, context: Tuple[str, ...], use_cache=True) -> Optional[NGramNode]:
        if use_cache and context in self._cache:
            return self._cache[context]
            
        conn = self._get_conn()
        cursor = conn.cursor()
        ctx_hash = self._hash_context(context)
        
        cursor.execute("""
            SELECT id, total_count, vector_x, vector_y, vector_z, last_updated 
            FROM ngrams WHERE context_hash=?
        """, (ctx_hash,))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        ngram_id, total_count, vx, vy, vz, last_updated = row
        
        # Get transitions
        cursor.execute("SELECT next_word, weight FROM transitions WHERE ngram_id=?", (ngram_id,))
        transitions = {word: weight for word, weight in cursor.fetchall()}
        
        # Reconstruct NGramNode
        node = NGramNode(context=context)
        node.next_words = defaultdict(float, transitions)
        node.total_count = total_count
        node.vector_position = (vx, vy, vz) if vx is not None else None
        
        # Parse timestamp if needed, but ignoring for now
        
        if use_cache:
            self._update_cache(context, node)
            
        return node
        
    def get_all_ngrams(self, level: int) -> Dict[Tuple[str, ...], NGramNode]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, context_json, total_count, vector_x, vector_y, vector_z FROM ngrams WHERE level=?", (level,))
        results = {}
        
        for row in cursor.fetchall():
            ngram_id, ctx_json, total, vx, vy, vz = row
            context = tuple(json.loads(ctx_json))
            
            sub_cursor = conn.cursor()
            sub_cursor.execute("SELECT next_word, weight FROM transitions WHERE ngram_id=?", (ngram_id,))
            transitions = {word: weight for word, weight in sub_cursor.fetchall()}
            
            node = NGramNode(context=context)
            node.next_words = defaultdict(float, transitions)
            node.total_count = total
            node.vector_position = (vx, vy, vz) if vx is not None else None
            
            results[context] = node
            sub_cursor.close()
            
        return results
        
    def get_most_common_words(self, limit: int = 10) -> Dict[str, float]:
        """Optimization: SQL aggregation instead of iterating all objects"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT next_word, SUM(weight) as total_freq 
            FROM transitions 
            GROUP BY next_word 
            ORDER BY total_freq DESC 
            LIMIT ?
        """, (limit,))
        
        results = {word: freq for word, freq in cursor.fetchall()}
        
        # Normalize
        total = sum(results.values())
        if total > 0:
            return {w: f/total for w, f in results.items()}
        return {}
        
    
    def batch_add_transitions(self, transitions: List[Tuple[Tuple[str, ...], str, float]]):
        """Batch insert transitions for performance"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # We need to ensure parent N-grams exist first
        # Extract unique contexts
        contexts = {t[0] for t in transitions}
        
        # 1. Ensure N-Grams exist
        for context in contexts:
            ctx_hash = self._hash_context(context)
            cursor.execute("""
                INSERT OR IGNORE INTO ngrams (context_hash, context_json, level, last_updated)
                VALUES (?, ?, ?, ?)
            """, (ctx_hash, json.dumps(context), len(context), datetime.datetime.now()))
            
        # 2. Add Transitions
        # Get IDs map
        cursor.execute("SELECT context_hash, id FROM ngrams WHERE context_hash IN ({})".format(
            ','.join(['?']*len(contexts))
        ), [self._hash_context(c) for c in contexts])
        
        hash_to_id = {row[0]: row[1] for row in cursor.fetchall()}
        
        batch_data = []
        for context, next_word, weight in transitions:
            ctx_hash = self._hash_context(context)
            ngram_id = hash_to_id.get(ctx_hash)
            if ngram_id:
                batch_data.append((ngram_id, next_word, weight, weight))
                
        cursor.executemany("""
            INSERT INTO transitions (ngram_id, next_word, weight)
            VALUES (?, ?, ?)
            ON CONFLICT(ngram_id, next_word) 
            DO UPDATE SET weight = weight + ?
        """, batch_data)
        
        conn.commit()

    def commit(self):
        conn = self._get_conn()
        conn.commit()
        
    def close(self):
        if hasattr(self.local, 'conn'):
            self.local.conn.close()
            del self.local.conn

