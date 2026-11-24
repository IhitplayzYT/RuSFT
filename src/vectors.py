import numpy as np

import random
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import time
from loguru import logger
from collections import defaultdict

@dataclass
class VectorEntry:
    id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = 0.0

class TopicVectorDB:
    def __init__(self,topic_name: str,embed_dim: int = 768,db_path: Optional[str] = None,idx_type: str = "flat"):
        self.topic_name = topic_name
        self.embed_dim = embed_dim
        self.db_path = Path(db_path) if db_path else None
        self.idx_type = idx_type
        self.index = None
        self.entries: Dict[str, VectorEntry] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.next_id = 0
        self._init_index()
        if self.db_path and self.db_path.exists():  self.load()
    
    def _init_index(self):
        if self.idx_type == "flat": self.index = faiss.IndexFlatL2(self.embed_dim)
        elif self.idx_type == "ivf": self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.embed_dim), self.embed_dim, 100)
        elif self.idx_type == "hnsw": self.index = faiss.IndexHNSWFlat(self.embed_dim, 32)
        else: self.index = faiss.IndexFlatL2(self.embed_dim)
        logger.info(f"Inited {self.idx_type} index for topic {self.topic_name}")
    
    def add(self,vector: np.ndarray,text: str,metadata: Optional[Dict[str, Any]] = None,entry_id: Optional[str] = None) -> str:
        if entry_id is None:
            entry_id = f"{self.topic_name}_{self.next_id}"
            self.next_id += 1
        vector = vector.astype(np.float32).reshape(1, -1)
        self.entries[entry_id] = VectorEntry(entry_id,vector[0],text,metadata or {},0,0.0)
        self.id_to_index[entry_id] = self.index.ntotal
        self.index.add(vector)
        return entry_id
    
    def search(self,query_vector: np.ndarray,k: int = 5,update_access: bool = True) -> List[Tuple[VectorEntry, float]]:
        dists, indices = self.index.search(query_vector.astype(np.float32).reshape(1, -1), k)
        ret = []
        for dist, idx in zip(dists[0], indices[0]):
            if idx >= 0 and idx < self.index.ntotal:
                entry_id = None
                for eid, eidx in self.id_to_index.items():
                    if eidx == idx:
                        entry_id = eid
                        break
                if entry_id and entry_id in self.entries:
                    entry = self.entries[entry_id]
                    if update_access:
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                    ret.append((entry, float(dist)))
        return ret
    
    def get_entry(self, entry_id: str) -> Optional[VectorEntry]: return self.entries.get(entry_id)
    
    def get_all_entries(self) -> List[VectorEntry]: return list(self.entries.values())
    
    def get_frequent_entries(self,threshold: int = 5,limit: Optional[int] = None) -> List[VectorEntry]:
        f = [entry for entry in self.entries.values() if entry.access_count >= threshold]
        f.sort(key=lambda x: x.access_count, reverse=True)
        if limit: f = f[:limit]
        return f 
    
    def delete(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            del self.entries[entry_id]
            if entry_id in self.id_to_index: del self.id_to_index[entry_id]
            return True
        return False
    
    def save(self, save_path: Optional[str] = None):
        save_path = Path(save_path) if save_path else self.db_path
        if not save_path: raise ValueError("No save path provided")
        save_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        data = {
            "entries": [ {"id": entry.id,"vector": entry.vector.tolist(),"text": entry.text,"metadata": entry.metadata,"access_count": entry.access_count,"last_accessed": entry.last_accessed} for entry in self.entries.values() ],
            "id_to_index": self.id_to_index,
            "next_id": self.next_id
        }
        with open(save_path / "entries.pkl", "wb") as f: pickle.dump(data, f)
        logger.info(f"Saved VDB Topic: {self.topic_name} To: {save_path}")
    
    def load(self, load_path: Optional[str] = None):
        load_path = Path(load_path) if load_path else self.db_path
        if not load_path or not load_path.exists(): raise ValueError(f"Load path not exist: {load_path}")
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        with open(load_path / "entries.pkl", "rb") as f: entries_data = pickle.load(f)
        self.entries = {}
        for entry_data in entries_data["entries"]:
            self.entries[entry_data["id"]] = VectorEntry(id=entry_data["id"],vector=np.array(entry_data["vector"], dtype=np.float32),text=entry_data["text"],metadata=entry_data["metadata"],access_count=entry_data["access_count"],last_accessed=entry_data["last_accessed"])
        self.id_to_index = entries_data["id_to_index"]
        self.next_id = entries_data["next_id"]
        logger.info(f"Loaded VDB Topic: {self.topic_name} To: {load_path}")


class RouterVectorDB:
    """
    Special vector database for the router containing summarized vectors
    from expert databases
    """
    
    def __init__(self,embed_dim: int = 768,db_path: Optional[str] = None,promotion_threshold: int = 10,promotion_window: int = 100,max_entries: int = 10000):
        self.embed_dim = embed_dim
        self.db_path = Path(db_path) if db_path else None
        self.promotion_threshold = promotion_threshold
        self.promotion_window = promotion_window
        self.max_entries = max_entries
        self.index = faiss.IndexFlatL2(embed_dim)
        self.entries: Dict[str, VectorEntry] = {}
        self.id_to_index: Dict[str, int] = {}
        self.next_id = 0
        self.promotion_candidates: Dict[str, List[VectorEntry]] = defaultdict(list)
        self.promotion_history: List[Dict[str, Any]] = []
        if self.db_path and self.db_path.exists(): self.load()
    
    def add(self,vector: np.ndarray,text: str,metadata: Optional[Dict[str, Any]] = None,entry_id: Optional[str] = None,source_topic: Optional[str] = None) -> str:
        if entry_id is None:
            entry_id = f"router_{self.next_id}"
            self.next_id += 1
        vector = vector.astype(np.float32).reshape(1, -1)
        if metadata is None: metadata = {}
        if source_topic: metadata["source_topic"] = source_topic
        entry = VectorEntry(entry_id,vector[0],text,metadata,0,0.0)
        self.entries[entry_id] = entry
        self.id_to_index[entry_id] = self.index.ntotal
        self.index.add(vector)
        self._evict()
        return entry_id
    
    def _evict(self):
        if len(self.entries) > self.max_entries:
            sorted_entries = sorted(self.entries.values(),key=lambda x: (x.access_count, x.last_accessed))
            num_to_remove = len(self.entries) - self.max_entries
            for entry in sorted_entries[:num_to_remove]: self.delete(entry.id)
    
    def search(self,query_vector: np.ndarray,k: int = 5,update_access: bool = True) -> List[Tuple[VectorEntry, float]]:
        distances, indices = self.index.search( query_vector.astype(np.float32).reshape(1, -1), k)
        rets = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < self.index.ntotal:
                entry_id = None
                for eid, eidx in self.id_to_index.items():
                    if eidx == idx:
                        entry_id = eid
                        break
                if entry_id and entry_id in self.entries:
                    entry = self.entries[entry_id]
                    if update_access:
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                    rets.append((entry, float(dist)))
        return rets
    
    def promote(self,expert_db: TopicVectorDB,num_to_promote: int = 5,summary_method: str = "centroid") -> List[str]:
        """ Promote frequently accessed vectors from expert DB to router DB """
        frequent_entries = expert_db.get_frequent_entries(threshold=self.promotion_threshold,limit=num_to_promote * 2)
        if not frequent_entries: return []
        promoted_ids = []
        if summary_method == "centroid":
            centroid = np.mean(np.array([entry.vector for entry in frequent_entries]), axis=0)
            summary_text = f"Summary of {len(frequent_entries)} frequent queries from {expert_db.topic_name}"
            entry_id = self.add(vector=centroid,text=summary_text,metadata={"type": "summary","source_topic": expert_db.topic_name,"num_entries": len(frequent_entries),"summary_method": "centroid"})
            promoted_ids.append(entry_id)
            
        elif summary_method == "top":
            for entry in frequent_entries[:num_to_promote]:
                entry_id = self.add(vector=entry.vector,text=entry.text,metadata={"type": "individual","source_topic": expert_db.topic_name,"original_access_count": entry.access_count},source_topic=expert_db.topic_name)
                promoted_ids.append(entry_id)
        
        elif summary_method == "random":
            # Randomly sample from frequent entries
            sampled = random.sample(frequent_entries, min(num_to_promote, len(frequent_entries)))
            for entry in sampled:
                entry_id = self.add(vector=entry.vector,text=entry.text,metadata={"type": "individual","source_topic": expert_db.topic_name,"original_access_count": entry.access_count},source_topic=expert_db.topic_name)
                promoted_ids.append(entry_id)
        self.promotion_history.append({"timestamp": time.time(),"source_topic": expert_db.topic_name,"num_promoted": len(promoted_ids),"method": summary_method,"promoted_ids": promoted_ids})
        logger.info(f"Promoted {len(promoted_ids)} entries from {expert_db.topic_name} to router DB using {summary_method} method")
        return promoted_ids
    
    def batch_promote(self,expert_dbs: Dict[str, TopicVectorDB],num_per_expert: int = 3,summary_method: str = "centroid") -> Dict[str, List[str]]:
        ret = {}
        for name, expert in expert_dbs.items(): ret[name] = self.promote(expert,num_per_expert,summary_method)
        return ret
    
    def get_entry(self, entry_id: str) -> Optional[VectorEntry]: return self.entries.get(entry_id)
    
    def delete(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            del self.entries[entry_id]
            if entry_id in self.id_to_index: del self.id_to_index[entry_id]
            return True
        return False
    
    def save(self, save_path: Optional[str] = None):
        save_path = Path(save_path) if save_path else self.db_path
        if not save_path: raise ValueError("No save path provided")
        save_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        entries_data = {
            "entries": [ {"id": entry.id,"vector": entry.vector.tolist(),"text": entry.text,"metadata": entry.metadata,"access_count": entry.access_count,"last_accessed": entry.last_accessed} for entry in self.entries.values() ],
            "id_to_index": self.id_to_index,
            "next_id": self.next_id,
            "promotion_history": self.promotion_history,
        }
        with open(save_path / "entries.pkl", "wb") as f: pickle.dump(entries_data, f)
        logger.info(f"Saved router VDB: {save_path}")
    
    def load(self, load_path: Optional[str] = None):
        load_path = Path(load_path) if load_path else self.db_path
        if not load_path or not load_path.exists(): raise ValueError(f"Load path not exist: {load_path}")
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        with open(load_path / "entries.pkl", "rb") as f: entries_data = pickle.load(f)
        self.entries = {}
        for entry_data in entries_data["entries"]:
            self.entries[entry_data["id"]] = VectorEntry(entry_data["id"],np.array(entry_data["vector"], dtype=np.float32),entry_data["text"],entry_data["metadata"],entry_data["access_count"],entry_data["last_accessed"])
        self.id_to_index = entries_data["id_to_index"]
        self.next_id = entries_data["next_id"]
        self.promotion_history = entries_data.get("promotion_history", [])
        logger.info(f"Loaded router VDB: {load_path}")


class VectorDBWrangler:
    """ Handles topic specific DBs and the router DB with autopromotion """
    
    def __init__(self,topic_names: List[str],embed_dim: int = 768,base_db_path: str = "./data/vector_dbs",embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",router_promotion_threshold: int = 10,router_promotion_window: int = 100):
        self.topic_names = topic_names
        self.embed_dim = embed_dim
        self.base_db_path = Path(base_db_path)
        self.embedding_model = embedding_model
        self.router_promotion_threshold = router_promotion_threshold
        self.router_promotion_window = router_promotion_window
        self.embedder = SentenceTransformer(embedding_model)
        self.topic_dbs: Dict[str, TopicVectorDB] = {}
        for topic in topic_names:
            topic_path = self.base_db_path / topic
            self.topic_dbs[topic] = TopicVectorDB(topic_name=topic,embed_dim=embed_dim,db_path=str(topic_path))
        router_path = self.base_db_path / "router"
        self.router_db = RouterVectorDB(embed_dim,str(router_path),router_promotion_threshold,router_promotion_window)
        logger.info(f"Initialized VDBwrangler with {len(topic_names)} topics")
    
    def embed(self, text: str) -> np.ndarray: return self.embedder.encode(text, convert_to_numpy=True)
    
    def add_topic_db(self,topic: str,text: str,metadata: Optional[Dict[str, Any]] = None,entry_id: Optional[str] = None) -> str:
        if topic not in self.topic_dbs: raise ValueError(f"Unknown topic: {topic}")
        vector = self.embed(text)
        return self.topic_dbs[topic].add(vector,text,metadata,entry_id)
    
    def search_topic_db(self,topic: str,query: str,k: int = 5,update_access: bool = True) -> List[Tuple[VectorEntry, float]]:
        if topic not in self.topic_dbs: raise ValueError(f"Unknown topic: {topic}")
        query_vector = self.embed(query)
        return self.topic_dbs[topic].search(query_vector,k,update_access)
    
    def search_router_db(self,query: str,k: int = 5,update_access: bool = True) -> List[Tuple[VectorEntry, float]]:
        query_vector = self.embed(query)
        return self.router_db.search(query_vector,k,update_access)
    
    def promote(self,topics: Optional[List[str]] = None,num_per_topic: int = 3,summary_method: str = "centroid") -> Dict[str, List[str]]:
        if topics is None: topics = list(self.topic_dbs.keys())
        return self.router_db.batch_promote({topic: self.topic_dbs[topic] for topic in topics},num_per_topic,summary_method)
    
    def autoregress(self,interval: int = 100,num_promotions: int = 5) -> Dict[str, Any]:
        pass

    
    def get_topic_db(self, topic: str) -> TopicVectorDB: return self.topic_dbs.get(topic)
    
    def get_router_db(self) -> RouterVectorDB: return self.router_db
    
    def save_all(self):
        for topic, db in self.topic_dbs.items(): db.save()
        self.router_db.save()
        logger.info("Saved all vector databases")
    
    def load_all(self):
        for topic, db in self.topic_dbs.items():
            if db.db_path and db.db_path.exists(): db.load()
        if self.router_db.db_path and self.router_db.db_path.exists(): self.router_db.load()
        logger.info("Loaded all vector databases")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {"topic_dbs": {},"router_db": {"total_entries": len(self.router_db.entries),"promotion_history_length": len(self.router_db.promotion_history)}}
        for topic, db in self.topic_dbs.items():
            frequent_entries = db.get_frequent_entries(threshold=5)
            stats["topic_dbs"][topic] = {"total_entries": len(db.entries),"frequent_entries": len(frequent_entries),"total_accesses": sum(e.access_count for e in db.entries.values())}
        return stats