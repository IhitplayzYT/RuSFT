from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import toml
from pathlib import Path

@dataclass
class ModelConfig:
    model: str = "Qwen/Qwen3.5-8B-Instruct"
    in_8bit: bool = True
    in_4bit: bool = False
    dev: str = "cuda"

@dataclass
class LoRAConfig:
    rank: int = 16  
    alpha: int = 32   # alpha = 2 * rank
    dropout: float = 0.05
    trgt_mods: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    bias: str = "none"

@dataclass
class RouterConfig:
    embed_dim: int = 768
    hidden_dim: int = 256
    mx_experts: int = 3 # 3 max experts per request
    dropout: float = 0.1
    lr: float = 0.001
    w_decay: float = 0.01


@dataclass
class VectorDBConfig:
    embed_dim: int = 768
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    db_path: str = "./data/vector_dbs"
    idx_type: str = "flat"
    promote_thresh: int = 10  # Min references to be promoted
    promote_window: int = 100 # The window in which min ref shoudl happen for promotion
    mx_router_entries: int = 10000


@dataclass
class RAGConfig:
    top_k: int = 20
    similarity_thresh: float = 0.7
    mx_ctx_len: int = 10000
    rerank: bool = True
    rerank_top_k: int = 8
    ctx_template: str = "Context: {context}\n\nQuestion: {question}"
    combine_ctxs: bool = True
    mx_ctx_per_expert: int = 3

@dataclass
class SFTConfig:
    output_dir: str = "./outputs/sft"
    n_epochs: int = 20
    per_dev_train_batch_sz: int = 4
    per_dev_eval_batch_sz: int = 4
    grad_accum_steps: int = 4
    lr: float = 0.0001
    warmup_steps: int = 100
    log_freq: int = 60 # Log after every 60s
    save_freq: int = 100
    eval_steps: int = 100
    max_seq_len: int = 512
    fp16: bool = True
    bf16: bool = False
    grad_checkpoint: bool = True
    lr_scheduler: str = "cosine"
    w_decay: float = 0.001
    mx_saves: int = 10


@dataclass
class PPOConfig:
    output_dir: str = "./outputs/ppo"
    lr: float = 0.000015
    batch_sz: int = 128
    mini_batch_sz: int = 16
    grad_accum_steps: int = 4
    ppo_epochs: int = 4
    kl_coef: float = 0.1
    clip: float = 0.2
    max_grad_norm: float = 1.0
    reward_model_path: Optional[str] = None
    max_length: int = 512
    generation_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 2048,"temperature": 0.7,"top_p": 0.9,"do_sample": True})


@dataclass
class TopicConfig:
    name: str
    model_path: Optional[str] = None
    lora_path: Optional[str] = None
    db_path: Optional[str] = None
    enabled: bool = True
    weight: float = 1.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    topics: List[TopicConfig] = field(default_factory=list)
    
    @classmethod
    def from_toml(cls, config_path: str) -> "Config":
        config_path = Path(config_path)
        if not config_path.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
        data = toml.load(config_path)
        x = []
        for table in data.get("topics", []): x.append(TopicConfig(**table))
        return cls(model=ModelConfig(**data.get("model", {})),lora=LoRAConfig(**data.get("lora", {})),router=RouterConfig(**data.get("router", {})),vector_db=VectorDBConfig(**data.get("vector_db", {})),rag=RAGConfig(**data.get("rag", {})),sft=SFTConfig(**data.get("sft", {})),ppo=PPOConfig(**data.get("ppo", {})),topics=x) 

    @classmethod    
    def to_toml(self, save_path: str):
        pth = Path(save_path)
        pth.parent.mkdir(parents=True, exist_ok=True)
        data = {"model": self.model.__dict__,"lora": self.lora.__dict__,"router": self.router.__dict__,"vector_db": self.vector_db.__dict__,"rag": self.rag.__dict__,"sft": self.sft.__dict__,"ppo": self.ppo.__dict__,"topics": [topic.__dict__ for topic in self.topics]}
        with open(pth, "w") as f:
            toml.dump(data, f)
    
    @classmethod
    def get_default_config(cls) -> "Config":
        cfg = cls()
        cfg.topics = [TopicConfig(name="coding"),TopicConfig(name="mathematics"),TopicConfig(name="science"),TopicConfig(name="general")]
        for topic in cfg.topics:
            if topic.model_path is None: topic.model_path = cfg.model.base_model
        return cfg


def create_default_config_file(save_path: str = "./config/moe_config.toml"):
    ret = Config.get_default_config()
    ret.to_toml(save_path)
    return ret 
