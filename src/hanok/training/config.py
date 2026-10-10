"""Original training defaults; values mirror the v4 TrainingConfig."""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TrainingConfig:
    stage: str = "auto"
    dataset_specs: Optional[List[str]] = None
    batch_size: int = 2
    max_steps: int = 100000
    pretrain_steps: int = 100000
    sft_steps: int = 10000
    accumulation_steps: int = 8
    learning_rate: float = 5e-5
    sft_learning_rate: float = 1.5e-5
    warmup_steps: int = 200
    eval_interval: int = 5000
    max_seq_len: int = 1024
    num_workers: int = 4
    use_bfloat16: bool = True
    use_8bit_optimizer: bool = True
    weight_decay: float = 0.1
    validation_split: float = 0.02
    validation_batches: int = 32
    enable_gui: bool = True # 기본 값
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    force_redownload: bool = False
    stream_datasets: bool = False
    samples_per_dataset: Optional[int] = None

