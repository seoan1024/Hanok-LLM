"""Reproducibility helpers retaining the v4 seed offset by rank."""
import random
import torch

def setup_distributed(seed: int, rank: int=0, world_size: int=1):
    random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed+rank)

def seed_dataloader_worker(worker_id: int):
    """Seed Python random from PyTorch's per-worker initial seed."""
    seed=torch.initial_seed() % (2**32)
    random.seed(seed)
