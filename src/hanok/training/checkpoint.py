"""Checkpoint helpers preserving the legacy .pth payload format."""
import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import logging
logger=logging.getLogger(__name__)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    checkpoint_path: str,
    stage: Optional[str] = None,
):
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'lora_rank': getattr(model, 'lora_rank', None),
        'lora_alpha': getattr(model, 'lora_alpha', None),
        'stage': stage,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    load_training_state: bool = True,
) -> int:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_state = dict(checkpoint['model_state_dict'])
        model_lora = any(name.endswith('.lora_A') for name in model_state)
        if getattr(model, 'lora_rank', None) and not model_lora:
            legacy_sft_path = Path(checkpoint_path).parent.name.lower() == 'sft'
            if (checkpoint.get('stage') == 'sft' or legacy_sft_path) and load_training_state:
                raise ValueError(
                    "This is a pre-LoRA SFT checkpoint and cannot resume into LoRA. "
                    "Restart SFT from the pretraining checkpoint."
                )
            for name, value in model.state_dict().items():
                if '.lora_' in name:
                    model_state[name] = value
        # RoPE tables are derived from the configured context length. Reuse
        # the current model's tables so older checkpoints with a different
        # cached table size remain loadable.
        for name in ('f_cos', 'f_sin'):
            if name in model_state:
                model_state[name] = getattr(model, name)
        model.load_state_dict(model_state)
        if hasattr(model,"reset_rope_cache"):
            model.reset_rope_cache()
        logger.info("✅ Model state loaded")

        if not load_training_state or (getattr(model, 'lora_rank', None) and not model_lora):
            logger.info("✅ Model-only resume: optimizer and scheduler were reset")
            return 0

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("✅ Optimizer state loaded")

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("✅ Scheduler state loaded")

        start_step = checkpoint['step']
        logger.info(f"✅ Checkpoint loaded from step {start_step}")
        return start_step

    except Exception:
        logger.exception("Checkpoint loading failed: %s", checkpoint_path)
        raise

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = checkpoints[-1]
    latest_path = os.path.join(checkpoint_dir, latest)
    logger.info(f"Found latest checkpoint: {latest}")
    return latest_path

