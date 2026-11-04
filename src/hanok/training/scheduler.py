"""Original cosine warmup scheduler policy."""
from transformers import get_cosine_schedule_with_warmup

def build_scheduler(optimizer, config):
    return get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=config.warmup_steps,num_training_steps=config.max_steps)
