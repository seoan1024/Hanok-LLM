"""Original optimizer selection, unchanged from v4."""
import logging
import torch
logger=logging.getLogger(__name__)

def build_optimizer(model, config, device):
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters")
    try:
        import bitsandbytes as bnb
    except ImportError:
        bnb=None
    if config.use_8bit_optimizer and bnb is not None and device.type=='cuda':
        optimizer=bnb.optim.AdamW8bit(trainable_parameters,lr=config.learning_rate,weight_decay=config.weight_decay)
        logger.info('Using bitsandbytes AdamW8bit')
        return optimizer
    logger.info('Using torch AdamW')
    return torch.optim.AdamW(trainable_parameters,lr=config.learning_rate,weight_decay=config.weight_decay)
