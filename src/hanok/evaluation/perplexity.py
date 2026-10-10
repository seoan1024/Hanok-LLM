"""Likelihood metrics for held-out token batches."""
import math
import torch

@torch.no_grad()
def perplexity(model, batches, device, *, max_batches=None):
    was_training=model.training
    model.eval()
    total_nll=0.0; total_tokens=0
    try:
        for index,batch in enumerate(batches):
            if max_batches is not None and index>=max_batches: break
            input_ids=batch['input_ids'].to(device); labels=batch['labels'].to(device)
            _,loss,_=model(input_ids,labels=labels)
            count=int((labels[...,1:]!=-100).sum().item())
            if count:
                total_nll+=float(loss.item())*count; total_tokens+=count
    finally:
        model.train(was_training)
    if not total_tokens: return {'loss':None,'perplexity':None,'tokens':0}
    mean=total_nll/total_tokens
    return {'loss':mean,'perplexity':math.exp(min(mean,80.0)),'tokens':total_tokens}
