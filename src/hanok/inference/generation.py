"""Local generation using the existing KV-cache implementation."""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str = "안녕? 너는 누구니?",
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.3,
    do_sample: bool = True,
    device: torch.device = None
) -> str:
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if device is None:
        device = next(model.parameters()).device

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite number greater than zero")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be a finite number greater than zero")
    was_training = model.training
    model.eval()

    try:
        prompt_text = f"### 질문: {prompt}\n### 응답:\n"
        # Training encodes examples with add_special_tokens=False; matching
        # that convention avoids injecting an unseen BOS token at inference.
        tokens = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        # Keep every forward pass inside the model's precomputed RoPE window.
        # Preserve the end of the prompt so the answer marker remains visible.
        context_limit = int(getattr(model, "f_cos", torch.empty(0)).shape[0])
        if context_limit and tokens.shape[1] > context_limit:
            tokens = tokens[:, -context_limit:]
        # Repetition penalty applies only to generated tokens, including when
        # the prompt was shortened to fit the model context window.
        prompt_length = tokens.shape[1]
        if context_limit:
            max_tokens = min(max_tokens, max(0, context_limit - tokens.shape[1]))

        kv_caches = None
        output_tokens = tokens

        for _ in range(max_tokens):
            input_tokens = output_tokens[:, -1:] if kv_caches is not None else output_tokens

            with torch.no_grad():
                logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)

            next_logits = logits[:, -1, :] / temperature

            # Padding is an input-only token; sampling it can expose the
            # training padding convention as visible junk in generated text.
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is not None and 0 <= pad_token_id < next_logits.size(-1):
                next_logits[:, pad_token_id] = float("-inf")

            if repetition_penalty != 1.0:
                for token_id in set(output_tokens[0, prompt_length:].tolist()):
                    if next_logits[0, token_id] < 0:
                        next_logits[0, token_id] *= repetition_penalty
                    else:
                        next_logits[0, token_id] /= repetition_penalty

            if do_sample and top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Keep the token that crosses the probability threshold;
                    # remove tokens only when the mass before them reaches top_p.
                    sorted_indices_to_remove = (cumsum_probs - sorted_probs) >= top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
                    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            output_tokens = torch.cat([output_tokens, next_token], dim=1)

            stop_token_ids = {tokenizer.eos_token_id}
            for token_id in getattr(tokenizer, "additional_special_tokens_ids", []):
                token_text = tokenizer.convert_ids_to_tokens(token_id)
                if token_text in {"<|eot_id|>", "<|end_of_turn|>"}:
                    stop_token_ids.add(token_id)
            if next_token.item() in stop_token_ids:
                break

        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response = generated_text.split("### 응답:")[-1].strip() if "### 응답:" in generated_text else generated_text

        return response
    finally:
        model.train(was_training)
