"""Export Hanok checkpoints as a standard Transformers Llama model.

Hanok's decoder uses the same tensor layout and operations as a decoder-only
Llama model (including split-half RoPE), so weights can be mapped directly to
the built-in ``LlamaForCausalLM`` format without custom remote code.
"""
import json
from pathlib import Path


def _llama_state_dict(model):
    """Map Hanok parameter names to the equivalent Transformers Llama names."""
    state = {}
    source = model.state_dict()
    def weight(name):
        module = model.get_submodule(name)
        merged = model.merged_linear_weight(module)
        return merged.detach().cpu().contiguous()
    state["model.embed_tokens.weight"] = source["embed.weight"].detach().cpu().contiguous()

    for index in range(len(model.layers)):
        src = f"layers.{index}."
        dst = f"model.layers.{index}."
        mapping = {
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.k_proj.weight": "attention.wk.weight",
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "mlp.up_proj.weight": "feed_forward.w3.weight",
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
        }
        for target_name, source_name in mapping.items():
            module_name = src + source_name.removesuffix(".weight")
            state[dst + target_name] = weight(module_name)

    state["model.norm.weight"] = source["norm.weight"].detach().cpu().contiguous()
    # LlamaConfig ties lm_head.weight to model.embed_tokens.weight. Omitting
    # the duplicate is required for safe serialization and preserves tying.
    return state


def save_hanok_bundle(model, tokenizer, output_dir, *, stage="base", metadata=None):
    """Write a directly loadable ``AutoModelForCausalLM`` bundle.

    The bundle contains standard Llama-format safetensors, config and tokenizer
    files. A native Hanok state dict is also written for exact local reloads.
    """
    try:
        from safetensors.torch import save_file
        from transformers import LlamaConfig
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-specific message
        raise RuntimeError(
            "HF export requires transformers and safetensors to be installed"
        ) from exc

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    hidden_size = model.dim
    num_layers = len(model.layers)
    intermediate_size = model.layers[0].feed_forward.w1.out_features
    max_positions = int(model.f_cos.shape[0])
    model_dtype = next(model.parameters()).dtype

    config = LlamaConfig(
        vocab_size=model.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=model.n_heads,
        num_key_value_heads=model.n_heads,
        max_position_embeddings=max_positions,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        dtype=model_dtype,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.architectures = ["LlamaForCausalLM"]
    config.hanok_stage = stage
    config.hanok_metadata = metadata or {}
    config.save_pretrained(root)

    mapped = _llama_state_dict(model)
    save_file(mapped, str(root / "model.safetensors"), metadata={"format": "pt"})
    torch.save(model.state_dict(), root / "model_state_dict.pth")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}### 시스템: {{ message['content'] }}\n"
        "{% elif message['role'] == 'user' %}### 질문: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}### 응답: {{ message['content'] }}\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}### 응답:{% endif %}"
    )
    tokenizer.save_pretrained(root)

    manifest = {
        "format": "transformers-llama",
        "model_class": "LlamaForCausalLM",
        "stage": stage,
        "source": "Hanok KoreanLLM",
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": model.n_heads,
        "intermediate_size": intermediate_size,
        "vocab_size": model.vocab_size,
        "max_position_embeddings": max_positions,
        "tied_embeddings": True,
        "metadata": metadata or {},
    }
    (root / "hanok_export.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def export_checkpoint(checkpoint_path, output_dir, *, device="cpu", max_seq_len=1024,
                      stage="trained", metadata=None):
    """Load a native checkpoint and export it as a Transformers model bundle."""
    from ..inference.loader import load_model

    model, tokenizer, _ = load_model(
        checkpoint_path, device=device, max_seq_len=max_seq_len,
        preserve_checkpoint_dtype=True,
    )
    return save_hanok_bundle(
        model, tokenizer, output_dir, stage=stage,
        metadata={"source_checkpoint": str(checkpoint_path), **(metadata or {})},
    )
