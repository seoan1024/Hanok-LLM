"""Native checkpoint loading for local inference."""
import torch
from pathlib import Path
from transformers import AutoTokenizer
from ..model.architecture import (
    HANOK_DIM,
    HANOK_INTERMEDIATE_SIZE,
    HANOK_N_HEADS,
    HANOK_N_LAYERS,
    HANOK_VOCAB_SIZE,
    KoreanLLM,
)
from ..training.checkpoint import load_checkpoint

def load_model(checkpoint_path, *, device=None, max_seq_len=1024, preserve_checkpoint_dtype=False):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    source=Path(checkpoint_path)
    if source.is_dir():
        native_path=source/'model_state_dict.pth'
        if not native_path.is_file():
            raise FileNotFoundError(
                f"No native Hanok model_state_dict.pth found in export directory: {source}"
            )
        checkpoint_path=native_path
        tokenizer_source=str(source)
        local_files_only=True
    else:
        tokenizer_dir=source.parent/'tokenizer'
        tokenizer_source=(str(tokenizer_dir) if tokenizer_dir.is_dir()
                          else (str(source.parent) if (source.parent/'tokenizer.json').is_file()
                                else 'EleutherAI/polyglot-ko-1.3b'))
        local_files_only=tokenizer_source!='EleutherAI/polyglot-ko-1.3b'

    tokenizer=AutoTokenizer.from_pretrained(
        tokenizer_source,
        clean_up_tokenization_spaces=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token_id==tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token':'<|pad|>'})
    if len(tokenizer) > HANOK_VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) exceeds the fixed model vocabulary "
            f"({HANOK_VOCAB_SIZE})"
        )
    model=KoreanLLM(vocab_size=HANOK_VOCAB_SIZE,pad_token_id=tokenizer.pad_token_id,
                    dim=HANOK_DIM,n_layers=HANOK_N_LAYERS,n_heads=HANOK_N_HEADS,
                    intermediate_size=HANOK_INTERMEDIATE_SIZE,
                    max_seq_len=max_seq_len).to(device)
    payload=torch.load(checkpoint_path,map_location=device)
    state=payload.get('model_state_dict',payload)
    lora_keys=[key for key in state if key.endswith('.lora_A')]
    if lora_keys:
        rank=state[lora_keys[0]].shape[0]
        model.enable_lora(rank, payload.get('lora_alpha'))
    checkpoint_dtype=state['embed.weight'].dtype if 'embed.weight' in state else None
    state=dict(state)
    for name in ('f_cos','f_sin'):
        if name in state:
            state[name]=getattr(model,name)
    model.load_state_dict(state)
    if preserve_checkpoint_dtype and checkpoint_dtype in (torch.float16,torch.bfloat16,torch.float32):
        model.to(dtype=checkpoint_dtype)
    model.reset_rope_cache()
    model.eval()
    return model,tokenizer,device
