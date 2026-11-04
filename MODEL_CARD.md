# Model Card: Hanok LLM

## Summary

Hanok is a Korean-focused, 1B parameter decoder-only Transformer implementation. This repository contains training and inference code, not trained model weights. No benchmark score or model-quality result is claimed.

## Architecture

The model instantiates 33 Transformer blocks, hidden size 1,280, 10 attention heads (head dimension 128), and SwiGLU intermediate size 5,881. With the 30,004-entry model vocabulary and tied input/output embeddings, the model has 1,000,000,000 parameters (1B). Blocks use pre-normalized residual connections, RMSNorm (`eps=1e-6`), RoPE (theta 10,000), and tied input/output embeddings. Attention uses PyTorch scaled dot-product attention and supports an inference KV cache. Training enables non-reentrant activation checkpointing.

The model constructor defaults to a 4,096-token RoPE table. The training and CLI paths default to 1,024. The tokenizer is loaded from `EleutherAI/polyglot-ko-1.3b`; runtime code adds a separate `<|pad|>` token when needed. The model embedding table has 30,004 entries, and the trainer rejects tokenizers larger than that capacity.

## Training

The default pipeline performs pretraining followed by SFT. Pretraining defaults to `eliceai/korean-webtext-edu`, `wikimedia/wikipedia`, and `kuotient/gsm8k-ko`; SFT defaults to BCCard, KoAlpaca, one pinned AI Hub daily multi-session conversation shard, and local Hanok identity/sanity examples. AI Hub terms and access conditions apply to that third-party mirror. SFT requires every configured dataset to load, requires a Hanok checkpoint, freezes base weights, and trains rank-64 LoRA adapters with alpha 128. For multi-turn rows, preceding turns are masked context and the last user's first following assistant answer is supervised. This makes the training target conversation-conditioned, but does not guarantee free-form dialogue ability or factual accuracy. See [DATA_CARD.md](DATA_CARD.md) for implementation details and data limitations.

## Intended use

The code supports local experimentation, education, and research on Korean-language model training and inference. Any production or commercial use must first assess the repository license, dataset and tokenizer terms, output quality, privacy, safety, and applicable requirements. The repository does not provide a deployment or safety certification.

## Limitations and risks

- No trained weights, verified training run, benchmark score, or quality evaluation is included.
- The bundled evaluation is a small hand-authored regression suite, not a broad or official capability benchmark.
- Outputs may be incorrect, biased, unsafe, repetitive, or low quality. Korean fluency and factual accuracy are not guaranteed.
- Data licensing, consent, privacy, contamination, and corpus quality are not audited by this project.
- Longer context settings are configurable but have not been quality-validated; memory use increases with sequence length.
- No safety evaluation, privacy audit, or deployment assessment has been performed.

## Evaluation

`hanok evaluate` runs the bundled 15-item Korean multiple-choice suite and three fixed Korean generation prompts. It reports accuracy, answer-format rate, Hangul-letter ratio, repeated-token ratio, empty-response rate, and perplexity over the reference choice labels. These measurements are intended only for regression checks on a specified checkpoint. The training monitor separately shows pretraining and SFT loss curves and uses deterministic CPU chat checks. No results are claimed in this card.

## Checkpoints and export

Native `.pth` checkpoints can be loaded for local inference and exported to a Transformers Llama bundle. GGUF conversion uses an existing llama.cpp checkout; optional quantization and Ollama registration require their respective local executables. Export maps or repackages the supplied checkpoint; it does not provide pretrained Llama weights or improve model quality.
