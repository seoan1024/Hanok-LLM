# Model Card: Hanok LLM

## Summary

Hanok is a Korean-focused, decoder-only Transformer implementation migrated from the Korean-llm-v4 codebase. This repository contains training and inference code, not trained model weights. No benchmark score or model-quality result is claimed.

## Architecture

The training configuration instantiates 20 Transformer blocks, hidden size 1,920, 10 attention heads (head dimension 192), and SwiGLU intermediate size 4,800. Blocks use pre-normalized residual connections, RMSNorm (`eps=1e-6`), RoPE (theta 10,000), and tied input/output embeddings. Attention uses PyTorch scaled dot-product attention and supports an inference KV cache. Training enables non-reentrant activation checkpointing.

The model constructor defaults to a 4,096-token RoPE table. The training and CLI paths default to 1,024. Vocabulary size is derived from `beomi/Llama-3-Open-Ko-8B`; runtime code adds a separate `<|pad|>` token if the tokenizer has no pad token or aliases pad to EOS. The resulting parameter count depends on the tokenizer vocabulary and is not specified as a verified fixed total.

## Training

The default pipeline performs pretraining followed by SFT. Pretraining defaults to `eliceai/korean-webtext-edu` (`train`); SFT defaults to `BCCard/gemma-4-31B-korean-on-policy-150k` plus local Hanok identity examples. SFT requires a Hanok checkpoint, freezes base weights, and trains rank-32 LoRA adapters with alpha 64. It masks prompt tokens and calculates loss on response tokens. Start rank-32 SFT from a pretraining checkpoint instead of resuming an older rank-8 SFT checkpoint. See [DATA_CARD.md](DATA_CARD.md) for implementation details and data limitations.

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

`hanok evaluate` runs the bundled 15-item Korean multiple-choice suite and three fixed Korean generation prompts. It reports accuracy, answer-format rate, Hangul-letter ratio, repeated-token ratio, empty-response rate, and perplexity over the reference choice labels. These measurements are intended only for regression checks on a specified checkpoint. No results are claimed in this card.

## Checkpoints and export

Native `.pth` checkpoints can be loaded for local inference and exported to a Transformers Llama bundle. GGUF conversion uses an existing llama.cpp checkout; optional quantization and Ollama registration require their respective local executables. Export maps or repackages the supplied checkpoint; it does not provide pretrained Llama weights or improve model quality.
