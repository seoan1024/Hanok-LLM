# Model Card: Hanok LLM

## Overview

Hanok is the modular project name for the Korean-llm-v4 custom decoder Transformer. This repository contains code, not model weights. No benchmark or quality result is claimed.

## Architecture

20 transformer blocks, hidden dimension 1,920, 10 attention heads, SwiGLU intermediate dimension 4,800, RMSNorm, RoPE, pre-normalized residual blocks, tied embeddings/output projection, and a KV cache. The vocabulary size follows the loaded tokenizer including the legacy runtime PAD-token behavior. Total parameter count therefore depends on that vocabulary and must be measured from the instantiated model; the README's approximate 1.09B figure is not independently runtime-verified here.

## Tokenizer and training

Tokenizer source: `beomi/Llama-3-Open-Ko-8B`. Training workflow: pretraining then SFT; original defaults and dataset configuration are documented in README and `TrainingConfig`. No weights, benchmark runs, or reproducible training outputs are included.

## Intended use and limitations

Personal experimentation, education, and research intent follows the project owner's stated goal; actual rights are governed by the tracked LICENSE and third-party terms. The model may produce incorrect, biased, unsafe, or low-quality outputs. No safety evaluation, benchmark, privacy audit, or deployment assessment has been performed.

## Evaluation and local distribution

The repository includes a 15-question Korean multiple-choice sanity suite and response perplexity reporting. The suite is for regression checks only; it is not an official or comprehensive capability evaluation, and no score is claimed until it is run against a specific checkpoint.

Native checkpoints can be exported to a standard Transformers `LlamaForCausalLM` bundle. GGUF conversion and Ollama model creation use an existing local llama.cpp checkout and, optionally, its quantizer and an Ollama installation. Export changes the file format only; it does not improve model quality or supply pretrained Llama weights.
