# Data Card: Hanok LLM

## Dataset sources in code

- SFT: `nlpai-lab/kullm-v2`, config `None`, split `train`
- SFT: `beomi/KoAlpaca-v1.1a`, config `None`, split `train`
- Pretraining: `AdaMLLab/KorMix`, config `minhash_deduped`, split `train`

These are code defaults, not evidence that a training run has downloaded or used them. Dataset licenses, versions, availability, and current terms have not been independently verified. Check each source before use; no license is guessed here.

## Existing preprocessing

HF Datasets downloads are cached under `./datasets/cache`; downloaded rows are stored as Parquet and recorded in `./datasets/datasets_manifest.json`. Existing parsing supports text/content/document/body, instruction/input/output, question/response, question/answer, and prompt/response fields. SFT labels mask the prompt and train on response tokens only. EOS is appended; PAD positions are ignored in loss.

The added `hanok.data.pipeline` utilities offer conservative Unicode/whitespace normalization, configurable language and heuristic quality checks, optional email/phone masking, SHA-256 exact and normalized deduplication, tokenizer chunking, sequence packing, JSONL sharding, and manifests. These are opt-in and not wired into the legacy default training path. The PII patterns are incomplete and cannot certify anonymization. Reports are only generated for explicit shard preparation; original source rows are not rewritten.

## Reproducibility and limitations

The training config has seed 42, random split seed, and worker count 4. Full deterministic CUDA behavior and dataloader state resume are not implemented. Dataset versions and licenses are not pinned. Existing checkpoint state does not include data-loader RNG state or preprocessing fingerprints. No actual corpus statistics or token counts are available in this checkout.

The bundled `src/hanok/evaluation/benchmark_ko.jsonl` is a small hand-authored multiple-choice sanity suite. It is not a training source and is not representative of broad Korean ability. Its perplexity is measured on those reference choices only and should not be compared with unrelated datasets or published benchmark scores.
