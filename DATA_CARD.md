# Data Card: Hanok LLM

## Default sources in the training code

| Stage | Dataset | Configuration | Split | Fields used |
|---|---|---|---|---|
| Pretraining | `eliceai/korean-webtext-edu` | revision `c949654b` · `data-*-of-00018.arrow` | `train` | `text` (educational score is not used by the trainer) |
| Pretraining | `wikimedia/wikipedia` | `20231101.ko` | `train` | `text` |
| Pretraining | `kuotient/gsm8k-ko` | default | `train` | `question`, `answer` |
| SFT | `BCCard/gemma-4-31B-korean-on-policy-150k` | default | `train` | `instruction`, optional `input`, `output` |
| SFT | `beomi/KoAlpaca-v1.1a` | default | `train` | `instruction`, optional `input`, `output` |
| SFT | `mkd-chanwoo/aihub_conversation` | revision `c1f42ab` · `aihub_daily_conversation_shard_0000.jsonl` | `train` | ChatML `text`; daily Korean dialogue with prior-session summaries; AI Hub terms apply |
| SFT | `hanok-local/hanok-identity-v1` | local | `train` | Curated `instruction`, optional `input`, `output`; repeated 2 times |
| SFT | `hanok-local/hanok-sanity-v1` | local | `train` | 29 basic QA examples; repeated 4 times |

These are defaults declared by the source code, not evidence that a dataset was downloaded or used for training. The default pretraining sources combine educational Korean web text (`eliceai/korean-webtext-edu` with pinned revision `c949654b0a18b4e9cb457364be6676d97551b377` and `data-*-of-00018.arrow`), the Korean Wikipedia dump (`wikimedia/wikipedia` 20231101.ko), and the GSM8K Korean math translation (`kuotient/gsm8k-ko`). The SFT mix combines single-turn instruction/QA data with one pinned shard from the third-party [AI Hub conversation mirror](https://huggingface.co/datasets/mkd-chanwoo/aihub_conversation). The mirror's [pinned JSONL file](https://huggingface.co/datasets/mkd-chanwoo/aihub_conversation/blob/c1f42abfd0346560fde7a1ed81b163b77660852d/aihub_daily_conversation_shard_0000.jsonl) contains ChatML dialogue; the [upstream AI Hub record](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=71678&topMenu=100) describes daily multi-session Korean conversations. AI Hub terms and access conditions apply. The BCCard source is not revision-pinned; its card reports Gemma-generated responses but leaves the exact verifier model as a placeholder. Users must review upstream provenance, terms, and fitness for their intended use. The local Hanok identity and sanity examples are included with this repository (repeated 2 and 4 times, respectively); the 29 basic QA rows have not been independently fact-checked by this project.

## Loading, caching, and parsing

The dataset manager downloads with Hugging Face Datasets and stores a Parquet file under `./datasets/cache/`; `./datasets/datasets_manifest.json` records the dataset configuration, local path, row count, and download strategy. `--force-redownload` requests a fresh download. Streaming is bounded: it requires `--samples-per-dataset` and caches at most that many records per dataset.

The parser recognizes these schemas:

- `instruction` plus `output`, with optional `input`;
- `question` plus `response`, with optional `system_prompt`;
- `question` plus `answer`;
- `prompt` plus `response`;
- role/content arrays in `messages`, `conversations`, or `query_and_response`, plus ChatML transcripts;
- one of `text`, `content`, `document`, or `body`.

For multi-turn rows, the parser selects the last user turn and the first following assistant turn as the target, retaining preceding recognized turns as masked context. It does not supervise every assistant turn in a row. Rows without a recognized non-empty sample (at least six characters after trimming) cannot be used. Pretraining appends EOS and packs document token streams into fixed-length blocks. SFT masks prompt labels with `-100`; when the prompt exceeds half the configured sequence length, it keeps the most recent prompt tokens (including the answer boundary) so at least half the sequence remains available for the target and EOS. Padding labels are ignored by the loss. SFT stops if any configured dataset fails to load, avoiding accidental local-only training. The pinned AI Hub shard has not been downloaded or audited by this repository; parsing compatibility alone does not establish content quality or rights for a particular use.

## Splits and reproducibility

The default validation fraction is 0.02. Pretraining holds out row groups selected by the row-group sampler; SFT uses a seeded random split. Evaluation is limited to `validation_batches` per interval. The project seeds Python and PyTorch, but does not guarantee deterministic CUDA execution or save data-loader RNG state in checkpoints. Source dataset revisions and preprocessing fingerprints are not pinned in checkpoint metadata, so a resumed run may not reproduce the exact same data.

## Evaluation data

`src/hanok/evaluation/benchmark_ko.jsonl` is a separate, hand-authored 15-item multiple-choice sanity suite. It is not a training dataset, representative Korean corpus, or official benchmark. The evaluation command also generates on three fixed prompts. Its choice-label perplexity is specific to the reference answers in this small suite and should not be compared with training validation loss or unrelated benchmark results.

The bundled `src/hanok/data/hanok_sanity.jsonl` is a small training curriculum for basic facts, arithmetic, identity, and greetings. Repetition improves exposure but does not make the examples a representative corpus or guarantee factual accuracy.

## Privacy and limitations

The default training path does not run a PII detector, anonymizer, language filter, or an additional quality filter. Upstream data curation does not guarantee that every record is correct or appropriate. Users are responsible for reviewing data rights, sensitive content, privacy, and sharing restrictions. Training logs and generated reports may contain prompts or text; inspect them before sharing. No full corpus audit is included in this repository.
