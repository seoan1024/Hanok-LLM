# Data Card: Hanok LLM

## Default sources in the training code

| Stage | Dataset | Configuration | Split | Fields used |
|---|---|---|---|---|
| Pretraining | `eliceai/korean-webtext-edu` | default | `train` | `text` (educational score is not used by the trainer) |
| SFT | `BCCard/gemma-4-31B-korean-on-policy-150k` | default | `train` | `instruction`, `output` |
| SFT | `hanok-local/hanok-identity-v1` | local | `train` | Curated `instruction`, optional `input`, `output`; repeated 16 times for identity emphasis |

These are defaults declared by the source code, not evidence that a dataset was downloaded or used for training. The default pretraining source pins revision `c949654b0a18b4e9cb457364be6676d97551b377` and selects `data-*-of-00018.arrow`; the BCCard SFT source is not revision-pinned. The Elice dataset card describes educational-value filtering and MIT licensing; the BCCard dataset card reports Apache-2.0 and Gemma-generated responses, but its verifier-model details should be reviewed because the card leaves a placeholder. Users must review upstream provenance, terms, and fitness for their intended use. The local Hanok identity examples are included with this repository.

## Loading, caching, and parsing

The legacy dataset manager downloads with Hugging Face Datasets and stores a Parquet file under `./datasets/cache/`; `./datasets/datasets_manifest.json` records the dataset configuration, local path, row count, and download strategy. `--force-redownload` requests a fresh download. Streaming is bounded: it requires `--samples-per-dataset` and caches at most that many records per dataset.

The parser recognizes these schemas:

- `instruction` plus `output`, with optional `input`;
- `question` plus `response`, with optional `system_prompt`;
- `question` plus `answer`;
- `prompt` plus `response`;
- one of `text`, `content`, `document`, or `body`.

Rows without a recognized non-empty text sample (at least six characters after trimming) cannot be used. Pretraining appends EOS and packs document token streams into fixed-length blocks. SFT formats prompt and response markers, masks prompt labels with `-100`, and truncates to the configured sequence length while reserving space for EOS. Padding labels are ignored by the loss.

## Splits and reproducibility

The default validation fraction is 0.02. Pretraining holds out row groups selected by the row-group sampler; SFT uses a seeded random split. Evaluation is limited to `validation_batches` per interval. The project seeds Python and PyTorch, but does not guarantee deterministic CUDA execution or save data-loader RNG state in checkpoints. Source dataset revisions and preprocessing fingerprints are not pinned in checkpoint metadata, so a resumed run may not reproduce the exact same data.

## Evaluation data

`src/hanok/evaluation/benchmark_ko.jsonl` is a separate, hand-authored 15-item multiple-choice sanity suite. It is not a training dataset, representative Korean corpus, or official benchmark. The evaluation command also generates on three fixed prompts. Its choice-label perplexity is specific to the reference answers in this small suite and should not be compared with training validation loss or unrelated benchmark results.

## Privacy and limitations

The default training path does not run a PII detector, anonymizer, language filter, or an additional quality filter. Upstream data curation does not guarantee that every record is correct or appropriate. Users are responsible for reviewing data rights, sensitive content, privacy, and sharing restrictions. Training logs and generated reports may contain prompts or text; inspect them before sharing. No full corpus audit is included in this repository.
