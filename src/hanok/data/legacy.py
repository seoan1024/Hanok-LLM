"""Existing dataset manager and parquet loader, retained for v4 compatibility."""
import itertools, json, hashlib, logging, time, traceback
from collections import OrderedDict
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset as HFDataset

logger=logging.getLogger(__name__)
DATASETS_DIR=Path('./datasets')
DATASETS_CACHE_DIR=DATASETS_DIR/'cache'
DATASETS_MANIFEST_FILE=DATASETS_DIR/'datasets_manifest.json'
def ensure_datasets_dir():
 DATASETS_DIR.mkdir(parents=True,exist_ok=True); DATASETS_CACHE_DIR.mkdir(parents=True,exist_ok=True)

class DatasetManager:
    DEFAULT_SFT_DATASETS = [
        {
            "name": "nlpai-lab/kullm-v2",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "input", "output"]
        },
        {
            "name": "beomi/KoAlpaca-v1.1a",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "input", "output"]
        }
    ]

    def __init__(self, dataset_configs: Optional[List[Dict]] = None, cache_dir: Path = DATASETS_CACHE_DIR):
        self.cache_dir = cache_dir
        self.dataset_configs = dataset_configs or self.DEFAULT_SFT_DATASETS
        self.manifest = self._load_manifest()
        ensure_datasets_dir()

    def _load_manifest(self) -> Dict:
        if DATASETS_MANIFEST_FILE.exists():
            with open(DATASETS_MANIFEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(DATASETS_MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _get_dataset_hash(self, config: Dict) -> str:
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def download_dataset(self, config: Dict, force: bool = False) -> Optional[str]:
        dataset_hash = self._get_dataset_hash(config)
        dataset_name = config["name"]

        if dataset_hash in self.manifest and not force:
            cached_path = self.manifest[dataset_hash].get("path")
            if cached_path and Path(cached_path).exists():
                logger.info(f"✅ Using cached dataset: {dataset_name}")
                return cached_path

        logger.info(f"📥 Downloading {dataset_name}...")

        last_error = None
        error_details = []

        if config.get("streaming"):
            # Explicit streaming is for large corpora: cache only the requested
            # finite shard instead of downloading the whole source dataset first.
            strategies = [{"name": "streaming", "streaming": True, "force_redownload": False}]
        else:
            strategies = [
                {"name": "standard", "streaming": False, "force_redownload": False},
                {"name": "streaming", "streaming": True, "force_redownload": False},
                {"name": "force_redownload", "streaming": False, "force_redownload": True},
            ]

        for attempt, strategy in enumerate(strategies, 1):
            try:
                logger.info(f"🔄 Attempt {attempt}/{len(strategies)} - Strategy: {strategy['name']}")

                load_kwargs = {
                    "path": config["name"],
                    "split": config["split"],
                    "cache_dir": str(self.cache_dir),
                }

                if config.get("config"):
                    load_kwargs["name"] = config["config"]

                if strategy["streaming"]:
                    load_kwargs["streaming"] = True

                if strategy["force_redownload"]:
                    load_kwargs["download_mode"] = "force_redownload"

                ds = load_dataset(**load_kwargs)

                if strategy["streaming"]:
                    logger.info("📥 Converting streaming dataset to regular dataset...")
                    # Never materialize an unbounded streaming iterator in memory.
                    max_examples = config.get("max_examples")
                    if max_examples is None:
                        raise ValueError("Streaming fallback requires max_examples")
                    ds = HFDataset.from_list(list(itertools.islice(ds, max_examples)))

                if config.get("max_examples") and not strategy["streaming"]:
                    ds = ds.select(range(min(len(ds), config["max_examples"])))

                local_path = self.cache_dir / f"{dataset_hash}"
                local_path.mkdir(exist_ok=True, parents=True)

                ds.to_parquet(str(local_path / "data.parquet"))

                self.manifest[dataset_hash] = {
                    "name": dataset_name,
                    "config": config,
                    "path": str(local_path),
                    "num_examples": len(ds),
                    "download_strategy": strategy["name"]
                }
                self._save_manifest()

                logger.info(f"✅ Dataset saved: {local_path} ({len(ds)} examples) via {strategy['name']}")
                return str(local_path)

            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)
                tb_str = traceback.format_exc()

                error_details.append({
                    "attempt": attempt,
                    "strategy": strategy["name"],
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "traceback": tb_str
                })

                logger.warning(f"⚠️ Attempt {attempt} failed ({strategy['name']})")
                logger.warning(f"   ↳ Error Type : {error_type}")
                logger.warning(f"   ↳ Error Msg  : {error_msg}")

                if attempt < len(strategies):
                    wait_time = attempt * 2
                    logger.info(f"⏳ Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)

        logger.error("=" * 80)
        logger.error(f"❌ Failed to download {dataset_name} after {len(strategies)} attempts")
        logger.error("=" * 80)

        for detail in error_details:
            logger.error(f"[Attempt {detail['attempt']}] Strategy: {detail['strategy']}")
            logger.error(f"  - Type   : {detail['error_type']}")
            logger.error(f"  - Message: {detail['error_msg']}")
            logger.error(f"  - Traceback:\n{detail['traceback']}")
            logger.error("-" * 60)

        logger.error(f"📌 Last error summary: {type(last_error).__name__}: {last_error}")
        logger.error("=" * 80)
        return None

    def get_or_download_all(self, force: bool = False) -> List[str]:
        paths = []
        failed = []

        for config in self.dataset_configs:
            path = self.download_dataset(config, force=force)
            if path:
                paths.append(path)
            else:
                failed.append(config["name"])

        logger.info(f"✅ Ready with {len(paths)} datasets")

        if failed:
            logger.warning("=" * 60)
            logger.warning(f"⚠️ 다음 데이터셋 다운로드 실패 ({len(failed)}개):")
            for name in failed:
                logger.warning(f"   - {name}")
            logger.warning("=" * 60)

        return paths

class LocalKoreanDataset(Dataset):
    def __init__(self, dataset_paths: List[str], tokenizer, max_len: int = 256,
                 data_samples_per_dataset: Optional[int] = None,
                 train_on_response_only: bool = True,
                 cache_dir: Path = DATASETS_CACHE_DIR):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_on_response_only = train_on_response_only
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = []
        self.offsets = []
        self.total_samples = 0
        self._parquet_handles = {}
        self._row_group_cache = OrderedDict()
        self._row_group_cache_limit = 2

        logger.info("📚 Loading local datasets (RAM-efficient Parquet mode)...")
        logger.info(f"📂 Dataset cache directory: {self.cache_dir.absolute()}")

        for dataset_path in dataset_paths:
            try:
                parquet_file = Path(dataset_path) / "data.parquet"
                if not parquet_file.exists():
                    logger.warning(f"⚠️ Parquet file not found: {parquet_file}")
                    continue

                logger.info(f"📖 Opening parquet directly: {parquet_file}")

                # 여기서는 메타데이터 확인용으로만 열고,
                # ParquetFile 객체 자체는 self.datasets에 저장하지 않음
                pf = pq.ParquetFile(str(parquet_file))
                num_rows = pf.metadata.num_rows
                num_row_groups = pf.num_row_groups

                if data_samples_per_dataset is not None:
                    num_rows = min(num_rows, data_samples_per_dataset)

                if num_rows <= 0:
                    logger.warning(f"⚠️ Dataset is empty: {parquet_file}")
                    continue

                row_groups = []
                accumulated = 0

                for group_idx in range(num_row_groups):
                    group_rows = pf.metadata.row_group(group_idx).num_rows

                    if accumulated >= num_rows:
                        break

                    rows_to_use = min(group_rows, num_rows - accumulated)
                    row_groups.append((group_idx, accumulated, rows_to_use))
                    accumulated += rows_to_use

                # ★ pickle 가능한 값만 저장
                dataset_info = {
                    "path": str(parquet_file),
                    "num_rows": num_rows,
                    "row_groups": row_groups
                }

                self.offsets.append(self.total_samples)
                self.datasets.append(dataset_info)
                self.total_samples += num_rows

                logger.info(
                    f"✅ Registered {num_rows:,} samples from {Path(dataset_path).name}"
                )
                logger.info(f"   Row groups: {num_row_groups:,}")

            except Exception as e:
                logger.error(f"❌ Error loading dataset from {dataset_path}: {e}")
                logger.error(f"   Full traceback:\n{traceback.format_exc()}")

        logger.info(f"✅ Total samples available: {self.total_samples:,}")

    def _clean_text(self, value) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _build_instruction_sample(
        self,
        instruction: str,
        input_text: str,
        output: str
    ) -> str:
        instruction, input_text, output = map(
            self._clean_text,
            (instruction, input_text, output)
        )

        parts = []
        if instruction:
            parts.append(f"### 질문: {instruction}")
        if input_text:
            parts.append(f"### 입력: {input_text}")

        parts.append(f"### 응답: {output}")
        return "\n".join(parts)

    def _extract_text(self, item: Dict) -> Optional[str]:
        if "instruction" in item and "output" in item:
            text = self._build_instruction_sample(
                item.get("instruction", ""),
                item.get("input", ""),
                item.get("output", "")
            )

        elif "question" in item and "response" in item:
            q, r, s = map(
                self._clean_text,
                (
                    item.get("question", ""),
                    item.get("response", ""),
                    item.get("system_prompt", "")
                )
            )

            if s:
                text = f"### 시스템: {s}\n### 질문: {q}\n### 응답: {r}"
            else:
                text = f"### 질문: {q}\n### 응답: {r}"

        elif "question" in item and "answer" in item:
            text = (
                f"### 질문: {self._clean_text(item.get('question', ''))}\n"
                f"### 응답: {self._clean_text(item.get('answer', ''))}"
            )

        elif "prompt" in item and "response" in item:
            text = (
                f"### 질문: {self._clean_text(item.get('prompt', ''))}\n"
                f"### 응답: {self._clean_text(item.get('response', ''))}"
            )
        elif item.get("text"):
            text = self._clean_text(item["text"])

        elif item.get("content"):
            text = self._clean_text(item["content"])

        elif item.get("document"):
            text = self._clean_text(item["document"])

        elif item.get("body"):
            text = self._clean_text(item["body"])

        else:
            text = None

        return text if text and len(text) > 5 else None  

    def __len__(self):
        return self.total_samples

    def _find_dataset_and_local_index(self, idx: int):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)

        for i, offset in enumerate(self.offsets):
            end = (
                self.offsets[i + 1]
                if i + 1 < len(self.offsets)
                else self.total_samples
            )

            if offset <= idx < end:
                return i, idx - offset

        raise IndexError(idx)

    def _get_item(self, idx: int) -> Dict:
        dataset_idx, local_idx = self._find_dataset_and_local_index(idx)
        info = self.datasets[dataset_idx]

        # Keep one Parquet reader and a small row-group LRU per DataLoader worker.
        # The previous implementation reopened the file and decoded the entire
        # row group for every individual sample.
        pf = self._parquet_handles.get(dataset_idx)
        if pf is None:
            pf = pq.ParquetFile(info["path"])
            self._parquet_handles[dataset_idx] = pf

        for group_idx, group_start, group_rows in info["row_groups"]:
            if group_start <= local_idx < group_start + group_rows:
                row_in_group = local_idx - group_start

                cache_key = (dataset_idx, group_idx)
                table = self._row_group_cache.get(cache_key)
                if table is None:
                    table = pf.read_row_group(group_idx)
                    self._row_group_cache[cache_key] = table
                    while len(self._row_group_cache) > self._row_group_cache_limit:
                        self._row_group_cache.popitem(last=False)
                else:
                    self._row_group_cache.move_to_end(cache_key)
                row = table.slice(row_in_group, 1).to_pylist()[0]

                return row

        raise IndexError(idx)

    def _encode_sample(self, text: str) -> Tuple[List[int], List[int]]:
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        if eos_id is None or pad_id is None:
            raise ValueError(
                "Tokenizer must define separate eos_token_id and pad_token_id"
            )

        marker = "### 응답:"

        if self.train_on_response_only and marker in text:
            prompt, response = text.split(marker, 1)

            prompt_ids = self.tokenizer.encode(
                prompt + marker + "\n",
                add_special_tokens=False
            )

            response_ids = self.tokenizer.encode(
                response,
                add_special_tokens=False
            )

            # Preserve the full prompt whenever it fits. Only truncate it when
            # the prompt alone would leave no room for a response token and EOS.
            prompt_budget = max(0, self.max_len - 2)
            if len(prompt_ids) > prompt_budget:
                # Keep the question's beginning and the answer boundary when a
                # long prompt must be shortened to satisfy total sequence length.
                marker_ids = self.tokenizer.encode(
                    marker + "\n", add_special_tokens=False
                )
                marker_budget = min(len(marker_ids), prompt_budget)
                if marker_budget:
                    prompt_ids = (
                        prompt_ids[:prompt_budget - marker_budget]
                        + marker_ids[-marker_budget:]
                    )
                else:
                    prompt_ids = prompt_ids[:prompt_budget]

            # The final 1 is reserved for EOS, so prompt + response + EOS
            # always fits the configured maximum sequence length.
            response_budget = max(0, self.max_len - len(prompt_ids) - 1)
            response_ids = response_ids[:max(0, response_budget)]

            ids = prompt_ids + response_ids + [eos_id]
            labels = [-100] * len(prompt_ids) + response_ids + [eos_id]

        else:
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            ids.append(eos_id)
            labels = list(ids)

            # Pretraining returns the complete document stream. The collator
            # packs it into full context windows instead of discarding tails.
            if not self.train_on_response_only:
                return ids, labels

        ids = ids[:self.max_len]
        labels = labels[:self.max_len]

        padding_length = self.max_len - len(ids)

        if padding_length > 0:
            ids += [pad_id] * padding_length
            labels += [-100] * padding_length

        return ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._get_item(idx)
        text = self._extract_text(item)
        if not text:
            raise ValueError(f"Empty or unsupported dataset sample at index {idx}")
        encoded, labels = self._encode_sample(text)
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack(
            [sample["input_ids"] for sample in batch]
        ),
        "labels": torch.stack(
            [sample["labels"] for sample in batch]
        )
    }


class SequencePackingCollator:
    """Pack complete pretraining documents into fixed-length causal-LM blocks."""
    def __init__(self, max_len: int, max_blocks: int):
        self.max_len = max_len
        self.max_blocks = max_blocks
        self.buffer = []

    def __call__(self, batch):
        for sample in batch:
            self.buffer.extend(sample["input_ids"].tolist())
        block_count = min(len(self.buffer) // self.max_len, self.max_blocks)
        if block_count == 0:
            empty = torch.empty((0, self.max_len), dtype=torch.long)
            return {"input_ids": empty, "labels": empty.clone()}
        take = block_count * self.max_len
        packed = torch.tensor(self.buffer[:take], dtype=torch.long).view(-1, self.max_len)
        del self.buffer[:take]
        return {"input_ids": packed, "labels": packed.clone()}


class RowGroupBatchSampler:
    """Shuffle Parquet row groups, then shuffle rows locally for reader locality."""
    def __init__(self, dataset, batch_size: int, seed: int, *, validation=False,
                 validation_fraction=0.02):
        self.dataset = dataset
        self.batch_size = batch_size
        rng = random.Random(seed)
        groups = []
        for dataset_idx, info in enumerate(dataset.datasets):
            for group_idx, start, count in info["row_groups"]:
                groups.append([dataset.offsets[dataset_idx] + start, count])
        rng.shuffle(groups)
        total = sum(count for _, count in groups)
        desired = int(total * validation_fraction)
        held_out = set()
        held_count = 0
        if validation and desired:
            for index in range(len(groups) - 1, -1, -1):
                if held_count >= desired:
                    break
                held_out.add(index)
                held_count += groups[index][1]
        elif not validation:
            # Training receives every group except the held-out suffix.
            validation_indices = set()
            held_count = 0
            for index in range(len(groups) - 1, -1, -1):
                if held_count >= desired:
                    break
                validation_indices.add(index)
                held_count += groups[index][1]
            held_out = validation_indices
        self.groups = [group for index, group in enumerate(groups)
                       if (index in held_out) == validation]
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed + random.randrange(1_000_000))
        groups = list(self.groups)
        rng.shuffle(groups)
        for start, count in groups:
            rows = list(range(start, start + count))
            rng.shuffle(rows)
            for offset in range(0, len(rows), self.batch_size):
                yield rows[offset:offset + self.batch_size]

    def __len__(self):
        return sum((count + self.batch_size - 1) // self.batch_size
                   for _, count in self.groups)

