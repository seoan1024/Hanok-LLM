"""Deterministic Korean sanity benchmark and response perplexity report."""
import json
import re
import unicodedata
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .perplexity import perplexity
from ..inference.generation import generate

DEFAULT_BENCHMARK = Path(__file__).with_name("benchmark_ko.jsonl")
KOREAN_GENERATION_PROMPTS = (
    "초등학생도 이해할 수 있게 비가 내리는 이유를 한국어 두 문장으로 설명해 주세요.",
    "봄날 공원에서 가족이 산책하는 장면을 자연스러운 한국어 두 문장으로 묘사해 주세요.",
    "‘회의가 내일로 연기되었다’를 뜻은 유지하면서 더 공손한 한국어 문장으로 바꿔 주세요.",
)


def load_benchmark(path=DEFAULT_BENCHMARK):
    records = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item.get("question"), str) or not item["question"].strip():
                raise ValueError(f"Missing question at {path}:{line_number}")
            choices = item.get("choices")
            answer = item.get("answer")
            if not isinstance(choices, dict) or not choices or answer not in choices:
                raise ValueError(f"Invalid choices/answer at {path}:{line_number}")
            records.append(item)
    if not records:
        raise ValueError(f"Benchmark contains no examples: {path}")
    return records


def _render_question(item):
    choices = "\n".join(f"{key}. {value}" for key, value in item["choices"].items())
    return (
        f"{item['question'].strip()}\n{choices}\n"
        "정답은 선택지의 알파벳 하나만 답하세요."
    )


def _normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w가-힣]+", "", text, flags=re.UNICODE)


def _parse_choice(response, choices):
    first_line = response.strip().splitlines()[0] if response.strip() else ""
    normalized_first_line = _normalized(first_line)
    for label, text in choices.items():
        normalized_choice = _normalized(str(text))
        if normalized_choice and normalized_first_line.startswith(normalized_choice):
            return label

    match = re.search(
        r"(?:^|[\s:：])([A-Z])(?=$|[\s.,:：)\]가-힣])",
        first_line.upper(),
    )
    if match and match.group(1) in choices:
        return match.group(1)
    return None


def _hangul_ratio(text):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    hangul = sum("가" <= char <= "힣" or "ㄱ" <= char <= "ㆎ" for char in letters)
    return hangul / len(letters)


def _repeated_token_ratio(text):
    tokens = re.findall(r"[가-힣]+|[A-Za-z]+|\d+", text.casefold())
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def _make_perplexity_batches(records, tokenizer, max_seq_len, batch_size=2):
    samples = []
    for item in records:
        prompt_ids = tokenizer.encode(
            f"### 질문: {_render_question(item)}\n### 응답:",
            add_special_tokens=False,
        )
        answer_ids = tokenizer.encode(str(item["answer"]), add_special_tokens=False)
        eos_id = tokenizer.eos_token_id
        if eos_id is None or not answer_ids:
            continue
        answer_ids = answer_ids + [eos_id]
        prompt_budget = max_seq_len - len(answer_ids)
        if prompt_budget < 0:
            answer_ids = answer_ids[: max_seq_len - 1]
            prompt_budget = 1
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
        token_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        padding = max_seq_len - len(token_ids)
        if padding < 0:
            token_ids = token_ids[-max_seq_len:]
            labels = labels[-max_seq_len:]
        else:
            token_ids.extend([tokenizer.pad_token_id] * padding)
            labels.extend([-100] * padding)
        samples.append({
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    def collate(batch):
        return {
            "input_ids": torch.stack([sample["input_ids"] for sample in batch]),
            "labels": torch.stack([sample["labels"] for sample in batch]),
        }

    return DataLoader(samples, batch_size=batch_size, collate_fn=collate)


@torch.no_grad()
def evaluate_benchmark(model, tokenizer, *, path=DEFAULT_BENCHMARK,
                       device=None, max_new_tokens=8, language_new_tokens=64,
                       max_seq_len=1024):
    """Score answer accuracy, Korean generation shape and target perplexity.

    This small fixed suite is a regression/sanity check, not an official or
    comprehensive Korean language benchmark.
    """
    records = load_benchmark(path)
    device = torch.device(device or next(model.parameters()).device)
    was_training = model.training
    model.eval()
    results = []
    language_results = []
    try:
        for item in records:
            response = generate(
                model,
                tokenizer,
                prompt=_render_question(item),
                max_tokens=max_new_tokens,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                do_sample=False,
                device=device,
            )
            prediction = _parse_choice(response, item["choices"])
            results.append({
                "id": item.get("id"),
                "question": item["question"],
                "expected": item["answer"],
                "predicted": prediction,
                "correct": prediction == item["answer"],
                "response": response,
            })
        for prompt in KOREAN_GENERATION_PROMPTS:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=language_new_tokens,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                do_sample=False,
                device=device,
            )
            language_results.append({
                "prompt": prompt,
                "response": response,
                "hangul_letter_ratio": round(_hangul_ratio(response), 4),
                "repeated_token_ratio": round(_repeated_token_ratio(response), 4),
                "empty": not bool(response.strip()),
            })
    finally:
        model.train(was_training)

    loader = _make_perplexity_batches(records, tokenizer, max_seq_len)
    ppl = perplexity(model, loader, device)
    correct = sum(result["correct"] for result in results)
    format_count = sum(result["predicted"] is not None for result in results)
    return {
        "benchmark": "Hanok Korean sanity benchmark",
        "benchmark_kind": "local_regression_suite_not_official",
        "examples": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "answer_format_rate": format_count / len(results),
        "korean_generation_examples": len(language_results),
        "mean_hangul_letter_ratio": sum(x["hangul_letter_ratio"] for x in language_results) / len(language_results),
        "mean_repeated_token_ratio": sum(x["repeated_token_ratio"] for x in language_results) / len(language_results),
        "empty_generation_rate": sum(x["empty"] for x in language_results) / len(language_results),
        "choice_letter_perplexity": ppl,
        "results": results,
        "korean_generation_results": language_results,
    }
