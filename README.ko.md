<div align="center">

# 🏯 Hanok LLM 1.0

### 한국어 중심 모듈형 언어 모델 스택

🌐 **언어 선택:** [English](README.md) · [한국어](README.ko.md)

**한국어 LLM 실험 · 학습 · 추론 · 체크포인트 · 데이터 파이프라인을 위한 순수 PyTorch 구현 — 완전한 상용화를 목표로 설계되었습니다.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=111827" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Datasets-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=111827" alt="Datasets"/>
  <img src="https://img.shields.io/badge/Windows%2011-0078D4?style=for-the-badge&logo=windows11&logoColor=white" alt="Windows 11"/>
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/License-See%20LICENSE-6B7280?style=for-the-badge" alt="License"/>
</p>

<img src="assets/svg/hanok-overview.svg" alt="Hanok LLM overview" width="100%"/>

<p>
  <b>20 Layers</b> · <b>1,920 Hidden</b> · <b>10 Heads</b> · <b>SwiGLU 4,800</b> · <b>~1.09B Parameters</b>
</p>

**저장소:** [https://github.com/seoan1024/Hanok-LLM.git](https://github.com/seoan1024/Hanok-LLM.git)  
**연구 기반:** [https://github.com/seoan1024/Korean-llm.git](https://github.com/seoan1024/Korean-llm.git)

</div>

---

## ✦ Hanok은 무엇인가?

Hanok LLM은 지금까지 연구해 온 [Korean-llm](https://github.com/seoan1024/Korean-llm.git) 프로젝트를 밑받침으로 만들어진  
모듈형 · 상용화 지향 한국어 언어 모델 스택입니다. **완전한 상용화**를 명확한 목표로 설계되었습니다.

모든 시스템은 `src/hanok/` 패키지 아래에서 엄격하게 분리되어 있습니다.  
모델 · 데이터 · 학습 · 추론 · 평가 · Export · 모니터링 각 계층을 독립적으로 검사하고 확장할 수 있습니다.

| 계층 | 역할 |
|:-----|:-----|
| **Model** | 순수 PyTorch 디코더 전용 Transformer (`KoreanLLM`) — RMSNorm, RoPE, SwiGLU, Tied Embedding, KV Cache |
| **Data** | Hugging Face 데이터셋 로딩, Parquet 캐싱, 한국어 친화적 필드 파싱, SFT 응답 전용 마스킹 |
| **Training** | Pretrain → SFT 오케스트레이션, Gradient Accumulation, Cosine Schedule, 체크포인트 재개, 선택적 GUI |
| **Inference** | 네이티브 `.pth` 로딩 + temperature / top-k / top-p / repetition-penalty 생성 |
| **Evaluation** | Perplexity + 재현 가능한 한국어 객관식 회귀 벤치마크 |
| **Export** | 표준 Transformers Llama 체크포인트, llama.cpp GGUF, Ollama Modelfile 및 등록 |
| **UI** | 선택적 실시간 학습 모니터 GUI |

> **직접 들여다볼 수 있고, 실제로 출시할 수 있는 한국어 LLM 스택을 만드세요.**  
> 순수 PyTorch. 명시적인 기본값. 로컬 체크포인트. 실용적인 Windows + CUDA 워크플로.

학습된 모델 가중치는 별도로 제공되지 않습니다. Export 명령은 사용자가 가진 Hanok 체크포인트를 변환합니다.

---

## 🧠 모델 아키텍처 상세

<img src="assets/svg/architecture-ui.svg" alt="Hanok architecture" width="100%"/>

### 핵심 설정

| 파라미터                     | 값                             | 비고 |
|:-----------------------------|:------------------------------:|:-----|
| Transformer 블록             | **20**                         | `n_layers` |
| Hidden dimension             | **1,920**                      | `dim` |
| Attention heads              | **10**                         | `n_heads` |
| Head dimension               | **192**                        | `dim // n_heads` |
| SwiGLU intermediate          | **4,800**                      | `dim × 2.5` |
| Normalization                | **RMSNorm**                    | `eps=1e-6` |
| Positional encoding          | **RoPE**                       | θ = 10,000 |
| Residual style               | **Pre-norm**                   | Attention & FFN |
| Embedding / output           | **Tied weights**               | `output.weight = embed.weight` |
| KV cache                     | **지원**                       | 레이어별, detach |
| 기본 학습 시퀀스 길이        | **1,024**                      | `TrainingConfig.max_seq_len`; 설정 가능 |
| 모델 직접 생성 기본값        | **4,096**                      | `KoreanLLM(max_seq_len=4096)` RoPE 테이블 길이 |
| 학습 / CLI 컨텍스트 기본값   | **1,024**                      | 학습, 추론, 평가, Export 경로 기본값 |
| Vocab size (기본)            | **128,256**                    | 로드된 토크나이저 기준 |
| 인스턴스화된 파라미터 수     | **약 1.09B**                   | 최종 vocab size에 따라 변동 |

시퀀스 한도는 아키텍처 전체에 고정된 상한이 아니라, 각 모델 인스턴스에서 생성하는 RoPE 테이블 길이입니다. `KoreanLLM` 클래스를 직접 만들 때 기본값은 4,096 토큰이지만, 학습과 CLI 로더 경로의 기본값은 1,024입니다. 더 긴 컨텍스트는 `--max-seq-len`으로 설정할 수 있습니다. 생성할 때 프롬프트와 새 토큰을 합쳐 이 한도를 적용하며, 프롬프트가 한도를 넘으면 앞부분부터 잘라냅니다. 1,024보다 긴 컨텍스트는 설정 가능하지만 기본 학습 길이가 아니며 품질을 검증하지 않았습니다. 시퀀스가 길어지면 메모리 사용량도 증가합니다.

### 구성 요소

**RMSNorm**  
평균 중심화 없이 Root-Mean-Square로 정규화합니다. 깊은 Residual 스택에서도 가볍고 안정적입니다.

**RoPE (Rotary Positional Embeddings)**  
Query와 Key 모두에 적용됩니다. 미리 계산된 cos/sin 테이블을 버퍼로 등록하고 절대 위치로 슬라이스하여 KV-cache 기반 점진적 디코딩을 지원합니다.

**Attention**  
- `F.scaled_dot_product_attention` 기반 Multi-head self-attention  
- 기본적으로 Causal masking (`is_causal=True`)  
- 외부 attention mask 선택적 지원  
- 시퀀스 차원으로 KV cache를 이어 붙여 효율적인 생성 지원

**SwiGLU Feed-Forward**  
Bias 없는 세 개의 Linear (`w1`, `w2`, `w3`):  
`w2(SiLU(w1(x)) * w3(x))`. Intermediate 크기는 `dim × 2.5 = 4,800`.

**TransformerBlock**  
Pre-norm residual 구조:  
`x = x + Attention(RMSNorm(x))`  
`x = x + SwiGLU(RMSNorm(x))`

**KoreanLLM**  
- Embedding → 20 × TransformerBlock → 최종 RMSNorm → Tied output projection  
- 가중치 초기화: Linear / Embedding 모두 normal(std=0.02)  
- 학습 시 `torch.utils.checkpoint` (non-reentrant)로 메모리 효율 확보  
- Loss: `ignore_index=-100`인 token-level cross-entropy, 유효 타깃 개수로 나눔 (빈 타깃 배치도 안전하게 처리)

### 설계 철학

아키텍처는 현대적인 디코더 전용 관행(RMSNorm + RoPE + SwiGLU + Tied Embedding)을 따르면서도  
완전히 투명하게 유지됩니다. 불투명한 커스텀 CUDA 커널이나 폐쇄형 컴포넌트는 없으며,  
모든 텐서 연산이 순수 PyTorch로 드러나 있습니다.

---

## ⚡ 학습 스택

<img src="assets/svg/training-ui.svg" alt="Training pipeline" width="100%"/>

### 기본 `TrainingConfig`

```text
stage                      auto | pretrain | sft
batch_size                 2
accumulation_steps         8          # 유효 배치 = 16
learning_rate              5e-5
sft_learning_rate          1.5e-5
warmup_steps               200
scheduler                  Cosine
weight_decay               0.1
max_steps                  100,000    # 수동 stage 기본값
pretrain_steps             100,000
sft_steps                  10,000
eval_interval              5,000
max_seq_len                1024
num_workers                4
use_bfloat16               True
use_8bit_optimizer         True       # AdamW8bit → AdamW 폴백
validation_split           0.02
validation_batches         32
enable_gui                 True
seed                       42
checkpoint_dir             checkpoints
force_redownload           False
stream_datasets            False
samples_per_dataset        None
resume_from_checkpoint     None
```

### 학습 단계

```text
                    ┌──────────────────┐
                    │   DATA SOURCES   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ CACHE / PARQUET  │
                    └────────┬─────────┘
                             │
                 ┌───────────▼───────────┐
                 │       PRETRAIN        │
                 │  KorMix / 커스텀 데이터 │
                 │  (기본 100k steps)    │
                 └───────────┬───────────┘
                             │ checkpoint
                 ┌───────────▼───────────┐
                 │          SFT          │
                 │ KuLLM + KoAlpaca      │
                 │ (기본 10k steps)      │
                 │ 응답 토큰만 Loss 계산 │
                 └───────────┬───────────┘
                             │
                    ┌────────▼────────┐
                    │  NATIVE .PTH    │
                    └─────────────────┘
```

- **`auto` 모드**: 먼저 전체 Pretraining을 수행한 뒤, 최신 Pretrain 체크포인트에서 SFT를 시작합니다 (SFT용 옵티마이저·스케줄은 새로 생성).  
- **체크포인트 재개**: 두 단계 모두 지원.  
- **SFT 라벨 마스킹**: 응답 토큰만 Loss에 기여하고, 프롬프트 토큰은 `-100`으로 설정.  
- **Pretraining 패킹**: 문서 전체 토큰을 이어 1,024 토큰 블록을 만들고, Parquet row group 안에서 행을 섞어 반복 읽기를 줄입니다.
- **SFT 업데이트**: 사전학습 가중치는 고정하고 rank-8 LoRA 어댑터를 학습합니다. 네이티브 체크포인트는 어댑터를 보존하고 HF Export는 어댑터를 기본 가중치에 병합합니다.
- **Validation**: Pretraining은 Parquet row group 기준 약 2%를 분리하고, SFT는 무작위 2% 분할을 사용합니다. `eval_interval`마다 평가합니다 (`validation_batches` 상한).
- **GUI**: 선택적 실시간 Loss 모니터 (`enable_gui=True` 기본값, `--no-gui`로 비활성화).

### 옵티마이저 & 스케줄러

- 기본: **AdamW8bit** (bitsandbytes 사용 가능 시)  
- 폴백: 일반 **AdamW**  
- Learning-rate 스케줄: **Linear Warmup + Cosine Annealing** (Warmup 200 steps)

### 메모리 & Precision

- CUDA에서 `use_bfloat16=True`일 때 BF16 autocast  
- 학습 중 Gradient Checkpointing 활성화  
- CUDA 디바이스에서 DataLoader pin_memory 사용
- SFT는 rank-8 LoRA 어댑터만 학습해 학습 대상 파라미터와 옵티마이저 메모리를 줄입니다.

---

## 🖥️ 권장 / 검증된 워크스테이션

<img src="assets/svg/windows11-5090-ui.svg" alt="Windows 11 + RTX 5090 profile" width="100%"/>

| 구성 요소     | 권장 사양                                                   |
|:--------------|:------------------------------------------------------------|
| OS            | **Windows 11**                                              |
| GPU           | **NVIDIA GeForce RTX 시리즈** |
| VRAM          | **24 GB 이상 권장** (Pretraining 예상 사용량 약 18–21 GB)   |
| 시스템 RAM    | **64 GB 이상**                                              |
| CPU           | 고성능 멀티코어                                             |
| Python        | **3.10+** (문서화 환경: 3.11)                               |
| Precision     | **BF16** (지원 CUDA 하드웨어)                               |
| 저장 공간     | 빠른 SSD (데이터셋 + 체크포인트가 빠르게 증가)              |

> **메모리 참고**  
> 시퀀스 1,024, 기본 배치 크기·누적 횟수, FP32 모델 가중치 + BF16 autocast, AdamW8bit, Gradient Checkpointing 기준 Pretraining 피크 VRAM은 **약 18–21 GB로 추정**합니다. 이 값은 실측 결과가 아니며, GPU와 CUDA 메모리 할당 상태에 따라 달라집니다. LoRA SFT는 메모리 사용 양상이 다릅니다.
> 실제 사용량은 배치 크기, 시퀀스 길이, 옵티마이저 상태, Gradient Checkpointing 여부, 학습/추론 여부에 따라 달라집니다.  
> 더 긴 시퀀스나 더 큰 유효 배치를 사용하면 VRAM이 더 필요하거나 Accumulation을 줄여야 합니다.

---

## 🧪 검증 및 로드맵 상태

<img src="assets/svg/status-ui.svg" alt="Validation status" width="100%"/>

| 주장                                                   | 상태 |
|:-------------------------------------------------------|:----:|
| Windows 11 / RTX 5090 Laptop GPU 로컬 테스트           | ✅   |
| 모델 생성 및 네이티브 추론                             | ✅   |
| 학습 / 체크포인트 / 재개 워크플로                      | ✅   |
| 아키텍처 / 데이터 / 이동 동등성 테스트                 | ✅   |
| Model Card & Data Card 포함                            | ✅   |
| 완전한 상용화 목표                                     | ✅   |
| 학습된 가중치 + 공식 벤치마크 점수                     | 🔜 릴리스 예정 |
| Transformers Llama 형식 로컬 Export                   | ✅ 코드 제공    |
| llama.cpp GGUF / Ollama Export                       | ✅ 외부 llama.cpp / Ollama 필요 |
| 공식 한국어 품질 벤치마크 결과                        | 미실행          |

---

## 🧩 기술 스택

<img src="assets/svg/stack-ui.svg" alt="Technology stack" width="100%"/>

| 카테고리          | 라이브러리 / 도구                                          |
|:------------------|:-----------------------------------------------------------|
| Core              | Python 3.10+, PyTorch 2.x                                  |
| Tokenization      | Hugging Face Transformers (`beomi/Llama-3-Open-Ko-8B`)     |
| Data              | Hugging Face Datasets, PyArrow, Pandas                     |
| Optimization      | bitsandbytes (선택적 8-bit AdamW)                          |
| Monitoring        | matplotlib + 선택적 GUI 모니터                             |
| Packaging         | setuptools, `pyproject.toml`, CLI 엔트리포인트 `hanok`     |

---

## 📦 프로젝트 구조

```text
Hanok-LLM/
├── src/hanok/
│   ├── model/
│   │   └── architecture.py          # KoreanLLM (RMSNorm · RoPE · SwiGLU · KV cache)
│   ├── data/
│   │   └── legacy.py                # DatasetManager, LocalKoreanDataset, collate, Parquet 캐시
│   ├── training/
│   │   ├── config.py                # TrainingConfig dataclass (모든 기본값)
│   │   ├── trainer.py               # 메인 오케스트레이션 (auto / pretrain / sft)
│   │   ├── optimizer.py             # AdamW / AdamW8bit 선택
│   │   ├── scheduler.py             # Cosine + Warmup
│   │   ├── checkpoint.py            # 저장 / 로드 / find_latest
│   │   └── reproducibility.py       # Seed & 분산 헬퍼
│   ├── inference/
│   │   ├── loader.py                # 체크포인트 + 토크나이저 로딩
│   │   └── generation.py            # KV cache 기반 Autoregressive 생성
│   ├── evaluation/
│   │   ├── perplexity.py
│   │   ├── benchmark.py
│   │   └── benchmark_ko.jsonl       # 로컬 sanity benchmark (공식 점수 아님)
│   ├── export/
│   │   ├── huggingface.py           # Hanok -> 표준 Transformers Llama 가중치 매핑
│   │   └── ollama.py                # llama.cpp GGUF 변환 + Modelfile + Ollama 등록
│   ├── ui/
│   │   └── monitor.py               # 선택적 학습 GUI
│   └── cli.py                       # `hanok` 엔트리포인트 (train / infer)
├── assets/svg/                      # README 비주얼 키트
├── MODEL_CARD.md
├── DATA_CARD.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
└── requirements.txt
```

---

## 🛠️ 설치 (Windows 11)

```powershell
# 1. Python 3.10+ 환경 생성 / 활성화
python --version

# 2. 드라이버에 맞는 CUDA용 PyTorch 설치
#    → https://pytorch.org (CUDA 버전을 신중히 선택)

# 3. 프로젝트 편집 가능 모드로 설치
python -m pip install -e .

# 4. (선택) 8-bit 옵티마이저 지원
python -m pip install -e ".[8bit]"

# 5. CLI 사용 가능 여부 확인
hanok --help
```

### 의존성 (`requirements.txt` / `pyproject.toml` 기준)

```text
torch
transformers
datasets
accelerate
bitsandbytes          # 선택, 8-bit AdamW용
pyarrow
pandas
requests
tqdm
matplotlib
setuptools
wheel
```

---

## 🎯 빠른 시작

### SFT (사전학습 체크포인트에서 시작)

```powershell
hanok train --stage sft --resume-from-checkpoint checkpoints/pretrain/hanok_llm_10000.pth --no-gui
```

### Pretraining

```powershell
hanok train --stage pretrain --no-gui
```

### Auto 파이프라인 (Pretrain → SFT)

```powershell
hanok train --no-gui
```

자동 경로는 먼저 Pretraining을 완료(또는 최신 Pretrain 체크포인트에서 재개)한 뒤,  
해당 체크포인트에서 새로운 옵티마이저·스케줄로 SFT를 시작합니다.
학습 파이프라인을 바꾼 뒤 새로 시작할 때는 기존 `checkpoints/pretrain`과 `checkpoints/sft` 폴더를 다른 이름으로 옮겨 두세요. Auto 모드는 발견한 최신 Pretraining 체크포인트에서 자동 재개하며, 기존 전체 가중치 SFT 체크포인트는 LoRA 방식으로 이어서 재개할 수 없습니다.

### 유용한 학습 플래그

| 플래그                        | 설명                                          |
|:------------------------------|:----------------------------------------------|
| `--stage {auto,pretrain,sft}` | 학습 단계                                     |
| `--no-gui`                    | 선택적 학습 모니터 비활성화                   |
| `--stream-datasets`           | 전체 다운로드 대신 스트리밍                   |
| `--samples-per-dataset N`     | 데이터셋당 예제 수 제한 (스모크 / 디버그)     |
| `--max-steps N`               | 최대 학습 스텝 오버라이드                     |
| `--batch-size N`              | 배치 크기 오버라이드                          |
| `--max-seq-len N`             | RoPE 컨텍스트와 학습 시퀀스 길이 설정 (기본값: 1,024) |
| `--resume-from-checkpoint`    | 경로 또는 `latest`                            |
| `--force-redownload`          | 데이터셋 강제 재다운로드                      |

---

## 💬 추론 (Inference)

```powershell
hanok infer `
  --checkpoint checkpoints/korean_llm_00010.pth `
  --prompt "안녕하세요. 너는 누구니?" `
  --max-tokens 128
```

### 생성 제어 옵션

| 플래그                   | 기본값 | 설명                                    |
|:-------------------------|:------:|:----------------------------------------|
| `--temperature`          | 0.6    | Softmax 온도 (0보다 커야 함)            |
| `--top-k`                | 40     | 상위 k개 로짓만 유지                    |
| `--top-p`                | 0.95   | Nucleus sampling                        |
| `--repetition-penalty`   | 1.3    | 이미 생성된 토큰에 페널티 적용          |
| `--max-seq-len`          | 1024   | 로드 시 사용하는 컨텍스트 윈도우        |
| `--device`               | auto   | `cuda` / `cpu`                          |

### 생성 동작 방식

1. 프롬프트를 Instruction 형식으로 감쌉니다:  
   `### 질문: {prompt}\n### 응답:`
2. **KV cache**를 사용해 Autoregressive 방식으로 토큰을 생성합니다.
3. Temperature 스케일링 → (선택) top-k 필터링 → Softmax → (선택) top-p Nucleus 필터링 → Multinomial 샘플링.
4. 이미 등장한 토큰에 Repetition Penalty를 적용합니다.
5. EOS를 만나거나 시퀀스 길이 안전 제한에 도달하면 중단합니다.
6. `### 응답:` 이후의 텍스트만 반환합니다.

---

## 🗂️ 데이터 소스 & 파이프라인

### 기본 데이터셋

| 단계         | 데이터셋                   | Config              | Split | 비고 |
|:-------------|:---------------------------|:--------------------|:------|:-----|
| Pretraining  | `AdaMLLab/KorMix`          | `minhash_deduped`   | train | 텍스트 코퍼스 |
| SFT          | `nlpai-lab/kullm-v2`       | default             | train | Instruction |
| SFT          | `beomi/KoAlpaca-v1.1a`     | default             | train | Instruction |

**토크나이저:** `beomi/Llama-3-Open-Ko-8B`  
토크나이저에 별도 pad 토큰이 없으면 `<|pad|>`를 추가합니다.

### 데이터 처리 상세

- Hugging Face 다운로드는 `./datasets/cache`에 캐시됩니다.
- 다운로드된 행은 **Parquet**으로 저장되고 매니페스트에 기록됩니다.
- 여러 가지 일반적인 스키마를 지원합니다:  
  `text` / `content` / `document` / `body`,  
  `instruction` / `input` / `output`,  
  `question` / `response`, `question` / `answer`, `prompt` / `response`.
- **SFT 모드**에서는 프롬프트 부분을 마스킹하고, 응답 토큰만 Loss에 기여합니다.
- EOS를 붙이고, PAD 위치는 `ignore_index=-100`으로 무시합니다.
- 빠른 반복을 위해 스트리밍과 `samples_per_dataset` 제한을 사용할 수 있습니다.

> 학습 전에 반드시 원본 데이터셋의 라이선스와 이용 약관을 확인하세요.  
> 이 저장소는 제3자 데이터를 재배포하거나 재라이선스하지 않습니다.

---

## 📤 체크포인트 Export 및 평가

Hanok 체크포인트는 표준 Transformers `LlamaForCausalLM` 폴더로 변환할 수 있습니다. 이 폴더에는 `model.safetensors`, Llama 설정, tokenizer 파일, 원본 Hanok 가중치가 포함됩니다.

```powershell
# 표준 Transformers Llama 형식
hanok export --checkpoint checkpoints/sft/hanok_llm_10000.pth --format hf --output exports/hanok-hf

# GGUF 변환 (llama.cpp 체크아웃이 필요합니다)
hanok export --checkpoint checkpoints/sft/hanok_llm_10000.pth --format gguf `
  --output exports/hanok-f16.gguf --llama-cpp-dir D:\tools\llama.cpp

# Q4_K_M 양자화 후 Ollama 모델 생성 (llama-quantize와 Ollama 설치 필요)
hanok export --checkpoint checkpoints/sft/hanok_llm_10000.pth --format ollama `
  --output exports/hanok-f16.gguf --llama-cpp-dir D:\tools\llama.cpp `
  --quantization Q4_K_M --ollama-name hanok:latest
```

Transformers 폴더는 `AutoModelForCausalLM.from_pretrained("exports/hanok-hf")`로 로드합니다. GGUF/Ollama 변환은 외부 `llama.cpp` 및 선택적 양자화 도구를 실행합니다.

`hanok evaluate --checkpoint ...`는 `src/hanok/evaluation/benchmark_ko.jsonl`의 객관식 sanity suite와 별도의 한국어 서술형 프롬프트를 결정적으로 평가합니다. 정답률·답변 형식 준수율·한글 비율·반복률·빈 답변률·선택지 글자 기준 perplexity와 생성 답변을 JSON 보고서에 저장합니다. 선택지 글자 perplexity는 이 소규모 시험 형식에만 해당하며 학습 검증 손실과 직접 비교하지 않습니다. 15개 객관식과 3개 서술형 프롬프트는 회귀 확인용이며 공식 성능 벤치마크가 아닙니다. 높은 점수도 일반적인 한국어 지능이나 사실 정확성을 보장하지 않습니다.

이 저장소는 학습된 가중치를 제공하지 않습니다. 모델은 무작위 초기화부터 학습하므로 체크포인트 품질은 학습 데이터와 학습량에 달려 있습니다. Export는 학습 품질을 올리거나 사전학습된 Llama 가중치를 가져오지 않습니다.

---

## 🔬 Model Card & Data Card

| 문서 | 목적 |
|:-----|:-----|
| [`MODEL_CARD.md`](MODEL_CARD.md) | 아키텍처 요약, 의도된 사용, 한계, 토크나이저 노트 |
| [`DATA_CARD.md`](DATA_CARD.md)   | 데이터셋 소스, 전처리 동작, 재현성 관련 주의사항 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 기여 방법 |
| [`SECURITY.md`](SECURITY.md)     | 보안 보고 절차 |

---

## 🧭 설계 원칙

1. **Inspectability (들여다볼 수 있음)** — 모든 계층이 일반 PyTorch이며, 블랙박스 커널이 없습니다.
2. **Explicit defaults (명시적 기본값)** — 학습 하이퍼파라미터는 한곳(`TrainingConfig`)에 선언됩니다.
3. **Local-first (로컬 우선)** — 체크포인트는 원격 레지스트리 없이도 로드할 수 있는 네이티브 `.pth` 파일입니다.
4. **Windows + CUDA 실용성** — 문서화된 경로는 이론적인 Linux 전용 랩이 아니라, 실제 워크스테이션 워크플로입니다.
5. **Commercial trajectory (상용화 궤적)** — 아키텍처와 도구는 프로덕션 릴리스(가중치 + 벤치마크 + HF/Ollama)가 이 코드베이스 위에 깔끔하게 올라갈 수 있도록 설계되었습니다.

---

## 📜 라이선스

[`LICENSE`](LICENSE) 파일을 참고하세요.  
제3자 데이터셋, 토크나이저, 의존성에는 별도의 이용 약관이 적용됩니다.  
상용 사용 전에 반드시 확인하시기 바랍니다.

---

<div align="center">

### 🏯 HANOK · KOREAN LLM ENGINEERING

**들여다볼 수 있는 아키텍처 · 실용적인 학습 · 네이티브 추론 · Windows/CUDA 워크플로 · 상용화를 위해 설계됨**

**저장소:** [Hanok-LLM](https://github.com/seoan1024/Hanok-LLM.git) · **연구 기반:** [Korean-llm](https://github.com/seoan1024/Korean-llm.git)

<img src="assets/svg/hanok-overview.svg" alt="Hanok closing panel" width="100%"/>

</div>
