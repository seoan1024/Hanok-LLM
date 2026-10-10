"""Convert a Transformers Llama bundle to GGUF and create Ollama models."""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def convert_hf_to_gguf(model_dir, output_path, llama_cpp_dir, *, outtype="f16",
                       quantization=None, quantize_executable="llama-quantize"):
    """Run the official llama.cpp HF-to-GGUF converter on a local model bundle.

    llama.cpp is an external dependency; this function never downloads or
    clones it. The source bundle must be a Hanok export (standard Llama format).
    """
    model = Path(model_dir).resolve()
    llama_root = Path(llama_cpp_dir).resolve()
    converter = llama_root / "convert_hf_to_gguf.py"
    if not model.is_dir() or not (model / "config.json").is_file():
        raise FileNotFoundError(f"Transformers model bundle not found: {model}")
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp converter not found: {converter}")
    if outtype not in {"f16", "bf16", "f32", "q8_0", "tq1_0", "tq2_0", "auto"}:
        raise ValueError(f"Unsupported GGUF output type: {outtype}")
    if quantization and outtype not in {"f16", "bf16", "f32", "auto"}:
        raise ValueError("Quantization requires an unquantized f16/bf16/f32/auto source")

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(converter), str(model),
        "--outfile", str(target), "--outtype", outtype,
    ]
    subprocess.run(command, cwd=llama_root, check=True)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"llama.cpp did not produce a GGUF file at {target}")

    if quantization:
        quantizer = shutil.which(quantize_executable)
        if quantizer is None and Path(quantize_executable).is_file():
            quantizer = str(Path(quantize_executable).resolve())
        if quantizer is None:
            raise FileNotFoundError(
                f"llama-quantize executable not found: {quantize_executable}"
            )
        quantized = target.with_name(f"{target.stem}-{quantization}.gguf")
        subprocess.run(
            [quantizer, str(target), str(quantized), quantization], check=True
        )
        if not quantized.is_file() or quantized.stat().st_size == 0:
            raise RuntimeError(f"llama-quantize did not produce {quantized}")
        return quantized
    return target


def write_modelfile(gguf_path, output_path, *, template=None, context=1024,
                    temperature=0.6):
    """Write a Hanok-compatible Ollama Modelfile for an existing GGUF."""
    model = Path(gguf_path).resolve()
    if model.suffix.lower() != ".gguf" or not model.is_file():
        raise ValueError("A converted GGUF model file is required")
    if context < 1:
        raise ValueError("context must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite number greater than zero")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    template = template or (
        "{{ if .System }}### 시스템: {{ .System }}\n{{ end }}"
        "{{ if .Prompt }}### 질문: {{ .Prompt }}\n{{ end }}"
        "### 응답:"
    )
    escaped_model = json.dumps(model.as_posix(), ensure_ascii=False)
    content = (
        f"FROM {escaped_model}\n"
        f'TEMPLATE """{template}"""\n'
        f"PARAMETER num_ctx {context}\n"
        f"PARAMETER temperature {temperature}\n"
        'PARAMETER stop "### 질문:"\n'
        'PARAMETER stop "<|eot_id|>"\n'
    )
    target.write_text(content, encoding="utf-8")
    return target


def create_ollama_model(modelfile_path, name, *, executable="ollama"):
    """Register an exported GGUF in a running/local Ollama installation."""
    modelfile = Path(modelfile_path).resolve()
    if not modelfile.is_file():
        raise FileNotFoundError(f"Modelfile not found: {modelfile}")
    ollama = shutil.which(executable)
    if ollama is None and Path(executable).is_file():
        ollama = str(Path(executable).resolve())
    if ollama is None:
        raise FileNotFoundError(f"Ollama executable not found: {executable}")
    subprocess.run([ollama, "create", name, "-f", str(modelfile)], check=True)
    return name
