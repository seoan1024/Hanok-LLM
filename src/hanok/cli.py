from pathlib import Path
import argparse, json, sys
from datetime import datetime
from .training.trainer import main, parse_args

def _infer(argv):
    parser=argparse.ArgumentParser(prog='hanok infer',description='Run local inference from a Hanok .pth checkpoint')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--temperature',type=float,default=.6)
    parser.add_argument('--top-k',type=int,default=40)
    parser.add_argument('--top-p',type=float,default=.95)
    parser.add_argument('--repetition-penalty',type=float,default=1.3)
    parser.add_argument('--max-seq-len',type=int,default=1024)
    parser.add_argument('--device',default=None)
    parser.add_argument('--greedy',action='store_true',help='Use deterministic greedy decoding')
    args=parser.parse_args(argv)
    from .inference.loader import load_model
    from .inference.generation import generate
    model,tokenizer,device=load_model(args.checkpoint,device=args.device,max_seq_len=args.max_seq_len)
    print(generate(model,tokenizer,prompt=args.prompt,max_tokens=args.max_tokens,
                   temperature=args.temperature,top_k=args.top_k,top_p=args.top_p,
                   repetition_penalty=args.repetition_penalty,do_sample=not args.greedy,device=device))


def _export(argv):
    parser=argparse.ArgumentParser(prog='hanok export',description='Export a native Hanok checkpoint')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True,help='HF directory or destination GGUF file')
    parser.add_argument('--format',choices=['hf','gguf','ollama'],default='hf')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-seq-len',type=int,default=1024)
    parser.add_argument('--stage',default='trained')
    parser.add_argument('--llama-cpp-dir',help='Existing llama.cpp checkout for GGUF conversion')
    parser.add_argument('--gguf-outtype',choices=['f16','bf16','f32','q8_0','tq1_0','tq2_0','auto'],default='f16')
    parser.add_argument('--quantization',help='Optional llama-quantize type, e.g. Q4_K_M')
    parser.add_argument('--ollama-name',help='Required for --format ollama, e.g. hanok:latest')
    parser.add_argument('--ollama-executable',default='ollama')
    parser.add_argument('--llama-quantize-executable',default='llama-quantize')
    args=parser.parse_args(argv)

    from .export import export_checkpoint, convert_hf_to_gguf, write_modelfile
    if args.format=='hf':
        result=export_checkpoint(args.checkpoint,args.output,device=args.device,
                                 max_seq_len=args.max_seq_len,stage=args.stage)
        print(f'Hugging Face model exported: {result}')
        return

    if not args.llama_cpp_dir:
        parser.error('--llama-cpp-dir is required for GGUF and Ollama export')
    if args.format=='ollama' and not args.ollama_name:
        parser.error('--ollama-name is required for Ollama export')

    import tempfile
    from .export import create_ollama_model
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hanok-hf-export-',dir=target.parent) as temp_dir:
        hf_dir=Path(temp_dir)/'hf'
        export_checkpoint(args.checkpoint,hf_dir,device=args.device,
                          max_seq_len=args.max_seq_len,stage=args.stage)
        gguf_path=convert_hf_to_gguf(
            hf_dir,target,args.llama_cpp_dir,outtype=args.gguf_outtype,
            quantization=args.quantization,
            quantize_executable=args.llama_quantize_executable,
        )
    print(f'GGUF model exported: {gguf_path}')

    if args.format=='ollama':
        modelfile=Path(gguf_path).with_suffix('.Modelfile')
        write_modelfile(gguf_path,modelfile)
        create_ollama_model(modelfile,args.ollama_name,executable=args.ollama_executable)
        print(f'Ollama model created: {args.ollama_name}')


def _evaluate(argv):
    parser=argparse.ArgumentParser(prog='hanok evaluate',description='Run Hanok Korean sanity benchmark and perplexity')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--benchmark',default=None,help='JSONL benchmark; defaults to the bundled Korean sanity suite')
    parser.add_argument('--output',default=None,help='JSON report destination')
    parser.add_argument('--device',default=None)
    parser.add_argument('--max-new-tokens',type=int,default=8)
    parser.add_argument('--language-new-tokens',type=int,default=64)
    parser.add_argument('--max-seq-len',type=int,default=1024)
    args=parser.parse_args(argv)

    from .inference.loader import load_model
    from .evaluation import DEFAULT_BENCHMARK, evaluate_benchmark
    model,tokenizer,device=load_model(args.checkpoint,device=args.device,max_seq_len=args.max_seq_len)
    benchmark_path=args.benchmark or DEFAULT_BENCHMARK
    report=evaluate_benchmark(
        model,tokenizer,path=benchmark_path,device=device,
        max_new_tokens=args.max_new_tokens,
        language_new_tokens=args.language_new_tokens,
        max_seq_len=args.max_seq_len,
    )
    report['checkpoint']=str(args.checkpoint)
    report['benchmark_path']=str(benchmark_path)
    report['evaluated_at']=datetime.now().astimezone().isoformat(timespec='seconds')
    output=Path(args.output) if args.output else (
        Path('evaluation')/'reports'/f"{Path(args.checkpoint).stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Accuracy: {report['correct']}/{report['examples']} ({report['accuracy']:.1%})")
    print(f"Answer format rate: {report['answer_format_rate']:.1%}")
    print(f"Mean Hangul letter ratio: {report['mean_hangul_letter_ratio']:.1%}")
    print(f"Mean repeated-token ratio: {report['mean_repeated_token_ratio']:.1%}")
    print(f"Empty Korean responses: {report['empty_generation_rate']:.1%}")
    print(f"Choice-letter perplexity: {report['choice_letter_perplexity']['perplexity']}")
    print(f"Report: {output.resolve()}")

def entrypoint():
    argv=sys.argv[1:]
    if argv and argv[0]=='infer':
        _infer(argv[1:])
    elif argv and argv[0]=='export':
        _export(argv[1:])
    elif argv and argv[0]=='evaluate':
        _evaluate(argv[1:])
    else:
        if argv and argv[0]=='train':
            sys.argv=[sys.argv[0],*argv[1:]]
        main(parse_args())

