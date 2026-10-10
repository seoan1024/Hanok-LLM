"""Training orchestration migrated from v4; defaults and optimizer policy preserved."""
import argparse, logging, random, time, threading, json, os
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from ..model.architecture import KoreanLLM
from ..inference.generation import generate
from ..training.checkpoint import save_checkpoint,load_checkpoint,find_latest_checkpoint
from ..training.config import TrainingConfig
from ..training.optimizer import build_optimizer
from ..training.scheduler import build_scheduler
from ..training.reproducibility import setup_distributed
from ..data.legacy import (DatasetManager,LocalKoreanDataset,collate_fn,
                           SequencePackingCollator,RowGroupBatchSampler,ensure_datasets_dir)
from ..ui.monitor import TrainingMonitorGUI

LOG_DIR=Path('./logs')
LOG_DIR.mkdir(parents=True,exist_ok=True)
LOSS_HISTORY_FILE=LOG_DIR/'loss_history.json'
loss_history=[]
gui_monitor=None
DATASETS_DIR=Path("./datasets")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR/"training.log",encoding="utf-8")
    ]
)
logger=logging.getLogger(__name__)


def save_loss_history():
    try:
        LOSS_HISTORY_FILE.write_text(json.dumps(loss_history,indent=2,ensure_ascii=False),encoding="utf-8")
    except OSError:
        logger.exception("Loss history save failed")


def load_loss_history():
    global loss_history
    if LOSS_HISTORY_FILE.exists():
        loss_history=json.loads(LOSS_HISTORY_FILE.read_text(encoding="utf-8"))


def build_dataset_configs(config: TrainingConfig) -> List[Dict]:
    """Turn CLI dataset names into DatasetManager configurations.

    For SFT, the original two instruction datasets remain the default. For
    pretraining, require an explicit dataset so users intentionally choose a
    licensed Korean text corpus instead of accidentally training on SFT data.
    Dataset syntax: `repository_id` or `repository_id:configuration`.
    """
    if not config.dataset_specs:
        if config.stage=="sft":
            if config.stream_datasets:
                return [{**dataset,"streaming":True,"max_examples":config.samples_per_dataset} for dataset in DatasetManager.DEFAULT_SFT_DATASETS]
            return DatasetManager.DEFAULT_SFT_DATASETS

        if config.stage=="pretrain":
            return [{
                "name":"AdaMLLab/KorMix",
                "config":"minhash_deduped",
                "split":"train",
                "text_keys":["text","content","document","body"],
                "max_examples":config.samples_per_dataset,
                "streaming":config.stream_datasets,
            }]

        raise ValueError("Pretraining requires --dataset REPOSITORY[:CONFIG]")

    configs=[]
    for spec in config.dataset_specs:
        name,separator,dataset_config=spec.partition(":")
        configs.append({
            "name":name,
            "config":dataset_config if separator else None,
            "split":"train",
            "text_keys":["text","content","document","body"],
            "max_examples":config.samples_per_dataset,
            "streaming":config.stream_datasets,
        })
    return configs


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_bfloat16: bool, max_batches: int) -> Optional[float]:
    if len(loader)==0:
        return None

    was_training=model.training
    model.eval()
    total_nll=0.0
    total_targets=0

    try:
        evaluated_batches=0
        for batch in loader:
            if evaluated_batches>=max_batches:
                break

            input_ids=batch["input_ids"].to(device,non_blocking=True)
            labels=batch["labels"].to(device,non_blocking=True)
            if input_ids.numel()==0:
                continue

            if device.type=="cuda" and use_bfloat16:
                with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                    _,loss,_=model(input_ids,labels=labels)
            else:
                _,loss,_=model(input_ids,labels=labels)

            target_count=int((labels[..., 1:]!=-100).sum().item())
            if target_count:
                total_nll+=loss.item()*target_count
                total_targets+=target_count
                evaluated_batches+=1
    finally:
        model.train(was_training)

    return total_nll/total_targets if total_targets else None


def main(config: TrainingConfig=TrainingConfig(),load_training_state: bool=True):
    global gui_monitor

    if config.stage=="sft" and not config.resume_from_checkpoint:
        raise ValueError(
            "SFT requires --resume-from-checkpoint pointing to a pretrained Hanok checkpoint; "
            "use --stage auto to run pretraining followed by SFT."
        )

    if config.stage=="auto":
        logger.info("🚀 Auto mode: pretraining followed by SFT")

        pretrain_dir="checkpoints/pretrain"
        pretrain_resume=config.resume_from_checkpoint

        if pretrain_resume is None and find_latest_checkpoint(pretrain_dir):
            pretrain_resume="latest"

        pretrain_config=replace(config,stage="pretrain",max_steps=config.pretrain_steps,checkpoint_dir=pretrain_dir,resume_from_checkpoint=pretrain_resume,enable_gui=False,learning_rate=config.learning_rate)
        pretrain_result=main(pretrain_config)

        if pretrain_result is False:
            logger.error("Auto mode stopped: pretraining did not complete successfully.")
            return False

        pretrain_checkpoint=find_latest_checkpoint(pretrain_dir)

        if pretrain_checkpoint is None:
            logger.error("Auto mode stopped: no pretraining checkpoint was produced, so SFT cannot start.")
            return False

        sft_dir="checkpoints/sft"
        sft_checkpoint=find_latest_checkpoint(sft_dir)

        if sft_checkpoint is not None:
            logger.info(f"✅ SFT checkpoint found. Resuming from {sft_checkpoint}")
            sft_resume=sft_checkpoint
            load_sft_training_state=True
        else:
            logger.info(f"✅ No SFT checkpoint found. Starting SFT from {pretrain_checkpoint}")
            sft_resume=pretrain_checkpoint
            load_sft_training_state=False

        sft_config=replace(
            config,
            stage="sft",
            dataset_specs=None,
            max_steps=config.sft_steps,
            checkpoint_dir=sft_dir,
            resume_from_checkpoint=sft_resume,
            learning_rate=config.sft_learning_rate,
        )
        return main(sft_config,load_training_state=load_sft_training_state)

    setup_distributed(config.seed)
    ensure_datasets_dir()
    load_loss_history()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"📂 Datasets directory: {DATASETS_DIR.absolute()}")
    logger.info(f"📂 Logs directory: {LOG_DIR.absolute()}")

    # ============================================
    # 1. 토크나이저 로드
    # ============================================
    logger.info("Loading tokenizer...")

    tokenizer=AutoTokenizer.from_pretrained(
        "beomi/Llama-3-Open-Ko-8B",
        clean_up_tokenization_spaces=False
    )

    if tokenizer.pad_token is None or tokenizer.pad_token_id==tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
        logger.info("✅ Added separate pad token: <|pad|>")

    logger.info(f"Tokenizer: vocab_size={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, pad_id={tokenizer.pad_token_id}")

    # ============================================
    # 2. 데이터셋 다운로드 및 로드
    # ============================================
    logger.info("Setting up datasets...")

    try:
        dataset_configs=build_dataset_configs(config)
    except ValueError as e:
        logger.error(str(e))
        return False

    manager=DatasetManager(dataset_configs=dataset_configs)
    dataset_paths=manager.get_or_download_all(force=config.force_redownload)

    if not dataset_paths:
        logger.error("❌ No datasets available!")
        return False

    dataset=LocalKoreanDataset(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        data_samples_per_dataset=config.samples_per_dataset,
        train_on_response_only=(config.stage=="sft")
    )

    if len(dataset)==0:
        logger.error("❌ Dataset is empty!")
        return False

    if config.stage == "pretrain":
        train_sampler=RowGroupBatchSampler(dataset,config.batch_size,config.seed,
                                           validation_fraction=config.validation_split)
        validation_sampler=RowGroupBatchSampler(dataset,config.batch_size,config.seed,
                                                validation=True,
                                                validation_fraction=config.validation_split)
        loader=DataLoader(dataset,batch_sampler=train_sampler,num_workers=config.num_workers,
                          collate_fn=SequencePackingCollator(config.max_seq_len,config.batch_size),
                          pin_memory=(device.type=="cuda"),persistent_workers=(config.num_workers>0))
        validation_loader=DataLoader(dataset,batch_sampler=validation_sampler,
                                     num_workers=config.num_workers,
                                     collate_fn=SequencePackingCollator(config.max_seq_len,config.batch_size),
                                     pin_memory=(device.type=="cuda"),
                                     persistent_workers=(config.num_workers>0)) if len(validation_sampler) else None
        train_size=len(dataset)-int(len(dataset)*config.validation_split)
        validation_size=int(len(dataset)*config.validation_split)
    else:
        validation_size=max(1,int(len(dataset)*config.validation_split)) if len(dataset)>1 and config.validation_split>0 else 0
        train_size=len(dataset)-validation_size
        if train_size==0:
            logger.error("Validation split leaves no training examples")
            return False
        split_generator=torch.Generator().manual_seed(config.seed)
        train_dataset,validation_dataset=random_split(dataset,[train_size,validation_size],generator=split_generator)
        loader=DataLoader(train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,
                          shuffle=True,collate_fn=collate_fn,pin_memory=(device.type=="cuda"))
        validation_loader=DataLoader(validation_dataset,batch_size=config.batch_size,num_workers=config.num_workers,
                                     shuffle=False,collate_fn=collate_fn,
                                     pin_memory=(device.type=="cuda")) if validation_size else None

    # ============================================
    # 3. 모델 생성 (VRAM 최적화: BF16 변환)
    # ============================================
    logger.info("Creating model...")

    model_config=dict(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        dim=1920,
        n_layers=20,
        n_heads=10,
        max_seq_len=config.max_seq_len
    )

    # Keep trainable parameters in fp32. Autocast below still runs eligible
    # CUDA operations in bf16, while the optimizer updates fp32 weights rather
    # than rounding every update back into bf16 parameter precision.
    model=KoreanLLM(**model_config).to(device)
    if config.stage=="sft":
        model.enable_lora(rank=8,alpha=16)

    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {total_params/1e6:.1f}M total params, {trainable_params/1e6:.1f}M trainable")

    # ============================================
    # GUI 시작
    # ============================================
    def start_gui():
        global gui_monitor
        gui_monitor=TrainingMonitorGUI(tokenizer,device,model_config,config.checkpoint_dir)
        import hanok.ui.monitor as monitor_module
        monitor_module.loss_history=loss_history
        gui_monitor.run()

    if config.enable_gui:
        gui_thread=threading.Thread(target=start_gui,daemon=True)
        gui_thread.start()
        time.sleep(1.5)
        logger.info("🖥️  Monitoring GUI started (Loss graph + Chat)")

    # ============================================
    # 4. 옵티마이저와 스케줄러 (VRAM 최적화: 8-bit Optimizer)
    # ============================================
    optimizer=build_optimizer(model,config,device)
    scheduler=build_scheduler(optimizer,config)

    # BF16 사용 시 Scaler는 기본적으로 필요 없으나, 하위 호환성을 위해 유지
    scaler=torch.amp.GradScaler("cuda") if (device.type=="cuda" and not config.use_bfloat16) else None

    # ============================================
    # 5. 체크포인트 로드
    # ============================================
    start_step=0

    if config.resume_from_checkpoint:
        checkpoint_path=config.resume_from_checkpoint

        if checkpoint_path.lower()=="latest":
            checkpoint_path=find_latest_checkpoint(config.checkpoint_dir)

            if checkpoint_path is None:
                logger.warning("No checkpoint found, starting from scratch")

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")

            start_step=load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                device,
                load_training_state=load_training_state,
            )

            if gui_monitor:
                gui_monitor.notify_checkpoint(checkpoint_path)

        elif checkpoint_path and checkpoint_path.lower()!="latest":
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {checkpoint_path}")

    # ============================================
    # 6. 학습 루프
    # ============================================
    logger.info(f"🚀 Starting training from step {start_step}...")
    logger.info(f"📊 Dataset size: train={train_size}, validation={validation_size}")
    logger.info(f"📊 Total batches per epoch: {len(loader)}")

    model.train()
    optimizer.zero_grad()

    running_loss=0.0
    running_target_tokens=0
    accumulated_target_tokens=0
    step=0
    interrupted=False

    try:
        epoch=0

        while True:
            epoch+=1
            logger.info(f"\n📍 Epoch {epoch}")

            for batch_idx,batch in enumerate(loader):
                actual_step=(step//config.accumulation_steps)+start_step

                if actual_step>=config.max_steps:
                    logger.info(f"Reached max steps ({config.max_steps}), stopping training")
                    break

                input_ids=batch["input_ids"].to(device,non_blocking=True)
                labels=batch["labels"].to(device,non_blocking=True)
                target_count=int((labels[..., 1:]!=-100).sum().item())
                if target_count == 0 and config.stage == "pretrain":
                    continue
                if target_count == 0:
                    raise ValueError(
                        "Training batch has no supervised next-token targets; "
                        "check the dataset and max_seq_len."
                    )
                accumulated_target_tokens+=target_count

                if device.type=="cuda" and config.use_bfloat16:
                    with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                        _,loss,_=model(input_ids,labels=labels)
                        loss_scaled=loss*target_count

                    loss_scaled.backward()
                else:
                    _,loss,_=model(input_ids,labels=labels)
                    loss_scaled=loss*target_count
                    loss_scaled.backward()

                running_loss+=loss.item()*target_count
                running_target_tokens+=target_count

                if step%4==0:
                    print(".",end="",flush=True)

                if (step+1)%config.accumulation_steps==0:
                    if device.type=="cuda" and config.use_bfloat16:
                        pass
                    elif scaler:
                        scaler.unscale_(optimizer)

                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_target_tokens)

                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

                    if scaler and not config.use_bfloat16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                    actual_step=(step+1)//config.accumulation_steps+start_step
                    avg_loss=running_loss/max(1,running_target_tokens)
                    lr=scheduler.get_last_lr()[0]

                    validation_loss=(
                        evaluate(model,validation_loader,device,config.use_bfloat16,config.validation_batches)
                        if validation_loader and actual_step%config.eval_interval==0 else None
                    )

                    validation_text=f" | Val: {validation_loss:.4f}" if validation_loss is not None else ""
                    log_msg=f"[Step {actual_step:5d}] Train: {avg_loss:.4f}{validation_text} | LR: {lr:.2e} | Tokens/update: {running_target_tokens}"
                    logger.info(f"\n{log_msg}")

                    loss_history.append({
                        "step":actual_step,
                        "loss":float(avg_loss),
                        "validation_loss":float(validation_loss) if validation_loss is not None else None,
                        "lr":float(lr),
                        "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    save_loss_history()

                    if gui_monitor:
                        gui_monitor.notify_loss()

                    running_loss=0.0
                    running_target_tokens=0
                    accumulated_target_tokens=0

                    if actual_step%config.eval_interval==0:
                        logger.info("\n📝 Generating samples...")

                        prompts=[
                            "한국의 수도는",
                            "인공지능이란",
                            "안녕?"
                        ]

                        for prompt in prompts:
                            response=generate(
                                model,
                                tokenizer,
                                prompt=prompt,
                                max_tokens=50,
                                temperature=0.7,
                                top_p=0.95,
                                device=device
                            )
                            logger.info(f"  Q: {prompt}\n  A: {response}")

                        checkpoint_path=f"{config.checkpoint_dir}/hanok_llm_{actual_step:05d}.pth"
                        save_checkpoint(model,optimizer,scheduler,actual_step,checkpoint_path,stage=config.stage)

                        if gui_monitor:
                            gui_monitor.notify_checkpoint(checkpoint_path)
                            gui_monitor.notify_log(f"체크포인트 저장됨: {Path(checkpoint_path).name}")

                step+=1

            if actual_step>=config.max_steps:
                break

    except KeyboardInterrupt:
        interrupted=True
        logger.info("\n⚠️ Training interrupted by user")

        actual_step=(step//config.accumulation_steps)+start_step
        checkpoint_path=f"{config.checkpoint_dir}/hanok_llm_interrupted_{actual_step:05d}.pth"

        save_checkpoint(model,optimizer,scheduler,actual_step,checkpoint_path,stage=config.stage)

        if gui_monitor:
            gui_monitor.notify_checkpoint(checkpoint_path)

    except Exception:
        logger.exception("Training failed")
        save_loss_history()
        raise

    if interrupted:
        logger.info("Training interrupted; recovery checkpoint saved.")
    else:
        final_step=(step//config.accumulation_steps)+start_step

        if final_step>start_step:
            final_checkpoint=f"{config.checkpoint_dir}/hanok_llm_{final_step:05d}.pth"
            save_checkpoint(model,optimizer,scheduler,final_step,final_checkpoint,stage=config.stage)

            if gui_monitor:
                gui_monitor.notify_checkpoint(final_checkpoint)

        logger.info("🎉 Training completed!")

    save_loss_history()

    if gui_monitor and gui_monitor.running:
        logger.info("GUI가 열려 있습니다. 창을 닫으면 종료됩니다.")

        while gui_monitor.running:
            time.sleep(1)

    return not interrupted


def parse_args() -> TrainingConfig:
    parser=argparse.ArgumentParser(description="Train KoreanLLM from scratch: pretraining or SFT")
    parser.add_argument("--stage",choices=["auto","pretrain","sft"],default="auto",help="Default auto runs pretraining and then SFT. Select a stage only for manual control.")
    parser.add_argument("--dataset",action="append",dest="dataset_specs",help="Hugging Face dataset as repository_id or repository_id:config. Repeat for multiple datasets.")
    parser.add_argument("--batch-size",type=int,default=2)
    parser.add_argument("--accumulation-steps",type=int,default=8)
    parser.add_argument("--max-steps",type=int,default=100_000,help="Steps for a manually selected stage.")
    parser.add_argument("--pretrain-steps",type=int,default=100_000)
    parser.add_argument("--sft-steps",type=int,default=10_000)          # ★ 변경: 25000 → 10000
    parser.add_argument("--max-seq-len",type=int,default=1024)
    parser.add_argument("--learning-rate",type=float,default=5e-5)
    parser.add_argument("--sft-learning-rate",type=float,default=1.5e-5)  # ★ 새로 추가
    parser.add_argument("--warmup-steps",type=int,default=200)
    parser.add_argument("--weight-decay",type=float,default=0.1)
    parser.add_argument("--eval-interval",type=int,default=5000)
    parser.add_argument("--validation-split",type=float,default=0.02)
    parser.add_argument("--validation-batches",type=int,default=32)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--samples-per-dataset",type=int)
    parser.add_argument("--resume-from-checkpoint",default=None)
    parser.add_argument("--force-redownload",action="store_true")
    parser.add_argument("--stream-datasets",action="store_true",help="Stream and cache only --samples-per-dataset records; required for huge corpora.")
    parser.add_argument("--no-bfloat16",action="store_false",dest="use_bfloat16")
    parser.add_argument("--no-8bit-optimizer",action="store_false",dest="use_8bit_optimizer")

    gui_group=parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui",action="store_true",dest="enable_gui",default=True,help="Enable the training monitor GUI (enabled by default).")
    gui_group.add_argument("--no-gui",action="store_false",dest="enable_gui",help="Disable the training monitor GUI.")

    args=parser.parse_args()
    return TrainingConfig(**vars(args))
