"""Self-supervised MAE pretraining on unlabeled CRISM cubes."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from identification.crism_common import resolve_device, format_torch_runtime

from .dataset import (
    PrefetchWindowLoader,
    UnlabeledWindowDataset,
    discover_unlabeled_files,
)
from .defaults import default_pretrain_args, mae_data_dir, save_last_pretrain
from .model import SpatialSpectralMAE, encoder_from_config


def cosine_warmup_lr(step: int, total_steps: int, warmup_frac: float, base_lr: float, min_lr: float) -> float:
    warmup_steps = max(1, int(round(total_steps * warmup_frac)))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _config_from_args(args: Dict) -> Dict:
    keys = (
        "crop", "spatial_patch", "spectral_patch", "bands", "d_model",
        "encoder_depth", "encoder_heads", "decoder_dim", "decoder_depth",
        "decoder_heads", "p_feat", "p_cont", "mask_ratio",
    )
    return {k: args[k] for k in keys if k in args}


def run_pretrain(config: Dict, log=None) -> Dict:
    args = default_pretrain_args()
    args.update({k: v for k, v in config.items() if v is not None})
    device = resolve_device(args.get("device", "cpu"))
    output_dir = Path(args.get("output_dir") or mae_data_dir() / "pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg)

    _log(format_torch_runtime())
    files = discover_unlabeled_files(args["data_path"], args.get("input_pattern") or "*")
    _log(f"无标签立方体 {len(files)} 个：{args['data_path']}")
    ds = UnlabeledWindowDataset(
        files,
        crop=int(args["crop"]),
        samples_per_epoch=int(args["samples_per_epoch"]),
        data_key=args.get("data_key") or None,
        data_layout=str(args.get("data_layout", "HWB")),
        preprocess_mode=str(args.get("preprocess_mode", "crop")),
        seed=int(args.get("seed", 0)),
    )
    _log(
        f"可读文件 {len(ds.meta)}，跳过 {ds.skipped}；"
        f"每轮 {len(ds)} 个 {args['crop']}×{args['crop']} 窗口"
    )
    batch_size = int(args["batch_size"])
    num_readers = max(1, int(args.get("num_readers", 4)))
    crops_per_read = max(1, int(args.get("crops_per_read", 4)))
    prefetch_batches = max(1, int(args.get("prefetch_batches", 2)))
    loader = PrefetchWindowLoader(
        ds,
        batch_size,
        num_readers=num_readers,
        crops_per_read=crops_per_read,
        drop_last=len(ds) >= batch_size,
        pin_memory=device.type == "cuda",
        prefetch_batches=prefetch_batches,
    )
    mae_cfg = _config_from_args(args)
    model = SpatialSpectralMAE(
        encoder=encoder_from_config(mae_cfg),
        decoder_dim=int(args["decoder_dim"]),
        decoder_depth=int(args["decoder_depth"]),
        decoder_heads=int(args["decoder_heads"]),
        p_feat=float(args["p_feat"]),
        p_cont=float(args["p_cont"]),
        mask_ratio=float(args["mask_ratio"]),
    ).to(device)
    n_enc = model.encoder.num_parameters()
    _log(
        f"编码器参数 {n_enc:,}；tokens={model.encoder.n_patches} "
        f"({model.encoder.n_h}×{model.encoder.n_w}×{model.encoder.n_c})"
    )
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        _log(
            f"实际训练设备：{device}（{gpu_name}），"
            f"模型已上 GPU，当前显存约 {mem:.0f} MiB。"
            "读盘与计算重叠进行；若日志里「读盘」仍远大于「计算」，"
            "再加大读盘线程或 batch，不要加每轮窗口数。"
        )
    else:
        _log(
            f"实际训练设备：{device}（CPU）。"
            "若你选了 cuda:0 却看到这一行，说明当前 PyTorch 是 CPU 版，"
            "请看日志开头的 cuda.is_available。"
        )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(args["lr"]),
        weight_decay=float(args["weight_decay"]),
        betas=(0.9, 0.95),
    )
    epochs = int(args["epochs"])
    steps_per_epoch = max(1, len(loader))
    total_steps = max(1, epochs * steps_per_epoch)
    log_path = output_dir / "pretrain_log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["epoch", "step", "lr", "loss", "feat_loss", "cont_loss", "seconds"])

    use_amp = device.type == "cuda" and bool(args.get("use_amp", True))
    _log(
        f"开始训练：{epochs} 轮 × 每轮 {len(ds)} 窗口 / batch={batch_size} "
        f"= 每轮 {steps_per_epoch} 个 batch；"
        f"读盘 {num_readers} 线程，每次开文件抽 {crops_per_read} 个窗口，"
        f"预取 {prefetch_batches} 个 batch。"
        "利用率低时先加读盘线程和 batch；轮数/每轮窗口只决定训多久。"
    )
    model.train()
    step = 0
    avg_loss = 0.0
    lr_now = float(args["lr"])
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_sum = 0.0
        n_steps = 0
        t_ep = time.perf_counter()
        _log(f"—— 第 {epoch}/{epochs} 轮开始，共 {steps_per_epoch} 个 batch ——")
        data_iter = iter(loader)
        while True:
            if n_steps == 0:
                _log(
                    "正在从磁盘组装第 1 个 batch（每个窗口一次随机裁剪，"
                    "可能要几分钟；这期间 GPU 会空转）…"
                )
            t_load = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            cube = batch.to(device, non_blocking=True)
            load_dt = time.perf_counter() - t_load
            lr_now = cosine_warmup_lr(
                step, total_steps, float(args["warmup_frac"]),
                float(args["lr"]), float(args["min_lr"]),
            )
            for group in optim.param_groups:
                group["lr"] = lr_now
            optim.zero_grad(set_to_none=True)
            t_fw = time.perf_counter()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(cube)
                    loss = out["loss"]
                loss.backward()
            else:
                out = model(cube)
                loss = out["loss"]
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            n_steps += 1
            step += 1
            log_this = n_steps <= 3 or n_steps % 10 == 0 or n_steps == steps_per_epoch
            if log_this and device.type == "cuda":
                torch.cuda.synchronize(device)
            fw_dt = time.perf_counter() - t_fw
            ep_sum += float(loss.detach())
            if log_this:
                mem_note = ""
                if device.type == "cuda":
                    mem_note = (
                        f"  显存 {torch.cuda.memory_allocated(device) / (1024 ** 2):.0f} MiB"
                    )
                _log(
                    f"[ep {epoch}/{epochs}  batch {n_steps}/{steps_per_epoch}] "
                    f"loss {float(loss.detach()):.4f}  "
                    f"读盘 {load_dt:.1f}s  计算 {fw_dt:.2f}s  "
                    f"设备 {device}{mem_note}"
                )
        avg_loss = ep_sum / max(1, n_steps)
        dt = time.perf_counter() - t_ep
        writer.writerow([epoch, step, f"{lr_now:.6e}", f"{avg_loss:.6f}", "", "", f"{dt:.2f}"])
        log_f.flush()
        _log(f"[ep {epoch}/{epochs}] 本轮平均 loss {avg_loss:.4f}  lr {lr_now:.2e}  ({dt:.1f}s)")
        if epoch % 10 == 0 or epoch == epochs:
            ckpt = {
                "encoder_state_dict": model.encoder_state_dict(),
                "pretrain_full_state_dict": model.state_dict(),
                "config": mae_cfg,
                "pretrain_args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in args.items()
                    if k != "log"
                },
                "epoch_done": epoch,
                "final_loss": avg_loss,
            }
            path = ckpt_dir / f"encoder_ep{epoch}.pt"
            torch.save(ckpt, path)
            latest = ckpt_dir / "encoder.pt"
            torch.save(ckpt, latest)
            _log(f"已保存 {latest}")

    log_f.close()
    latest = ckpt_dir / "encoder.pt"
    record = {
        "checkpoint_path": str(latest),
        "output_dir": str(output_dir),
        "n_files": len(ds.meta),
        "epochs": epochs,
        "final_loss": avg_loss,
        "elapsed_s": time.perf_counter() - t0,
        "config": mae_cfg,
    }
    save_last_pretrain(record)
    _log(f"预训练完成。编码器：{latest}  用时 {record['elapsed_s']/60:.1f} min")
    return record
