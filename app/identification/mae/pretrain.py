"""Self-supervised MAE pretraining on unlabeled CRISM cubes."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from identification.crism_common import resolve_device, format_torch_runtime

from .dataset import UnlabeledWindowDataset, discover_unlabeled_files
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
    loader = DataLoader(
        ds,
        batch_size=int(args["batch_size"]),
        shuffle=False,
        num_workers=int(args.get("num_workers", 0)),
        drop_last=True if len(ds) >= int(args["batch_size"]) else False,
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
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(args["lr"]),
        weight_decay=float(args["weight_decay"]),
        betas=(0.9, 0.95),
    )
    epochs = int(args["epochs"])
    total_steps = max(1, epochs * max(1, len(loader)))
    log_path = output_dir / "pretrain_log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow(["epoch", "step", "lr", "loss", "feat_loss", "cont_loss", "seconds"])

    use_amp = device.type == "cuda" and bool(args.get("use_amp", True))
    model.train()
    step = 0
    avg_loss = 0.0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_sum = 0.0
        n_steps = 0
        t_ep = time.perf_counter()
        for batch in loader:
            cube = batch.to(device, non_blocking=True)
            lr_now = cosine_warmup_lr(
                step, total_steps, float(args["warmup_frac"]),
                float(args["lr"]), float(args["min_lr"]),
            )
            for group in optim.param_groups:
                group["lr"] = lr_now
            optim.zero_grad(set_to_none=True)
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
            ep_sum += float(loss.detach())
            n_steps += 1
            step += 1
        avg_loss = ep_sum / max(1, n_steps)
        dt = time.perf_counter() - t_ep
        writer.writerow([epoch, step, f"{lr_now:.6e}", f"{avg_loss:.6f}", "", "", f"{dt:.2f}"])
        log_f.flush()
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            _log(f"[ep {epoch}/{epochs}] loss {avg_loss:.4f}  lr {lr_now:.2e}  ({dt:.1f}s)")
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
