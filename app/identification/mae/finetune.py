"""Few-shot fine-tune of a pretrained MAE encoder for mineral classes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from identification.crism_common import format_torch_runtime, resolve_device
from identification.defaults import default_class_names

from .dataset import LabeledCropDataset, collect_labeled_records
from .defaults import default_finetune_args, mae_data_dir, save_last_finetune
from .model import MineralMAEClassifier, load_encoder_from_checkpoint


def run_finetune(config: Dict, log=None) -> Dict:
    args = default_finetune_args()
    args.update({k: v for k, v in config.items() if v is not None})
    device = resolve_device(args.get("device", "cpu"))

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg)

    encoder_path = Path(args["encoder_path"])
    if not encoder_path.is_file():
        raise FileNotFoundError(f"找不到预训练编码器：{encoder_path}")
    _log(format_torch_runtime())
    encoder, payload = load_encoder_from_checkpoint(encoder_path, device="cpu")
    enc_cfg = dict(payload.get("config") or {})
    _log(f"加载编码器 {encoder_path}")
    num_classes = int(args["num_classes"])
    tiles, label_map, points = collect_labeled_records(
        args["data_path"],
        args["label_path"],
        num_classes=num_classes,
        data_key=args.get("data_key") or None,
        label_key=args.get("label_key") or None,
        data_layout=str(args.get("data_layout", "HWB")),
        input_pattern=str(args.get("input_pattern") or "*"),
        max_per_class=int(args.get("max_per_class") or 0),
        seed=int(args.get("seed", 0)),
    )
    _log(f"标注像元 {len(points)}，类别数 {num_classes}")
    dataset = LabeledCropDataset(
        tiles,
        label_map,
        points,
        crop=int(enc_cfg.get("crop", args["crop"])),
        spatial_patch=int(enc_cfg.get("spatial_patch", args["spatial_patch"])),
        data_key=args.get("data_key") or None,
        data_layout=str(args.get("data_layout", "HWB")),
        preprocess_mode=str(args.get("preprocess_mode", "full")),
        num_classes=num_classes,
    )
    val_frac = float(args.get("val_fraction", 0.2))
    n_val = max(1, int(round(len(dataset) * val_frac))) if len(dataset) > 4 else 0
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_set, val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(int(args.get("seed", 0))),
        )
    else:
        train_set, val_set = dataset, None
    train_loader = DataLoader(
        train_set,
        batch_size=int(args["batch_size"]),
        shuffle=True,
        num_workers=int(args.get("num_workers", 0)),
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=int(args["batch_size"]),
            shuffle=False,
            num_workers=int(args.get("num_workers", 0)),
        )

    model = MineralMAEClassifier(encoder, num_classes).to(device)
    freeze = bool(args.get("freeze_encoder", False))
    if freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False
        _log("已冻结编码器（只训练分类头）")
    param_groups = [
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": float(args["head_lr"])},
        {"params": [p for p in model.spatial_head.parameters() if p.requires_grad], "lr": float(args["head_lr"])},
    ]
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": float(args["lr"])})
    optim = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    smoothing = float(args.get("label_smoothing", 0.02))
    spatial_w = float(args.get("spatial_loss_weight", 0.5))
    epochs = int(args["epochs"])
    best_acc = -1.0
    output_dir = Path(args.get("output_dir") or mae_data_dir() / "finetune")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best.pth"

    def _eval():
        if val_loader is None:
            return 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for cube, y, _blocks in val_loader:
                logits = model(cube.to(device))
                pred = logits.argmax(dim=1).cpu()
                correct += int((pred == y).sum())
                total += int(y.numel())
        return correct / max(total, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for cube, y, blocks in train_loader:
            cube = cube.to(device)
            y = y.to(device)
            blocks = blocks.to(device)
            optim.zero_grad(set_to_none=True)
            logits, spatial = model(cube, return_spatial=True)
            loss = F.cross_entropy(logits, y, label_smoothing=smoothing)
            flat = spatial.reshape(-1, num_classes)
            tgt = blocks.reshape(-1)
            valid = tgt >= 0
            if int(valid.sum()) > 0:
                loss = loss + spatial_w * F.cross_entropy(
                    flat[valid], tgt[valid], label_smoothing=smoothing
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            loss_sum += float(loss.item()) * len(y)
            n += len(y)
        val_acc = _eval()
        train_loss = loss_sum / max(n, 1)
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            _log(
                f"[ep {epoch}/{epochs}] loss {train_loss:.4f}  "
                f"val_acc {val_acc * 100:.2f}%"
            )
        if val_acc >= best_acc:
            best_acc = val_acc
            names = args.get("class_names") or default_class_names(num_classes)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "config": {
                        **enc_cfg,
                        "num_classes": num_classes,
                        "class_names": list(names),
                        "crop": int(enc_cfg.get("crop", args["crop"])),
                        "spatial_patch": int(enc_cfg.get("spatial_patch", args["spatial_patch"])),
                    },
                    "val_acc": best_acc,
                    "epoch": epoch,
                    "freeze_encoder": freeze,
                },
                best_path,
            )

    record = {
        "checkpoint_path": str(best_path),
        "encoder_path": str(encoder_path),
        "output_dir": str(output_dir),
        "num_classes": num_classes,
        "n_points": int(len(points)),
        "val_acc": float(best_acc),
        "freeze_encoder": freeze,
    }
    save_last_finetune(record)
    _log(f"微调完成。最佳模型：{best_path}  val_acc={best_acc * 100:.2f}%")
    return record
