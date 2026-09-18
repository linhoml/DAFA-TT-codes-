#!/usr/bin/env python3
"""Headless MAE pretrain / fine-tune / test / apply (no Qt GUI).

For HPC/Slurm. Example:

  python scripts/run_mae.py pretrain --data /scratch/crism --output /scratch/mae/pretrain \\
      --epochs 30 --samples-per-epoch 8192 --batch-size 64 --device cuda:0 --resume

  python scripts/run_mae.py finetune --encoder /scratch/mae/pretrain/checkpoints/encoder.pt \\
      --data /scratch/labels/cubes --label /scratch/labels/maps --num-classes 23 \\
      --max-per-class 200 --epochs 40 --batch-size 32 --device cuda:0 --freeze-encoder
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _log(message: str) -> None:
    print(message, flush=True)


def _cmd_pretrain(args: argparse.Namespace):
    from identification.mae.pipeline import pretrain

    cfg = {
        "data_path": args.data,
        "output_dir": args.output,
        "input_pattern": args.pattern,
        "data_layout": args.layout,
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "num_readers": args.num_readers,
        "preprocess_mode": args.preprocess,
        "device": args.device,
        "resume": args.resume,
        "resume_path": args.resume_path,
    }
    return pretrain(cfg, log=_log)


def _cmd_finetune(args: argparse.Namespace):
    from identification.mae.pipeline import finetune

    cfg = {
        "encoder_path": args.encoder,
        "data_path": args.data,
        "label_path": args.label,
        "output_dir": args.output,
        "input_pattern": args.pattern,
        "data_layout": args.layout,
        "num_classes": args.num_classes,
        "max_per_class": args.max_per_class,
        "freeze_encoder": args.freeze_encoder,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "preprocess_mode": args.preprocess,
    }
    return finetune(cfg, log=_log)


def _cmd_test(args: argparse.Namespace):
    from identification.mae.pipeline import run_test

    cfg = {
        "checkpoint_path": args.checkpoint,
        "data_path": args.data,
        "label_path": args.label,
        "output_dir": args.output,
        "input_pattern": args.pattern,
        "data_layout": args.layout,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    return run_test(cfg, log=_log)


def _cmd_apply(args: argparse.Namespace):
    from identification.io import list_input_files
    from identification.mae.apply import apply_paths
    from identification.mae.defaults import mae_data_dir

    paths = list_input_files(args.data, args.pattern)
    return apply_paths(
        paths,
        checkpoint_path=args.checkpoint,
        save_dir=args.output or str(mae_data_dir() / "apply"),
        device_cfg=args.device,
        batch_size=args.batch_size,
        data_layout=args.layout,
        log=_log,
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layout", default="HWB", choices=["HWB", "BHW"])
    p.add_argument("--pattern", default="*", help="Folder glob when --data is a directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CRISM 3D-MAE without the desktop GUI (HPC / Slurm).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("pretrain", help="Self-supervised MAE pre-training")
    p_pre.add_argument("--data", required=True, help="Unlabeled cube file or folder")
    p_pre.add_argument("--output", required=True, help="Output directory (checkpoints/)")
    p_pre.add_argument("--epochs", type=int, default=30)
    p_pre.add_argument("--samples-per-epoch", type=int, default=8192)
    p_pre.add_argument("--num-readers", type=int, default=8)
    p_pre.add_argument("--preprocess", default="crop", choices=["crop", "full"])
    p_pre.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p_pre.add_argument("--resume-path", default=None, help="Optional encoder.pt to continue from")
    _add_common(p_pre)
    p_pre.set_defaults(func=_cmd_pretrain)

    p_ft = sub.add_parser("finetune", help="Few-shot / labeled fine-tuning")
    p_ft.add_argument("--encoder", required=True, help="Pretrained encoder.pt")
    p_ft.add_argument("--data", required=True, help="Labeled cube file or folder")
    p_ft.add_argument("--label", required=True, help="Label map (0=unlabeled, 1..K classes)")
    p_ft.add_argument("--output", required=True)
    p_ft.add_argument("--num-classes", type=int, required=True)
    p_ft.add_argument("--max-per-class", type=int, default=0, help="0 = use all labeled pixels")
    p_ft.add_argument("--epochs", type=int, default=40)
    p_ft.add_argument("--freeze-encoder", action="store_true")
    p_ft.add_argument("--preprocess", default="full", choices=["crop", "full"])
    _add_common(p_ft)
    p_ft.set_defaults(func=_cmd_finetune)

    p_te = sub.add_parser("test", help="Classify one cube and score against a label map")
    p_te.add_argument("--checkpoint", required=True, help="model_best.pth")
    p_te.add_argument("--data", required=True)
    p_te.add_argument("--label", required=True)
    p_te.add_argument("--output", required=True)
    _add_common(p_te)
    p_te.set_defaults(func=_cmd_test)

    p_ap = sub.add_parser("apply", help="Write ENVI *_MAE_classification.img")
    p_ap.add_argument("--checkpoint", required=True, help="model_best.pth")
    p_ap.add_argument("--data", required=True)
    p_ap.add_argument("--output", required=True)
    _add_common(p_ap)
    p_ap.set_defaults(func=_cmd_apply)

    args = parser.parse_args()
    record = args.func(args)
    if isinstance(record, dict):
        ckpt = record.get("checkpoint_path") or record.get("envi_path")
        if ckpt:
            _log(f"DONE {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
