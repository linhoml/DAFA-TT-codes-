"""Train and apply the vendored crism_ml Hierarchical Bayesian Model."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from identification.defaults import identification_data_dir
from identification.io import write_envi_class_map

from . import ensure_crism_ml
from .adapt import cube_to_if_mat, mineral_names, remap_prediction


LogFn = Callable[[str], None]


def hbm_data_dir() -> Path:
    return identification_data_dir() / "hbm"


def default_dataset_dir() -> Path:
    return hbm_data_dir() / "datasets"


def default_work_dir() -> Path:
    return hbm_data_dir() / "runs"


def last_hbm_record_path() -> Path:
    return hbm_data_dir() / "last_trained.json"


def load_last_hbm() -> Optional[Dict]:
    path = last_hbm_record_path()
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_last_hbm(record: Dict) -> None:
    path = last_hbm_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)


def _log(log: Optional[LogFn], message: str) -> None:
    if log is not None:
        log(message)
    else:
        print(message)


class _LogHandler(logging.Handler):
    def __init__(self, log: LogFn):
        super().__init__()
        self._log = log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log(self.format(record))
        except Exception:
            pass


def _attach_logger(log: Optional[LogFn]) -> Optional[_LogHandler]:
    if log is None:
        return None
    handler = _LogHandler(log)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return handler


def _detach_logger(handler: Optional[_LogHandler]) -> None:
    if handler is not None:
        logging.getLogger().removeHandler(handler)


def _configure_runtime(workdir: str | Path, n_jobs: int = 1) -> Path:
    ensure_crism_ml()
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    import crism_ml
    import crism_ml.io as cio
    import crism_ml.preprocessing as cpre
    import crism_ml.train as ctrain

    jobs = max(int(n_jobs), 1)
    crism_ml.N_JOBS = jobs
    ctrain.N_JOBS = jobs
    cpre.N_JOBS = jobs
    cio.CACHE_DIR = str(cache)
    return work


def _assert_datasets(datadir: str | Path) -> Path:
    root = Path(datadir)
    bland = root / "CRISM_bland_unratioed.mat"
    mineral = root / "CRISM_labeled_pixels_ratioed.mat"
    missing = [p.name for p in (bland, mineral) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "HBM 训练需要 Plebani 等发布的数据集，缺少：\n"
            + "\n".join(missing)
            + f"\n请放到：{root}\n"
            "下载：https://zenodo.org/records/13338091"
        )
    return root


def _count_classes(model) -> int:
    classes = getattr(model, "classes", None)
    if classes is None:
        return 0
    return int(np.size(classes))


def train_hbm(datadir: str | Path, workdir: str | Path, n_jobs: int = 1,
              log: Optional[LogFn] = None) -> Dict:
    """Train bland-pixel and mineral HBM models (cached under workdir/cache)."""
    data = _assert_datasets(datadir)
    work = _configure_runtime(workdir, n_jobs=n_jobs)
    handler = _attach_logger(log)
    try:
        from crism_ml.train import feat_masks, train_model, train_model_bland

        _log(log, f"数据集目录：{data}")
        _log(log, f"缓存 / 输出：{work / 'cache'}")
        fin0, fin = feat_masks()
        _log(log, "训练 bland-pixel HBM…")
        bmodels = train_model_bland(str(data), fin0)
        _log(log, "训练矿物 HBM…")
        models = train_model(str(data), fin)
        n_classes = _count_classes(models[0]) if models else 0
        record = {
            "datadir": str(data),
            "workdir": str(work),
            "cache_dir": str(work / "cache"),
            "n_mineral_models": len(models),
            "n_bland_models": len(bmodels),
            "n_classes": n_classes,
        }
        save_last_hbm(record)
        _log(log, f"训练完成。矿物模型数={len(models)}，类别数={n_classes}")
        return record
    except Exception:
        import traceback

        _log(log, traceback.format_exc())
        raise
    finally:
        _detach_logger(handler)


def _load_mat_from_path(path: str | Path):
    from crism_ml.io import load_image

    return load_image(str(path))


def classify_mat(
    mat: Dict,
    datadir: str | Path,
    workdir: str | Path,
    thresholds=(0.5, 0.7),
    n_jobs: int = 1,
    log: Optional[LogFn] = None,
    source_name: str = "scene",
    save_dir: Optional[str | Path] = None,
) -> Dict:
    """Classify one CRISM-ML IF cube and optionally write an ENVI class map."""
    work = _configure_runtime(workdir, n_jobs=n_jobs)
    data = Path(datadir)
    if not (work / "cache").exists() or not any((work / "cache").glob("*model.pkl")):
        _assert_datasets(data)
    handler = _attach_logger(log)
    try:
        import crism_ml.preprocessing as cp
        from crism_ml.io import image_shape
        from crism_ml.train import (
            compute_bland_scores,
            compute_scores,
            feat_masks,
            filter_predictions,
            iteration_weights,
            train_model,
            train_model_bland,
        )

        fin0, fin = feat_masks()
        _log(log, "加载 / 训练 bland 与矿物模型（有缓存则直接读取）…")
        bmodels = train_model_bland(str(data), fin0)
        models = train_model(str(data), fin)
        ww_ = iteration_weights(models[0].classes)

        ts_if = np.asarray(mat["IF"])
        im_shape = image_shape(mat)
        _log(log, f"{source_name} 尺寸 {im_shape[0]}×{im_shape[1]}，光谱 {ts_if.shape}")

        if_, rem = cp.filter_bad_pixels(ts_if)
        if1 = cp.remove_spikes_column(
            if_.reshape(*im_shape, -1), 3, 5
        ).reshape(if_.shape)
        slog = compute_bland_scores(if1, (bmodels, fin0))
        slog_inf = cp.replace(slog, rem, -np.inf).reshape(im_shape)
        if2 = cp.ratio(if1.reshape(*im_shape, -1), slog_inf).reshape(if_.shape)
        from crism_ml import CONF

        ifm = cp.remove_spikes(if2.copy(), CONF["despike_params"])
        sumlog = compute_scores(ifm, (models, fin), ww_)
        thr = tuple(thresholds) if thresholds is not None else (0.5, 0.7)
        if len(thr) == 1:
            pred, pred0, pp_ = filter_predictions(sumlog, models[0].classes, thr=float(thr[0]))
        else:
            pred, pred0, pp_ = filter_predictions(
                sumlog, models[0].classes, kls_thr=(float(thr[0]), float(thr[1]))
            )

        pred_map = np.asarray(pred, dtype=np.int32).reshape(im_shape)
        raw_map = np.asarray(pred0, dtype=np.int32).reshape(im_shape)
        conf_map = np.asarray(pp_, dtype=np.float32).reshape(im_shape)
        display, names, codes = remap_prediction(pred_map)

        result = {
            "display_prediction": display,
            "raw_prediction": raw_map.astype(np.int16, copy=False),
            "hbm_codes": pred_map,
            "confidence": conf_map,
            "num_classes": max(len(names), 1),
            "class_names": names,
            "class_codes": codes,
            "shape": im_shape,
            "source_name": source_name,
        }
        out_dir = Path(save_dir) if save_dir else work
        out_dir.mkdir(parents=True, exist_ok=True)
        envi_path = write_envi_class_map(
            out_dir / f"{source_name}_hbm_class.img",
            pred_map.astype(np.int16, copy=False),
            names if names else None,
        )
        result["envi_path"] = str(envi_path)
        pkl_path = out_dir / f"{source_name}_hbm.pkl"
        with open(pkl_path, "wb") as handle:
            pickle.dump(
                {"pred": pred_map, "pred0": raw_map, "confidence": conf_map},
                handle,
                pickle.HIGHEST_PROTOCOL,
            )
        result["pkl_path"] = str(pkl_path)
        _log(log, f"已保存 ENVI 分类图：{envi_path}")
        return result
    finally:
        _detach_logger(handler)


def classify_path(
    path: str | Path,
    datadir: str | Path,
    workdir: str | Path,
    thresholds=(0.5, 0.7),
    n_jobs: int = 1,
    log: Optional[LogFn] = None,
    save_dir: Optional[str | Path] = None,
) -> Dict:
    _configure_runtime(workdir, n_jobs=n_jobs)
    mat = _load_mat_from_path(path)
    return classify_mat(
        mat,
        datadir=datadir,
        workdir=workdir,
        thresholds=thresholds,
        n_jobs=n_jobs,
        log=log,
        source_name=Path(path).stem,
        save_dir=save_dir,
    )


def classify_cube(
    cube: np.ndarray,
    datadir: str | Path,
    workdir: str | Path,
    data_layout: str = "HWB",
    thresholds=(0.5, 0.7),
    n_jobs: int = 1,
    log: Optional[LogFn] = None,
    save_dir: Optional[str | Path] = None,
    source_name: str = "opened_cube",
) -> Dict:
    mat = cube_to_if_mat(cube, data_layout=data_layout)
    return classify_mat(
        mat,
        datadir=datadir,
        workdir=workdir,
        thresholds=thresholds,
        n_jobs=n_jobs,
        log=log,
        source_name=source_name,
        save_dir=save_dir,
    )


def classify_paths(
    paths: Sequence[str | Path],
    datadir: str | Path,
    workdir: str | Path,
    save_dir: str | Path,
    thresholds=(0.5, 0.7),
    n_jobs: int = 1,
    log: Optional[LogFn] = None,
    progress_cb=None,
) -> Dict:
    path_list = [Path(p) for p in paths]
    if not path_list:
        raise FileNotFoundError("没有可分类的立方体文件。")
    saved: List[str] = []
    last = None
    total = len(path_list)
    for index, path in enumerate(path_list):
        if progress_cb:
            progress_cb(index, total, path.name)
        last = classify_path(
            path,
            datadir=datadir,
            workdir=workdir,
            thresholds=thresholds,
            n_jobs=n_jobs,
            log=log,
            save_dir=save_dir,
        )
        saved.append(str(last.get("envi_path") or ""))
    if progress_cb:
        progress_cb(total, total, "完成")
    return {
        "saved": saved,
        "last": last,
        "count": len(saved),
        "save_dir": str(save_dir),
    }


def evaluate_prediction(pred_map: np.ndarray, label_map: np.ndarray, k: Optional[int] = None) -> Dict:
    """OA / AA / Kappa on labeled pixels (label ids = HBM mineral codes)."""
    from identification.crism_common import format_evaluation_report

    pred = np.asarray(pred_map)
    lab = np.asarray(label_map)
    if pred.shape != lab.shape:
        raise ValueError(
            "标签图尺寸必须与影像一致，才能计算检验精度。"
            f" 标签={tuple(int(x) for x in lab.shape)}，"
            f"影像={tuple(int(x) for x in pred.shape)}。"
        )
    max_id = int(max(int(pred.max() or 0), int(lab.max() or 0), int(k or 0)))
    n = max(max_id, 1)
    valid = (lab >= 1) & (lab <= n) & (pred >= 1)
    cm = np.zeros((n, n), dtype=np.int64)
    if np.any(valid):
        gt0 = np.clip(lab[valid].astype(np.int64) - 1, 0, n - 1)
        pr0 = np.clip(pred[valid].astype(np.int64) - 1, 0, n - 1)
        np.add.at(cm, (gt0, pr0), 1)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    total = int(cm.sum())
    recall = np.divide(tp, support, out=np.full_like(tp, np.nan), where=support > 0)
    present = support > 0
    oa = float(tp.sum() / max(total, 1))
    aa = float(np.nanmean(recall[present])) if np.any(present) else np.nan
    row_sum, col_sum = cm.sum(axis=1), cm.sum(axis=0)
    pe = float(np.dot(row_sum, col_sum)) / max(float(total * total), 1.0)
    kappa = (oa - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 0.0
    metrics = {
        "OA": oa,
        "AA": aa,
        "Kappa": float(kappa),
        "macro_F1": aa,
        "recall": recall,
        "support": support.astype(np.int64),
        "total": total,
    }
    names = [f"class_{i}" for i in range(1, n + 1)]
    try:
        names = mineral_names(range(1, n + 1))
    except Exception:
        pass
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": float(kappa),
        "macro_F1": aa,
        "total": total,
        "accuracy_report": format_evaluation_report(metrics, names),
        "confusion_matrix": cm,
    }
