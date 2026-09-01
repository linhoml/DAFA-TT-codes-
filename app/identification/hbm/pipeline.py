"""Train and apply the vendored crism_ml Hierarchical Bayesian Model."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from identification.defaults import identification_data_dir
from identification.io import classification_stem, is_classification_output, write_envi_class_map

from . import ensure_crism_ml
from .adapt import (
    build_hbm_display,
    class_pixel_counts,
    cube_to_if_mat,
    mineral_names,
    normalize_if_values,
)


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


def _first_file(candidates) -> Optional[Path]:
    for path in candidates:
        path = Path(path)
        if path.is_file():
            return path
    return None


def find_trained_model_files(workdir: str | Path | None = None) -> tuple[Path, Path]:
    """Locate bland + mineral pickles written by 模型训练.

    When ``workdir`` is given, only that training run is used (cache/ then the
    directory itself). Otherwise fall back to last_trained.json / default workdir.
    """
    last = load_last_hbm() or {}
    search_dirs: list[Path] = []
    bland_named: list = []
    mineral_named: list = []

    work = Path(workdir) if workdir and str(workdir).strip() else None
    if work is not None:
        search_dirs.append(work / "cache")
        search_dirs.append(work)
        last_work = last.get("workdir")
        try:
            if last_work and Path(last_work).resolve() == work.resolve():
                bland_named.append(last.get("bland_model_path"))
                mineral_named.append(last.get("mineral_model_path"))
        except OSError:
            pass
    else:
        bland_named.append(last.get("bland_model_path"))
        mineral_named.append(last.get("mineral_model_path"))
        last_work = last.get("workdir")
        if last_work:
            search_dirs.append(Path(last_work) / "cache")
            search_dirs.append(Path(last_work))
        search_dirs.append(default_work_dir() / "cache")
        search_dirs.append(default_work_dir())

    bland_globs: list[Path] = []
    mineral_globs: list[Path] = []
    seen = set()
    for directory in search_dirs:
        key = str(directory)
        if key in seen or not directory.is_dir():
            continue
        seen.add(key)
        bland_named.append(directory / "default_bmodel.pkl")
        mineral_named.append(directory / "default_model.pkl")
        bland_globs.extend(sorted(directory.glob("*bmodel.pkl")))
        mineral_globs.extend(
            p for p in sorted(directory.glob("*model.pkl"))
            if not p.name.endswith("bmodel.pkl")
        )

    bland = _first_file([p for p in bland_named if p] + bland_globs)
    mineral = _first_file([p for p in mineral_named if p] + mineral_globs)
    if bland is None or mineral is None:
        where = str(work / "cache") if work is not None else str(default_work_dir() / "cache")
        raise FileNotFoundError(
            "没有找到已训练的 HBM 模型。请先运行 Identification → HBM → 模型训练。\n"
            f"期望位置：{where}/default_bmodel.pkl 与 default_model.pkl。"
        )
    return bland, mineral


def load_trained_models(workdir: str | Path | None = None):
    """Load bland and mineral HBM models saved by training (no dataset needed)."""
    ensure_crism_ml()
    bland_path, mineral_path = find_trained_model_files(workdir)
    with open(bland_path, "rb") as handle:
        bmodels = pickle.load(handle)
    with open(mineral_path, "rb") as handle:
        models = pickle.load(handle)
    if not bmodels or not models:
        raise ValueError(f"模型文件为空：{bland_path} / {mineral_path}")
    return bmodels, models, bland_path, mineral_path


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
        bland_path, mineral_path = find_trained_model_files(work)
        record = {
            "datadir": str(data),
            "workdir": str(work),
            "cache_dir": str(work / "cache"),
            "bland_model_path": str(bland_path),
            "mineral_model_path": str(mineral_path),
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


def _coerce_if_matrix(if_arr: np.ndarray) -> np.ndarray:
    """Make IF an (npix, nchan) float32 array."""
    arr = np.asarray(if_arr)
    if arr.ndim == 3:
        if arr.shape[0] in (248, 350, 438) and arr.shape[-1] not in (248, 350, 438):
            arr = np.transpose(arr, (1, 2, 0))
        height, width, bands = arr.shape
        arr = arr.reshape(height * width, bands)
    elif arr.ndim == 2 and arr.shape[0] in (248, 350, 438) and arr.shape[1] not in (248, 350, 438):
        arr = arr.T
    return np.ascontiguousarray(arr, dtype=np.float32)


def _prepare_if_mat(mat: Dict, log: Optional[LogFn] = None) -> Dict:
    """Flatten IF, scale 0–10000 I/F to 0–1, keep x/y."""
    prepared = dict(mat)
    if_arr, scale = normalize_if_values(_coerce_if_matrix(mat["IF"]))
    prepared["IF"] = if_arr
    if scale != 1.0:
        _log(log, f"I/F 中值偏大，已除以 {scale:g} 缩放到 0–1（否则会被当成坏像元滤掉）")
    for key in ("x", "y"):
        if key in prepared:
            prepared[key] = np.asarray(prepared[key]).reshape(-1)
    return prepared


def _paint_regions(avgs, im_shape) -> np.ndarray:
    region_map = np.zeros(im_shape, dtype=np.int32)
    if not avgs:
        return region_map
    height, width = im_shape
    for avg in avgs:
        xy = np.asarray(avg.get("coords"))
        if xy.size == 0:
            continue
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        xs = xy[:, 0].astype(np.int32, copy=False)
        ys = xy[:, 1].astype(np.int32, copy=False)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        region_map[ys[valid], xs[valid]] = int(avg["pred"])
    return region_map


def classify_mat(
    mat: Dict,
    workdir: str | Path,
    thresholds=(0.5, 0.7),
    n_jobs: int = 1,
    log: Optional[LogFn] = None,
    source_name: str = "scene",
    save_dir: Optional[str | Path] = None,
) -> Dict:
    """Classify one CRISM-ML IF cube using models from 模型训练."""
    work = _configure_runtime(workdir, n_jobs=n_jobs)
    handler = _attach_logger(log)
    try:
        import crism_ml.preprocessing as cp
        from crism_ml import CONF
        from crism_ml.io import image_shape
        from crism_ml.train import (
            compute_bland_scores,
            compute_scores,
            feat_masks,
            filter_predictions,
            iteration_weights,
        )

        bmodels, models, bland_path, mineral_path = load_trained_models(work)
        _log(log, f"使用已训练模型：\n  bland  {bland_path}\n  mineral  {mineral_path}")
        fin0, fin = feat_masks()
        ww_ = iteration_weights(models[0].classes)

        mat = _prepare_if_mat(mat, log=log)
        ts_if = np.asarray(mat["IF"])
        im_shape = image_shape(mat)
        n_pix = int(np.prod(im_shape))
        if ts_if.shape[0] != n_pix:
            raise ValueError(
                f"光谱像素数 {ts_if.shape[0]} 与影像尺寸 {im_shape[0]}×{im_shape[1]}={n_pix} 不一致。"
            )
        _log(log, f"{source_name} 尺寸 {im_shape[0]}×{im_shape[1]}，光谱 {ts_if.shape}")

        if_, rem = cp.filter_bad_pixels(ts_if.copy())
        _log(log, f"坏像元 {int(np.sum(rem))} / {rem.size}")
        if1 = cp.remove_spikes_column(
            if_.reshape(*im_shape, -1), 3, 5
        ).reshape(if_.shape)
        slog = compute_bland_scores(if1, (bmodels, fin0))
        slog_inf = cp.replace(slog, rem, -np.inf).reshape(im_shape)
        if2 = cp.ratio(if1.reshape(*im_shape, -1), slog_inf).reshape(if_.shape)
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
        finite_pp = conf_map[np.isfinite(conf_map)]
        if finite_pp.size:
            _log(
                log,
                "置信度百分位 "
                f"p50={float(np.percentile(finite_pp, 50)):.3f} "
                f"p90={float(np.percentile(finite_pp, 90)):.3f} "
                f"p99={float(np.percentile(finite_pp, 99)):.3f}；"
                f"过滤后矿物像元 {int(np.sum(pred_map > 0))}",
            )

        region_map = np.zeros(im_shape, dtype=np.int32)
        try:
            from crism_ml.train import evaluate_regions

            avgs = evaluate_regions(
                if2, im_shape, cp.replace(pred, rem, 0), pp_, if0=if_,
            )
            region_map = _paint_regions(avgs, im_shape)
            _log(log, f"HBM 斑块 {len(avgs)} 个，斑块像元 {int(np.sum(region_map > 0))}")
        except Exception as exc:
            _log(log, f"斑块检测跳过：{exc}")

        display, names, codes, mode = build_hbm_display(
            pred_map, unfiltered=raw_map, region=region_map,
        )
        if mode == "unfiltered":
            _log(log, "置信度过滤后没有矿物像元，改为显示未过滤 argmax（已隐藏 bland/artifact）")
        elif mode == "empty":
            _log(log, "未检出矿物。请确认输入是 CRISM TRR3 I/F，或降低易分/难分类别阈值。")
        elif mode == "regions":
            _log(log, "叠加显示 HBM 检出斑块")
        counts = class_pixel_counts(display, names)
        if counts:
            _log(log, "类别像元数：\n  " + "\n  ".join(counts))

        result = {
            "display_prediction": display,
            "raw_prediction": raw_map.astype(np.int16, copy=False),
            "hbm_codes": pred_map,
            "confidence": conf_map,
            "num_classes": len(names),
            "class_names": names,
            "class_codes": codes,
            "shape": im_shape,
            "source_name": source_name,
            "display_mode": mode,
            "class_counts": counts,
        }
        out_dir = Path(save_dir) if save_dir else work
        out_dir.mkdir(parents=True, exist_ok=True)
        out_stem = classification_stem(source_name, "HBM")
        envi_path = write_envi_class_map(
            out_dir / f"{out_stem}.img",
            display.astype(np.int16, copy=False),
            names if names else None,
        )
        result["envi_path"] = str(envi_path)
        _log(log, f"已保存 ENVI 分类图：{envi_path}")
        return result
    finally:
        _detach_logger(handler)


def classify_path(
    path: str | Path,
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
        workdir=workdir,
        thresholds=thresholds,
        n_jobs=n_jobs,
        log=log,
        source_name=Path(path).stem,
        save_dir=save_dir,
    )


def classify_cube(
    cube: np.ndarray,
    workdir: str | Path,
    data_layout: str = "HWB",
    thresholds=(0.5, 0.7),
    n_jobs: int = 1,
    log: Optional[LogFn] = None,
    save_dir: Optional[str | Path] = None,
    source_name: str = "opened_cube",
    wavelengths=None,
) -> Dict:
    mat = cube_to_if_mat(cube, data_layout=data_layout, wavelengths=wavelengths)
    return classify_mat(
        mat,
        workdir=workdir,
        thresholds=thresholds,
        n_jobs=n_jobs,
        log=log,
        source_name=source_name,
        save_dir=save_dir,
    )


def classify_paths(
    paths: Sequence[str | Path],
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
        if is_classification_output(path):
            continue
        if progress_cb:
            progress_cb(index, total, path.name)
        last = classify_path(
            path,
            workdir=workdir,
            thresholds=thresholds,
            n_jobs=n_jobs,
            log=log,
            save_dir=save_dir,
        )
        saved.append(str(last.get("envi_path") or ""))
    if last is None:
        raise FileNotFoundError("没有可分类的立方体文件（已跳过分类结果）。")
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
