"""Qt dialogs for HBM (crism_ml) train / test / apply."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from identification.dialogs import _Worker, _path_row
from identification.io import DEFAULT_INPUT_PATTERN, FILE_FILTER_CUBE, FILE_FILTER_LABEL

from .pipeline import (
    classify_path,
    default_dataset_dir,
    default_work_dir,
    evaluate_prediction,
    find_trained_model_files,
    load_last_hbm,
    train_hbm,
)


HBM_HELP = """\
HBM（Hierarchical Bayesian Model）嵌入自：
https://github.com/Banus/crism_ml
Plebani et al., Icarus 2022, doi:10.1016/j.icarus.2021.114849

训练（仅此步需要 Zenodo 数据集 13338091）：
  CRISM_bland_unratioed.mat
  CRISM_labeled_pixels_ratioed.mat
训练完成后模型保存在工作目录 cache/（default_bmodel.pkl、default_model.pkl）。

应用 / 测试：直接使用上面训练好的模型，不再需要数据集目录。
对象为 CRISM TRR3 I/F 影像（ENVI .img/.hdr/.lbl，约 438 波段）。
检验标签须与影像同高同宽，类别编号为 HBM 矿物代码（见 crism_ml.lab）。
"""


def _trained_model_hint(workdir: str) -> str:
    try:
        bland, mineral = find_trained_model_files(workdir)
        return (
            "将直接使用「模型训练」写出的模型（无需数据集目录）：\n"
            f"  平场：{bland}\n"
            f"  矿物：{mineral}"
        )
    except Exception:
        return "尚未找到已训练模型。请先运行 Identification → HBM → 模型训练。"


def _thr_row(parent, low=0.5, high=0.7):
    row = QHBoxLayout()
    spin_lo = QDoubleSpinBox()
    spin_lo.setRange(0.0, 1.0)
    spin_lo.setSingleStep(0.05)
    spin_lo.setDecimals(2)
    spin_lo.setValue(low)
    spin_hi = QDoubleSpinBox()
    spin_hi.setRange(0.0, 1.0)
    spin_hi.setSingleStep(0.05)
    spin_hi.setDecimals(2)
    spin_hi.setValue(high)
    row.addWidget(QLabel("易分/难分类别阈值："))
    row.addWidget(spin_lo)
    row.addWidget(spin_hi)
    row.addStretch()
    parent.addLayout(row)
    return spin_lo, spin_hi


def _require_trained_models(parent, workdir: str) -> bool:
    if not (workdir or "").strip():
        QMessageBox.warning(parent, "参数不完整", "请指定工作目录（训练时保存模型的目录）。")
        return False
    try:
        find_trained_model_files(workdir)
        return True
    except FileNotFoundError as exc:
        QMessageBox.warning(parent, "缺少已训练模型", str(exc))
        return False


class HbmTrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — HBM 模型训练")
        self.resize(680, 640)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self.result_record = None

        layout = QVBoxLayout(self)
        help_box = QPlainTextEdit()
        help_box.setReadOnly(True)
        help_box.setPlainText(HBM_HELP)
        help_box.setMaximumHeight(220)
        layout.addWidget(help_box)

        form = QFormLayout()
        last = load_last_hbm() or {}
        self.edit_data, data_row = _path_row(self, "选择数据集目录", directory=True)
        self.edit_data.setText(str(last.get("datadir") or default_dataset_dir()))
        form.addRow("数据集目录：", data_row)

        self.edit_work, work_row = _path_row(self, "选择工作目录", directory=True)
        self.edit_work.setText(str(last.get("workdir") or default_work_dir()))
        form.addRow("工作目录（缓存模型）：", work_row)

        self.spin_jobs = QSpinBox()
        self.spin_jobs.setRange(1, 64)
        self.spin_jobs.setValue(1)
        form.addRow("并行进程数：", self.spin_jobs)
        layout.addLayout(form)

        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self._start)
        layout.addWidget(self.btn_start)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append_log(self, text: str):
        self.log.appendPlainText(text.rstrip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _start(self):
        datadir = self.edit_data.text().strip()
        workdir = self.edit_work.text().strip()
        if not datadir:
            QMessageBox.warning(self, "参数不完整", "请指定数据集目录。")
            return
        if not workdir:
            QMessageBox.warning(self, "参数不完整", "请指定工作目录。")
            return
        n_jobs = int(self.spin_jobs.value())
        self.btn_start.setEnabled(False)
        self._append_log("======== 开始 HBM 训练 ========")

        def job(log_fn):
            return train_hbm(datadir, workdir, n_jobs=n_jobs, log=log_fn)

        self._thread = QThread(self)
        self._worker = _Worker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_done(self, record):
        self.result_record = record
        self.btn_start.setEnabled(True)
        QMessageBox.information(
            self, "HBM 训练",
            f"训练完成。\n工作目录：{(record or {}).get('workdir')}",
        )

    def _on_fail(self, message: str):
        self.btn_start.setEnabled(True)
        self._append_log("失败：" + message)
        QMessageBox.critical(self, "HBM 训练失败", message)

    def reject(self):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "训练进行中", "请等待训练结束再关闭。")
            return
        super().reject()


class HbmTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — HBM 模型测试")
        self.resize(720, 760)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self.result_summary = None

        layout = QVBoxLayout(self)
        help_box = QPlainTextEdit()
        help_box.setReadOnly(True)
        help_box.setPlainText(
            HBM_HELP
            + "\n测试必须提供与影像同尺寸的标签，日志写出 OA / AA / Kappa。"
        )
        help_box.setMaximumHeight(240)
        layout.addWidget(help_box)

        last = load_last_hbm() or {}
        form = QFormLayout()
        self.edit_data, data_row = _path_row(
            self, "选择测试立方体", filter_str=FILE_FILTER_CUBE,
        )
        form.addRow("测试影像：", data_row)

        self.edit_label, label_row = _path_row(
            self, "选择检验标签图", filter_str=FILE_FILTER_LABEL,
        )
        form.addRow("检验标签（须与影像同尺寸）：", label_row)

        self.edit_label_key = QLineEdit()
        form.addRow("标签变量名：", self.edit_label_key)

        self.edit_work, work_row = _path_row(self, "选择工作目录", directory=True)
        self.edit_work.setText(str(last.get("workdir") or default_work_dir()))
        form.addRow("工作目录（已训练模型）：", work_row)

        self.model_hint = QLabel(_trained_model_hint(self.edit_work.text().strip()))
        self.model_hint.setWordWrap(True)
        form.addRow("", self.model_hint)
        self.edit_work.textChanged.connect(self._refresh_model_hint)

        self.edit_output, out_row = _path_row(self, "选择输出目录", directory=True)
        self.edit_output.setText(str(default_work_dir() / "test_runs"))
        form.addRow("结果保存路径：", out_row)

        self.spin_jobs = QSpinBox()
        self.spin_jobs.setRange(1, 64)
        self.spin_jobs.setValue(1)
        form.addRow("并行进程数：", self.spin_jobs)
        layout.addLayout(form)
        self.spin_thr_lo, self.spin_thr_hi = _thr_row(layout)

        self.btn_start = QPushButton("开始测试")
        self.btn_start.clicked.connect(self._start)
        layout.addWidget(self.btn_start)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append_log(self, text: str):
        self.log.appendPlainText(text.rstrip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _refresh_model_hint(self) -> None:
        self.model_hint.setText(_trained_model_hint(self.edit_work.text().strip()))

    def _start(self):
        image = self.edit_data.text().strip()
        label = self.edit_label.text().strip()
        workdir = self.edit_work.text().strip()
        output = self.edit_output.text().strip()
        if not image or not Path(image).exists():
            QMessageBox.warning(self, "参数不完整", "请选择测试影像。")
            return
        if not label or not Path(label).exists():
            QMessageBox.warning(
                self, "参数不完整",
                "HBM 测试必须提供与影像同尺寸的标签图，才能计算检验精度。",
            )
            return
        if not workdir or not output:
            QMessageBox.warning(self, "参数不完整", "请指定工作目录和保存路径。")
            return
        if not _require_trained_models(self, workdir):
            return
        n_jobs = int(self.spin_jobs.value())
        thresholds = (float(self.spin_thr_lo.value()), float(self.spin_thr_hi.value()))
        label_key = self.edit_label_key.text().strip() or None
        self.result_summary = None
        self.btn_start.setEnabled(False)
        self._append_log("======== 开始 HBM 测试 ========")

        def job(log_fn):
            from identification.io import load_label_array

            result = classify_path(
                image,
                workdir=workdir,
                thresholds=thresholds,
                n_jobs=n_jobs,
                log=log_fn,
                save_dir=output,
            )
            label_map = load_label_array(label, key=label_key)
            metrics = evaluate_prediction(result["hbm_codes"], label_map)
            report = metrics.get("accuracy_report") or ""
            if report:
                log_fn(report)
            result.update(metrics)
            return result

        self._thread = QThread(self)
        self._worker = _Worker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_done(self, summary):
        self.result_summary = summary
        self.btn_start.setEnabled(True)
        oa = (summary or {}).get("OA")
        extra = ""
        try:
            if oa is not None and oa == oa:
                extra = f"\nOA={float(oa)*100:.2f}%  AA={float(summary.get('AA'))*100:.2f}%"
        except (TypeError, ValueError):
            extra = ""
        QMessageBox.information(
            self, "HBM 测试",
            f"测试完成。{extra}\n检验精度已写入日志。\n"
            "关闭窗口后分类图叠加显示在主界面左下方。",
        )

    def _on_fail(self, message: str):
        self.btn_start.setEnabled(True)
        self._append_log("失败：" + message)
        QMessageBox.critical(self, "HBM 测试失败", message)

    def reject(self):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "测试进行中", "请等待测试结束再关闭。")
            return
        super().reject()


class HbmApplyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — HBM 模型应用")
        self.resize(640, 680)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "对 CRISM I/F 影像做 bland-pixel ratio 后 HBM 矿物分类。\n"
            "直接加载「HBM → 模型训练」写出的模型，无需再选数据集目录。\n"
            "可选当前已打开影像、单个文件或文件夹；结果写出 ENVI *.img。"
        ))

        src_group = QGroupBox("输入数据")
        src_layout = QVBoxLayout(src_group)
        self.radio_opened = QRadioButton("当前已打开的高光谱影像")
        self.radio_file = QRadioButton("单个文件")
        self.radio_folder = QRadioButton("文件夹（处理其中全部立方体）")
        has_opened = getattr(parent, "current_data", None) is not None
        self.radio_opened.setEnabled(bool(has_opened))
        if has_opened:
            self.radio_opened.setChecked(True)
        else:
            self.radio_file.setChecked(True)
        src_layout.addWidget(self.radio_opened)
        src_layout.addWidget(self.radio_file)
        src_layout.addWidget(self.radio_folder)
        layout.addWidget(src_group)

        last = load_last_hbm() or {}
        form = QFormLayout()
        self.edit_data, data_row = _path_row(
            self, "选择立方体文件", filter_str=FILE_FILTER_CUBE,
        )
        btn_dir = QPushButton("选文件夹")
        btn_dir.clicked.connect(self._pick_data_dir)
        data_row.layout().addWidget(btn_dir)
        self.edit_data.textChanged.connect(self._on_data_path_changed)
        form.addRow("文件或文件夹：", data_row)

        self.edit_pattern = QLineEdit(DEFAULT_INPUT_PATTERN)
        form.addRow("文件夹匹配模式：", self.edit_pattern)

        self.edit_work, work_row = _path_row(self, "选择工作目录", directory=True)
        self.edit_work.setText(str(last.get("workdir") or default_work_dir()))
        form.addRow("工作目录（已训练模型）：", work_row)

        self.model_hint = QLabel(_trained_model_hint(self.edit_work.text().strip()))
        self.model_hint.setWordWrap(True)
        form.addRow("", self.model_hint)
        self.edit_work.textChanged.connect(self._refresh_model_hint)

        self.edit_save, save_row = _path_row(self, "选择结果保存目录", directory=True)
        self.edit_save.setText(str(default_work_dir() / "apply_runs"))
        form.addRow("结果保存路径：", save_row)

        self.spin_jobs = QSpinBox()
        self.spin_jobs.setRange(1, 64)
        self.spin_jobs.setValue(1)
        form.addRow("并行进程数：", self.spin_jobs)
        layout.addLayout(form)
        self.spin_thr_lo, self.spin_thr_hi = _thr_row(layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("开始应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择立方体文件夹", self.edit_data.text())
        if path:
            self.edit_data.setText(path)
            self.radio_folder.setChecked(True)

    def _on_data_path_changed(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        path = Path(text)
        if path.is_dir():
            self.radio_folder.setChecked(True)
        elif path.is_file():
            self.radio_file.setChecked(True)

    def _source(self) -> str:
        if self.radio_file.isChecked():
            return "file"
        if self.radio_folder.isChecked():
            return "folder"
        return "opened"

    def _refresh_model_hint(self) -> None:
        self.model_hint.setText(_trained_model_hint(self.edit_work.text().strip()))

    def _on_accept(self):
        cfg = self.params()
        if not cfg["workdir"] or not cfg["save_dir"]:
            QMessageBox.warning(self, "参数不完整", "请指定工作目录和保存路径。")
            return
        if not _require_trained_models(self, cfg["workdir"]):
            return
        source = cfg["source"]
        if source == "opened" and getattr(self.parent(), "current_data", None) is None:
            QMessageBox.warning(self, "缺少影像", "当前没有已打开的影像。请选择文件或文件夹。")
            return
        if source == "file" and (not cfg["data_path"] or not Path(cfg["data_path"]).is_file()):
            QMessageBox.warning(self, "缺少输入文件", "请选择要分类的立方体文件。")
            return
        if source == "folder" and (not cfg["data_path"] or not Path(cfg["data_path"]).is_dir()):
            QMessageBox.warning(self, "缺少输入文件夹", "请选择包含立方体的文件夹。")
            return
        self.accept()

    def params(self) -> dict:
        source = self._source()
        data_path = self.edit_data.text().strip()
        if source == "folder" and data_path and Path(data_path).is_file():
            source = "file"
        if source == "file" and data_path and Path(data_path).is_dir():
            source = "folder"
        return {
            "source": source,
            "data_path": data_path,
            "input_pattern": self.edit_pattern.text().strip() or DEFAULT_INPUT_PATTERN,
            "workdir": self.edit_work.text().strip(),
            "save_dir": self.edit_save.text().strip(),
            "n_jobs": int(self.spin_jobs.value()),
            "thresholds": (float(self.spin_thr_lo.value()), float(self.spin_thr_hi.value())),
        }
