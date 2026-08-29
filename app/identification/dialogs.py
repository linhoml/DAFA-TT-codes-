"""Qt dialogs for Identification: train, test, and apply."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from .defaults import (
    DATA_FORMAT_HELP,
    builtin_checkpoint_path,
    builtin_preprocess_path,
    identification_data_dir,
    load_last_trained,
)
from .io import DEFAULT_INPUT_PATTERN, FILE_FILTER_CUBE, FILE_FILTER_LABEL


def _preferred_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _preferred_batch(device: str) -> int:
    return 256 if device.startswith("cuda") else 64


def _path_row(parent: QWidget, label: str, directory: bool = False, filter_str: str = "") -> tuple:
    edit = QLineEdit()
    btn = QPushButton("浏览…")

    def browse():
        if directory:
            path = QFileDialog.getExistingDirectory(parent, label, edit.text() or str(Path.home()))
        else:
            path, _ = QFileDialog.getOpenFileName(
                parent, label, edit.text() or str(Path.home()), filter_str or "All Files (*)"
            )
        if path:
            edit.setText(path)

    btn.clicked.connect(browse)
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(btn)
    return edit, row


class _LogIO:
    def __init__(self, emit):
        self._emit = emit
        self._buf = ""

    def write(self, text):
        self._buf += str(text)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._emit(line)

    def flush(self):
        if self._buf.strip():
            self._emit(self._buf.rstrip())
        self._buf = ""


class _Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def run(self):
        import sys

        stream = _LogIO(self.log.emit)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = stream
        sys.stderr = stream
        try:
            result = self._fn(lambda msg: self.log.emit(str(msg)))
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            stream.flush()
            sys.stdout = old_out
            sys.stderr = old_err


class IdentificationTrainDialog(QDialog):
    """Collect training data paths and run preprocess + LSGA training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — 模型训练")
        self.resize(720, 780)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self.result_record = None

        layout = QVBoxLayout(self)
        help_box = QPlainTextEdit()
        help_box.setReadOnly(True)
        help_box.setPlainText(DATA_FORMAT_HELP)
        help_box.setMaximumHeight(210)
        layout.addWidget(QLabel("请先阅读输入数据格式，再选择训练立方体与标签。"))
        layout.addWidget(help_box)

        form = QFormLayout()
        self.edit_data, data_row = _path_row(
            self, "选择训练立方体或文件夹",
            directory=False,
            filter_str=FILE_FILTER_CUBE,
        )
        btn_data_dir = QPushButton("选文件夹")
        btn_data_dir.clicked.connect(self._pick_data_dir)
        data_row.layout().addWidget(btn_data_dir)
        form.addRow("训练数据（文件或文件夹）：", data_row)

        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["HWB", "BHW"])
        form.addRow("数据排布：", self.combo_layout)

        self.edit_data_key = QLineEdit()
        self.edit_data_key.setPlaceholderText("仅 .mat/.npz 需要；ENVI/img/dat 请留空")
        form.addRow("MAT/NPZ 变量名：", self.edit_data_key)

        self.edit_pattern = QLineEdit(DEFAULT_INPUT_PATTERN)
        form.addRow("文件夹匹配模式：", self.edit_pattern)

        self.edit_label, label_row = _path_row(
            self, "选择标签图", filter_str=FILE_FILTER_LABEL
        )
        form.addRow("标签（1..K，0=背景）：", label_row)

        self.edit_label_key = QLineEdit()
        self.edit_label_key.setPlaceholderText("留空则使用第一个二维数组")
        form.addRow("标签变量名：", self.edit_label_key)

        default_out = str(identification_data_dir() / "runs")
        self.edit_output, out_row = _path_row(self, "选择输出目录", directory=True)
        self.edit_output.setText(default_out)
        form.addRow("输出目录：", out_row)

        self.chk_preprocessed = QCheckBox("输入已是预处理后的 240 波段立方体（变量名 data）")
        form.addRow("", self.chk_preprocessed)

        self.spin_classes = QSpinBox()
        self.spin_classes.setRange(2, 64)
        self.spin_classes.setValue(24)
        form.addRow("类别数：", self.spin_classes)

        self.spin_patch = QSpinBox()
        self.spin_patch.setRange(3, 21)
        self.spin_patch.setSingleStep(2)
        self.spin_patch.setValue(9)
        form.addRow("patch 大小：", self.spin_patch)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4096)
        self.spin_batch.setValue(_preferred_batch(_preferred_device()))
        form.addRow("batch size：", self.spin_batch)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(130)
        form.addRow("训练轮数：", self.spin_epochs)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(0)
        form.addRow("随机种子：", self.spin_seed)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "cuda:0"])
        preferred = _preferred_device()
        self.combo_device.setCurrentText(preferred)
        form.addRow("计算设备：", self.combo_device)

        layout.addLayout(form)
        speed_hint = QLabel(
            "加速：有 NVIDIA 显卡时务必选 cuda:0（不要用 cpu）。"
            "16GB 显存建议 batch=256。预处理只做一次；若立方体已预处理请勾选上方复选框。"
        )
        speed_hint.setWordWrap(True)
        layout.addWidget(speed_hint)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self._start)
        btn_row.addWidget(self.btn_start)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志…")
        layout.addWidget(self.log, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练数据文件夹", self.edit_data.text())
        if path:
            self.edit_data.setText(path)

    def _append_log(self, text: str):
        self.log.appendPlainText(text.rstrip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _collect(self) -> dict:
        data_path = self.edit_data.text().strip()
        label_path = self.edit_label.text().strip()
        output_dir = self.edit_output.text().strip()
        if not data_path or not Path(data_path).exists():
            raise ValueError("请选择有效的训练数据文件或文件夹。")
        if not label_path or not Path(label_path).exists():
            raise ValueError("训练必须提供标签图（.mat / .img / .dat / .hdr 等，二维，类别 1..K）。")
        if not output_dir:
            raise ValueError("请指定输出目录。")
        return {
            "data_path": data_path,
            "already_preprocessed": self.chk_preprocessed.isChecked(),
            "data_key": self.edit_data_key.text().strip() or None,
            "data_layout": self.combo_layout.currentText(),
            "input_pattern": self.edit_pattern.text().strip() or DEFAULT_INPUT_PATTERN,
            "label_path": label_path,
            "label_key": self.edit_label_key.text().strip() or None,
            "output_dir": output_dir,
            "num_classes": int(self.spin_classes.value()),
            "patch_size": int(self.spin_patch.value()),
            "batch_size": int(self.spin_batch.value()),
            "epochs": int(self.spin_epochs.value()),
            "seed": int(self.spin_seed.value()),
            "device": self.combo_device.currentText(),
        }

    def _start(self):
        try:
            config = self._collect()
        except Exception as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return

        self.btn_start.setEnabled(False)
        self._append_log("======== 开始训练 ========")

        def job(log_fn):
            from .pipeline import run_training
            return run_training(config, log=log_fn)

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
        self._append_log("训练成功结束。")
        QMessageBox.information(
            self, "模型训练",
            "训练完成。\n\n"
            f"最佳模型：{record.get('checkpoint_path')}\n"
            f"预处理模型：{record.get('preprocess_model_path')}\n\n"
            "请看日志里的 train / val_OA / val_AA。\n"
            "若精度只略高于「随机猜」那一行（例如 24 类约 4%，10 类约 10%），"
            "通常是标签没对齐或数据不是 0–1 的 I/F。\n\n"
            "可在「模型应用」中选择「本次训练的新模型」。",
        )

    def _on_fail(self, message: str):
        self.btn_start.setEnabled(True)
        self._append_log("失败：" + message)
        QMessageBox.critical(self, "训练失败", message)

    def reject(self):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "训练进行中", "请等待训练结束再关闭。")
            return
        super().reject()


class IdentificationTestDialog(QDialog):
    """External-scene test / full-image evaluation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — 模型测试")
        self.resize(720, 780)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self.result_summary = None

        layout = QVBoxLayout(self)
        help_box = QPlainTextEdit()
        help_box.setReadOnly(True)
        help_box.setPlainText(
            DATA_FORMAT_HELP
            + "\n测试说明：\n"
            "• 有标签时计算 OA / AA / Kappa 与分类图；无标签只做预测。\n"
            "• 未预处理的测试数据必须复用训练得到的 process_model.pkl。\n"
        )
        help_box.setMaximumHeight(230)
        layout.addWidget(help_box)

        form = QFormLayout()
        self.edit_data, data_row = _path_row(
            self, "选择测试立方体",             filter_str=FILE_FILTER_CUBE,
        )
        btn_dir = QPushButton("选文件夹")
        btn_dir.clicked.connect(self._pick_data_dir)
        data_row.layout().addWidget(btn_dir)
        form.addRow("测试数据（文件或文件夹）：", data_row)

        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["HWB", "BHW"])
        form.addRow("数据排布：", self.combo_layout)

        self.edit_data_key = QLineEdit()
        self.edit_data_key.setPlaceholderText("仅 .mat/.npz 需要；ENVI/img/dat 请留空")
        form.addRow("MAT/NPZ 变量名：", self.edit_data_key)

        self.edit_pattern = QLineEdit(DEFAULT_INPUT_PATTERN)
        form.addRow("文件夹匹配模式：", self.edit_pattern)

        self.chk_preprocessed = QCheckBox("输入已是预处理后的 240 波段立方体")
        form.addRow("", self.chk_preprocessed)

        last = load_last_trained() or {}
        self.edit_prep, prep_row = _path_row(
            self, "选择 preprocess_model.pkl",
            filter_str="Pickle (*.pkl);;All Files (*)",
        )
        if last.get("preprocess_model_path"):
            self.edit_prep.setText(str(last["preprocess_model_path"]))
        form.addRow("预处理模型 process_model.pkl：", prep_row)

        self.edit_ckpt, ckpt_row = _path_row(
            self, "选择分类 checkpoint",
            filter_str="PyTorch (*.pth);;All Files (*)",
        )
        if last.get("checkpoint_path"):
            self.edit_ckpt.setText(str(last["checkpoint_path"]))
        form.addRow("分类模型 *.pth：", ckpt_row)

        self.edit_label, label_row = _path_row(
            self, "选择标签图（可选）", filter_str=FILE_FILTER_LABEL
        )
        form.addRow("标签（可选）：", label_row)

        self.edit_label_key = QLineEdit()
        form.addRow("标签变量名：", self.edit_label_key)

        self.edit_output, out_row = _path_row(self, "选择输出目录", directory=True)
        self.edit_output.setText(str(identification_data_dir() / "test_runs"))
        form.addRow("输出目录：", out_row)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "cuda:0"])
        self.combo_device.setCurrentText(_preferred_device())
        form.addRow("计算设备：", self.combo_device)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4096)
        self.spin_batch.setValue(_preferred_batch(_preferred_device()))
        form.addRow("推理 batch size：", self.spin_batch)

        layout.addLayout(form)

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

    def _pick_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择测试数据文件夹", self.edit_data.text())
        if path:
            self.edit_data.setText(path)

    def _append_log(self, text: str):
        self.log.appendPlainText(text.rstrip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _collect(self) -> dict:
        data_path = self.edit_data.text().strip()
        ckpt = self.edit_ckpt.text().strip()
        output_dir = self.edit_output.text().strip()
        if not data_path or not Path(data_path).exists():
            raise ValueError("请选择有效的测试数据。")
        if not ckpt or not Path(ckpt).exists():
            raise ValueError("请选择训练得到的 *.pth 分类模型。")
        if not output_dir:
            raise ValueError("请指定输出目录。")
        already = self.chk_preprocessed.isChecked()
        prep = self.edit_prep.text().strip()
        if not already and (not prep or not Path(prep).exists()):
            raise ValueError("未预处理的测试数据必须提供训练阶段的 process_model.pkl。")
        return {
            "data_path": data_path,
            "already_preprocessed": already,
            "data_key": self.edit_data_key.text().strip() or None,
            "data_layout": self.combo_layout.currentText(),
            "input_pattern": self.edit_pattern.text().strip() or DEFAULT_INPUT_PATTERN,
            "preprocess_model_path": prep,
            "checkpoint_path": ckpt,
            "label_path": self.edit_label.text().strip() or "",
            "label_key": self.edit_label_key.text().strip() or None,
            "output_dir": output_dir,
            "device": self.combo_device.currentText(),
            "batch_size": int(self.spin_batch.value()),
        }

    def _start(self):
        try:
            config = self._collect()
        except Exception as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.btn_start.setEnabled(False)
        self._append_log("======== 开始测试 ========")

        def job(log_fn):
            from .pipeline import run_testing
            return run_testing(config, log=log_fn)

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
        oa = summary.get("OA")
        aa = summary.get("AA")
        extra = ""
        if oa == oa and aa == aa:  # not NaN
            extra = f"\nOA={float(oa)*100:.2f}%  AA={float(aa)*100:.2f}%"
        QMessageBox.information(
            self, "模型测试",
            f"测试完成。\n输出：{summary.get('output_dir')}{extra}",
        )

    def _on_fail(self, message: str):
        self.btn_start.setEnabled(True)
        self._append_log("失败：" + message)
        QMessageBox.critical(self, "测试失败", message)

    def reject(self):
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "测试进行中", "请等待测试结束再关闭。")
            return
        super().reject()


class IdentificationApplyDialog(QDialog):
    """Choose builtin vs newly trained model for the currently opened cube."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — 模型应用")
        self.resize(560, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "对当前已打开的高光谱影像进行分类。\n"
                "应用波段范围：1.02–2.58 μm（重采样为 240 通道后预处理）。\n"
                "分类图显示在软件左下方结果区。"
            )
        )

        group = QGroupBox("选择模型")
        g_layout = QVBoxLayout(group)
        self.radio_builtin = QRadioButton("默认内置已训练模型")
        self.radio_trained = QRadioButton("使用「模型训练 / 模型测试」得到的新模型")
        self.radio_builtin.setChecked(True)
        g_layout.addWidget(self.radio_builtin)
        g_layout.addWidget(self.radio_trained)
        layout.addWidget(group)

        form = QFormLayout()
        self.edit_ckpt, ckpt_row = _path_row(
            self, "选择分类模型", filter_str="PyTorch (*.pth);;All Files (*)"
        )
        self.edit_prep, prep_row = _path_row(
            self, "选择预处理模型", filter_str="Pickle (*.pkl);;All Files (*)"
        )
        form.addRow("分类模型 *.pth：", ckpt_row)
        form.addRow("预处理模型 *.pkl：", prep_row)

        builtin_ckpt = builtin_checkpoint_path()
        builtin_prep = builtin_preprocess_path()
        last = load_last_trained() or {}
        self.edit_ckpt.setText(str(builtin_ckpt))
        self.edit_prep.setText(str(builtin_prep))

        self.radio_builtin.toggled.connect(
            lambda checked: self._fill_paths(checked, builtin_ckpt, builtin_prep, last)
        )
        if last.get("checkpoint_path") and not builtin_ckpt.exists():
            self.radio_trained.setChecked(True)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "cuda:0"])
        self.combo_device.setCurrentText(_preferred_device())
        form.addRow("计算设备：", self.combo_device)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4096)
        self.spin_batch.setValue(_preferred_batch(_preferred_device()))
        form.addRow("推理 batch size：", self.spin_batch)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setDecimals(3)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.0)
        self.spin_conf.setToolTip("最大 softmax 置信度低于该阈值的像元显示为背景 0")
        form.addRow("置信度阈值：", self.spin_conf)
        layout.addLayout(form)

        hint = QLabel(
            f"内置模型目录：{builtin_dir_text()}\n"
            "请将训练好的 model_best.pth 与 preprocess_model.pkl 放到该目录，"
            "或改用本次训练的新模型。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("开始应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _fill_paths(self, builtin_checked, builtin_ckpt, builtin_prep, last):
        if builtin_checked:
            self.edit_ckpt.setText(str(builtin_ckpt))
            self.edit_prep.setText(str(builtin_prep))
        else:
            self.edit_ckpt.setText(str(last.get("checkpoint_path") or ""))
            self.edit_prep.setText(str(last.get("preprocess_model_path") or ""))

    def _on_accept(self):
        cfg = self.params()
        if not Path(cfg["checkpoint_path"]).exists():
            QMessageBox.warning(
                self, "缺少分类模型",
                f"找不到分类模型：\n{cfg['checkpoint_path']}\n\n"
                "请先完成「模型训练」，或把内置 model_best.pth 放到指定目录。",
            )
            return
        if not Path(cfg["preprocess_model_path"]).exists():
            QMessageBox.warning(
                self, "缺少预处理模型",
                f"找不到预处理模型：\n{cfg['preprocess_model_path']}",
            )
            return
        self.accept()

    def params(self) -> dict:
        return {
            "use_builtin": self.radio_builtin.isChecked(),
            "checkpoint_path": self.edit_ckpt.text().strip(),
            "preprocess_model_path": self.edit_prep.text().strip(),
            "device": self.combo_device.currentText(),
            "batch_size": int(self.spin_batch.value()),
            "confidence_threshold": float(self.spin_conf.value()),
        }


def builtin_dir_text() -> str:
    return str(builtin_checkpoint_path().parent)
