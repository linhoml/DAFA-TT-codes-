"""Qt dialogs for Identification: train, test, and apply."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
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
    identification_data_dir,
    load_last_trained,
)
from .io import DEFAULT_INPUT_PATTERN, FILE_FILTER_CUBE, FILE_FILTER_LABEL


def _torch_cuda_status_text() -> str:
    try:
        from .crism_common import python_executable, torch_cuda_status
        py = python_executable()
        info = torch_cuda_status()
    except Exception as exc:
        try:
            from .crism_common import python_executable
            py = python_executable()
        except Exception:
            py = "（未知）"
        return (
            f"启动用的 Python：{py}\n"
            f"无法导入 PyTorch（{exc}）。选 cuda:0 不会用到显卡。"
            f"请在命令行运行：\"{py}\" -m pip install torch"
        )
    if info["available"] and info["gpu_name"]:
        return (
            f"启动用的 Python：{py}\n"
            f"已检测到 GPU：{info['gpu_name']}。"
            f"PyTorch {info['torch_version']}（CUDA {info['cuda_built']}）。"
            "选 cuda:0 才会用显卡训练。"
        )
    if info["available"]:
        return (
            f"启动用的 Python：{py}\n"
            f"PyTorch {info['torch_version']} 报告 CUDA 可用，"
            "但没读到 GPU 名称。选 cuda:0 尝试使用显卡。"
        )
    quoted = f'"{py}"' if " " in py else py
    return (
        f"启动用的 Python：{py}\n"
        f"这个解释器里的 PyTorch {info['torch_version']} "
        f"看不到 NVIDIA GPU（cuda.is_available=False，"
        f"torch.version.cuda={info['cuda_built']}）。\n"
        "下拉框选 cuda:0 也不会真正用显卡。请用上面这一条安装 GPU 版，例如：\n"
        f"{quoted} -m pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu128"
    )


def _preferred_device() -> str:
    try:
        from .crism_common import torch_cuda_status
        if torch_cuda_status()["available"]:
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _preferred_batch(device: str) -> int:
    return 256 if device.startswith("cuda") else 64


def _add_device_row(form: QFormLayout) -> QComboBox:
    combo = QComboBox()
    combo.addItems(["cpu", "cuda:0"])
    combo.setCurrentText(_preferred_device())
    form.addRow("计算设备：", combo)
    hint = QLabel(_torch_cuda_status_text())
    hint.setWordWrap(True)
    form.addRow("", hint)
    return combo


def _ensure_device_usable(parent: QWidget, device_text: str) -> bool:
    try:
        from .crism_common import resolve_device
        resolve_device(device_text)
        return True
    except Exception as exc:
        QMessageBox.critical(parent, "计算设备不可用", str(exc))
        return False


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
            import traceback

            self.log.emit(traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            stream.flush()
            sys.stdout = old_out
            sys.stderr = old_err


class IdentificationTrainDialog(QDialog):
    """Collect training data paths and run preprocess + LSGA training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — LSGA 模型训练")
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

        self.combo_device = _add_device_row(form)

        layout.addLayout(form)
        speed_hint = QLabel(
            "读取后自动截取 1.02–2.6 μm，并做去尖峰 / 空间修补 / SG / L2，无需预处理模型。"
            "有 NVIDIA 显卡且上方显示已检测到 GPU 时请选 cuda:0。"
            "16GB 显存建议 batch=256。"
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

        if not _ensure_device_usable(self, config["device"]):
            return

        self.btn_start.setEnabled(False)
        self._append_log("======== 开始训练 ========")
        self._append_log(f"界面选择的计算设备：{config['device']}")

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
        metrics = (record or {}).get("metrics") or {}
        lines = [
            "训练完成。\n",
            f"最佳分类模型（.pth）：{record.get('checkpoint_path')}",
            "",
            "读入数据时已自动截取 1.02–2.6 μm 并完成去尖峰 / SG / L2 等预处理，",
            "不另存 preprocess_model.pkl。「模型应用」会对打开的影像做同样处理。",
        ]
        if metrics.get("test_all_OA") is not None:
            lines.append(
                f"【与原 main.py 相同口径】标注区整体 OA（含训练像元）="
                f"{metrics['test_all_OA'] * 100:.2f}%  "
                f"AA={metrics['test_all_AA'] * 100:.2f}%"
            )
        if metrics.get("heldout_OA") is not None:
            lines.append(
                f"空间留出测试 OA={metrics['heldout_OA'] * 100:.2f}%  "
                f"AA={metrics['heldout_AA'] * 100:.2f}%"
            )
        lines.append(
            "\n每轮日志里的 val_OA 是空间验证，通常低于上面的整体 OA。"
            "可在「模型应用」中选择「本次训练的新模型」。"
        )
        QMessageBox.information(self, "模型训练", "\n".join(lines))

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
        self.setWindowTitle("Identification — LSGA 模型测试")
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
            "• 必须提供与测试影像同高同宽的标签图，才能计算检验精度。\n"
            "• 日志会写出 OA / AA / Kappa 以及各类别召回率。\n"
            "• 测试与训练相同：按波长截取 1.02–2.6 μm 后自动预处理，不需要 .pkl。\n"
            "• 分类图写出 ENVI *.img / *.hdr；关闭窗口后叠加显示在主界面左下方。\n"
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

        last = load_last_trained() or {}
        self.edit_ckpt, ckpt_row = _path_row(
            self, "选择分类 checkpoint",
            filter_str="PyTorch (*.pth);;All Files (*)",
        )
        if last.get("checkpoint_path"):
            self.edit_ckpt.setText(str(last["checkpoint_path"]))
        form.addRow("分类模型 *.pth：", ckpt_row)

        self.edit_label, label_row = _path_row(
            self, "选择检验标签图", filter_str=FILE_FILTER_LABEL
        )
        form.addRow("检验标签（须与影像同尺寸）：", label_row)

        self.edit_label_key = QLineEdit()
        form.addRow("标签变量名：", self.edit_label_key)

        self.edit_output, out_row = _path_row(self, "选择输出目录", directory=True)
        self.edit_output.setText(str(identification_data_dir() / "test_runs"))
        form.addRow("输出目录：", out_row)

        self.combo_device = _add_device_row(form)

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
        label_path = self.edit_label.text().strip()
        if not data_path or not Path(data_path).exists():
            raise ValueError("请选择有效的测试数据。")
        if not ckpt or not Path(ckpt).is_file():
            raise ValueError("请选择训练得到的 *.pth 分类模型。")
        if not label_path or not Path(label_path).exists():
            raise ValueError(
                "模型测试必须提供与影像空间尺寸一致的标签图"
                "（.mat / .img / .dat / .hdr 等，二维，类别 1..K），才能计算检验精度。"
            )
        if not output_dir:
            raise ValueError("请指定输出目录。")
        return {
            "data_path": data_path,
            "data_key": self.edit_data_key.text().strip() or None,
            "data_layout": self.combo_layout.currentText(),
            "input_pattern": self.edit_pattern.text().strip() or DEFAULT_INPUT_PATTERN,
            "checkpoint_path": ckpt,
            "label_path": label_path,
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
        if not _ensure_device_usable(self, config["device"]):
            return
        self.result_summary = None
        self.btn_start.setEnabled(False)
        self._append_log("======== 开始测试 ========")
        self._append_log(f"界面选择的计算设备：{config['device']}")

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
        kappa = summary.get("Kappa")
        extra = ""
        try:
            if oa is not None and aa is not None and oa == oa and aa == aa:
                extra = (
                    f"\nOA={float(oa) * 100:.2f}%  "
                    f"AA={float(aa) * 100:.2f}%"
                )
                if kappa is not None and kappa == kappa:
                    extra += f"  Kappa={float(kappa):.4f}"
        except (TypeError, ValueError):
            extra = ""
        QMessageBox.information(
            self, "模型测试",
            f"测试完成。\n输出：{summary.get('output_dir')}{extra}\n\n"
            "检验精度已写入上方日志。\n"
            "关闭本窗口后，分类图会叠加显示在主界面左下方；"
            "可在「分类显示类别」输入数字只显示某一类矿物。",
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
    """Classify the opened cube, a file, or every cube in a folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — LSGA 模型应用")
        self.resize(640, 640)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "对影像做 1.02–2.6 μm 截取与自动预处理后分类。\n"
                "可选当前已打开的影像、单个文件，或文件夹内全部立方体。\n"
                "分类图叠加显示在左下方；同时写出 ENVI *.img / *.hdr。"
            )
        )

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

        form = QFormLayout()
        self.edit_data, data_row = _path_row(
            self, "选择立方体文件", filter_str=FILE_FILTER_CUBE,
        )
        btn_dir = QPushButton("选文件夹")
        btn_dir.clicked.connect(self._pick_data_dir)
        data_row.layout().addWidget(btn_dir)
        self.edit_data.textChanged.connect(self._on_data_path_changed)
        form.addRow("文件或文件夹：", data_row)

        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["HWB", "BHW"])
        form.addRow("数据排布：", self.combo_layout)

        self.edit_data_key = QLineEdit()
        self.edit_data_key.setPlaceholderText("仅 .mat/.npz 需要；ENVI 请留空")
        form.addRow("MAT/NPZ 变量名：", self.edit_data_key)

        self.edit_pattern = QLineEdit(DEFAULT_INPUT_PATTERN)
        form.addRow("文件夹匹配模式：", self.edit_pattern)

        model_group = QGroupBox("选择模型")
        g_layout = QVBoxLayout(model_group)
        self.radio_builtin = QRadioButton("默认内置已训练模型")
        self.radio_trained = QRadioButton("使用「模型训练 / 模型测试」得到的新模型")
        self.radio_builtin.setChecked(True)
        g_layout.addWidget(self.radio_builtin)
        g_layout.addWidget(self.radio_trained)
        layout.addWidget(model_group)

        self.edit_ckpt, ckpt_row = _path_row(
            self, "选择分类模型", filter_str="PyTorch (*.pth);;All Files (*)"
        )
        form.addRow("分类模型 *.pth：", ckpt_row)

        builtin_ckpt = builtin_checkpoint_path()
        last = load_last_trained() or {}
        last_ckpt = str(last.get("checkpoint_path") or "")
        last_ckpt_ok = bool(last_ckpt) and Path(last_ckpt).is_file()
        builtin_ok = builtin_ckpt.is_file()

        self.radio_builtin.toggled.connect(
            lambda checked: self._fill_paths(checked, builtin_ckpt, last)
        )
        if last_ckpt_ok and not builtin_ok:
            self.radio_trained.setChecked(True)
        self._fill_paths(self.radio_builtin.isChecked(), builtin_ckpt, last)

        self.combo_device = _add_device_row(form)

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

        self.edit_save, save_row = _path_row(
            self, "选择结果保存目录", directory=True,
        )
        self.edit_save.setText(str(identification_data_dir() / "apply_runs"))
        form.addRow("结果保存路径：", save_row)
        layout.addLayout(form)

        hint = QLabel(
            f"内置模型目录：{builtin_dir_text()}\n"
            "每个输入立方体会写出 输入文件名_classification.img 与同名 .hdr（ENVI 分类图）。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

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

    def _fill_paths(self, builtin_checked, builtin_ckpt, last):
        if builtin_checked:
            self.edit_ckpt.setText(str(builtin_ckpt) if Path(builtin_ckpt).is_file() else "")
            return
        self.edit_ckpt.setText(str(last.get("checkpoint_path") or ""))

    def _source(self) -> str:
        if self.radio_file.isChecked():
            return "file"
        if self.radio_folder.isChecked():
            return "folder"
        return "opened"

    def _on_accept(self):
        cfg = self.params()
        ckpt = Path(cfg["checkpoint_path"].strip()) if cfg["checkpoint_path"].strip() else None
        if ckpt is None or not ckpt.is_file():
            QMessageBox.warning(
                self, "缺少分类模型",
                f"找不到分类模型：\n{cfg['checkpoint_path']}\n\n"
                "请先完成「模型训练」，或把内置 model_best.pth 放到指定目录。",
            )
            return
        if not cfg["save_dir"]:
            QMessageBox.warning(self, "缺少保存路径", "请指定结果保存目录。")
            return
        source = cfg["source"]
        data_path = cfg["data_path"]
        if source == "opened":
            parent = self.parent()
            if getattr(parent, "current_data", None) is None:
                QMessageBox.warning(
                    self, "缺少影像",
                    "当前没有已打开的高光谱影像。请选择单个文件或文件夹。",
                )
                return
        if source == "file":
            if not data_path or not Path(data_path).is_file():
                QMessageBox.warning(self, "缺少输入文件", "请选择要分类的立方体文件。")
                return
        if source == "folder":
            if not data_path or not Path(data_path).is_dir():
                QMessageBox.warning(self, "缺少输入文件夹", "请选择包含立方体的文件夹。")
                return
        if not _ensure_device_usable(self, cfg["device"]):
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
            "data_layout": self.combo_layout.currentText(),
            "data_key": self.edit_data_key.text().strip() or None,
            "input_pattern": self.edit_pattern.text().strip() or DEFAULT_INPUT_PATTERN,
            "use_builtin": self.radio_builtin.isChecked(),
            "checkpoint_path": self.edit_ckpt.text().strip(),
            "device": self.combo_device.currentText(),
            "batch_size": int(self.spin_batch.value()),
            "confidence_threshold": float(self.spin_conf.value()),
            "save_dir": self.edit_save.text().strip(),
        }


def builtin_dir_text() -> str:
    return str(builtin_checkpoint_path().parent)
