"""Qt dialogs for Identification → MAE SSL."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread
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
    QVBoxLayout,
)

from identification.defaults import identification_data_dir
from identification.dialogs import (
    _Worker,
    _add_device_row,
    _ensure_device_usable,
    _path_row,
    _preferred_device,
)
from identification.io import DEFAULT_INPUT_PATTERN, FILE_FILTER_CUBE, FILE_FILTER_LABEL

from .defaults import MAE_HELP, mae_data_dir
from .pipeline import (
    default_encoder_path,
    default_finetune_checkpoint,
    finetune,
    pretrain,
    run_test,
)


def _start_worker(dialog, fn, on_ok):
    dialog.btn_start.setEnabled(False)
    dialog._thread = QThread(dialog)
    dialog._worker = _Worker(fn)
    dialog._worker.moveToThread(dialog._thread)
    dialog._thread.started.connect(dialog._worker.run)
    dialog._worker.log.connect(dialog._append_log)
    dialog._worker.finished.connect(on_ok)
    dialog._worker.failed.connect(dialog._failed)
    dialog._worker.finished.connect(dialog._thread.quit)
    dialog._worker.failed.connect(dialog._thread.quit)
    dialog._thread.finished.connect(lambda: dialog.btn_start.setEnabled(True))
    dialog._thread.start()


class _MaeBaseDialog(QDialog):
    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _failed(self, message: str) -> None:
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, self.windowTitle(), message)


class MaePretrainDialog(_MaeBaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — MAE 自监督预训练")
        self.resize(720, 760)
        self._thread: Optional[QThread] = None
        self._worker = None
        layout = QVBoxLayout(self)
        help_box = QPlainTextEdit()
        help_box.setReadOnly(True)
        help_box.setPlainText(MAE_HELP)
        help_box.setMaximumHeight(220)
        layout.addWidget(help_box)
        form = QFormLayout()
        self.edit_data, data_row = _path_row(
            self, "选择无标签立方体或文件夹", filter_str=FILE_FILTER_CUBE
        )
        btn_dir = QPushButton("选文件夹")
        btn_dir.clicked.connect(self._pick_dir)
        data_row.layout().addWidget(btn_dir)
        form.addRow("无标签 CRISM（文件或文件夹）：", data_row)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["HWB", "BHW"])
        form.addRow("数据排布：", self.combo_layout)
        self.edit_pattern = QLineEdit(DEFAULT_INPUT_PATTERN)
        form.addRow("文件夹匹配模式：", self.edit_pattern)
        self.edit_output, out_row = _path_row(self, "输出目录", directory=True)
        self.edit_output.setText(str(mae_data_dir() / "pretrain"))
        form.addRow("输出目录：", out_row)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 2000)
        self.spin_epochs.setValue(50)
        form.addRow("预训练轮数：", self.spin_epochs)
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(32, 200000)
        self.spin_samples.setValue(2048)
        form.addRow("每轮随机窗口数：", self.spin_samples)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(8 if _preferred_device() == "cpu" else 32)
        form.addRow("batch size：", self.spin_batch)
        self.spin_readers = QSpinBox()
        self.spin_readers.setRange(1, 16)
        self.spin_readers.setValue(1 if _preferred_device() == "cpu" else 4)
        form.addRow("读盘线程数：", self.spin_readers)
        self.combo_prep = QComboBox()
        self.combo_prep.addItems(["crop", "full"])
        form.addRow("预处理：", self.combo_prep)
        self.combo_device = _add_device_row(form)
        layout.addLayout(form)
        hint = QLabel(
            "预处理 crop=只截 1.02–2.6 μm 并 L2（适合上万幅无标签图）；"
            "full=与 LSGA 相同的去尖峰/空间修补。\n"
            "轮数×每轮窗口=看到的块总数，加大只会更久。GPU 利用率低时："
            "batch 32–64（16GB 显存可用 64–128），读盘线程 4–8。"
            "1 万幅图建议轮数 100–200、每轮 8192–16384。"
            "数据请放本地硬盘；网盘上 .img 为 0 字节会被跳过。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.btn_start = QPushButton("开始预训练")
        self.btn_start.clicked.connect(self._start)
        layout.addWidget(self.btn_start)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择无标签立方体文件夹")
        if path:
            self.edit_data.setText(path)

    def _start(self):
        data = self.edit_data.text().strip()
        if not data or not Path(data).exists():
            QMessageBox.warning(self, "MAE 预训练", "请选择无标签立方体文件或文件夹。")
            return
        if not _ensure_device_usable(self, self.combo_device.currentText()):
            return
        cfg = {
            "data_path": data,
            "data_layout": self.combo_layout.currentText(),
            "input_pattern": self.edit_pattern.text().strip() or "*",
            "output_dir": self.edit_output.text().strip(),
            "epochs": int(self.spin_epochs.value()),
            "samples_per_epoch": int(self.spin_samples.value()),
            "batch_size": int(self.spin_batch.value()),
            "num_readers": int(self.spin_readers.value()),
            "preprocess_mode": self.combo_prep.currentText(),
            "device": self.combo_device.currentText(),
        }
        self.log.clear()
        _start_worker(self, lambda log: pretrain(cfg, log=log), self._ok)

    def _ok(self, record):
        QMessageBox.information(
            self, "MAE 预训练完成",
            f"编码器已保存：\n{record.get('checkpoint_path')}\n"
            f"loss={record.get('final_loss')}",
        )


class MaeFinetuneDialog(_MaeBaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — MAE 少量样本微调")
        self.resize(720, 780)
        self._thread: Optional[QThread] = None
        self._worker = None
        layout = QVBoxLayout(self)
        help_box = QPlainTextEdit()
        help_box.setReadOnly(True)
        help_box.setPlainText(MAE_HELP)
        help_box.setMaximumHeight(160)
        layout.addWidget(help_box)
        form = QFormLayout()
        self.edit_encoder, enc_row = _path_row(
            self, "选择预训练编码器", filter_str="PyTorch (*.pt *.pth);;All Files (*)"
        )
        self.edit_encoder.setText(default_encoder_path())
        form.addRow("预训练编码器：", enc_row)
        self.edit_data, data_row = _path_row(
            self, "选择有标签立方体", filter_str=FILE_FILTER_CUBE
        )
        btn_dir = QPushButton("选文件夹")
        btn_dir.clicked.connect(lambda: self._pick(self.edit_data, True))
        data_row.layout().addWidget(btn_dir)
        form.addRow("立方体（文件或文件夹）：", data_row)
        self.edit_label, label_row = _path_row(
            self, "选择标签", filter_str=FILE_FILTER_LABEL
        )
        btn_ldir = QPushButton("选文件夹")
        btn_ldir.clicked.connect(lambda: self._pick(self.edit_label, True))
        label_row.layout().addWidget(btn_ldir)
        form.addRow("标签：", label_row)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["HWB", "BHW"])
        form.addRow("数据排布：", self.combo_layout)
        self.spin_classes = QSpinBox()
        self.spin_classes.setRange(2, 64)
        self.spin_classes.setValue(24)
        form.addRow("类别数：", self.spin_classes)
        self.spin_few = QSpinBox()
        self.spin_few.setRange(0, 10000)
        self.spin_few.setValue(0)
        self.spin_few.setSpecialValueText("不限制（用全部标注）")
        form.addRow("每类最多样本：", self.spin_few)
        self.chk_freeze = QCheckBox("冻结编码器（linear probe / 极少样本）")
        form.addRow("", self.chk_freeze)
        self.edit_output, out_row = _path_row(self, "输出目录", directory=True)
        self.edit_output.setText(str(mae_data_dir() / "finetune"))
        form.addRow("输出目录：", out_row)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(40)
        form.addRow("微调轮数：", self.spin_epochs)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        form.addRow("batch size：", self.spin_batch)
        self.combo_device = _add_device_row(form)
        layout.addLayout(form)
        self.btn_start = QPushButton("开始微调")
        self.btn_start.clicked.connect(self._start)
        layout.addWidget(self.btn_start)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick(self, edit, directory: bool):
        if directory:
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if path:
            edit.setText(path)

    def _start(self):
        if not Path(self.edit_encoder.text().strip()).is_file():
            QMessageBox.warning(self, "MAE 微调", "请先完成自监督预训练，或选择 encoder.pt。")
            return
        if not Path(self.edit_data.text().strip()).exists():
            QMessageBox.warning(self, "MAE 微调", "请选择立方体。")
            return
        if not Path(self.edit_label.text().strip()).exists():
            QMessageBox.warning(self, "MAE 微调", "请选择标签。")
            return
        if not _ensure_device_usable(self, self.combo_device.currentText()):
            return
        cfg = {
            "encoder_path": self.edit_encoder.text().strip(),
            "data_path": self.edit_data.text().strip(),
            "label_path": self.edit_label.text().strip(),
            "data_layout": self.combo_layout.currentText(),
            "num_classes": int(self.spin_classes.value()),
            "max_per_class": int(self.spin_few.value()),
            "freeze_encoder": self.chk_freeze.isChecked(),
            "output_dir": self.edit_output.text().strip(),
            "epochs": int(self.spin_epochs.value()),
            "batch_size": int(self.spin_batch.value()),
            "device": self.combo_device.currentText(),
            "preprocess_mode": "full",
        }
        self.log.clear()
        _start_worker(self, lambda log: finetune(cfg, log=log), self._ok)

    def _ok(self, record):
        QMessageBox.information(
            self, "MAE 微调完成",
            f"分类模型：\n{record.get('checkpoint_path')}\n"
            f"val_acc={float(record.get('val_acc') or 0) * 100:.2f}%",
        )


class MaeTestDialog(_MaeBaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — MAE 模型测试")
        self.resize(700, 700)
        self._thread: Optional[QThread] = None
        self._worker = None
        self.result_summary = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.edit_ckpt, ckpt_row = _path_row(
            self, "选择微调模型", filter_str="PyTorch (*.pth *.pt);;All Files (*)"
        )
        self.edit_ckpt.setText(default_finetune_checkpoint())
        form.addRow("微调模型：", ckpt_row)
        self.edit_data, data_row = _path_row(
            self, "测试立方体", filter_str=FILE_FILTER_CUBE
        )
        btn_dir = QPushButton("选文件夹")
        btn_dir.clicked.connect(lambda: self._pick_dir(self.edit_data))
        data_row.layout().addWidget(btn_dir)
        form.addRow("测试影像：", data_row)
        self.edit_label, label_row = _path_row(
            self, "测试标签", filter_str=FILE_FILTER_LABEL
        )
        form.addRow("标签：", label_row)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["HWB", "BHW"])
        form.addRow("数据排布：", self.combo_layout)
        self.edit_output, out_row = _path_row(self, "输出目录", directory=True)
        self.edit_output.setText(str(mae_data_dir() / "test"))
        form.addRow("输出目录：", out_row)
        self.combo_device = _add_device_row(form)
        layout.addLayout(form)
        self.btn_start = QPushButton("开始测试")
        self.btn_start.clicked.connect(self._start)
        layout.addWidget(self.btn_start)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            edit.setText(path)

    def _start(self):
        if not Path(self.edit_ckpt.text().strip()).is_file():
            QMessageBox.warning(self, "MAE 测试", "请选择微调后的 model_best.pth。")
            return
        if not Path(self.edit_data.text().strip()).exists():
            QMessageBox.warning(self, "MAE 测试", "请选择测试影像。")
            return
        if not Path(self.edit_label.text().strip()).exists():
            QMessageBox.warning(self, "MAE 测试", "测试必须提供标签。")
            return
        if not _ensure_device_usable(self, self.combo_device.currentText()):
            return
        cfg = {
            "checkpoint_path": self.edit_ckpt.text().strip(),
            "data_path": self.edit_data.text().strip(),
            "label_path": self.edit_label.text().strip(),
            "data_layout": self.combo_layout.currentText(),
            "output_dir": self.edit_output.text().strip(),
            "device": self.combo_device.currentText(),
        }
        self.log.clear()
        _start_worker(self, lambda log: run_test(cfg, log=log), self._ok)

    def _ok(self, result):
        self.result_summary = result
        oa = result.get("OA")
        extra = f"OA={oa * 100:.2f}%" if oa is not None else ""
        QMessageBox.information(self, "MAE 测试完成", extra or "已完成。")


class MaeApplyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identification — MAE 模型应用")
        self.resize(640, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "用微调后的 spatial+spectral MAE 对 CRISM 分类，写出 *_MAE_classification.img。"
        ))
        src = QGroupBox("输入数据")
        src_l = QVBoxLayout(src)
        self.radio_opened = QRadioButton("当前已打开的高光谱影像")
        self.radio_file = QRadioButton("单个文件")
        self.radio_folder = QRadioButton("文件夹")
        has_opened = getattr(parent, "current_data", None) is not None
        self.radio_opened.setEnabled(bool(has_opened))
        if has_opened:
            self.radio_opened.setChecked(True)
        else:
            self.radio_file.setChecked(True)
        src_l.addWidget(self.radio_opened)
        src_l.addWidget(self.radio_file)
        src_l.addWidget(self.radio_folder)
        layout.addWidget(src)
        form = QFormLayout()
        self.edit_data, data_row = _path_row(
            self, "立方体", filter_str=FILE_FILTER_CUBE
        )
        btn_dir = QPushButton("选文件夹")
        btn_dir.clicked.connect(self._pick_dir)
        data_row.layout().addWidget(btn_dir)
        form.addRow("文件或文件夹：", data_row)
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["HWB", "BHW"])
        form.addRow("数据排布：", self.combo_layout)
        self.edit_ckpt, ckpt_row = _path_row(
            self, "微调模型", filter_str="PyTorch (*.pth *.pt);;All Files (*)"
        )
        self.edit_ckpt.setText(default_finetune_checkpoint())
        form.addRow("微调模型：", ckpt_row)
        self.edit_save, save_row = _path_row(self, "保存目录", directory=True)
        self.edit_save.setText(str(identification_data_dir() / "mae_apply"))
        form.addRow("保存目录：", save_row)
        self.combo_device = _add_device_row(form)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self.edit_data.setText(path)

    def _accept(self):
        if self.radio_opened.isChecked():
            self.accept()
            return
        if not Path(self.edit_data.text().strip()).exists():
            QMessageBox.warning(self, "MAE 应用", "请选择立方体文件或文件夹。")
            return
        if not Path(self.edit_ckpt.text().strip()).is_file():
            QMessageBox.warning(self, "MAE 应用", "请选择微调模型。")
            return
        self.accept()

    def params(self) -> dict:
        if self.radio_opened.isChecked():
            source = "opened"
        elif self.radio_folder.isChecked():
            source = "folder"
        else:
            source = "file"
        return {
            "source": source,
            "data_path": self.edit_data.text().strip(),
            "checkpoint_path": self.edit_ckpt.text().strip(),
            "save_dir": self.edit_save.text().strip(),
            "device": self.combo_device.currentText(),
            "data_layout": self.combo_layout.currentText(),
            "batch_size": 8,
            "input_pattern": "*",
        }
