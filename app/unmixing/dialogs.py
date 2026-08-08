"""Qt dialogs for Hapke endmember physical parameters."""

from __future__ import annotations

from typing import List, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)

from .hapke_rt import HapkeEndmember


class HapkeEndmemberParamDialog(QDialog):
    """
    Table to review/edit spectrum ID, density ρ, real index n, mean grain size D.
    Checkboxes select which endmembers participate in unmixing.
    Lab incidence / emission angles are shown (defaults 30° / 0°).
    """

    COL_USE = 0
    COL_ID = 1
    COL_D = 2
    COL_RHO = 3
    COL_N = 4

    def __init__(self, endmembers: Sequence[HapkeEndmember], parent=None,
                 lab_incidence_deg: float = 30.0, lab_emission_deg: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("Hapke 端元物理参数")
        self.resize(820, 420)
        self._endmembers = list(endmembers)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "端元光谱为反射率因子 REFF（非图像 I/F）。\n"
                "Excel 已读入：光谱ID、平均粒径 D、密度 ρ、折射率实部 n；可在此修改。\n"
                "勾选「选用」的端元才会参与后续「单光谱计算」和「图像处理」。\n"
                "解混时：图像 I/F = 模型 REFF × cos(太阳入射角)。"
            )
        )

        btn_row = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_none = QPushButton("全不选")
        btn_all.clicked.connect(lambda: self._set_all_selected(True))
        btn_none.clicked.connect(lambda: self._set_all_selected(False))
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.table = QTableWidget(len(endmembers), 5, self)
        self.table.setHorizontalHeaderLabels(
            ["选用", "光谱ID", "平均粒径 D (μm)", "密度 ρ", "折射率 n"]
        )
        self.table.horizontalHeader().setSectionResizeMode(self.COL_USE, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_ID, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_D, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_RHO, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_N, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

        self._checks: List[QCheckBox] = []
        self._id_edits: List[QLineEdit] = []

        for i, em in enumerate(endmembers):
            chk = QCheckBox()
            chk.setChecked(bool(getattr(em, "selected", True)))
            chk.setToolTip("勾选后参与单光谱计算 / 图像处理")
            self.table.setCellWidget(i, self.COL_USE, chk)
            self._checks.append(chk)

            id_edit = QLineEdit(em.spectrum_id or em.name)
            self.table.setCellWidget(i, self.COL_ID, id_edit)
            self._id_edits.append(id_edit)

            for col, value, lo, hi, step, decimals in (
                (self.COL_D, em.grain_size_um, 0.1, 5000.0, 1.0, 2),
                (self.COL_RHO, em.density, 0.1, 20.0, 0.1, 3),
                (self.COL_N, em.n, 1.01, 5.0, 0.01, 3),
            ):
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setValue(float(value))
                self.table.setCellWidget(i, col, spin)

        layout.addWidget(self.table)

        # Lab geometry used when inverting k from endmember reflectance
        form = QFormLayout()
        self.lab_inc = QDoubleSpinBox()
        self.lab_inc.setRange(0.0, 89.0)
        self.lab_inc.setDecimals(2)
        self.lab_inc.setValue(float(lab_incidence_deg))
        self.lab_emi = QDoubleSpinBox()
        self.lab_emi.setRange(0.0, 89.0)
        self.lab_emi.setDecimals(2)
        self.lab_emi.setValue(float(lab_emission_deg))
        form.addRow("端元 REFF 测量入射角 i (°)", self.lab_inc)
        form.addRow("端元 REFF 测量发射角 e (°)", self.lab_emi)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all_selected(self, selected: bool):
        for chk in self._checks:
            chk.setChecked(bool(selected))

    def _on_accept(self):
        selected_rows = [i for i, chk in enumerate(self._checks) if chk.isChecked()]
        if not selected_rows:
            QMessageBox.warning(
                self, "未选用端元",
                "请至少勾选一个端元，才能进行单光谱计算 / 图像处理。",
            )
            return
        for i in selected_rows:
            dens = self.table.cellWidget(i, self.COL_RHO).value()
            n = self.table.cellWidget(i, self.COL_N).value()
            d = self.table.cellWidget(i, self.COL_D).value()
            sid = self._id_edits[i].text().strip()
            if not sid:
                QMessageBox.warning(
                    self, "参数错误",
                    f"第 {i+1} 行光谱ID为空。请填写后再次点击「确定」。",
                )
                return
            if dens <= 0 or n <= 1.0 or d <= 0:
                QMessageBox.warning(
                    self, "参数错误",
                    f"第 {i+1} 行参数无效（要求：ρ>0，n>1，D>0）。\n"
                    "请修改后再次点击「确定」。",
                )
                return
        self.done(QDialog.Accepted)

    def result_endmembers(self) -> List[HapkeEndmember]:
        """Return only checked endmembers, with edited physical parameters."""
        out: List[HapkeEndmember] = []
        for i, em in enumerate(self._endmembers):
            if not self._checks[i].isChecked():
                continue
            sid = self._id_edits[i].text().strip() or em.spectrum_id or em.name
            out.append(
                HapkeEndmember(
                    name=sid,
                    wavelengths=em.wavelengths,
                    reflectance=em.reflectance,
                    density=float(self.table.cellWidget(i, self.COL_RHO).value()),
                    n=float(self.table.cellWidget(i, self.COL_N).value()),
                    grain_size_um=float(self.table.cellWidget(i, self.COL_D).value()),
                    k=None,
                    ssa=None,
                    spectrum_id=sid,
                    source=em.source,
                    selected=True,
                )
            )
        return out

    def lab_geometry(self):
        return float(self.lab_inc.value()), float(self.lab_emi.value())
