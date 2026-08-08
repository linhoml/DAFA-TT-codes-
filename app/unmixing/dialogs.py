"""Qt dialogs for Hapke endmember physical parameters."""

from __future__ import annotations

from typing import List, Sequence

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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
    Table to review/edit mineral name, density ρ, real index n, mean grain size D,
    and per-mineral REFF lab incidence / emission angles.
    Checkboxes select which endmembers participate in unmixing.
    """

    COL_USE = 0
    COL_NAME = 1
    COL_D = 2
    COL_RHO = 3
    COL_N = 4
    COL_INC = 5
    COL_EMI = 6

    def __init__(self, endmembers: Sequence[HapkeEndmember], parent=None,
                 lab_incidence_deg: float = 30.0, lab_emission_deg: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("Hapke 端元物理参数")
        self.resize(980, 440)
        self._endmembers = list(endmembers)
        self._default_inc = float(lab_incidence_deg)
        self._default_emi = float(lab_emission_deg)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "端元光谱为反射率因子 REFF（非图像 I/F）。\n"
                "Excel 已读入：矿物名称（第1行）、平均粒径 D、密度 ρ、折射率实部 n；可在此修改。\n"
                "每个矿物可单独设置 REFF 测量入射角 i、发射角 e（默认 30° / 0°）。\n"
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

        self.table = QTableWidget(len(endmembers), 7, self)
        self.table.setHorizontalHeaderLabels(
            [
                "选用",
                "矿物名称",
                "平均粒径 D (μm)",
                "密度 ρ",
                "折射率 n",
                "入射角 i (°)",
                "发射角 e (°)",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(self.COL_USE, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        for col in (self.COL_D, self.COL_RHO, self.COL_N, self.COL_INC, self.COL_EMI):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

        self._checks: List[QCheckBox] = []
        self._name_edits: List[QLineEdit] = []

        for i, em in enumerate(endmembers):
            chk = QCheckBox()
            chk.setChecked(bool(getattr(em, "selected", True)))
            chk.setToolTip("勾选后参与单光谱计算 / 图像处理")
            self.table.setCellWidget(i, self.COL_USE, chk)
            self._checks.append(chk)

            # 矿物名称：Excel 第1行
            name_edit = QLineEdit(em.name or em.spectrum_id or f"EM{i+1}")
            self.table.setCellWidget(i, self.COL_NAME, name_edit)
            self._name_edits.append(name_edit)

            inc0 = float(getattr(em, "lab_incidence_deg", self._default_inc) or self._default_inc)
            emi0 = float(getattr(em, "lab_emission_deg", self._default_emi) or self._default_emi)

            for col, value, lo, hi, step, decimals in (
                (self.COL_D, em.grain_size_um, 0.1, 5000.0, 1.0, 2),
                (self.COL_RHO, em.density, 0.1, 20.0, 0.1, 3),
                (self.COL_N, em.n, 1.01, 5.0, 0.01, 3),
                (self.COL_INC, inc0, 0.0, 89.0, 0.5, 2),
                (self.COL_EMI, emi0, 0.0, 89.0, 0.5, 2),
            ):
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setValue(float(value))
                self.table.setCellWidget(i, col, spin)

        layout.addWidget(self.table)

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
            name = self._name_edits[i].text().strip()
            if not name:
                QMessageBox.warning(
                    self, "参数错误",
                    f"第 {i+1} 行矿物名称为空。请填写后再次点击「确定」。",
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
            name = self._name_edits[i].text().strip() or em.name or f"EM{i+1}"
            out.append(
                HapkeEndmember(
                    name=name,
                    wavelengths=em.wavelengths,
                    reflectance=em.reflectance,
                    density=float(self.table.cellWidget(i, self.COL_RHO).value()),
                    n=float(self.table.cellWidget(i, self.COL_N).value()),
                    grain_size_um=float(self.table.cellWidget(i, self.COL_D).value()),
                    k=None,
                    ssa=None,
                    spectrum_id=em.spectrum_id or "",
                    lab_incidence_deg=float(self.table.cellWidget(i, self.COL_INC).value()),
                    lab_emission_deg=float(self.table.cellWidget(i, self.COL_EMI).value()),
                    source=em.source,
                    selected=True,
                )
            )
        return out

    def lab_geometry(self):
        """
        Backward-compatible helper: return geometry of the first selected
        endmember, or defaults if none.
        """
        for i, chk in enumerate(self._checks):
            if chk.isChecked():
                return (
                    float(self.table.cellWidget(i, self.COL_INC).value()),
                    float(self.table.cellWidget(i, self.COL_EMI).value()),
                )
        return self._default_inc, self._default_emi
