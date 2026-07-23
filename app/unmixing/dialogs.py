"""Qt dialogs for Hapke endmember physical parameters."""

from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHeaderView,
    QAbstractItemView,
)

from .hapke_rt import HapkeEndmember


class HapkeEndmemberParamDialog(QDialog):
    """
    Table to edit density ρ, real index n, mean grain size D for each mineral.
    """

    def __init__(self, endmembers: Sequence[HapkeEndmember], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hapke 端元物理参数")
        self.resize(640, 360)
        self._endmembers = list(endmembers)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "请为每个矿物端元输入：密度 ρ (g/cm³)、折射率实部 n（全波段共用）、"
                "平均粒径 D (μm)。\n"
                "程序将用 Hapke 模型反演各矿物的折射率虚部 k(λ)。"
            )
        )

        self.table = QTableWidget(len(endmembers), 4, self)
        self.table.setHorizontalHeaderLabels(["矿物端元", "密度 ρ", "折射率 n", "平均粒径 D (μm)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

        for i, em in enumerate(endmembers):
            name_item = QTableWidgetItem(em.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, name_item)

            for col, value, lo, hi, step, decimals in (
                (1, em.density, 0.1, 20.0, 0.1, 3),
                (2, em.n, 1.01, 5.0, 0.01, 3),
                (3, em.grain_size_um, 0.1, 5000.0, 1.0, 2),
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
        self.lab_inc.setValue(30.0)
        self.lab_emi = QDoubleSpinBox()
        self.lab_emi.setRange(0.0, 89.0)
        self.lab_emi.setDecimals(2)
        self.lab_emi.setValue(0.0)
        form.addRow("端元光谱测量入射角 i (°)", self.lab_inc)
        form.addRow("端元光谱测量发射角 e (°)", self.lab_emi)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        for i in range(self.table.rowCount()):
            dens = self.table.cellWidget(i, 1).value()
            n = self.table.cellWidget(i, 2).value()
            d = self.table.cellWidget(i, 3).value()
            if dens <= 0 or n <= 1.0 or d <= 0:
                QMessageBox.warning(self, "参数错误", f"第 {i+1} 行参数无效。")
                return
        self.accept()

    def result_endmembers(self) -> List[HapkeEndmember]:
        out: List[HapkeEndmember] = []
        for i, em in enumerate(self._endmembers):
            out.append(
                HapkeEndmember(
                    name=em.name,
                    wavelengths=em.wavelengths,
                    reflectance=em.reflectance,
                    density=float(self.table.cellWidget(i, 1).value()),
                    n=float(self.table.cellWidget(i, 2).value()),
                    grain_size_um=float(self.table.cellWidget(i, 3).value()),
                    k=None,
                    ssa=None,
                    source=em.source,
                )
            )
        return out

    def lab_geometry(self):
        return float(self.lab_inc.value()), float(self.lab_emi.value())
