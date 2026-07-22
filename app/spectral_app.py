import sys
import os
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import spectral.io.envi as envi

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QMenuBar, QFileDialog, QMessageBox,
                               QLineEdit, QPushButton, QLabel, QSplitter)
from PySide6.QtCore import Qt

# ================= 全局字体配置 =================
try:
    fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'SimSun',           # Windows 宋体
    'Microsoft YaHei',  # Windows 微软雅黑
    'SimHei',           # Windows 黑体
    'Songti SC',        # Mac 宋体
    'STSong',           # Mac 华文宋体
    'Arial'             # 英文备用
]
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

class SpectralApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("火星高光谱图像分析系统")
        self.resize(1200, 800)

        # 数据存储占位
        self.current_data = None
        self.rgb_image = None
        self.wavelengths = None
        self.ratio_mode = None
        self.click_coords = []      # 记录点击光谱用于手动比值
        self.click_positions = []   # 记录手动比值点击位置 (row, col)
        self.rgb_cbar = None              # 【需求1】RGB图的隐藏Colorbar
        self.manual_ratio_first_pos = None # 【需求4】手动比值记录第一个点的(row, col)
        self.manual_col_lines = []         # 【需求4】手动比值时的黄色辅助线对象列表

        self.selected_pos = None    # 记录当前选中点 (row, col)

        # 当前选中的原始光谱数据（用于右侧图表交互吸附）
        self.current_raw_spectrum = None
        # 当前比值光谱数据（用于比值图十字读数）
        self.current_ratio_spectrum = None
        # 已叠加的 RELAB 参考谱（波长μm, 反射率）；生成新比值光谱时一并清除
        self.relab_overlay = None

        # 当前显示的参数图像及标题（用于手动拉伸刷新）
        self.current_param_img = None
        self.current_param_title = None

        # 图像标记引用
        self.marker_rgb = None
        self.marker_result = None

        # 原始光谱图十字线与文本框引用
        self.raw_crosshair_vline = None
        self.raw_crosshair_hline = None
        self.raw_crosshair_text = None

        # 比值光谱图十字线与文本框引用
        self.ratio_crosshair_vline = None
        self.ratio_crosshair_hline = None
        self.ratio_crosshair_text = None

        self.init_ui()
        self.init_menu()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ================= 左侧：图像及结果显示区 =================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 1. 显示高光谱 RGB 图像
        self.fig_rgb = Figure()
        self.canvas_rgb = FigureCanvas(self.fig_rgb)
        self.ax_rgb = self.fig_rgb.add_subplot(111)
        self.ax_rgb.set_title("高光谱RGB图像")
        self._apply_image_layout(self.fig_rgb, self.ax_rgb, hide_cbar=True)
        self.canvas_rgb.setCursor(Qt.CrossCursor)

        # 2. 显示参数结果图 & 图例
        self.fig_result = Figure()
        self.canvas_result = FigureCanvas(self.fig_result)
        self.ax_result = self.fig_result.add_subplot(111)
        self.ax_result.set_title("结果图")
        self._apply_image_layout(self.fig_result, self.ax_result, hide_cbar=True)

        # 统一绑定点击事件（RGB图与结果图）
        self.canvas_rgb.mpl_connect('button_press_event', self.on_image_clicked)
        self.canvas_result.mpl_connect('button_press_event', self.on_image_clicked)

        left_layout.addWidget(self.canvas_rgb)
        left_layout.addWidget(self.canvas_result)
        splitter.addWidget(left_widget)

        # ================= 右侧：光谱显示及处理区 =================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        pixel_window_layout = QHBoxLayout()
        pixel_window_layout.addWidget(QLabel("像元窗口(N×N):"))
        self.window_input = QLineEdit("1")
        self.window_input.setPlaceholderText("N")
        self.window_input.setFixedWidth(60)

        self.window_input.returnPressed.connect(self.on_window_input_enter)
        pixel_window_layout.addWidget(self.window_input)
        pixel_window_layout.addStretch()

        right_layout.addLayout(pixel_window_layout)
        # 1. 原始光谱显示（使用 $\mu$m 解决字符不显示问题）
        self.fig_raw_spec = Figure()
        self.canvas_raw_spec = FigureCanvas(self.fig_raw_spec)
        self.ax_raw_spec = self.fig_raw_spec.add_subplot(111)
        self.ax_raw_spec.set_title("原始光谱")
        self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_raw_spec.set_ylabel("Reflectance")
        self.canvas_raw_spec.setCursor(Qt.CrossCursor)
        self.canvas_raw_spec.mpl_connect('button_press_event', self.on_raw_spec_clicked)

        # 2. 比值光谱显示
        self.fig_ratio_spec = Figure()
        self.canvas_ratio_spec = FigureCanvas(self.fig_ratio_spec)
        self.ax_ratio_spec = self.fig_ratio_spec.add_subplot(111)
        self.ax_ratio_spec.set_title("比值光谱")
        self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_ratio_spec.set_ylabel("Scaled Reflectance")
        self.canvas_ratio_spec.setCursor(Qt.CrossCursor)
        self.canvas_ratio_spec.mpl_connect('button_press_event', self.on_ratio_spec_clicked)

        # 统一右侧两图边距，保证波长轴位置对齐
        self._sync_spectrum_axes()

        # 3. 底部操作区
        bottom_tools_layout = QVBoxLayout()

        # 第一行：波长指示线 + RELAB
        row1_layout = QHBoxLayout()
        self.wavelength_input = QLineEdit()
        self.wavelength_input.setPlaceholderText("波长(μm)，逗号分隔，按回车画线")
        self.wavelength_input.returnPressed.connect(self.draw_wavelength_lines)

        self.btn_open_relab = QPushButton("Open RELAB文件")
        self.btn_open_relab.clicked.connect(self.open_relab_file)

        row1_layout.addWidget(QLabel("辅助波长:"))
        row1_layout.addWidget(self.wavelength_input)
        row1_layout.addWidget(self.btn_open_relab)
        row1_layout.addStretch()

        # 第二行：参数手动拉伸控制
        row2_layout = QHBoxLayout()
        self.vmin_input = QLineEdit()
        self.vmin_input.setPlaceholderText("Min")
        self.vmin_input.setFixedWidth(80)
        self.vmax_input = QLineEdit()
        self.vmax_input.setPlaceholderText("Max")
        self.vmax_input.setFixedWidth(80)

        self.btn_apply_stretch = QPushButton("拉伸显示")
        self.btn_apply_stretch.clicked.connect(self.apply_custom_stretch)
        self.vmin_input.returnPressed.connect(self.apply_custom_stretch)
        self.vmax_input.returnPressed.connect(self.apply_custom_stretch)

        row2_layout.addWidget(QLabel("参数拉伸:"))
        row2_layout.addWidget(QLabel("Min:"))
        row2_layout.addWidget(self.vmin_input)
        row2_layout.addWidget(QLabel("Max:"))
        row2_layout.addWidget(self.vmax_input)
        row2_layout.addWidget(self.btn_apply_stretch)
        row2_layout.addStretch()

        bottom_tools_layout.addLayout(row1_layout)
        bottom_tools_layout.addLayout(row2_layout)

        right_layout.addWidget(self.canvas_raw_spec)
        right_layout.addWidget(self.canvas_ratio_spec)
        right_layout.addLayout(bottom_tools_layout)

        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])

    def init_menu(self):
        menubar = self.menuBar()

        # 1. File
        file_menu = menubar.addMenu('File')
        file_menu.addAction('Open', self.open_file)
        file_menu.addAction('Save as', self.save_as_envi)
        file_menu.addAction('Close file', self.close_file)
        file_menu.addAction('Exit', self.close)

        # 2. Spectral parameter
        param_menu = menubar.addMenu('Spectral parameter')
        params = ['BD1400', 'BD1900', 'BD2100_2', 'BD2210_2',
                  'BD2250', 'BD2265', 'D2300', 'BD2500_2', 'SINDEX2']
        for p in params:
            param_menu.addAction(p, lambda name=p: self.calc_spectral_parameter(name))

        # 3. Identification
        id_menu = menubar.addMenu('Identification')
        for i in range(1, 4):
            id_menu.addAction(f'Model {i}', lambda m=i: self.run_identification_model(m))

        # 4. Unmixing
        unmix_menu = menubar.addMenu('Unmixing')
        unmix_menu.addAction('Hapke model', self.run_hapke_unmixing)
        unmix_menu.addAction('Sparse unmixing', self.run_sparse_unmixing)

        # 5. Tools
        tools_menu = menubar.addMenu('Tools')
        tools_menu.addAction('DISORT correction', self.run_disort)

        ratio_menu = tools_menu.addMenu('Ratio spectra')
        ratio_menu.addAction('自动提取', lambda: self.set_ratio_mode('auto'))
        ratio_menu.addAction('手动提取', lambda: self.set_ratio_mode('manual'))

    # ================= 文件加载与数据处理 =================

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "打开高光谱数据", "", "ENVI Header (*.hdr);;All Files (*)")
        if not filename:
            return

        try:
            self.marker_rgb = None
            self.marker_result = None
            self.current_param_img = None
            self.current_param_title = None

            img = envi.open(filename)
            self.current_data = np.array(img.load(), dtype=np.float32)

            # 根据文件名自动切割边缘 No Data 区域
            base_name = os.path.basename(filename).lower()
            if 'fr' in base_name:
                self.current_data[0, :, :] = np.nan
                self.current_data[-1, :, :] = np.nan
                self.current_data[:, 0:31, :] = np.nan
                self.current_data[:, -9:, :] = np.nan
                print(f"[{base_name}] 匹配为 FR 模式：切除上下1行、左31列、右9列")
            elif 'hr' in base_name:
                self.current_data[0, :, :] = np.nan
                self.current_data[-1, :, :] = np.nan
                self.current_data[:, 0:17, :] = np.nan
                self.current_data[:, -6:, :] = np.nan
                print(f"[{base_name}] 匹配为 HR 模式：切除上下1行、左17列、右6列")

            # 波长提取与归一化为微米 (μm)
            if img.bands.centers:
                self.wavelengths = np.array(img.bands.centers)
            elif 'wavelength' in img.metadata:
                self.wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
            else:
                bands_count = self.current_data.shape[2]
                self.wavelengths = np.arange(1, bands_count + 1)
                QMessageBox.warning(self, "波长缺失", "ENVI头文件中未找到波长信息，将使用波段索引代替。")

            if np.any(self.wavelengths > 100):
                self.wavelengths = self.wavelengths / 1000.0

            # 合成假彩色 RGB 图像 (按 2.53μm, 1.51μm, 1.08μm 真实波长)
            r_band = np.argmin(np.abs(self.wavelengths - 2.53))
            g_band = np.argmin(np.abs(self.wavelengths - 1.51))
            b_band = np.argmin(np.abs(self.wavelengths - 1.08))

            rgb = self.current_data[:, :, [r_band, g_band, b_band]]
            rgb_min = np.nanpercentile(rgb, 2)
            rgb_max = np.nanpercentile(rgb, 98)

            self.rgb_image = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-8), 0, 1)
            self.rgb_image[np.isnan(self.rgb_image)] = 0.0

            # 刷新左侧 RGB 画布（预留与结果图一致的 colorbar 占位，保证左对齐）
            self.fig_rgb.clf()
            self.ax_rgb = self.fig_rgb.add_subplot(111)
            self.ax_rgb.imshow(self.rgb_image)
            title_str = (
                f"假彩色图 (R {self.wavelengths[r_band]:.2f} $\mu$m, "
                f"G {self.wavelengths[g_band]:.2f} $\mu$m, "
                f"B {self.wavelengths[b_band]:.2f} $\mu$m)"
            )
            self.ax_rgb.set_title(title_str)
            self.ax_rgb.axis('off')
            self.marker_rgb = None
            self._apply_image_layout(self.fig_rgb, self.ax_rgb, hide_cbar=True)
            self.canvas_rgb.draw()

            # 打开数据后同步右侧光谱波长轴
            self._sync_spectrum_axes()
            self.canvas_raw_spec.draw()
            self.canvas_ratio_spec.draw()

            print(f"已加载文件: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"无法读取该 ENVI 文件:\n{str(e)}")

    def save_as_envi(self):
        QMessageBox.information(self, "保存", "功能：将右侧结果矩阵保存为 ENVI 格式 (待实现)")

    def close_file(self):
        self.current_data = None
        self.current_param_img = None
        self.current_param_title = None
        self.marker_rgb = None
        self.marker_result = None
        self.current_raw_spectrum = None
        self.selected_pos = None
        self.raw_crosshair_vline = None
        self.raw_crosshair_hline = None
        self.raw_crosshair_text = None
        self.rgb_cbar = None

        self.fig_rgb.clf()
        self.ax_rgb = self.fig_rgb.add_subplot(111)
        self.ax_rgb.set_title("高光谱RGB图像")
        self._apply_image_layout(self.fig_rgb, self.ax_rgb, hide_cbar=True)
        self.canvas_rgb.draw()

        self.fig_result.clf()
        self.ax_result = self.fig_result.add_subplot(111)
        self.ax_result.set_title("结果图")
        self._apply_image_layout(self.fig_result, self.ax_result, hide_cbar=True)
        self.canvas_result.draw()

        self.ax_raw_spec.clear()
        self.ax_raw_spec.set_title("原始光谱")
        self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_raw_spec.set_ylabel("Reflectance")
        self.canvas_raw_spec.draw()

        self.current_ratio_spectrum = None
        self.relab_overlay = None
        self.ratio_crosshair_vline = None
        self.ratio_crosshair_hline = None
        self.ratio_crosshair_text = None
        self.ax_ratio_spec.clear()
        self.ax_ratio_spec.set_title("比值光谱")
        self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_ratio_spec.set_ylabel("Scaled Reflectance")
        self._sync_spectrum_axes()
        self.canvas_ratio_spec.draw()

    # ================= 图像交互与光谱绘制 =================
    def _sync_spectrum_axes(self):
        """统一原始光谱与比值光谱的边距和 X 轴范围，使波长位置对齐。"""
        for fig in (self.fig_raw_spec, self.fig_ratio_spec):
            try:
                fig.set_layout_engine(None)
            except Exception:
                pass
            fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.16)

        if self.wavelengths is not None and len(self.wavelengths) > 0:
            xmin = float(np.nanmin(self.wavelengths))
            xmax = float(np.nanmax(self.wavelengths))
            if np.isfinite(xmin) and np.isfinite(xmax) and xmin < xmax:
                self.ax_raw_spec.set_xlim(xmin, xmax)
                self.ax_ratio_spec.set_xlim(xmin, xmax)

    def _reset_ratio_crosshair(self):
        self.ratio_crosshair_vline = None
        self.ratio_crosshair_hline = None
        self.ratio_crosshair_text = None

    def _clear_ratio_plot(self, title="比值光谱"):
        """生成新比值光谱前清空图中全部曲线（含 RELAB 叠加）。"""
        self.relab_overlay = None
        self._reset_ratio_crosshair()
        self.ax_ratio_spec.clear()
        self.ax_ratio_spec.set_title(title)
        self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_ratio_spec.set_ylabel("Scaled Reflectance")

    def _load_relab_txt(self, filename):
        """
        读取 RELAB txt：第1列波长，第2列反射率。
        自动跳过表头等非数字行；判断 nm/μm，统一换算为 μm。
        """
        rows = []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('#') or line.startswith(';') or line.startswith('%'):
                    continue

                # 兼容逗号/制表符/空格分隔
                parts = line.replace(',', ' ').replace('\t', ' ').split()
                if len(parts) < 2:
                    continue

                # 表头或含字母的行：无法转 float 则跳过
                try:
                    w = float(parts[0])
                    r = float(parts[1])
                except ValueError:
                    continue

                if np.isfinite(w) and np.isfinite(r):
                    rows.append((w, r))

        if not rows:
            raise ValueError("未读到有效的波长/反射率数据（请检查是否只有表头或格式不对）")

        arr = np.asarray(rows, dtype=float)
        wave = arr[:, 0]
        refl = arr[:, 1]

        # 多数点 >100 或中位数很大 → 按纳米处理
        if np.nanmax(wave) > 100.0 or np.nanmedian(wave) > 50.0:
            wave = wave / 1000.0

        order = np.argsort(wave)
        return wave[order], refl[order]

    def _continuum_remove(self, wave, refl):
        """
        上包络连续统去除：突出吸收等谱形特征，不做按 Y 轴幅值的比例缩放。
        """
        wave = np.asarray(wave, dtype=float)
        refl = np.asarray(refl, dtype=float)
        n = len(wave)
        if n < 2:
            return refl.copy()

        valid = np.isfinite(wave) & np.isfinite(refl)
        if np.count_nonzero(valid) < 2:
            return refl.copy()

        w = wave[valid]
        r = refl[valid]

        # 上凸包（连续统）：从左到右保留使反射率位于上方的点
        hull = [0]
        for i in range(1, len(w)):
            while len(hull) >= 2:
                i0, i1 = hull[-2], hull[-1]
                cross = (w[i1] - w[i0]) * (r[i] - r[i0]) - (r[i1] - r[i0]) * (w[i] - w[i0])
                # cross >= 0 表示 i1 不在上侧，弹出
                if cross >= 0:
                    hull.pop()
                else:
                    break
            hull.append(i)

        cont_valid = np.interp(w, w[hull], r[hull])
        with np.errstate(divide='ignore', invalid='ignore'):
            cr_valid = r / (cont_valid + 1e-12)
        cr_valid[~np.isfinite(cr_valid)] = 1.0

        # 填回原长度
        cr = np.ones_like(refl, dtype=float)
        cr[valid] = cr_valid
        return cr

    def _prepare_relab_for_compare(self, relab_wave, relab_refl):
        """
        若比值图已有光谱：对 RELAB 做连续统去除，只做谱形对比、突出特征；
        若无现有光谱：原样返回。
        不再按 Y 轴均值做比例 scale。
        """
        if self.current_ratio_spectrum is None:
            return relab_refl, False

        # 优先在与当前比值光谱重叠的波段上做连续统去除，特征更清晰
        if self.wavelengths is not None and len(self.wavelengths) > 0:
            w_min = max(float(np.nanmin(relab_wave)), float(np.nanmin(self.wavelengths)))
            w_max = min(float(np.nanmax(relab_wave)), float(np.nanmax(self.wavelengths)))
            if np.isfinite(w_min) and np.isfinite(w_max) and w_max > w_min:
                mask = (relab_wave >= w_min) & (relab_wave <= w_max)
                if np.count_nonzero(mask) >= 2:
                    cr = relab_refl.copy()
                    cr[mask] = self._continuum_remove(relab_wave[mask], relab_refl[mask])
                    # 重叠区外保持为 NaN，避免非对比区干扰观感
                    cr[~mask] = np.nan
                    return cr, True

        return self._continuum_remove(relab_wave, relab_refl), True

    def _get_window_size(self):
        """读取上方像元窗口 N，非法输入时回退为 1。"""
        try:
            w_size = int(self.window_input.text().strip())
        except ValueError:
            w_size = 1
        return max(1, w_size)

    def _extract_window_spectrum(self, row, col):
        """以 (row, col) 为中心，提取 N×N 像元平均光谱。"""
        rows, cols, _ = self.current_data.shape
        w_size = self._get_window_size()
        half = w_size // 2
        r_start = max(0, row - half)
        r_end = min(rows, row + half + 1)
        c_start = max(0, col - half)
        c_end = min(cols, col + half + 1)

        region = self.current_data[r_start:r_end, c_start:c_end, :]
        with np.errstate(all='ignore'):
            spectrum = np.nanmean(region, axis=(0, 1))
        return spectrum, w_size

    def _apply_image_layout(self, fig, ax, colorbar_mappable=None, hide_cbar=False):
        """
        统一假彩色图与结果图边距：右侧预留相同 colorbar 空间，保证左对齐；
        图像本身不显示坐标轴。
        """
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # 关闭 constrained，避免与固定边距冲突
        try:
            fig.set_layout_engine(None)
        except Exception:
            pass

        fig.subplots_adjust(left=0.02, right=0.86, top=0.90, bottom=0.05)
        cax = fig.add_axes([0.88, 0.05, 0.03, 0.80])

        if hide_cbar or colorbar_mappable is None:
            cax.set_visible(False)
            if fig is self.fig_rgb:
                self.rgb_cbar = None
        else:
            cbar = fig.colorbar(colorbar_mappable, cax=cax)
            if fig is self.fig_rgb:
                self.rgb_cbar = cbar

    def on_image_clicked(self, event):
        """统一处理 RGB 图和结果图上的鼠标点击"""
        if self.current_data is None:
            return
        if event.inaxes not in [self.ax_rgb, self.ax_result]:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        # 手动提取模式下，选第二个点时强制约束为第一点的同一列
        if self.ratio_mode == 'manual' and len(self.click_coords) == 1:
            if self.manual_ratio_first_pos is not None:
                col = self.manual_ratio_first_pos[1]

        rows, cols, _ = self.current_data.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return

        self.process_click_logic(row, col)

    def process_click_logic(self, row, col):
        """根据当前像元窗口 N，在原始光谱区显示以该点为中心的 N×N 平均光谱。"""
        if self.current_data is None or self.wavelengths is None:
            return

        rows, cols, _ = self.current_data.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return

        self.selected_pos = (row, col)
        spectrum, w_size = self._extract_window_spectrum(row, col)
        self.current_raw_spectrum = spectrum
        wave = self.wavelengths

        # 双向同步更新标记红十字
        self.update_image_markers(row, col)

        if self.ratio_mode != 'manual':
            self._clear_manual_lines()

            self.ax_raw_spec.clear()
            self.ax_raw_spec.plot(wave, spectrum, color='navy', linewidth=1.2)
            self.ax_raw_spec.set_title(
                f"原始光谱显示 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})"
            )
            self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
            self.ax_raw_spec.set_ylabel("Reflectance")
            self.ax_raw_spec.grid(True, linestyle='--', alpha=0.5)

            self.raw_crosshair_vline = None
            self.raw_crosshair_hline = None
            self.raw_crosshair_text = None

            if self.ratio_mode in ['auto', 'disort']:
                mean_val = np.nanmean(spectrum) + 1e-8
                ratio_spec = spectrum / mean_val
                self.current_ratio_spectrum = ratio_spec
                # 生成新比值光谱前先清空（含旧 RELAB），标题显示点击位置
                self._clear_ratio_plot(f"比值光谱 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})")
                self.ax_ratio_spec.plot(
                    wave, ratio_spec, color='crimson',
                    label=f'Ratio (X:{col}, Y:{row})'
                )
                self.ax_ratio_spec.legend(fontsize=8)
                self.ax_ratio_spec.grid(True, linestyle='--', alpha=0.5)

            self._sync_spectrum_axes()
            self.canvas_raw_spec.draw()
            self.canvas_ratio_spec.draw()

        elif self.ratio_mode == 'manual':
            if len(self.click_coords) == 0:
                self.ax_raw_spec.clear()
                self.click_positions = []
                self._clear_ratio_plot("等待选择分母...")
                self.current_ratio_spectrum = None

                self.ax_raw_spec.set_title(
                    f"手动比值: 分子 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})"
                )
                self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
                self.ax_raw_spec.set_ylabel("Reflectance")

                self.manual_ratio_first_pos = (row, col)

                self._clear_manual_lines()
                line_rgb = self.ax_rgb.axvline(x=col, color='yellow', linestyle='--', alpha=0.8)
                line_res = self.ax_result.axvline(x=col, color='yellow', linestyle='--', alpha=0.8)
                self.manual_col_lines.extend([line_rgb, line_res])
                self.canvas_rgb.draw()
                self.canvas_result.draw()
            else:
                self.ax_raw_spec.set_title(
                    f"手动比值: 分母完成 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})"
                )

            self.click_coords.append(spectrum)
            self.click_positions.append((row, col))
            self.ax_raw_spec.plot(
                wave, spectrum,
                label=f'Point {len(self.click_coords)} (X:{col}, Y:{row}, {w_size}x{w_size})'
            )
            self.ax_raw_spec.legend(fontsize=8)

            if len(self.click_coords) == 2:
                ratio = self.click_coords[0] / (self.click_coords[1] + 1e-8)
                self.current_ratio_spectrum = ratio
                r1, c1 = self.click_positions[0]
                r2, c2 = self.click_positions[1]
                # 生成新比值光谱前先清空（含旧 RELAB），标题显示分子/分母位置
                self._clear_ratio_plot(
                    f"比值光谱 (分子 X:{c1},Y:{r1} / 分母 X:{c2},Y:{r2})"
                )
                self.ax_ratio_spec.plot(
                    wave, ratio, color='crimson',
                    label=f'Ratio ({c1},{r1})/({c2},{r2})'
                )
                self.ax_ratio_spec.legend(fontsize=8)
                self.ax_ratio_spec.grid(True, linestyle='--', alpha=0.5)

                self.click_coords = []
                self.click_positions = []
                self.manual_ratio_first_pos = None
                self._clear_manual_lines()

            self._sync_spectrum_axes()
            self.canvas_raw_spec.draw()
            self.canvas_ratio_spec.draw()

    def on_raw_spec_clicked(self, event):
        """点击原始光谱图：显示吸附十字线及单行无背景框精确读数"""
        if event.inaxes != self.ax_raw_spec:
            return
        if event.xdata is None or self.wavelengths is None or self.current_raw_spectrum is None:
            return

        click_x = event.xdata

        # 1. 精确锁定最邻近的波长点
        idx = int(np.argmin(np.abs(self.wavelengths - click_x)))
        target_wave = self.wavelengths[idx]
        target_val = self.current_raw_spectrum[idx]

        # 2. 移除旧的十字线和数值文本
        if self.raw_crosshair_vline in self.ax_raw_spec.lines:
            self.raw_crosshair_vline.remove()
        if self.raw_crosshair_hline in self.ax_raw_spec.lines:
            self.raw_crosshair_hline.remove()
        if self.raw_crosshair_text in self.ax_raw_spec.texts:
            self.raw_crosshair_text.remove()

        # 3. 绘制贯穿坐标轴的红虚线十字准星
        self.raw_crosshair_vline = self.ax_raw_spec.axvline(
            x=target_wave, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8
        )
        self.raw_crosshair_hline = self.ax_raw_spec.axhline(
            y=target_val, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8
        )

        # 4. 在图表左上角渲染单行无背景框文本
        um_val = target_wave if target_wave < 100 else target_wave / 1000.0
        text_str = f" {um_val:.3f} $\mu$m {target_val:.4f}"
        self.raw_crosshair_text = self.ax_raw_spec.text(
            0.03, 0.95, text_str,
            transform=self.ax_raw_spec.transAxes,
            verticalalignment='top',
            fontsize=10,
            color='darkred'
        )

        self.canvas_raw_spec.draw()

    def on_ratio_spec_clicked(self, event):
        """点击比值光谱图：显示吸附十字线及波长/数值读数"""
        if event.inaxes != self.ax_ratio_spec:
            return
        if event.xdata is None or self.wavelengths is None or self.current_ratio_spectrum is None:
            return

        click_x = event.xdata
        idx = int(np.argmin(np.abs(self.wavelengths - click_x)))
        target_wave = self.wavelengths[idx]
        target_val = self.current_ratio_spectrum[idx]

        if self.ratio_crosshair_vline in self.ax_ratio_spec.lines:
            self.ratio_crosshair_vline.remove()
        if self.ratio_crosshair_hline in self.ax_ratio_spec.lines:
            self.ratio_crosshair_hline.remove()
        if self.ratio_crosshair_text in self.ax_ratio_spec.texts:
            self.ratio_crosshair_text.remove()

        self.ratio_crosshair_vline = self.ax_ratio_spec.axvline(
            x=target_wave, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8
        )
        self.ratio_crosshair_hline = self.ax_ratio_spec.axhline(
            y=target_val, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8
        )

        um_val = target_wave if target_wave < 100 else target_wave / 1000.0
        text_str = f" {um_val:.3f} $\mu$m {target_val:.4f}"
        self.ratio_crosshair_text = self.ax_ratio_spec.text(
            0.03, 0.95, text_str,
            transform=self.ax_ratio_spec.transAxes,
            verticalalignment='top',
            fontsize=10,
            color='darkred'
        )

        self.canvas_ratio_spec.draw()

    def update_image_markers(self, row, col):
        """同步更新左侧 RGB 图与参数结果图上的标记"""
        if hasattr(self, 'ax_rgb') and self.ax_rgb is not None:
            if self.marker_rgb in self.ax_rgb.lines:
                self.marker_rgb.remove()
            self.marker_rgb = self.ax_rgb.plot(col, row, 'r+', markersize=12, markeredgewidth=2)[0]
            self.canvas_rgb.draw()

        if hasattr(self, 'ax_result') and self.ax_result is not None:
            if self.marker_result in self.ax_result.lines:
                self.marker_result.remove()
            self.marker_result = self.ax_result.plot(col, row, 'r+', markersize=12, markeredgewidth=2)[0]
            self.canvas_result.draw()

    # ================= 科学计算模块 (基于物理波长定位) =================

    def get_band_mean_by_wave(self, target_wave_nm, num_bands=5):
        """
        根据目标物理波长 (nm 或 μm) 动态查找最接近的波段索引，
        并计算周围 num_bands 个波段的均值矩阵。
        """
        if self.current_data is None or self.wavelengths is None:
            return None

        target_um = target_wave_nm / 1000.0 if target_wave_nm > 100 else target_wave_nm
        closest_idx = int(np.argmin(np.abs(self.wavelengths - target_um)))

        half = num_bands // 2
        start_idx = max(0, closest_idx - half)
        end_idx = min(self.current_data.shape[2], closest_idx + half + 1)

        with np.errstate(all='ignore'):
            return np.nanmean(self.current_data[:, :, start_idx:end_idx], axis=2)

    def show_parameter_result(self, param_img, title_str, vmin=None, vmax=None):
        """
        渲染光谱参数结果图（叠加在 1.08μm 灰度底图上，50% 透明度）
        """
        self.current_param_img = param_img
        self.current_param_title = title_str

        self.fig_result.clf()
        self.ax_result = self.fig_result.add_subplot(111)
        self.marker_result = None

        # 1. 绘制底层 1.08 μm 灰度底图
        base_108 = self.get_band_mean_by_wave(1080, num_bands=5)
        if base_108 is not None:
            b_min, b_max = np.nanpercentile(base_108, [2, 98])
            base_norm = np.clip((base_108 - b_min) / (b_max - b_min + 1e-8), 0, 1)
            base_norm[np.isnan(base_norm)] = 0.0
            self.ax_result.imshow(base_norm, cmap='gray')

        # 2. 决定拉伸上下限 (vmin, vmax)
        valid_pixels = param_img[(param_img != 0) & np.isfinite(param_img)]

        if vmin is None or vmax is None:
            if len(valid_pixels) > 0:
                auto_vmin, auto_vmax = np.percentile(valid_pixels, [2, 98])
            else:
                auto_vmin, auto_vmax = 0.0, 1.0

            if vmin is None:
                vmin = auto_vmin
            if vmax is None:
                vmax = auto_vmax

            # 自动回显到 UI 输入框中
            self.vmin_input.setText(f"{vmin:.4f}")
            self.vmax_input.setText(f"{vmax:.4f}")

        # 3. 将 0 和 NaN/Inf 设置为透明掩膜
        param_masked = np.ma.masked_where((param_img == 0) | (~np.isfinite(param_img)), param_img)

        # 4. 叠加半透明参数图
        im = self.ax_result.imshow(param_masked, cmap='jet', vmin=vmin, vmax=vmax, alpha=0.5)

        self.ax_result.set_title(title_str)
        self._apply_image_layout(self.fig_result, self.ax_result, colorbar_mappable=im)

        # 5. 恢复选点标记
        if self.selected_pos is not None:
            r, c = self.selected_pos
            self.marker_result = self.ax_result.plot(c, r, 'r+', markersize=12, markeredgewidth=2)[0]

        self.canvas_result.draw()

    def apply_custom_stretch(self):
        """用户点击【拉伸显示】或按回车时执行"""
        if self.current_param_img is None:
            QMessageBox.warning(self, "提示", "当前没有显示的参数结果图！")
            return

        try:
            min_str = self.vmin_input.text().strip()
            max_str = self.vmax_input.text().strip()

            if not min_str or not max_str:
                QMessageBox.warning(self, "警告", "请确保已填写 Min 和 Max 数值！")
                return

            vmin = float(min_str)
            vmax = float(max_str)

            if vmin >= vmax:
                QMessageBox.warning(self, "警告", "最小值 (Min) 必须小于最大值 (Max)！")
                return

            self.show_parameter_result(self.current_param_img, self.current_param_title, vmin=vmin, vmax=vmax)

        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字！")

    def calc_spectral_parameter(self, param_name):
        """计算 CRISM 光谱参数 (全动态波长检索)"""
        if self.current_data is None:
            QMessageBox.warning(self, "警告", "请先打开高光谱图像数据！")
            return

        try:
            # 1. BD1400
            if param_name == 'BD1400':
                b1467 = self.get_band_mean_by_wave(1467, num_bands=5)
                b1330 = self.get_band_mean_by_wave(1330, num_bands=5)
                b1395 = self.get_band_mean_by_wave(1395, num_bands=5)

                r8 = ((1395.0 - 1330.0) / (1467.0 - 1330.0)) * b1467 + \
                     ((1467.0 - 1395.0) / (1467.0 - 1330.0)) * b1330

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (b1395 / (r8 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "BD1400")

            # 2. BD1900
            elif param_name == 'BD1900':
                b2067 = self.get_band_mean_by_wave(2067, num_bands=5)
                b1850 = self.get_band_mean_by_wave(1850, num_bands=5)
                b1930 = self.get_band_mean_by_wave(1930, num_bands=5)
                b1985 = self.get_band_mean_by_wave(1985, num_bands=5)

                valid_mask = ~(np.isnan(b2067) | np.isnan(b1850) | np.isnan(b1930) | np.isnan(b1985) |
                              (b2067 == 0) | (b1850 == 0))

                bd1900 = np.zeros_like(b1930, dtype=np.float32)

                if np.any(valid_mask):
                    r1 = ((1930.0 - 1850.0) / (2067.0 - 1850.0)) * b2067[valid_mask] + \
                         ((2067.0 - 1930.0) / (2067.0 - 1850.0)) * b1850[valid_mask]

                    r2 = ((1985.0 - 1850.0) / (2067.0 - 1850.0)) * b2067[valid_mask] + \
                         ((2067.0 - 1985.0) / (2067.0 - 1850.0)) * b1850[valid_mask]

                    valid_r = (r1 != 0) & (r2 != 0) & ~np.isnan(r1) & ~np.isnan(r2)

                    calc_mask = np.zeros_like(valid_mask, dtype=bool)
                    calc_mask[valid_mask] = valid_r

                    term1 = 1.0 - (b1930[calc_mask] / r1[valid_r])
                    term2 = 1.0 - (b1985[calc_mask] / r2[valid_r])

                    bd1900[calc_mask] = 0.5 * term1 + 0.5 * term2

                bd1900[~np.isfinite(bd1900)] = 0.0
                self.show_parameter_result(bd1900, "BD1900")

            # 3. BD2100_2
            elif param_name == 'BD2100_2':
                b2250 = self.get_band_mean_by_wave(2250, num_bands=3)
                b1930 = self.get_band_mean_by_wave(1930, num_bands=3)
                b2132 = self.get_band_mean_by_wave(2132, num_bands=5)

                r3 = ((2132.0 - 1930.0) / (2250.0 - 1930.0)) * b2250 + \
                     ((2250.0 - 2132.0) / (2250.0 - 1930.0)) * b1930

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (b2132 / (r3 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "BD2100_2")

            # 4. BD2210_2
            elif param_name == 'BD2210_2':
                b2250 = self.get_band_mean_by_wave(2250, num_bands=5)
                b2165 = self.get_band_mean_by_wave(2165, num_bands=5)
                b2210 = self.get_band_mean_by_wave(2210, num_bands=5)

                r4 = ((2210.0 - 2165.0) / (2250.0 - 2165.0)) * b2250 + \
                     ((2250.0 - 2210.0) / (2250.0 - 2165.0)) * b2165

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (b2210 / (r4 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "BD2210_2")

            # 5. BD2250
            elif param_name == 'BD2250':
                b2340 = self.get_band_mean_by_wave(2340, num_bands=3)
                b2120 = self.get_band_mean_by_wave(2120, num_bands=5)
                b2245 = self.get_band_mean_by_wave(2245, num_bands=7)

                r11 = ((2245.0 - 2120.0) / (2340.0 - 2120.0)) * b2340 + \
                      ((2340.0 - 2245.0) / (2340.0 - 2120.0)) * b2120

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (b2245 / (r11 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "BD2250")

            # 6. BD2265
            elif param_name == 'BD2265':
                b2340 = self.get_band_mean_by_wave(2340, num_bands=5)
                b2210 = self.get_band_mean_by_wave(2210, num_bands=5)
                b2265 = self.get_band_mean_by_wave(2265, num_bands=3)

                r9 = ((2265.0 - 2210.0) / (2340.0 - 2210.0)) * b2340 + \
                     ((2340.0 - 2265.0) / (2340.0 - 2210.0)) * b2210

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (b2265 / (r9 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "BD2265")

            # 7. D2300
            elif param_name == 'D2300':
                b2530 = self.get_band_mean_by_wave(2530, num_bands=5)
                b1815 = self.get_band_mean_by_wave(1815, num_bands=5)
                b2290_3 = self.get_band_mean_by_wave(2290, num_bands=3)
                b2320_3 = self.get_band_mean_by_wave(2320, num_bands=3)
                b2330_3 = self.get_band_mean_by_wave(2330, num_bands=3)
                b2120 = self.get_band_mean_by_wave(2120, num_bands=5)
                b2170 = self.get_band_mean_by_wave(2170, num_bands=5)
                b2210 = self.get_band_mean_by_wave(2210, num_bands=5)

                k_line = (b2530 - b1815) / (2530.0 - 1815.0)
                b_line = b2530 - k_line * 2530.0

                r5 = b2290_3 / (k_line * 2290.0 + b_line + 1e-8) + \
                     b2320_3 / (k_line * 2320.0 + b_line + 1e-8) + \
                     b2330_3 / (k_line * 2330.0 + b_line + 1e-8)

                r6 = b2120 / (k_line * 2120.0 + b_line + 1e-8) + \
                     b2170 / (k_line * 2170.0 + b_line + 1e-8) + \
                     b2210 / (k_line * 2210.0 + b_line + 1e-8)

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (r5 / (r6 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "D2300")

            # 8. BD2500_2
            elif param_name == 'BD2500_2':
                b2570 = self.get_band_mean_by_wave(2570, num_bands=5)
                b2364 = self.get_band_mean_by_wave(2364, num_bands=5)
                b2480 = self.get_band_mean_by_wave(2480, num_bands=5)

                r7 = ((2480.0 - 2364.0) / (2570.0 - 2364.0)) * b2570 + \
                     ((2570.0 - 2480.0) / (2570.0 - 2364.0)) * b2364

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (b2480 / (r7 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "BD2500_2")

            # 9. SINDEX2
            elif param_name == 'SINDEX2':
                b2400 = self.get_band_mean_by_wave(2400, num_bands=3)
                b2120 = self.get_band_mean_by_wave(2120, num_bands=5)
                b2290_7 = self.get_band_mean_by_wave(2290, num_bands=7)

                r10 = ((2290.0 - 2120.0) / (2400.0 - 2120.0)) * b2400 + \
                      ((2400.0 - 2290.0) / (2400.0 - 2120.0)) * b2120

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = 1.0 - (r10 / (b2290_7 + 1e-8))
                res[~np.isfinite(res)] = 0.0
                self.show_parameter_result(res, "SINDEX2")

            else:
                QMessageBox.information(self, "提示", f"参数【{param_name}】公式尚未配置。")

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算 {param_name} 时发生错误:\n{str(e)}")

    def run_identification_model(self, model_num):
        self.fig_result.clf()
        self.ax_result = self.fig_result.add_subplot(111)
        self.marker_result = None

        base_108 = self.get_band_mean_by_wave(1080, num_bands=5)
        if base_108 is not None:
            b_min, b_max = np.nanpercentile(base_108, [2, 98])
            base_norm = np.clip((base_108 - b_min) / (b_max - b_min + 1e-8), 0, 1)
            self.ax_result.imshow(base_norm, cmap='gray')

        dummy_model = np.random.randint(0, 4, (100, 100))
        im = self.ax_result.imshow(dummy_model, cmap='Set1', alpha=0.5)
        self.ax_result.set_title(f"Model {model_num} 矿物识别结果")
        self._apply_image_layout(self.fig_result, self.ax_result, colorbar_mappable=im)
        self.canvas_result.draw()

    def run_hapke_unmixing(self):
        print("运行 Hapke 模型解混...")

    def run_sparse_unmixing(self):
        print("运行稀疏解混模型...")

    def run_disort(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择DISORT输入数据")
        if filename:
            self.ratio_mode = 'disort'

    def _clear_manual_lines(self):
        """清除由于手动提取产生的辅助引导线"""
        if hasattr(self, 'manual_col_lines') and self.manual_col_lines:
            for line in self.manual_col_lines:
                try:
                    line.remove()
                except Exception:
                    pass
            self.manual_col_lines = []
            try:
                self.canvas_rgb.draw_idle()
                self.canvas_result.draw_idle()
            except Exception:
                pass

    def set_ratio_mode(self, mode):
        self.ratio_mode = mode
        self.click_coords = []
        self.click_positions = []
        self.manual_ratio_first_pos = None
        self._clear_manual_lines()
        QMessageBox.information(self, "模式切换", f"已切换为: {'自动提取' if mode == 'auto' else '手动提取'} 模式")

    def draw_wavelength_lines(self):
        text = self.wavelength_input.text()
        if not text:
            return
        try:
            waves = [float(w.strip()) for w in text.split(',') if w.strip()]

            # 清除旧灰色虚线
            for line in self.ax_ratio_spec.lines[:]:
                if line.get_linestyle() == '--' and line.get_color() == 'gray':
                    line.remove()

            for w in waves:
                if w > 100:  # 自动转为 μm
                    w = w / 1000.0
                self.ax_ratio_spec.axvline(x=w, color='gray', linestyle='--')

            self.canvas_ratio_spec.draw()

        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字，多个波长请用英文逗号分隔。")

    def open_relab_file(self):
        """
        打开 RELAB txt（第1列波长，第2列反射率）：
        - 自动判断 nm/μm，统一换算到 μm 后画到右下比值光谱图
        - 图中尚无光谱：直接绘制原始反射率
        - 图中已有比值光谱：对 txt 做连续统去除（谱形对比，突出特征），
          不再按 Y 轴数值做比例缩放
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开RELAB库文件", "", "Text Files (*.txt);;All Files (*)"
        )
        if not filename:
            return

        try:
            relab_wave, relab_refl = self._load_relab_txt(filename)
            plot_refl, did_shape = self._prepare_relab_for_compare(relab_wave, relab_refl)

            label = os.path.basename(filename)
            if did_shape:
                label = f"{label} (谱形)"

            self.relab_overlay = (relab_wave, plot_refl, label)
            self.ax_ratio_spec.plot(
                relab_wave,
                plot_refl,
                label=label,
                linestyle='-.',
                color='darkorange',
                linewidth=1.3,
            )
            self.ax_ratio_spec.set_ylabel("Scaled Reflectance")
            self.ax_ratio_spec.legend(fontsize=8)
            self.ax_ratio_spec.grid(True, linestyle='--', alpha=0.5)
            self._sync_spectrum_axes()
            self.canvas_ratio_spec.draw()

        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"无法读取 RELAB 文件:\n{str(e)}")

    def on_window_input_enter(self):
        """像元窗口输入回车：校验 N；若已有选点则立刻按新窗口刷新光谱。"""
        w_size = self._get_window_size()
        self.window_input.setText(str(w_size))

        # 回车后点击图像会用该 N 做平均；若已有选点则立即刷新
        if self.selected_pos is not None and self.current_data is not None:
            self.click_coords = []
            self.click_positions = []
            self.manual_ratio_first_pos = None
            self._clear_manual_lines()
            row, col = self.selected_pos
            self.process_click_logic(row, col)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpectralApp()
    window.show()
    sys.exit(app.exec())
