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
        self.click_coords = []      # 记录点击坐标/光谱用于手动比值
        self.rgb_cbar = None              # 【需求1】RGB图的隐藏Colorbar
        self.manual_ratio_first_pos = None # 【需求4】手动比值记录第一个点的(row, col)
        self.manual_col_lines = []         # 【需求4】手动比值时的黄色辅助线对象列表

        self.selected_pos = None    # 记录当前选中点 (row, col)

        # 当前选中的原始光谱数据（用于右侧图表交互吸附）
        self.current_raw_spectrum = None

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
        self.ax_rgb.axis('off')
        self.canvas_rgb.setCursor(Qt.CrossCursor)

        # 2. 显示参数结果图 & 图例
        self.fig_result = Figure()
        self.canvas_result = FigureCanvas(self.fig_result)
        self.ax_result = self.fig_result.add_subplot(111)
        self.ax_result.set_title("结果图")
        self.ax_result.axis('off')

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
        self.fig_raw_spec = Figure(layout='constrained')
        self.canvas_raw_spec = FigureCanvas(self.fig_raw_spec)
        self.ax_raw_spec = self.fig_raw_spec.add_subplot(111)
        self.ax_raw_spec.set_title("原始光谱")
        self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_raw_spec.set_ylabel("Reflectance")
        self.canvas_raw_spec.setCursor(Qt.CrossCursor)
        self.canvas_raw_spec.mpl_connect('button_press_event', self.on_raw_spec_clicked)

        # 2. 比值光谱显示
        self.fig_ratio_spec = Figure(layout='constrained')
        self.canvas_ratio_spec = FigureCanvas(self.fig_ratio_spec)
        self.ax_ratio_spec = self.fig_ratio_spec.add_subplot(111)
        self.ax_ratio_spec.set_title("比值光谱")
        self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")

        # 3. 底部操作区
        bottom_tools_layout = QVBoxLayout()

        # 第一行：波长指示线 + RELAB
        row1_layout = QHBoxLayout()
        self.wavelength_input = QLineEdit()
        self.wavelength_input.setPlaceholderText("波长(μm)，逗号分隔，按回车画线")
        self.wavelength_input.returnPressed.connect(self.draw_wavelength_lines)

        self.btn_open_relab = QPushButton("Open RELAB文件")
        self.btn_open_relab.clicked.connect(self.open_relab_file)

        self.window_input = QLineEdit("1")
        self.window_input.setPlaceholderText("N")
        self.window_input.setFixedWidth(40)

        row1_layout.addWidget(QLabel("辅助波长:"))
        row1_layout.addWidget(self.wavelength_input)
        row1_layout.addWidget(self.btn_open_relab)

        row1_layout.addWidget(QLabel(" 像元窗口(N×N):"))
        row1_layout.addWidget(self.window_input)

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

            # 刷新左侧 RGB 画布
            self.ax_rgb.clear()
            self.ax_rgb.imshow(self.rgb_image)
            title_str = f"假彩色图 (R {self.wavelengths[r_band]:.2f} $\mu$m, G {self.wavelengths[g_band]:.2f} $\mu$m, B {self.wavelengths[b_band]:.2f} $\mu$m)"
            self.ax_rgb.set_title(title_str)
            self.ax_rgb.axis('off')

            self.fig_rgb.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
            self.canvas_rgb.draw()

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
        self.raw_crosshair_vline = None
        self.raw_crosshair_hline = None
        self.raw_crosshair_text = None

        self.ax_rgb.clear()
        self.ax_rgb.set_title("高光谱RGB图像")
        self.ax_rgb.axis('off')
        self.canvas_rgb.draw()

        self.ax_raw_spec.clear()
        self.ax_raw_spec.set_title("原始光谱")
        self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_raw_spec.set_ylabel("Reflectance")
        self.canvas_raw_spec.draw()

    # ================= 图像交互与光谱绘制 =================
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

        # 👇 【需求4新增】：手动提取模式下，选第二个点时强制约束为第一点的同一列
        if self.ratio_mode == 'manual' and len(self.click_coords) == 1:
            if self.manual_ratio_first_pos is not None:
                col = self.manual_ratio_first_pos[1]  # 强行将列拉回第一点的列

        rows, cols, _ = self.current_data.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return

        self.selected_pos = (row, col)

        # 👇 【需求2新增】：解析输入框，实现 N乘N 的区域平均光谱
        try:
            w_size = int(self.window_input.text().strip())
        except ValueError:
            w_size = 1

        w_size = max(1, w_size)
        half = w_size // 2
        r_start = max(0, row - half)
        r_end = min(rows, row + half + 1)
        c_start = max(0, col - half)
        c_end = min(cols, col + half + 1)

        region = self.current_data[r_start:r_end, c_start:c_end, :]
        with np.errstate(all='ignore'):
            spectrum = np.nanmean(region, axis=(0, 1))

        self.current_raw_spectrum = spectrum
        wave = self.wavelengths

        # 双向同步更新标记红十字
        self.update_image_markers(row, col)

        if self.ratio_mode != 'manual':
            self._clear_manual_lines() # 自动模式下确保清理辅助线

            self.ax_raw_spec.clear()
            self.ax_raw_spec.plot(wave, spectrum, color='navy', linewidth=1.2)
            self.ax_raw_spec.set_title(f"1. 原始光谱显示 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})")
            self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
            self.ax_raw_spec.set_ylabel("Reflectance")
            self.ax_raw_spec.grid(True, linestyle='--', alpha=0.5)

            # 清理十字光标记录
            self.raw_crosshair_vline = None
            self.raw_crosshair_hline = None
            self.raw_crosshair_text = None
            self.canvas_raw_spec.draw()

            if self.ratio_mode in ['auto', 'disort']:
                self.ax_ratio_spec.clear()
                mean_val = np.nanmean(spectrum) + 1e-8
                self.ax_ratio_spec.plot(wave, spectrum / mean_val, color='crimson')
                self.ax_ratio_spec.set_title("比值光谱")
                self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")
                self.ax_ratio_spec.grid(True, linestyle='--', alpha=0.5)
                self.canvas_ratio_spec.draw()

        elif self.ratio_mode == 'manual':
            # 👇 【需求3新增】：手动选第一个点时，清空之前遗留的光谱画布
            if len(self.click_coords) == 0:
                self.ax_raw_spec.clear()
                self.ax_ratio_spec.clear()

                self.ax_raw_spec.set_title("手动比值: 选择分子 (目标位置)")
                self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
                self.ax_raw_spec.set_ylabel("Reflectance")
                self.ax_ratio_spec.set_title("等待选择分母...")
                self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")

                self.manual_ratio_first_pos = (row, col)

                # 👇 【需求4辅助新增】：在图像上画一条黄色虚线，提示用户当前被限制在这一列
                self._clear_manual_lines()
                line_rgb = self.ax_rgb.axvline(x=col, color='yellow', linestyle='--', alpha=0.8)
                line_res = self.ax_result.axvline(x=col, color='yellow', linestyle='--', alpha=0.8)
                self.manual_col_lines.extend([line_rgb, line_res])
                self.canvas_rgb.draw()
                self.canvas_result.draw()
            else:
                self.ax_raw_spec.set_title("手动比值: 选择分母完成！")

            self.click_coords.append(spectrum)
            self.ax_raw_spec.plot(wave, spectrum, label=f'Point {len(self.click_coords)} ({w_size}x{w_size})')
            self.ax_raw_spec.legend()
            self.canvas_raw_spec.draw()

            if len(self.click_coords) == 2:
                ratio = self.click_coords[0] / (self.click_coords[1] + 1e-8)
                self.ax_ratio_spec.clear()
                self.ax_ratio_spec.plot(wave, ratio, color='crimson')
                self.ax_ratio_spec.set_title("比值光谱")
                self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")
                self.ax_ratio_spec.grid(True, linestyle='--', alpha=0.5)
                self.canvas_ratio_spec.draw()

                # 重置，为下一对做准备，并清理黄色辅助线
                self.click_coords = []
                self.manual_ratio_first_pos = None
                self._clear_manual_lines()

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
        #text_str = f"波长 {um_val:.3f} $\mu$m 反射率 {target_val:.4f}"
        text_str = f" {um_val:.3f} $\mu$m {target_val:.4f}"
        self.raw_crosshair_text = self.ax_raw_spec.text(
            0.03, 0.95, text_str,
            transform=self.ax_raw_spec.transAxes,
            verticalalignment='top',
            fontsize=10,
            color='darkred'  # 设置深红色字体使其在坐标轴内醒目
        )

        self.canvas_raw_spec.draw()

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
        self.fig_result.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
        self.ax_result = self.fig_result.add_subplot(111)

        self.marker_result = None
        self.fig_result.set_layout_engine('constrained')
        self.ax_result = self.fig_result.add_subplot(111, sharex=self.ax_rgb, sharey=self.ax_rgb)
        self.marker_result = None
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
        self.ax_result.axis('off')

        self.fig_result.colorbar(im, ax=self.ax_result, fraction=0.046, pad=0.04)

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
        self.fig_result.set_layout_engine('constrained')
        self.ax_result = self.fig_result.add_subplot(111)

        base_108 = self.get_band_mean_by_wave(1080, num_bands=5)
        if base_108 is not None:
            b_min, b_max = np.nanpercentile(base_108, [2, 98])
            base_norm = np.clip((base_108 - b_min) / (b_max - b_min + 1e-8), 0, 1)
            self.ax_result.imshow(base_norm, cmap='gray')

        dummy_model = np.random.randint(0, 4, (100, 100))
        im = self.ax_result.imshow(dummy_model, cmap='Set1', alpha=0.5)
        self.ax_result.set_title(f"Model {model_num} 矿物识别结果")
        self.ax_result.axis('off')
        self.fig_result.colorbar(im, ax=self.ax_result, label='图例: 矿物类别')
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
        self.manual_ratio_first_pos = None
        self._clear_manual_lines()  # 👇 新增：切换模式时清理残余线
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
        filename, _ = QFileDialog.getOpenFileName(self, "打开RELAB库文件", "", "Text Files (*.txt)")
        if filename:
            wave = np.linspace(0.4, 2.5, 200)
            relab_ref = np.random.rand(200) * 0.5 + 0.5
            self.ax_ratio_spec.plot(wave, relab_ref, label='RELAB Ref', linestyle='-.')
            self.ax_ratio_spec.legend()
            self.canvas_ratio_spec.draw()

    def on_window_input_enter(self):
        """像元窗口输入回车后，立刻刷新当前选定点的光谱"""
        if self.selected_pos is not None:
            # 清空之前的比值状态，重新将其视为第一次点击（分子）
            self.click_coords = []
            self.manual_ratio_first_pos = None
            self._clear_manual_lines()

            row, col = self.selected_pos
            # 模拟鼠标点击该位置，触发光谱更新
            self.process_click_logic(row, col)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpectralApp()
    window.show()
    sys.exit(app.exec())
