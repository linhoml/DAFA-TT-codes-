import sys
import os
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import spectral.io.envi as envi

# 保证可导入同目录下的 disort 包（兼容 __file__ 无后缀 / 冻结路径等情况）
_APP_DIR = os.path.dirname(os.path.abspath(__file__ if '__file__' in globals() else sys.argv[0]))
for _p in (_APP_DIR, os.path.dirname(_APP_DIR)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


def _import_run_disort_correction():
    """Import DISORT runner; raise a clear error if the package folder is missing."""
    import importlib
    import types

    disort_dir = os.path.join(_APP_DIR, "disort")
    expected = os.path.join(disort_dir, "correction.py")

    if _APP_DIR not in sys.path:
        sys.path.insert(0, _APP_DIR)

    # 强制重新加载，避免本机残留旧版 disort.correction（无 observed_radiance 参数）
    for name in list(sys.modules):
        if name == "disort" or name.startswith("disort."):
            del sys.modules[name]

    try:
        from disort.correction import run_disort_correction
        return run_disort_correction
    except ImportError:
        pass

    # 按目录强制加载（解决工作目录 / 路径问题）
    if os.path.isdir(disort_dir) and os.path.isfile(expected):
        if "disort" not in sys.modules:
            pkg = types.ModuleType("disort")
            pkg.__path__ = [disort_dir]
            sys.modules["disort"] = pkg
        return importlib.import_module("disort.correction").run_disort_correction

    raise ModuleNotFoundError(
        "找不到 DISORT 模块 disort.correction。\n\n"
        "请把整个 disort 文件夹放到主程序同一目录下，例如：\n"
        f"  {_APP_DIR}\\spectral_app.py\n"
        f"  {_APP_DIR}\\disort\\__init__.py\n"
        f"  {_APP_DIR}\\disort\\correction.py\n"
        f"  {_APP_DIR}\\disort\\engine.py\n"
        f"  {_APP_DIR}\\disort\\io_input.py\n"
        f"  {_APP_DIR}\\disort\\optical_data.py\n"
        f"  {_APP_DIR}\\disort\\optical_properties.py\n"
        f"  {_APP_DIR}\\disort\\phase_function.py\n\n"
        f"当前检查：{expected}\n"
        f"是否存在：{os.path.isfile(expected)}"
    )


def _call_run_disort_correction(run_fn, **kwargs):
    """Call runner with keyword compatibility across old/new correction.py."""
    import inspect

    try:
        sig = inspect.signature(run_fn)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except Exception:
        params = {}
        accepts_var_kw = True

    call_kwargs = dict(kwargs)
    # 新版参数 → 旧版别名
    if (
        "observed_radiance" in call_kwargs
        and "observed_radiance" not in params
        and not accepts_var_kw
    ):
        if "observed_if" in params:
            call_kwargs["observed_if"] = call_kwargs.pop("observed_radiance")
        else:
            raise TypeError(
                "当前 disort/correction.py 过旧，不支持 observed_radiance。\n"
                "请用仓库最新的整个 app/disort/ 文件夹覆盖本机同名目录后重试。"
            )

    try:
        return run_fn(**call_kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "observed_radiance" in msg and "observed_if" in params:
            call_kwargs["observed_if"] = call_kwargs.pop("observed_radiance", None)
            return run_fn(**call_kwargs)
        raise TypeError(
            f"{msg}\n\n"
            "若提示 unexpected keyword argument 'observed_radiance'，\n"
            "说明本机 app/disort/correction.py 不是最新版。\n"
            "请覆盖更新整个 disort 文件夹（不要只更新 spectral_app.py）。"
        ) from exc

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QMenuBar, QFileDialog, QMessageBox,
                               QLineEdit, QPushButton, QLabel, QSplitter,
                               QProgressDialog, QInputDialog, QDialog)
from PySide6.QtCore import Qt, QObject, QThread, Signal


def _dialog_accepted(result) -> bool:
    """Robust QDialog.exec() result check across PySide6 enum/int variants."""
    try:
        # Qt documents QDialog.Accepted == 1
        return int(result) == 1
    except Exception:
        try:
            return int(result) == int(QDialog.Accepted)
        except Exception:
            return result == QDialog.Accepted

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
        # DISORT 校正结果（画在原始光谱区，支持十字线）
        self.disort_wavelength = None
        self.disort_albedo = None
        self.disort_observed_radiance = None
        self.disort_model_radiance = None
        self.disort_observed_if = None
        self.disort_model_if = None
        self.disort_s0 = None
        # 原始光谱 Y 轴：手动锁定后不随新光谱自动伸缩
        self.raw_ylim_locked = False
        # 已叠加的 RELAB 参考谱（波长μm, 反射率）；生成新比值光谱时一并清除
        self.relab_overlay = None
        # 比值图双 Y 轴（RELAB 谱形对比时使用）
        self.ax_ratio_twin = None

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
        self.ratio_twin_crosshair_hline = None  # RELAB 双轴上的水平准星

        # 自动比值：无光谱特征掩膜 & 每列分母光谱
        self.auto_featureless_mask = None   # (rows, cols) bool
        self.auto_col_denominators = None   # (cols, bands)

        # DISORT：辅助立方体与模式
        self.aux_data = None               # (rows, cols, bands) 辅助信息
        self.aux_path = None               # 辅助立方体 .lbl / .hdr 路径
        self.aux_img_path = None           # 对应 .img（PDS）
        self.aux_metadata = None           # PDS/ENVI metadata（含 UTC / Ls）
        self.disort_mode = None            # None | 'single' | 'image'
        self.disort_data_root = None       # input/ + optical/ 根目录
        self.disort_ls_deg = None          # 太阳经度 Ls（度）
        self.disort_utc_iso = None         # 由头文件解析的 UTC
        self.disort_ls_source = None       # SOLAR_LONGITUDE | UTC_COMPUTED | manual
        self.disort_band_step = 5
        self.disort_mcd_cache = None
        self.disort_albedo_cube = None     # 图像模式输出

        # Unmixing
        self.unmix_library = None          # SpectralLibrary (already on cube wavelengths when used)
        self.unmix_library_raw = None      # original library before resample
        self.unmix_abundance_cube = None   # (rows, cols, n_em)
        self.unmix_rmse_map = None
        self.unmix_last_result = None      # last single-pixel dict
        self.unmix_method = "nnls"
        self.unmix_sparsity = 3
        # Hapke RT endmembers (Excel + ρ, n, D → k)
        self.hapke_endmembers_raw = None   # list[HapkeEndmember] at native wavelengths
        self.hapke_endmembers = None       # prepared (k inverted), on cube wavelengths
        self.hapke_excel_path = None
        self.hapke_lab_incidence = 30.0
        self.hapke_lab_emission = 0.0
        self.hapke_mode = None             # None | 'single' | 'image'
        self.hapke_band_mask_cached = None
        self.hapke_background_endmember = None  # 图像背景端元（用于右下角显示）

        # Sparse unmixing (SUNSAL in SSA space)
        self.sparse_library_raw = None     # SpectralLibrary (REFF, native λ)
        self.sparse_endmember_ssa = None   # (bands, n_em) on cube wavelengths
        self.sparse_excel_path = None
        self.sparse_mode = None            # None | 'single'
        self.sparse_lab_incidence = 30.0
        self.sparse_lab_emission = 0.0
        self.sparse_lambda = 1e-4
        self.sparse_positivity = True
        self.sparse_addone = True
        self.sparse_al_iters = 100
        self.sparse_tol = 1e-4
        self.sparse_band_mask_cached = None

        # Identification (LSGA CRISM classifier)
        self.ident_class_map = None
        self.ident_confidence = None
        self.ident_class_names = None
        self.ident_num_classes = None
        self.ident_result = None

        self.init_ui()
        self.init_menu()

    SPECTRAL_PARAM_NAMES = [
        'BD1400', 'BD1900', 'BD2100_2', 'BD2210_2',
        'BD2250', 'BD2265', 'D2300', 'BD2500_2', 'SINDEX2'
    ]

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

        ident_filter_layout = QHBoxLayout()
        ident_filter_layout.addWidget(QLabel("分类显示类别:"))
        self.ident_class_filter_input = QLineEdit("0")
        self.ident_class_filter_input.setPlaceholderText("0=全部，输入数字只显示该类")
        self.ident_class_filter_input.setFixedWidth(160)
        self.ident_class_filter_input.setToolTip(
            "模型测试/应用之后：输入 0 显示全部类别，输入 1、2、… 只叠加显示该矿物。"
        )
        self.ident_class_filter_input.returnPressed.connect(
            self.apply_identification_class_filter
        )
        self.btn_ident_class_filter = QPushButton("显示该类")
        self.btn_ident_class_filter.clicked.connect(
            self.apply_identification_class_filter
        )
        ident_filter_layout.addWidget(self.ident_class_filter_input)
        ident_filter_layout.addWidget(self.btn_ident_class_filter)
        ident_filter_layout.addStretch()
        left_layout.addLayout(ident_filter_layout)

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

        # 原始光谱 Y 轴显示范围
        raw_ylim_layout = QHBoxLayout()
        raw_ylim_layout.addWidget(QLabel("原始光谱 Y轴:"))
        raw_ylim_layout.addWidget(QLabel("Min:"))
        self.raw_ymin_input = QLineEdit()
        self.raw_ymin_input.setPlaceholderText("Min")
        self.raw_ymin_input.setFixedWidth(80)
        self.raw_ymax_input = QLineEdit()
        self.raw_ymax_input.setPlaceholderText("Max")
        self.raw_ymax_input.setFixedWidth(80)
        self.btn_apply_raw_ylim = QPushButton("应用")
        self.btn_auto_raw_ylim = QPushButton("自动")
        self.btn_apply_raw_ylim.clicked.connect(self.apply_raw_ylim)
        self.btn_auto_raw_ylim.clicked.connect(self.auto_raw_ylim)
        self.raw_ymin_input.returnPressed.connect(self.apply_raw_ylim)
        self.raw_ymax_input.returnPressed.connect(self.apply_raw_ylim)
        raw_ylim_layout.addWidget(self.raw_ymin_input)
        raw_ylim_layout.addWidget(QLabel("Max:"))
        raw_ylim_layout.addWidget(self.raw_ymax_input)
        raw_ylim_layout.addWidget(self.btn_apply_raw_ylim)
        raw_ylim_layout.addWidget(self.btn_auto_raw_ylim)
        raw_ylim_layout.addStretch()

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

        # 光谱图保存按钮
        save_spec_layout = QHBoxLayout()
        self.btn_save_raw_spec = QPushButton("保存原始光谱图")
        self.btn_save_raw_spec.clicked.connect(
            lambda: self.save_spectrum_figure(self.fig_raw_spec, "raw_spectrum")
        )
        self.btn_save_ratio_spec = QPushButton("保存比值光谱图")
        self.btn_save_ratio_spec.clicked.connect(
            lambda: self.save_spectrum_figure(self.fig_ratio_spec, "ratio_spectrum")
        )
        save_spec_layout.addWidget(self.btn_save_raw_spec)
        save_spec_layout.addWidget(self.btn_save_ratio_spec)
        save_spec_layout.addStretch()

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
        right_layout.addLayout(raw_ylim_layout)
        right_layout.addWidget(self.canvas_ratio_spec)
        right_layout.addLayout(save_spec_layout)
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
        for p in self.SPECTRAL_PARAM_NAMES:
            param_menu.addAction(p, lambda name=p: self.calc_spectral_parameter(name))

        # 3. Identification
        id_menu = menubar.addMenu('Identification')
        lsga_menu = id_menu.addMenu('LSGA')
        lsga_menu.addAction('模型训练', self.identification_train)
        lsga_menu.addAction('模型测试', self.identification_test)
        lsga_menu.addAction('模型应用', self.identification_apply)
        hbm_menu = id_menu.addMenu('HBM')
        hbm_menu.addAction('模型训练', self.hbm_train)
        hbm_menu.addAction('模型测试', self.hbm_test)
        hbm_menu.addAction('模型应用', self.hbm_apply)

        # 4. Unmixing
        unmix_menu = menubar.addMenu('Unmixing')
        hapke_menu = unmix_menu.addMenu('Hapke model')
        hapke_menu.addAction('加载高光谱图像', self.open_hapke_hyperspectral)
        hapke_menu.addAction('加载辅助立方体', self.open_hapke_aux)
        hapke_menu.addAction('加载端元反射率 Excel…', self.load_hapke_excel_endmembers)
        hapke_menu.addSeparator()
        hapke_menu.addAction('单光谱计算', self.hapke_single_spectrum)
        hapke_menu.addAction('图像处理', self.hapke_image_mode)
        hapke_menu.addAction('退出 Hapke 单光谱模式', self.exit_hapke_mode)
        hapke_menu.addSeparator()
        hapke_menu.addAction('显示丰度图…', self.show_unmix_abundance_map)
        hapke_menu.addAction('显示 RMSE 图', self.show_unmix_rmse_map)
        hapke_menu.addAction('导出 k(λ)…', self.export_hapke_k)
        sparse_menu = unmix_menu.addMenu('Sparse unmixing')
        sparse_menu.addAction('加载高光谱图像', self.open_sparse_hyperspectral)
        sparse_menu.addAction('加载辅助立方体', self.open_sparse_aux)
        sparse_menu.addAction('加载端元反射率 Excel…', self.load_sparse_excel_endmembers)
        sparse_menu.addSeparator()
        sparse_menu.addAction('单光谱计算', self.sparse_single_spectrum)
        sparse_menu.addAction('图像处理', self.sparse_image_mode)
        sparse_menu.addAction('退出 Sparse 单光谱模式', self.exit_sparse_mode)
        sparse_menu.addSeparator()
        sparse_menu.addAction('显示丰度图…', self.show_unmix_abundance_map)
        sparse_menu.addAction('显示 RMSE 图', self.show_unmix_rmse_map)

        # 5. Tools
        tools_menu = menubar.addMenu('Tools')
        disort_menu = tools_menu.addMenu('DISORT correction')
        disort_menu.addAction('加载辐亮度图像', self.open_disort_radiance)
        disort_menu.addAction('加载辅助信息图像', self.open_disort_aux)
        disort_menu.addSeparator()
        disort_menu.addAction('配置/安装本地 MCD…', self.setup_local_mcd)
        disort_menu.addAction('查看本地 MCD 状态', self.show_mcd_status)
        disort_menu.addSeparator()
        disort_menu.addAction('单光谱计算', self.disort_single_spectrum_mode)
        disort_menu.addAction('图像处理', self.disort_image_mode)
        disort_menu.addAction('退出 DISORT 模式', self.exit_disort_mode)

        ratio_menu = tools_menu.addMenu('Ratio spectra')
        ratio_menu.addAction('自动提取', lambda: self.set_ratio_mode('auto'))
        ratio_menu.addAction('手动提取', lambda: self.set_ratio_mode('manual'))
        ratio_menu.addAction('退出', self.exit_ratio_mode)

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
            self.auto_featureless_mask = None
            self.auto_col_denominators = None

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
        self.disort_wavelength = None
        self.disort_albedo = None
        self.disort_observed_radiance = None
        self.disort_model_radiance = None
        self.disort_observed_if = None
        self.disort_model_if = None
        self.disort_s0 = None
        self.ident_class_map = None
        self.ident_confidence = None
        self.ident_class_names = None
        self.ident_num_classes = None
        self.ident_result = None
        self.raw_ylim_locked = False
        if self.ratio_mode == "disort":
            self.ratio_mode = None

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
        self._remove_ratio_twin()
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
            # DISORT 结果用自身波长轴对齐原始光谱区
            if (
                self.ratio_mode == "disort"
                and self.disort_wavelength is not None
                and np.any(np.isfinite(self.disort_wavelength))
            ):
                xmin = float(np.nanmin(self.disort_wavelength))
                xmax = float(np.nanmax(self.disort_wavelength))
            if np.isfinite(xmin) and np.isfinite(xmax) and xmin < xmax:
                self.ax_raw_spec.set_xlim(xmin, xmax)
                self.ax_ratio_spec.set_xlim(xmin, xmax)

    def _reset_ratio_crosshair(self):
        self.ratio_crosshair_vline = None
        self.ratio_crosshair_hline = None
        self.ratio_crosshair_text = None
        self.ratio_twin_crosshair_hline = None

    def _clear_ratio_crosshair_artists(self):
        """从图上真正移除比值十字线，保证同时只保留一组。"""
        for art in (
            self.ratio_crosshair_vline,
            self.ratio_crosshair_hline,
            self.ratio_crosshair_text,
            self.ratio_twin_crosshair_hline,
        ):
            if art is None:
                continue
            try:
                art.remove()
            except Exception:
                pass
        self._reset_ratio_crosshair()

    def _remove_artist_safe(self, artist, container):
        """安全移除 matplotlib 艺术家对象。"""
        if artist is None:
            return
        try:
            artist.remove()
        except Exception:
            pass

    def _remove_ratio_twin(self):
        """移除比值图上的双 Y 轴（RELAB 叠加轴）。"""
        self._clear_ratio_crosshair_artists()
        if getattr(self, 'ax_ratio_twin', None) is not None:
            try:
                self.ax_ratio_twin.remove()
            except Exception:
                pass
            self.ax_ratio_twin = None

    def _set_ylim_from_data(self, ax, values, pad_ratio=0.05):
        """按该光谱自身数值范围设置 Y 轴，使吸收特征更明显。"""
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        ymin = float(np.min(vals))
        ymax = float(np.max(vals))
        if ymin == ymax:
            pad = abs(ymin) * 0.05 + 1e-6
        else:
            pad = (ymax - ymin) * pad_ratio
        ax.set_ylim(ymin - pad, ymax + pad)

    def _sync_raw_ylim_inputs(self):
        """把当前原始光谱 Y 轴范围同步到输入框。"""
        ymin, ymax = self.ax_raw_spec.get_ylim()
        self.raw_ymin_input.setText(f"{ymin:.6g}")
        self.raw_ymax_input.setText(f"{ymax:.6g}")

    def _apply_raw_spec_ylim(self, values=None):
        """
        设置原始光谱 Y 轴：
        - 已锁定：沿用输入框中的 Min/Max
        - 未锁定：按数据自动伸缩，并回填输入框
        """
        if self.raw_ylim_locked:
            try:
                ymin = float(self.raw_ymin_input.text().strip())
                ymax = float(self.raw_ymax_input.text().strip())
                if ymin < ymax:
                    self.ax_raw_spec.set_ylim(ymin, ymax)
                    return
            except Exception:
                pass
            # 锁定但输入无效时回退自动
            self.raw_ylim_locked = False

        if values is not None:
            self._set_ylim_from_data(self.ax_raw_spec, values)
        self._sync_raw_ylim_inputs()

    def apply_raw_ylim(self):
        """手动应用原始光谱 Y 轴显示范围。"""
        try:
            ymin = float(self.raw_ymin_input.text().strip())
            ymax = float(self.raw_ymax_input.text().strip())
        except Exception:
            QMessageBox.warning(self, "输入错误", "请输入有效的 Y 轴 Min / Max 数值。")
            return
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin >= ymax:
            QMessageBox.warning(self, "输入错误", "需要 Min < Max，且均为有限数值。")
            return
        self.raw_ylim_locked = True
        self.ax_raw_spec.set_ylim(ymin, ymax)
        self.canvas_raw_spec.draw()

    def auto_raw_ylim(self):
        """按当前曲线数据自动设置原始光谱 Y 轴。"""
        self.raw_ylim_locked = False
        values = None
        if (
            self.ratio_mode == "disort"
            and self.disort_albedo is not None
        ):
            parts = [self.disort_albedo]
            if self.disort_observed_if is not None:
                parts.append(self.disort_observed_if)
            if self.disort_model_if is not None:
                parts.append(self.disort_model_if)
            values = np.concatenate(
                [np.asarray(p, dtype=float).ravel() for p in parts]
            )
        elif self.current_raw_spectrum is not None:
            values = self.current_raw_spectrum
        else:
            # 从图中线数据推断
            ys = []
            for line in self.ax_raw_spec.get_lines():
                yd = np.asarray(line.get_ydata(), dtype=float)
                ys.append(yd[np.isfinite(yd)])
            if ys:
                values = np.concatenate(ys) if any(a.size for a in ys) else None

        if values is None:
            QMessageBox.information(self, "提示", "当前原始光谱区无可用数据。")
            return
        self._apply_raw_spec_ylim(values)
        self.canvas_raw_spec.draw()

    def _clear_ratio_plot(self, title="比值光谱", show_y_labels=True):
        """生成新比值光谱前清空图中全部曲线（含 RELAB 叠加与双轴）。"""
        self.relab_overlay = None
        self._remove_ratio_twin()
        self._clear_ratio_crosshair_artists()
        self.ax_ratio_spec.clear()
        self.ax_ratio_spec.set_title(title)
        self.ax_ratio_spec.set_xlabel("Wavelength ($\mu$m)")
        self.ax_ratio_spec.set_ylabel("Scaled Reflectance")
        self.ax_ratio_spec.tick_params(axis='y', labelleft=show_y_labels)

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

        # 双击图像：退出 Ratio / DISORT / Hapke 模式
        if getattr(event, 'dblclick', False):
            if self.hapke_mode is not None:
                self.exit_hapke_mode()
                return
            if self.sparse_mode is not None:
                self.exit_sparse_mode()
                return
            if self.disort_mode is not None:
                self.exit_disort_mode()
                return
            if self.ratio_mode is not None:
                self.exit_ratio_mode()
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

        # Hapke 单光谱：点击左侧图像 → 用 Excel 端元拟合该像元矿物比例
        if self.hapke_mode == 'single' and event.inaxes == self.ax_rgb:
            self._hapke_run_single_pixel(row, col)
            return

        # Sparse 单光谱：点击左侧图像 → SSA + SUNSAL
        if self.sparse_mode == 'single' and event.inaxes == self.ax_rgb:
            self._sparse_run_single_pixel(row, col)
            return

        # DISORT 单光谱：点击上方辐亮度图触发校正
        if self.disort_mode == 'single' and event.inaxes == self.ax_rgb:
            self._disort_run_single_pixel(row, col)
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

            self.raw_crosshair_vline = None
            self.raw_crosshair_hline = None
            self.raw_crosshair_text = None

            if self.ratio_mode == 'auto':
                if self.auto_col_denominators is None:
                    QMessageBox.warning(self, "警告", "自动比值分母尚未准备好，请重新选择自动提取。")
                    return

                denom = self.auto_col_denominators[col]
                if (not np.any(np.isfinite(denom))) or np.nanmax(np.abs(denom)) < 1e-12:
                    QMessageBox.warning(
                        self, "警告",
                        f"第 {col} 列没有可用的无光谱特征区域，无法作为分母。"
                    )
                    return

                # 原始光谱区：分子 + 该列无特征区均值分母
                self.ax_raw_spec.clear()
                self.ax_raw_spec.plot(
                    wave, spectrum, color='navy', linewidth=1.2,
                    label=f'分子 (X:{col}, Y:{row})'
                )
                self.ax_raw_spec.plot(
                    wave, denom, color='gray', linewidth=1.2, linestyle='--',
                    label=f'分母 (列{col}无特征均值)'
                )
                self.ax_raw_spec.set_title(
                    f"自动比值 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})"
                )
                self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
                self.ax_raw_spec.set_ylabel("Reflectance")
                self.ax_raw_spec.grid(True, linestyle='--', alpha=0.5)
                self.ax_raw_spec.legend(fontsize=8)
                self._apply_raw_spec_ylim(
                    np.concatenate([
                        np.asarray(spectrum, dtype=float).ravel(),
                        np.asarray(denom, dtype=float).ravel(),
                    ])
                )

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_spec = spectrum / (denom + 1e-8)
                ratio_spec = np.asarray(ratio_spec, dtype=np.float32)
                ratio_spec[~np.isfinite(ratio_spec)] = np.nan
                self.current_ratio_spectrum = ratio_spec

                n_feat = 0
                if self.auto_featureless_mask is not None:
                    n_feat = int(np.count_nonzero(self.auto_featureless_mask[:, col]))
                self._clear_ratio_plot(
                    f"比值光谱 (分子 X:{col},Y:{row} / 列{col}无特征均值, n={n_feat})"
                )
                self.ax_ratio_spec.plot(
                    wave, ratio_spec, color='crimson',
                    label=f'Ratio (X:{col}, Y:{row})'
                )
                self.ax_ratio_spec.legend(fontsize=8)
                self.ax_ratio_spec.grid(True, linestyle='--', alpha=0.5)
                self._set_ylim_from_data(self.ax_ratio_spec, ratio_spec)

            elif self.ratio_mode == 'disort':
                # 保留 DISORT 地表反照率在原始光谱区；仅更新选中像元（供下次校正用）
                # current_raw_spectrum 已在上方赋值为当前像元光谱
                if self.disort_albedo is not None:
                    self._plot_disort_on_raw()
                else:
                    self.ax_raw_spec.clear()
                    self.ax_raw_spec.plot(wave, spectrum, color='navy', linewidth=1.2)
                    self.ax_raw_spec.set_title(
                        f"原始光谱显示 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})"
                    )
                    self.ax_raw_spec.set_xlabel(r"Wavelength ($\mu$m)")
                    self.ax_raw_spec.set_ylabel("Reflectance")
                    self.ax_raw_spec.grid(True, linestyle='--', alpha=0.5)
                    self._apply_raw_spec_ylim(spectrum)

            else:
                # 非比值模式：只显示原始光谱
                self.ax_raw_spec.clear()
                self.ax_raw_spec.plot(wave, spectrum, color='navy', linewidth=1.2)
                self.ax_raw_spec.set_title(
                    f"原始光谱显示 (X: {col}, Y: {row}, 均值: {w_size}x{w_size})"
                )
                self.ax_raw_spec.set_xlabel("Wavelength ($\mu$m)")
                self.ax_raw_spec.set_ylabel("Reflectance")
                self.ax_raw_spec.grid(True, linestyle='--', alpha=0.5)
                self._apply_raw_spec_ylim(spectrum)

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
            stacked = np.concatenate(
                [np.asarray(s, dtype=float).ravel() for s in self.click_coords]
            )
            self._apply_raw_spec_ylim(stacked)

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
                self._set_ylim_from_data(self.ax_ratio_spec, ratio)

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
        if event.xdata is None:
            return

        # DISORT 模式：对地表反照率光谱读数；否则对当前像元原始光谱读数
        if (
            self.ratio_mode == "disort"
            and self.disort_wavelength is not None
            and self.disort_albedo is not None
        ):
            wave_axis = np.asarray(self.disort_wavelength, dtype=np.float64)
            spectrum = np.asarray(self.disort_albedo, dtype=np.float64)
            value_label = "Albedo"
        else:
            if self.wavelengths is None or self.current_raw_spectrum is None:
                return
            wave_axis = np.asarray(self.wavelengths, dtype=np.float64)
            spectrum = np.asarray(self.current_raw_spectrum, dtype=np.float64)
            value_label = "R"

        if wave_axis.size == 0 or spectrum.size == 0:
            return

        click_x = event.xdata
        finite = np.isfinite(wave_axis) & np.isfinite(spectrum)
        if not np.any(finite):
            return
        idx_candidates = np.where(finite)[0]
        idx = int(idx_candidates[np.argmin(np.abs(wave_axis[finite] - click_x))])
        target_wave = float(wave_axis[idx])
        target_val = float(spectrum[idx])

        if self.raw_crosshair_vline in self.ax_raw_spec.lines:
            self.raw_crosshair_vline.remove()
        if self.raw_crosshair_hline in self.ax_raw_spec.lines:
            self.raw_crosshair_hline.remove()
        if self.raw_crosshair_text in self.ax_raw_spec.texts:
            self.raw_crosshair_text.remove()

        self.raw_crosshair_vline = self.ax_raw_spec.axvline(
            x=target_wave, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8
        )
        self.raw_crosshair_hline = self.ax_raw_spec.axhline(
            y=target_val, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8
        )

        um_val = target_wave if target_wave < 100 else target_wave / 1000.0
        text_str = f" {um_val:.3f} $\\mu$m {value_label}={target_val:.4f}"
        self.raw_crosshair_text = self.ax_raw_spec.text(
            0.03, 0.95, text_str,
            transform=self.ax_raw_spec.transAxes,
            verticalalignment='top',
            fontsize=10,
            color='darkred'
        )

        self.canvas_raw_spec.draw()

    def on_ratio_spec_clicked(self, event):
        """
        点击比值光谱图（含 RELAB 叠加/双轴）：显示唯一黑色十字线及波长/数值读数。
        """
        valid_axes = [self.ax_ratio_spec]
        if getattr(self, 'ax_ratio_twin', None) is not None:
            valid_axes.append(self.ax_ratio_twin)
        if event.inaxes not in valid_axes or event.xdata is None:
            return

        has_ratio = (
            self.current_ratio_spectrum is not None
            and self.wavelengths is not None
            and len(self.wavelengths) > 0
        )
        has_relab = self.relab_overlay is not None
        if not has_ratio and not has_relab:
            return

        click_x = float(event.xdata)
        clicked_twin = (
            self.ax_ratio_twin is not None and event.inaxes is self.ax_ratio_twin
        )

        ratio_val = None
        relab_val = None

        if has_ratio:
            idx = int(np.argmin(np.abs(self.wavelengths - click_x)))
            target_wave = float(self.wavelengths[idx])
            ratio_val = float(self.current_ratio_spectrum[idx])
            if has_relab:
                rw, rr, _ = self.relab_overlay
                relab_val = float(np.interp(target_wave, rw, rr))
        else:
            rw, rr, _ = self.relab_overlay
            idx = int(np.argmin(np.abs(rw - click_x)))
            target_wave = float(rw[idx])
            relab_val = float(rr[idx])

        # 先清掉旧十字线，保证始终只有一组
        self._clear_ratio_crosshair_artists()

        # 唯一黑色十字：竖线在主轴；横线只画一条（按点击轴决定）
        self.ratio_crosshair_vline = self.ax_ratio_spec.axvline(
            x=target_wave, color='black', linestyle='--', linewidth=1.2, alpha=0.9
        )

        use_relab_hline = False
        if clicked_twin and relab_val is not None and np.isfinite(relab_val):
            use_relab_hline = True
        elif (not has_ratio) and relab_val is not None and np.isfinite(relab_val):
            use_relab_hline = True
        elif ratio_val is not None and np.isfinite(ratio_val):
            use_relab_hline = False
        elif relab_val is not None and np.isfinite(relab_val):
            use_relab_hline = True

        if use_relab_hline:
            if self.ax_ratio_twin is not None:
                self.ratio_twin_crosshair_hline = self.ax_ratio_twin.axhline(
                    y=relab_val, color='black', linestyle='--', linewidth=1.2, alpha=0.9
                )
            else:
                self.ratio_crosshair_hline = self.ax_ratio_spec.axhline(
                    y=relab_val, color='black', linestyle='--', linewidth=1.2, alpha=0.9
                )
            active_val = relab_val
        else:
            self.ratio_crosshair_hline = self.ax_ratio_spec.axhline(
                y=ratio_val, color='black', linestyle='--', linewidth=1.2, alpha=0.9
            )
            active_val = ratio_val

        um_val = target_wave if target_wave < 100 else target_wave / 1000.0
        parts = [f"{um_val:.3f} $\mu$m"]
        if ratio_val is not None and np.isfinite(ratio_val):
            parts.append(f"Ratio {ratio_val:.4f}")
        if relab_val is not None and np.isfinite(relab_val):
            parts.append(f"RELAB {relab_val:.4f}")
        if active_val is not None and np.isfinite(active_val) and len(parts) == 1:
            parts.append(f"{active_val:.4f}")
        text_str = "  ".join(parts)

        self.ratio_crosshair_text = self.ax_ratio_spec.text(
            0.03, 0.95, text_str,
            transform=self.ax_ratio_spec.transAxes,
            verticalalignment='top',
            fontsize=10,
            color='black'
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

        if (
            self.ident_class_map is not None
            and self.current_param_title
            and str(self.current_param_title).startswith("Identification")
        ):
            self.show_identification_map()
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

    def _compute_spectral_parameter(self, param_name):
        """
        计算单个 CRISM 光谱参数，返回 (rows, cols) 浮点数组。
        无效像元置 0（与菜单显示逻辑一致）；不负责绘图。
        """
        if self.current_data is None or self.wavelengths is None:
            return None

        if param_name == 'BD1400':
            b1467 = self.get_band_mean_by_wave(1467, num_bands=5)
            b1330 = self.get_band_mean_by_wave(1330, num_bands=5)
            b1395 = self.get_band_mean_by_wave(1395, num_bands=5)
            r8 = ((1395.0 - 1330.0) / (1467.0 - 1330.0)) * b1467 + \
                 ((1467.0 - 1395.0) / (1467.0 - 1330.0)) * b1330
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 1.0 - (b1395 / (r8 + 1e-8))
            res[~np.isfinite(res)] = 0.0
            return res

        if param_name == 'BD1900':
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
            return bd1900

        if param_name == 'BD2100_2':
            b2250 = self.get_band_mean_by_wave(2250, num_bands=3)
            b1930 = self.get_band_mean_by_wave(1930, num_bands=3)
            b2132 = self.get_band_mean_by_wave(2132, num_bands=5)
            r3 = ((2132.0 - 1930.0) / (2250.0 - 1930.0)) * b2250 + \
                 ((2250.0 - 2132.0) / (2250.0 - 1930.0)) * b1930
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 1.0 - (b2132 / (r3 + 1e-8))
            res[~np.isfinite(res)] = 0.0
            return res

        if param_name == 'BD2210_2':
            b2250 = self.get_band_mean_by_wave(2250, num_bands=5)
            b2165 = self.get_band_mean_by_wave(2165, num_bands=5)
            b2210 = self.get_band_mean_by_wave(2210, num_bands=5)
            r4 = ((2210.0 - 2165.0) / (2250.0 - 2165.0)) * b2250 + \
                 ((2250.0 - 2210.0) / (2250.0 - 2165.0)) * b2165
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 1.0 - (b2210 / (r4 + 1e-8))
            res[~np.isfinite(res)] = 0.0
            return res

        if param_name == 'BD2250':
            b2340 = self.get_band_mean_by_wave(2340, num_bands=3)
            b2120 = self.get_band_mean_by_wave(2120, num_bands=5)
            b2245 = self.get_band_mean_by_wave(2245, num_bands=7)
            r11 = ((2245.0 - 2120.0) / (2340.0 - 2120.0)) * b2340 + \
                  ((2340.0 - 2245.0) / (2340.0 - 2120.0)) * b2120
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 1.0 - (b2245 / (r11 + 1e-8))
            res[~np.isfinite(res)] = 0.0
            return res

        if param_name == 'BD2265':
            b2340 = self.get_band_mean_by_wave(2340, num_bands=5)
            b2210 = self.get_band_mean_by_wave(2210, num_bands=5)
            b2265 = self.get_band_mean_by_wave(2265, num_bands=3)
            r9 = ((2265.0 - 2210.0) / (2340.0 - 2210.0)) * b2340 + \
                 ((2340.0 - 2265.0) / (2340.0 - 2210.0)) * b2210
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 1.0 - (b2265 / (r9 + 1e-8))
            res[~np.isfinite(res)] = 0.0
            return res

        if param_name == 'D2300':
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
            return res

        if param_name == 'BD2500_2':
            b2570 = self.get_band_mean_by_wave(2570, num_bands=5)
            b2364 = self.get_band_mean_by_wave(2364, num_bands=5)
            b2480 = self.get_band_mean_by_wave(2480, num_bands=5)
            r7 = ((2480.0 - 2364.0) / (2570.0 - 2364.0)) * b2570 + \
                 ((2570.0 - 2480.0) / (2570.0 - 2364.0)) * b2364
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 1.0 - (b2480 / (r7 + 1e-8))
            res[~np.isfinite(res)] = 0.0
            return res

        if param_name == 'SINDEX2':
            b2400 = self.get_band_mean_by_wave(2400, num_bands=3)
            b2120 = self.get_band_mean_by_wave(2120, num_bands=5)
            b2290_7 = self.get_band_mean_by_wave(2290, num_bands=7)
            r10 = ((2290.0 - 2120.0) / (2400.0 - 2120.0)) * b2400 + \
                  ((2400.0 - 2290.0) / (2400.0 - 2120.0)) * b2120
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 1.0 - (r10 / (b2290_7 + 1e-8))
            res[~np.isfinite(res)] = 0.0
            return res

        return None

    def _prepare_auto_ratio_denominators(self):
        """
        自动比值准备：
        1) 计算全部 Spectral parameter
        2) 对每个参数：整幅图中值 med、标准差 std；
           像元值 adj = 参数 - (med + std)；
           所有参数的 adj 都 < 0 的像元 = 无光谱特征区域
        3) 每一列无特征区域光谱均值 = 该列分母
        """
        if self.current_data is None:
            raise RuntimeError("请先打开高光谱数据")

        rows, cols, bands = self.current_data.shape
        param_list = []
        for name in self.SPECTRAL_PARAM_NAMES:
            p = self._compute_spectral_parameter(name)
            if p is None:
                raise RuntimeError(f"参数 {name} 计算失败")
            param_list.append(np.asarray(p, dtype=np.float32))

        stack = np.stack(param_list, axis=0)  # (n_param, rows, cols)

        # 每个参数：adj = param - (median + std)；全部 adj < 0 → 无特征
        adj_list = []
        with np.errstate(all='ignore'):
            for p in stack:
                # 统计时忽略无效像元（当前公式里无效常被置 0，这里用有限值全体）
                finite = p[np.isfinite(p)]
                if finite.size == 0:
                    med, std = 0.0, 0.0
                else:
                    med = float(np.nanmedian(finite))
                    std = float(np.nanstd(finite))
                adj_list.append(p - (med + std))
        adj_stack = np.stack(adj_list, axis=0)
        featureless = np.all(adj_stack < 0.0, axis=0)

        # 排除整波段无效像元
        with np.errstate(all='ignore'):
            valid_pix = np.any(np.isfinite(self.current_data), axis=2)
        featureless = featureless & valid_pix

        denoms = np.full((cols, bands), np.nan, dtype=np.float32)
        usable_cols = 0
        for c in range(cols):
            mask = featureless[:, c]
            if not np.any(mask):
                continue
            with np.errstate(all='ignore'):
                denoms[c] = np.nanmean(self.current_data[mask, c, :], axis=0)
            if np.any(np.isfinite(denoms[c])):
                usable_cols += 1

        self.auto_featureless_mask = featureless
        self.auto_col_denominators = denoms
        return featureless, denoms, usable_cols

    def calc_spectral_parameter(self, param_name):
        """计算 CRISM 光谱参数并显示"""
        if self.current_data is None:
            QMessageBox.warning(self, "警告", "请先打开高光谱图像数据！")
            return

        try:
            res = self._compute_spectral_parameter(param_name)
            if res is None:
                QMessageBox.information(self, "提示", f"参数【{param_name}】公式尚未配置。")
                return
            self.show_parameter_result(res, param_name)
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算 {param_name} 时发生错误:\n{str(e)}")

    def identification_train(self):
        """CRISM 监督分类：预处理 + LSGA 训练。"""
        try:
            from identification.dialogs import IdentificationTrainDialog
        except ImportError as exc:
            QMessageBox.critical(
                self, "缺少依赖",
                "模型训练需要 PyTorch、einops、scipy 等。\n\n"
                f"{exc}",
            )
            return
        dlg = IdentificationTrainDialog(self)
        dlg.exec()

    def identification_test(self):
        """外部场景定量测试 / 整图推理。"""
        try:
            from identification.dialogs import IdentificationTestDialog
        except ImportError as exc:
            QMessageBox.critical(
                self, "缺少依赖",
                "模型测试需要 PyTorch、einops、scipy 等。\n\n"
                f"{exc}",
            )
            return
        dlg = IdentificationTestDialog(self)
        dlg.exec()
        summary = getattr(dlg, "result_summary", None)
        if isinstance(summary, dict) and summary.get("display_prediction") is not None:
            self.show_identification_map(summary)

    def hbm_train(self):
        """HBM (crism_ml) bland + mineral model training."""
        try:
            from identification.hbm.dialogs import HbmTrainDialog
        except ImportError as exc:
            QMessageBox.critical(
                self, "缺少依赖",
                "HBM 训练需要 scipy、joblib、spectral 等。\n\n"
                f"{exc}",
            )
            return
        dlg = HbmTrainDialog(self)
        dlg.exec()

    def hbm_test(self):
        """HBM labeled evaluation on a CRISM I/F scene."""
        try:
            from identification.hbm.dialogs import HbmTestDialog
        except ImportError as exc:
            QMessageBox.critical(
                self, "缺少依赖",
                "HBM 测试需要 scipy、joblib、spectral 等。\n\n"
                f"{exc}",
            )
            return
        dlg = HbmTestDialog(self)
        dlg.exec()
        summary = getattr(dlg, "result_summary", None)
        if isinstance(summary, dict) and summary.get("display_prediction") is not None:
            self.show_identification_map(summary)

    def hbm_apply(self):
        """Classify CRISM I/F cubes with HBM and overlay the mineral map."""
        try:
            from identification.hbm.dialogs import HbmApplyDialog
        except ImportError as exc:
            QMessageBox.critical(
                self, "缺少依赖",
                "HBM 应用需要 scipy、joblib、spectral 等。\n\n"
                f"{exc}",
            )
            return

        dlg = HbmApplyDialog(self)
        if not _dialog_accepted(dlg.exec()):
            return
        params = dlg.params()
        source = params.get("source") or "opened"
        if source == "opened" and self.current_data is None:
            QMessageBox.information(
                self, "HBM 模型应用",
                "当前没有已打开的高光谱影像。请选择单个文件或文件夹。",
            )
            return

        progress = QProgressDialog("HBM 模型应用中…", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        cancelled = {"v": False}

        def cb(done, total, message=""):
            if progress.wasCanceled():
                cancelled["v"] = True
                return
            progress.setLabelText(message or "HBM 模型应用中…")
            progress.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            from pathlib import Path
            from identification.hbm.pipeline import classify_cube, classify_paths
            from identification.io import list_input_files

            save_dir = Path(params["save_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)
            common = dict(
                datadir=params["datadir"],
                workdir=params["workdir"],
                thresholds=params["thresholds"],
                n_jobs=int(params["n_jobs"]),
                save_dir=save_dir,
            )
            if source == "opened":
                result = classify_cube(
                    self.current_data,
                    data_layout="HWB",
                    source_name="opened_cube",
                    **common,
                )
                saved = [str(result.get("envi_path") or "")]
            else:
                paths = list_input_files(
                    params["data_path"],
                    params.get("input_pattern") or "*",
                )
                batch = classify_paths(
                    paths,
                    progress_cb=cb,
                    **common,
                )
                result = batch["last"]
                saved = list(batch.get("saved") or [])
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "HBM 模型应用失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()

        if cancelled["v"]:
            QMessageBox.information(self, "HBM 模型应用", "已取消。")
            return

        self.show_identification_map(result)
        saved_text = "\n".join(s for s in saved[:8] if s)
        QMessageBox.information(
            self, "HBM 模型应用完成",
            f"已写出 {len(saved)} 个 ENVI *.img 到：\n{params['save_dir']}\n"
            f"{saved_text}\n\n"
            "分类图已叠加显示在左下方。可在「分类显示类别」输入数字只显示某一类矿物。",
        )

    def identification_apply(self):
        """对打开影像 / 单个文件 / 文件夹分类，写出 ENVI *.img，叠加显示。"""
        try:
            from identification.dialogs import IdentificationApplyDialog
        except ImportError as exc:
            QMessageBox.critical(
                self, "缺少依赖",
                "模型应用需要 PyTorch、einops、scipy 等。\n\n"
                f"{exc}",
            )
            return

        dlg = IdentificationApplyDialog(self)
        if not _dialog_accepted(dlg.exec()):
            return
        params = dlg.params()
        source = params.get("source") or "opened"

        if source == "opened" and self.current_data is None:
            QMessageBox.information(
                self, "模型应用",
                "当前没有已打开的高光谱影像。请选择单个文件或文件夹。",
            )
            return

        progress = QProgressDialog("Identification 模型应用中…", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        cancelled = {"v": False}

        def cb(done, total, message=""):
            if progress.wasCanceled():
                cancelled["v"] = True
                return
            progress.setLabelText(message or "Identification 模型应用中…")
            progress.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            from pathlib import Path
            from identification.apply import apply_paths, apply_to_opened_cube
            from identification.io import list_input_files, write_envi_class_map

            save_dir = Path(params["save_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)

            if source == "opened":
                result = apply_to_opened_cube(
                    self.current_data,
                    self.wavelengths,
                    checkpoint_path=params["checkpoint_path"],
                    device_cfg=params["device"],
                    batch_size=int(params["batch_size"]),
                    confidence_threshold=float(params["confidence_threshold"]),
                    progress_cb=cb,
                )
                envi_path = write_envi_class_map(
                    save_dir / "opened_cube_class.img",
                    result["display_prediction"],
                    result.get("class_names"),
                )
                result["envi_path"] = str(envi_path)
                saved = [str(envi_path)]
            else:
                paths = list_input_files(
                    params["data_path"],
                    params.get("input_pattern") or "*",
                )
                batch = apply_paths(
                    paths,
                    checkpoint_path=params["checkpoint_path"],
                    save_dir=save_dir,
                    device_cfg=params["device"],
                    batch_size=int(params["batch_size"]),
                    confidence_threshold=float(params["confidence_threshold"]),
                    data_key=params.get("data_key"),
                    data_layout=params.get("data_layout") or "HWB",
                    progress_cb=cb,
                )
                result = batch["last"]
                saved = list(batch.get("saved") or [])
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "模型应用失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()

        if cancelled["v"]:
            QMessageBox.information(self, "模型应用", "已取消。")
            return

        self.show_identification_map(result)
        src = "内置模型" if params.get("use_builtin") else "本次训练的新模型"
        saved_text = "\n".join(saved[:8])
        if len(saved) > 8:
            saved_text += f"\n… 共 {len(saved)} 个文件"
        QMessageBox.information(
            self, "模型应用完成",
            f"来源：{src}\n"
            f"波段：1.02–2.6 μm → 240 通道（自动预处理）\n"
            f"类别数：{result.get('num_classes')}\n"
            f"模型：{params['checkpoint_path']}\n"
            f"已写出 {len(saved)} 个 ENVI *.img 到：\n{params['save_dir']}\n"
            f"{saved_text}\n\n"
            "分类图已叠加显示在左下方。可在「分类显示类别」输入数字只显示某一类矿物。",
        )

    def _ident_class_filter_id(self) -> int:
        text = ""
        if hasattr(self, "ident_class_filter_input"):
            text = self.ident_class_filter_input.text().strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError("请输入整数类别编号（0 表示显示全部）。") from exc

    def apply_identification_class_filter(self):
        """根据输入框只叠加显示某一类矿物。"""
        if self.ident_class_map is None:
            QMessageBox.information(
                self, "分类显示",
                "请先完成「模型测试」或「模型应用」。",
            )
            return
        try:
            class_id = self._ident_class_filter_id()
        except ValueError as exc:
            QMessageBox.warning(self, "分类显示", str(exc))
            return
        k = int(self.ident_num_classes or 0)
        if class_id < 0 or (k and class_id > k):
            QMessageBox.warning(
                self, "分类显示",
                f"类别编号应在 0–{k} 之间（0 表示全部）。",
            )
            return
        self.show_identification_map()

    def _identification_base_image(self, shape):
        """RGB 或 1.08 μm 灰度底图；空间尺寸不一致时返回 None。"""
        height, width = int(shape[0]), int(shape[1])
        if (
            self.rgb_image is not None
            and self.rgb_image.shape[0] == height
            and self.rgb_image.shape[1] == width
        ):
            return self.rgb_image
        if (
            self.current_data is not None
            and self.current_data.shape[0] == height
            and self.current_data.shape[1] == width
        ):
            base_108 = self.get_band_mean_by_wave(1080, num_bands=5)
            if base_108 is not None:
                b_min, b_max = np.nanpercentile(base_108, [2, 98])
                base_norm = np.clip((base_108 - b_min) / (b_max - b_min + 1e-8), 0, 1)
                base_norm[np.isnan(base_norm)] = 0.0
                return base_norm
        return None

    def show_identification_map(self, result: dict = None):
        """将分类图叠加在影像底图上（可按类别编号筛选）。"""
        from identification.io import filter_class_map
        from identification.test_full_image import class_tick_labels, make_cmap

        if result is not None:
            pred = np.asarray(result["display_prediction"])
            names = list(result.get("class_names") or [])
            k = int(result.get("num_classes") or max(int(np.nanmax(pred) or 0), 1))
            if not names:
                names = [f"class_{i}" for i in range(1, k + 1)]
            self.ident_class_map = pred
            self.ident_confidence = result.get("confidence")
            self.ident_class_names = names
            self.ident_num_classes = k
            self.ident_result = result

        if self.ident_class_map is None:
            return

        pred = np.asarray(self.ident_class_map)
        names = list(self.ident_class_names or [])
        k = int(self.ident_num_classes or max(int(np.nanmax(pred) or 0), 1))
        try:
            class_id = self._ident_class_filter_id()
        except ValueError:
            class_id = 0
        shown = filter_class_map(pred, class_id)

        self.current_param_img = shown.astype(float)
        if class_id > 0:
            label = names[class_id - 1] if 0 <= class_id - 1 < len(names) else f"class_{class_id}"
            title_str = f"Identification 矿物分类（仅类别 {class_id}  {label}）"
        else:
            title_str = "Identification 矿物分类（1.02–2.6 μm）"
        self.current_param_title = title_str

        self.fig_result.clf()
        self.ax_result = self.fig_result.add_subplot(111)
        self.marker_result = None

        base = self._identification_base_image(shown.shape)
        if base is not None:
            if np.ndim(base) == 2:
                self.ax_result.imshow(base, cmap="gray", interpolation="nearest")
            else:
                self.ax_result.imshow(base, interpolation="nearest")

        cmap, norm = make_cmap(k)
        masked = np.ma.masked_where(shown == 0, shown)
        cmap = cmap.copy()
        if base is not None:
            cmap.set_bad((0, 0, 0, 0))
            alpha = 0.55
        else:
            cmap.set_bad((0, 0, 0, 1))
            alpha = 1.0
        im = self.ax_result.imshow(
            masked, cmap=cmap, norm=norm, interpolation="nearest", alpha=alpha
        )
        self.ax_result.set_title(title_str)
        self._apply_image_layout(self.fig_result, self.ax_result, colorbar_mappable=im)
        try:
            cax = self.fig_result.axes[-1]
            ticks = np.arange(1, k + 1)
            cax.set_yticks(ticks)
            labels = class_tick_labels(names, include_background=False)
            cax.set_yticklabels(labels, fontsize=7)
        except Exception:
            pass

        if hasattr(self, "ident_class_filter_input") and self.ident_class_filter_input is not None:
            self.ident_class_filter_input.setPlaceholderText(
                f"0=全部，1–{k} 只显示一类"
            )
            self.ident_class_filter_input.setToolTip(
                f"输入 0 显示全部类别；输入 1–{k} 只叠加显示该矿物。"
            )

        if self.selected_pos is not None:
            r, c = self.selected_pos
            self.marker_result = self.ax_result.plot(
                c, r, "r+", markersize=12, markeredgewidth=2
            )[0]
        self.canvas_result.draw()

    def run_identification_model(self, model_num=None):
        """兼容旧入口：转到模型应用。"""
        self.identification_apply()

    def load_unmix_library(self):
        """加载端元光谱库：.mat（DAFA/TT）/ 多个 txt / 文件夹。"""
        from unmixing.library import load_library

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择端元光谱库",
            os.path.join(os.path.dirname(_APP_DIR), "data", "libraries"),
            "Spectral library (*.mat *.txt);;MATLAB (*.mat);;Text (*.txt);;All Files (*)",
        )
        # Also allow choosing a directory via a second prompt if cancelled? Offer dir button via question.
        if not path:
            reply = QMessageBox.question(
                self, "光谱库",
                "未选择文件。是否改为选择「含多个 txt 端元」的文件夹？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            path = QFileDialog.getExistingDirectory(
                self, "选择端元光谱文件夹",
                os.path.join(os.path.dirname(_APP_DIR), "data", "libraries"),
            )
            if not path:
                return
        try:
            lib = load_library(path)
            self.unmix_library_raw = lib
            self.unmix_library = None  # resample lazily when cube wavelengths known
            if self.wavelengths is not None:
                self.unmix_library = lib.resample(self.wavelengths)
            QMessageBox.information(
                self, "光谱库",
                f"已加载 {lib.n_endmembers} 个端元\n"
                f"波段：{lib.wavelengths.size}  "
                f"({lib.wavelengths[0]:.3f}–{lib.wavelengths[-1]:.3f} μm)\n"
                f"来源：{lib.source}\n\n"
                f"示例：{', '.join(lib.names[:5])}{'…' if lib.n_endmembers > 5 else ''}",
            )
            self.statusBar().showMessage(
                f"端元库：{lib.n_endmembers} 个 ({os.path.basename(path)})", 8000
            )
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def show_unmix_library_info(self):
        # Prefer Hapke endmembers if present
        if self.hapke_endmembers_raw:
            lines = []
            for em in self.hapke_endmembers_raw:
                kmin = float(np.nanmin(em.k)) if em.k is not None else float("nan")
                kmax = float(np.nanmax(em.k)) if em.k is not None else float("nan")
                inc = float(getattr(em, "lab_incidence_deg", 30.0))
                emi = float(getattr(em, "lab_emission_deg", 0.0))
                phase = float(getattr(em, "lab_phase_deg", 30.0))
                lines.append(
                    f"  {em.name}: ρ={em.density:g} g/cm³, n={em.n:g}, "
                    f"D={em.grain_size_um:g} μm, i={inc:g}°, e={emi:g}°, g={phase:g}°, "
                    f"k∈[{kmin:.3e},{kmax:.3e}]"
                )
            QMessageBox.information(
                self, "Hapke 端元",
                f"Excel：{os.path.basename(self.hapke_excel_path or '')}\n"
                f"已选用端元数：{len(self.hapke_endmembers_raw)}\n\n"
                + "\n".join(lines),
            )
            return
        lib = self.unmix_library or self.unmix_library_raw
        if lib is None:
            QMessageBox.information(self, "光谱库", "尚未加载 Sparse 光谱库或 Hapke Excel 端元。")
            return
        preview = "\n".join(f"  {i+1:3d}. {n}" for i, n in enumerate(lib.names[:40]))
        more = f"\n  … 共 {lib.n_endmembers} 个" if lib.n_endmembers > 40 else ""
        QMessageBox.information(
            self, "当前光谱库",
            f"端元数：{lib.n_endmembers}\n"
            f"波长：{lib.wavelengths[0]:.3f}–{lib.wavelengths[-1]:.3f} μm "
            f"({lib.wavelengths.size} bands)\n"
            f"来源：{lib.source}\n\n{preview}{more}",
        )

    def _ensure_unmix_library(self):
        if self.unmix_library_raw is None and self.unmix_library is None:
            # Try default DAFA/TT library shipped under data/libraries
            default = os.path.join(
                os.path.dirname(_APP_DIR), "data", "libraries", "TargetLibrary_paper.mat"
            )
            if os.path.isfile(default):
                reply = QMessageBox.question(
                    self, "光谱库",
                    "尚未加载端元库。是否使用自带的 DAFA/TT TargetLibrary_paper.mat"
                    "（蛇纹石 + 碳酸盐）？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    from unmixing.library import load_library
                    self.unmix_library_raw = load_library(default)
                else:
                    self.load_unmix_library()
            else:
                self.load_unmix_library()
        if self.unmix_library_raw is None and self.unmix_library is None:
            return False
        if self.wavelengths is None:
            QMessageBox.warning(self, "解混", "请先打开高光谱图像。")
            return False
        # Resample onto cube wavelengths
        raw = self.unmix_library_raw or self.unmix_library
        self.unmix_library = raw.resample(self.wavelengths)
        return True

    def _ask_unmix_options(self, title: str, default_method: str = "nnls"):
        """Return dict or None: scope, method, sparsity, sum_to_one, stride, wmin, wmax, i, e."""
        if self.current_data is None:
            QMessageBox.warning(self, title, "请先打开高光谱图像。")
            return None
        if not self._ensure_unmix_library():
            return None

        scope_items = ["当前选中像元（窗口平均）", "整图解混（可抽样）"]
        scope, ok = QInputDialog.getItem(self, title, "计算范围：", scope_items, 0, False)
        if not ok:
            return None
        is_image = scope.startswith("整图")

        methods = ["nnls (非负最小二乘)", "omp (稀疏 OMP)", "fcls (非负+归一)", "ucls (无约束)"]
        # pick default index
        def_idx = 0
        for i, m in enumerate(methods):
            if m.startswith(default_method):
                def_idx = i
                break
        method_label, ok = QInputDialog.getItem(self, title, "解混算法：", methods, def_idx, False)
        if not ok:
            return None
        method = method_label.split()[0].lower()

        sparsity = self.unmix_sparsity
        if method == "omp":
            sparsity, ok = QInputDialog.getInt(
                self, title, "稀疏度 K（最多选用几个端元）：",
                value=max(1, int(self.unmix_sparsity)), minValue=1,
                maxValue=min(20, self.unmix_library.n_endmembers),
            )
            if not ok:
                return None
            self.unmix_sparsity = int(sparsity)

        sum_to_one = True
        if method in ("nnls", "omp", "ucls"):
            reply = QMessageBox.question(
                self, title,
                "是否将丰度归一化（和为 1）？\n是：ASC 软约束  否：保留绝对尺度",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            sum_to_one = reply == QMessageBox.Yes

        stride = 1
        if is_image:
            stride, ok = QInputDialog.getInt(
                self, title, "空间步长（每隔 N 像元计算 1 个，加速整图）：",
                value=4, minValue=1, maxValue=50,
            )
            if not ok:
                return None

        # Optional wavelength window (CRISM IR mineral range default)
        wmin, wmax = 1.0, 2.6
        reply = QMessageBox.question(
            self, title,
            f"是否限制波长范围？\n是：使用 {wmin:.1f}–{wmax:.1f} μm（矿物常用）\n否：使用全部波段",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            wmin = wmax = None

        return {
            "is_image": is_image,
            "method": method,
            "sparsity": int(sparsity),
            "sum_to_one": bool(sum_to_one),
            "stride": int(stride),
            "wmin": wmin,
            "wmax": wmax,
        }

    def _unmix_endmember_matrix(self, wmin=None, wmax=None):
        lib = self.unmix_library
        A = lib.spectra.copy()
        mask = np.ones(A.shape[0], dtype=bool)
        if wmin is not None or wmax is not None:
            mask = lib.wavelength_mask(wmin, wmax)
        # also drop library columns that are all-nan in window
        return A, mask

    def _plot_unmix_fit(self, observed, reconstructed, title,
                        observed_label="原始 I/F", fitted_label="拟合 I/F",
                        ylabel="I/F"):
        """在右侧上方原始光谱区显示观测光谱与 Hapke 拟合光谱（均为 I/F）。"""
        self.fig_raw_spec.clf()
        self.ax_raw_spec = self.fig_raw_spec.add_subplot(111)
        self.raw_crosshair_vline = None
        self.raw_crosshair_hline = None
        self.raw_crosshair_text = None
        # 清除 DISORT 叠加状态，避免十字读数串台
        self.disort_wavelength = None
        self.disort_albedo = None
        self.disort_observed_if = None
        self.disort_model_if = None

        w = self.wavelengths
        self.ax_raw_spec.plot(w, observed, color="0.15", lw=1.3, label=observed_label)
        self.ax_raw_spec.plot(w, reconstructed, color="C3", lw=1.5, label=fitted_label)
        self.ax_raw_spec.set_title(title)
        self.ax_raw_spec.set_xlabel("Wavelength ($\\mu$m)")
        self.ax_raw_spec.set_ylabel(ylabel)
        self.ax_raw_spec.legend(loc="best", fontsize=9)
        self.ax_raw_spec.grid(True, alpha=0.3)
        self.current_raw_spectrum = np.asarray(observed, dtype=float)
        if not self.raw_ylim_locked:
            vals = np.concatenate([
                np.asarray(observed, dtype=float),
                np.asarray(reconstructed, dtype=float),
            ])
            self._apply_raw_spec_ylim(vals)
        self._sync_spectrum_axes()
        self.canvas_raw_spec.draw()

    def _format_abundance_text(self, abundance, names, top_n=12):
        ab = np.asarray(abundance, dtype=float)
        order = np.argsort(-np.nan_to_num(ab, nan=-1.0))
        lines = []
        for rank, i in enumerate(order[:top_n]):
            if not np.isfinite(ab[i]) or ab[i] <= 1e-6:
                continue
            name = names[i] if i < len(names) else f"EM{i+1}"
            lines.append(f"{rank+1:2d}. {name}: {ab[i]*100:.2f}%")
        if not lines:
            return "（无可报告丰度）"
        return "\n".join(lines)

    def _run_unmix_core(self, mode: str = "sparse"):
        """Sparse / linear library unmixing (not Hapke RT)."""
        opts = self._ask_unmix_options("Sparse unmixing", default_method="omp")
        if opts is None:
            return

        A_full, band_mask = self._unmix_endmember_matrix(opts["wmin"], opts["wmax"])
        if band_mask.sum() < 5:
            QMessageBox.warning(self, "解混", "波长窗口内有效波段过少。")
            return

        from unmixing.solvers import unmix_spectrum, unmix_cube

        names = self.unmix_library.names

        if not opts["is_image"]:
            if self.selected_pos is None:
                QMessageBox.information(self, "解混", "请先在假彩色图上点击选择像元。")
                return
            row, col = self.selected_pos
            spectrum, w_size = self._extract_window_spectrum(row, col)
            y = spectrum.copy()
            res = unmix_spectrum(
                y[band_mask],
                A_full[band_mask, :],
                method=opts["method"],
                sparsity=opts["sparsity"],
                sum_to_one=opts["sum_to_one"],
            )
            recon_full = np.full_like(y, np.nan)
            recon_full[band_mask] = res["reconstructed"]
            res["reconstructed"] = recon_full

            self.unmix_last_result = res
            self.unmix_method = opts["method"]
            title = f"Sparse unmix ({row},{col}) N={w_size}  RMSE={float(res['rmse']):.4g}"
            self._plot_unmix_fit(y, res["reconstructed"], title)
            ab_txt = self._format_abundance_text(res["abundance"], names)
            QMessageBox.information(
                self, "解混结果",
                f"像元 ({row}, {col})，窗口 {w_size}×{w_size}\n"
                f"算法：{res.get('method', opts['method'])}  RMSE={float(res['rmse']):.6f}\n\n"
                f"主要端元丰度：\n{ab_txt}",
            )
            return

        progress = QProgressDialog("整图解混中…", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        cancelled = {"v": False}

        def cb(done, total):
            if progress.wasCanceled():
                cancelled["v"] = True
                return
            progress.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        A_masked = A_full.copy()
        A_masked[~band_mask, :] = np.nan
        out = unmix_cube(
            self.current_data,
            A_masked,
            method=opts["method"],
            sparsity=opts["sparsity"],
            sum_to_one=opts["sum_to_one"],
            spatial_stride=opts["stride"],
            progress_cb=cb,
        )
        progress.close()
        if cancelled["v"]:
            QMessageBox.information(self, "解混", "已取消。")
            return

        self.unmix_abundance_cube = out["abundance"]
        self.unmix_rmse_map = out["rmse"]
        self.unmix_method = out.get("method", opts["method"])
        self.show_unmix_abundance_map(default_index=0)
        QMessageBox.information(
            self, "整图解混完成",
            f"算法：{self.unmix_method}\n"
            f"空间步长：{out.get('stride', opts['stride'])}\n"
            f"端元数：{self.unmix_abundance_cube.shape[2]}\n\n"
            "结果图已显示丰度。可用菜单「显示丰度图… / 显示 RMSE 图」切换。",
        )

    def open_hapke_hyperspectral(self):
        """Hapke：加载高光谱图像（辐亮度因子 I/F），显示在左侧上方。"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Hapke — 打开高光谱图像（I/F）",
            "",
            "ENVI Header (*.hdr);;All Files (*)",
        )
        if not filename:
            return
        try:
            self.open_file_path(filename, is_radiance=False)
            self.hapke_endmembers = None  # wavelengths may have changed
            self.hapke_band_mask_cached = None
            self.hapke_mode = None
            QMessageBox.information(
                self, "Hapke",
                "高光谱图像（辐亮度因子 I/F）已加载到左侧上方。\n\n"
                "注意：端元 Excel 为反射率因子 REFF；\n"
                "解混时用辅助立方体入射角 i，按 I/F = REFF×cos(i) 统一。\n\n"
                "请继续：加载辅助立方体 → 加载端元 Excel → 单光谱/图像处理。",
            )
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    def open_hapke_aux(self):
        """Hapke：加载辅助立方体（几何角），band13 显示在左下。"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Hapke — 打开辅助立方体",
            "",
            "CRISM DDR / PDS (*.lbl *.LBL *.img *.IMG);;ENVI Header (*.hdr);;All Files (*)",
        )
        if not filename:
            return
        try:
            aux, meta, lbl_path, img_path = self._load_aux_cube(filename)
            if aux.ndim != 3 or aux.shape[2] < 5:
                raise ValueError(
                    "辅助立方体至少需要几何波段："
                    "band1 入射角、band2 观测角、band3 相位角、band4 纬度、band5 经度。"
                )
            if self.current_data is not None:
                if aux.shape[0] != self.current_data.shape[0] or aux.shape[1] != self.current_data.shape[1]:
                    raise ValueError(
                        f"辅助图像尺寸 {aux.shape[:2]} 与高光谱图像 "
                        f"{self.current_data.shape[:2]} 不一致。"
                    )
            self.aux_data = aux
            self.aux_path = lbl_path
            self.aux_img_path = img_path
            self.aux_metadata = meta
            self._update_ls_from_aux_header()
            if aux.shape[2] >= 13:
                self._show_aux_local_time()
            else:
                # show incidence if no local-time band
                self.fig_result.clf()
                self.ax_result = self.fig_result.add_subplot(111)
                im = self.ax_result.imshow(aux[:, :, self.AUX_BAND_SOZ], cmap="viridis")
                self.ax_result.set_title("辅助信息 band1：太阳入射角 (°)")
                self.ax_result.axis("off")
                self.marker_result = None
                self.current_param_img = aux[:, :, self.AUX_BAND_SOZ]
                self.current_param_title = "Incidence (°)"
                self._apply_image_layout(self.fig_result, self.ax_result, colorbar_mappable=im)
                self.canvas_result.draw()
            QMessageBox.information(
                self, "Hapke",
                "辅助立方体已加载。\n"
                "几何角将用于 Hapke 单光谱 / 图像处理：\n"
                "  band1 太阳入射角 i（用于 I/F = REFF×cos(i)）\n"
                "  band2 观测角 e（Hapke 几何）。",
            )
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    def _ensure_hapke_ready(self, need_pixel=False):
        """Ensure hyperspectral + aux + Excel endmembers (with k) are ready."""
        if self.current_data is None or self.wavelengths is None:
            QMessageBox.information(self, "Hapke", "请先加载高光谱图像。")
            self.open_hapke_hyperspectral()
            if self.current_data is None:
                return False
        if self.aux_data is None:
            QMessageBox.information(self, "Hapke", "请先加载辅助立方体（提供入射角/观测角）。")
            self.open_hapke_aux()
            if self.aux_data is None:
                return False
        if self.aux_data.shape[0] != self.current_data.shape[0] or \
           self.aux_data.shape[1] != self.current_data.shape[1]:
            QMessageBox.critical(self, "Hapke", "高光谱与辅助立方体像元尺寸不一致。")
            return False
        if not self._ensure_hapke_endmembers():
            return False
        return True

    def _hapke_band_mask(self, ask=True):
        """Wavelength window for Hapke fitting; cache for click mode."""
        if self.hapke_band_mask_cached is not None and not ask:
            return self.hapke_band_mask_cached
        wmin, wmax = 1.0, 2.6
        use_window = True
        if ask:
            reply = QMessageBox.question(
                self, "Hapke",
                f"是否限制波长范围到 {wmin:.1f}–{wmax:.1f} μm？\n是：限制  否：全部波段",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            use_window = reply == QMessageBox.Yes
        band_mask = np.isfinite(self.wavelengths)
        if use_window:
            band_mask &= (self.wavelengths >= wmin) & (self.wavelengths <= wmax)
        for em in self.hapke_endmembers:
            band_mask &= np.isfinite(em.ssa) & np.isfinite(em.k)
        if band_mask.sum() < 5:
            QMessageBox.warning(self, "Hapke", "有效波段过少，请检查端元与波长范围。")
            return None
        self.hapke_band_mask_cached = band_mask
        return band_mask

    def hapke_single_spectrum(self):
        """
        进入 Hapke 单光谱模式：
        在左侧图像点击像元后，用已加载 Excel 端元解算矿物比例，
        并在右侧上方显示原始光谱与拟合光谱。
        不再重新选择/加载 Excel。
        """
        if not self._ensure_hapke_ready(need_pixel=False):
            return
        band_mask = self._hapke_band_mask(ask=True)
        if band_mask is None:
            return

        self.hapke_mode = "single"
        self.ratio_mode = None
        self.disort_mode = None
        excel_name = os.path.basename(self.hapke_excel_path or "") or "(已加载)"
        n_em = len(self.hapke_endmembers_raw or [])
        QMessageBox.information(
            self, "Hapke 单光谱计算",
            "已进入单光谱模式（直接使用已加载端元，不再重新打开 Excel）。\n\n"
            f"当前端元文件：{excel_name}\n"
            f"端元数量：{n_em}\n\n"
            "量纲约定：\n"
            "• 图像像元 = 辐亮度因子 I/F\n"
            "• 端元光谱 = 反射率因子 REFF\n"
            "• I/F = REFF × cos(i)，i 取自辅助立方体 band1\n\n"
            "请在左侧假彩色图上点击像元：\n"
            "• Hapke 非线性解混（拟合在 I/F 空间）\n"
            "• 右侧上方显示：原始 I/F + 拟合 I/F\n"
            "• 弹窗给出各矿物质量比例\n\n"
            "双击图像或菜单「退出 Hapke 单光谱模式」可退出。",
        )
        self.statusBar().showMessage(
            f"Hapke 单光谱模式：使用端元 {excel_name}，点击左侧图像像元计算", 0
        )

    def _hapke_run_single_pixel(self, row, col):
        """对点击像元做 Hapke 解混，显示进度，并绘制原始/拟合光谱。"""
        from unmixing.hapke_rt import fit_mass_fractions

        # 仅复用已加载端元；未加载时提示，不弹 Excel 对话框
        if not self._ensure_hapke_endmembers():
            return
        if self.aux_data is None:
            QMessageBox.warning(self, "Hapke", "缺少辅助立方体。")
            return

        band_mask = self._hapke_band_mask(ask=False)
        if band_mask is None:
            return

        self.selected_pos = (row, col)
        self.update_image_markers(row, col)

        soz, voz, phase, lat, lon, loct = self._aux_geometry_at(row, col)
        if not (np.isfinite(soz) and np.isfinite(voz)):
            QMessageBox.warning(
                self, "Hapke",
                f"像元 ({row},{col}) 辅助几何无效（入射角/观测角）。",
            )
            return
        inc, emi = float(soz), float(voz)

        progress = QProgressDialog(
            f"Hapke 单光谱计算中…\n像元 ({row}, {col})",
            None, 0, 100, self,
        )
        progress.setWindowTitle("计算进度")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setValue(5)
        progress.setLabelText(f"提取光谱…  像元 ({row}, {col})")
        QApplication.processEvents()

        spectrum, w_size = self._extract_window_spectrum(row, col)
        names = [em.name for em in self.hapke_endmembers]
        progress.setValue(20)
        progress.setLabelText(
            f"非线性最小二乘拟合中…\n"
            f"像元 ({row}, {col})，端元 {len(names)} 个"
        )
        QApplication.processEvents()

        nfev_state = {"n": 0, "max": max(50, 200 * len(names))}

        def _fit_progress(nfev, max_nfev):
            nfev_state["n"] = int(nfev)
            nfev_state["max"] = max(int(max_nfev), 1)
            pct = 20 + int(70 * min(nfev, max_nfev) / max_nfev)
            progress.setValue(min(90, pct))
            progress.setLabelText(
                f"非线性最小二乘拟合中… ({nfev}/{max_nfev})\n"
                f"像元 ({row}, {col})"
            )
            QApplication.processEvents()

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            res = fit_mass_fractions(
                spectrum,
                self.hapke_endmembers,
                incidence_deg=inc,
                emission_deg=emi,
                band_mask=band_mask,
                progress_cb=_fit_progress,
            )
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Hapke 解算失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        progress.setValue(95)
        progress.setLabelText("绘制原始光谱与拟合光谱…")
        QApplication.processEvents()

        self.unmix_last_result = res
        self.unmix_method = "hapke_nls"
        title = (
            f"Hapke 单光谱 ({row},{col})  窗口{w_size}×{w_size}  "
            f"RMSE={float(res['rmse']):.4g}"
        )
        self._plot_unmix_fit(spectrum, res["reconstructed"], title)
        progress.setValue(100)
        progress.close()

        ab_txt = self._format_abundance_text(res["abundance"], names)
        self.statusBar().showMessage(
            f"Hapke ({row},{col}) RMSE={float(res['rmse']):.4g} | "
            + ab_txt.replace("\n", "；"),
            15000,
        )
        mu0 = float(np.cos(np.radians(inc)))
        QMessageBox.information(
            self, "矿物比例",
            f"像元 ({row}, {col})，窗口 {w_size}×{w_size}\n"
            f"几何：i={inc:.2f}°, e={emi:.2f}°, g={phase:.2f}°\n"
            f"μ0=cos(i)={mu0:.4f}（I/F = REFF×μ0）\n"
            f"RMSE(I/F)={float(res['rmse']):.6f}\n"
            f"端元(REFF)：{os.path.basename(self.hapke_excel_path or '')}\n\n"
            f"矿物质量丰度：\n{ab_txt}\n\n"
            "右侧上方已显示：原始 I/F（黑）与拟合 I/F（红）。",
        )

    def exit_hapke_mode(self):
        was = self.hapke_mode
        self.hapke_mode = None
        self.hapke_band_mask_cached = None
        if was:
            self.statusBar().showMessage("已退出 Hapke 单光谱模式", 5000)

    def hapke_image_mode(self):
        """图像处理：整图 Hapke 非线性最小二乘；几何取自辅助立方体。"""
        self.hapke_mode = None
        if not self._ensure_hapke_ready(need_pixel=False):
            return
        from unmixing.hapke_rt import fit_cube_mass_fractions

        band_mask = self._hapke_band_mask(ask=True)
        if band_mask is None:
            return

        stride, ok = QInputDialog.getInt(
            self, "Hapke 图像处理",
            "空间步长（每隔 N 像元计算 1 个，其余置空）：",
            value=4, minValue=1, maxValue=50,
        )
        if not ok:
            return

        names = [em.name for em in self.hapke_endmembers]
        progress = QProgressDialog("Hapke 图像处理中…", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        cancelled = {"v": False}

        def cb(done, total):
            if progress.wasCanceled():
                cancelled["v"] = True
                return
            progress.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        def per_pix_geom(r, c):
            soz, voz, *_ = self._aux_geometry_at(r, c)
            if not (np.isfinite(soz) and np.isfinite(voz)):
                return np.nan, np.nan
            return float(soz), float(voz)

        try:
            out = fit_cube_mass_fractions(
                self.current_data,
                self.hapke_endmembers,
                incidence_deg=30.0,
                emission_deg=0.0,
                spatial_stride=int(stride),
                band_mask=band_mask,
                progress_cb=cb,
                per_pixel_geometry=per_pix_geom,
            )
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Hapke 图像处理失败", str(exc))
            return
        progress.close()
        if cancelled["v"]:
            QMessageBox.information(self, "Hapke", "已取消。")
            return

        self.unmix_abundance_cube = out["abundance"]
        self.unmix_rmse_map = out["rmse"]
        self.unmix_method = "hapke_nls"
        self.unmix_library = type("L", (), {"names": names})()
        self.show_unmix_abundance_map(default_index=0)
        QMessageBox.information(
            self, "Hapke 图像处理完成",
            f"端元数：{len(names)}\n"
            f"空间步长：{stride}\n"
            f"观测：图像 I/F；端元：REFF；I/F = REFF×cos(i)\n"
            f"几何：逐像元取自辅助立方体（band1 入射角，band2 观测角）\n\n"
            "可用「显示丰度图… / 显示 RMSE 图」切换。",
        )

    def run_hapke_unmixing(self):
        """兼容旧入口：转到单光谱计算模式。"""
        self.hapke_single_spectrum()

    def _plot_hapke_background_spectrum(self, em):
        """在右下方比值光谱显示区绘制图像背景端元（I/F）。"""
        if em is None:
            return
        w = np.asarray(getattr(em, "wavelengths", None), dtype=float)
        y = np.asarray(getattr(em, "reflectance", None), dtype=float)
        if w.size == 0 or y.size == 0 or w.shape != y.shape:
            return

        n_pix = None
        src = getattr(em, "source", "") or ""
        if "featureless_n=" in src:
            try:
                n_pix = int(src.split("featureless_n=")[-1].split(":")[0])
            except Exception:
                n_pix = None
        title = "图像背景端元（无特征像元平均 I/F）"
        if n_pix is not None:
            title = f"图像背景端元（无特征像元平均 I/F，n={n_pix}）"

        self._clear_ratio_plot(title=title, show_y_labels=True)
        self.ax_ratio_spec.set_ylabel("I/F")
        self.ax_ratio_spec.plot(
            w, y, color="C0", lw=1.4, label=em.name or "图像背景",
        )
        self.ax_ratio_spec.legend(fontsize=8, loc="best")
        self.ax_ratio_spec.grid(True, linestyle="--", alpha=0.5)
        self.current_ratio_spectrum = y.copy()
        self.relab_overlay = None
        self._set_ylim_from_data(self.ax_ratio_spec, y)
        self._sync_spectrum_axes()
        self.canvas_ratio_spec.draw()
        self.statusBar().showMessage("已在右下方显示图像背景端元光谱", 8000)

    def _build_hapke_background_endmember(self):
        """
        由图像无特征像元构建背景端元：
        - 光谱：无特征像元 I/F 均值（保持 I/F，解混时不再 ×cos(i)）
        - i/e/g：无特征像元辅助几何均值
        - ρ=3，n=1.8，D=10 μm
        """
        from unmixing.hapke_rt import HapkeEndmember

        if self.current_data is None or self.wavelengths is None:
            return None, "请先加载高光谱图像，才能生成「图像背景」端元。"

        try:
            if self.auto_featureless_mask is None:
                featureless, _, _ = self._prepare_auto_ratio_denominators()
            else:
                featureless = self.auto_featureless_mask
        except Exception as exc:
            return None, f"识别无特征像元失败：{exc}"

        n_feat = int(np.count_nonzero(featureless))
        if n_feat < 1:
            return None, "未找到无特征像元，无法生成「图像背景」端元。"

        # 剔除反射率 <0 或 >1 的像元：任一有限波段越界则整像元不参与平均
        data = np.asarray(self.current_data, dtype=float)
        with np.errstate(all="ignore"):
            finite = np.isfinite(data)
            in_range = (data >= 0.0) & (data <= 1.0)
            # 有有限值，且所有有限波段都在 [0, 1]
            reflectance_ok = np.any(finite, axis=2) & np.all((~finite) | in_range, axis=2)
            bg_mask = featureless & reflectance_ok

        n_pix = int(np.count_nonzero(bg_mask))
        n_rejected = n_feat - n_pix
        if n_pix < 1:
            return None, (
                f"无特征像元 {n_feat} 个，但反射率均不在 [0, 1] 内，"
                "无法生成「图像背景」端元。"
            )

        with np.errstate(all="ignore"):
            mean_iff = np.nanmean(data[bg_mask], axis=0)

        if self.aux_data is not None and self.aux_data.ndim == 3:
            soz = self.aux_data[:, :, self.AUX_BAND_SOZ]
            voz = self.aux_data[:, :, self.AUX_BAND_VOZ]
            phase = self.aux_data[:, :, self.AUX_BAND_PHASE]
            with np.errstate(all="ignore"):
                mean_i = float(np.nanmean(soz[bg_mask]))
                mean_e = float(np.nanmean(voz[bg_mask]))
                mean_g = float(np.nanmean(phase[bg_mask]))
            if not np.isfinite(mean_i):
                mean_i = 30.0
            if not np.isfinite(mean_e):
                mean_e = 0.0
            if not np.isfinite(mean_g):
                mean_g = 30.0
        else:
            mean_i, mean_e, mean_g = 30.0, 0.0, 30.0

        # 直接使用图像 I/F 均值；解混前向中对背景端元不再乘 cos(i)
        em = HapkeEndmember(
            name="图像背景",
            wavelengths=np.asarray(self.wavelengths, dtype=float).copy(),
            reflectance=np.asarray(mean_iff, dtype=float),
            density=3.0,
            n=1.8,
            grain_size_um=10.0,
            spectrum_id="BG",
            lab_incidence_deg=float(mean_i),
            lab_emission_deg=float(mean_e),
            lab_phase_deg=float(mean_g),
            source=(
                f"image_background_iff:featureless_n={n_pix}"
                f":rejected_out_of_range={n_rejected}"
            ),
            selected=True,
            is_background=True,
        )
        return em, None

    def load_hapke_excel_endmembers(self):
        """加载 Hapke 端元矿物反射率 Excel，并录入密度 / n / 粒径，反演 k(λ)。"""
        from unmixing.excel_endmembers import load_endmembers_excel, write_endmember_template
        from unmixing.dialogs import HapkeEndmemberParamDialog
        from unmixing.hapke_rt import prepare_endmembers_k

        lib_dir = os.path.join(os.path.dirname(_APP_DIR), "data", "libraries")
        os.makedirs(lib_dir, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择端元矿物反射率因子 REFF Excel",
            lib_dir,
            "Excel (*.xlsx *.xls);;All Files (*)",
        )
        if not path:
            reply = QMessageBox.question(
                self, "Hapke 端元",
                "未选择文件。是否生成一份 Excel 模板以便填写？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                tmpl = os.path.join(lib_dir, "hapke_endmembers_template.xlsx")
                write_endmember_template(tmpl)
                QMessageBox.information(
                    self, "模板已生成",
                    f"已写入：\n{tmpl}\n\n"
                    "表格排布：\n"
                    "  第1行：矿物名称\n"
                    "  第2行：光谱ID\n"
                    "  第3行：平均粒径 D (μm)\n"
                    "  第4行：密度 ρ\n"
                    "  第5行：折射率实部 n\n"
                    "  第6行起：第1列波长，其后各列为 REFF\n\n"
                    "填写后请重新选择该文件。",
                )
            return False
        try:
            ems = load_endmembers_excel(path)
        except Exception as exc:
            QMessageBox.critical(self, "读取 Excel 失败", str(exc))
            return False

        # 追加图像背景端元（无特征像元平均光谱）作为面板最后一行，并画到右下角
        bg_em, bg_err = self._build_hapke_background_endmember()
        if bg_em is None:
            self.hapke_background_endmember = None
            reply = QMessageBox.question(
                self, "图像背景端元",
                f"{bg_err}\n\n是否仍继续打开参数面板（不含背景端元）？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False
            ems_for_dlg = list(ems)
        else:
            self.hapke_background_endmember = bg_em
            self._plot_hapke_background_spectrum(bg_em)
            ems_for_dlg = list(ems) + [bg_em]

        dlg = HapkeEndmemberParamDialog(
            ems_for_dlg, self,
            lab_incidence_deg=30.0,
            lab_emission_deg=0.0,
            lab_phase_deg=30.0,
        )
        dlg_ret = dlg.exec()
        if not _dialog_accepted(dlg_ret):
            self.statusBar().showMessage("已取消加载 Hapke 端元 Excel", 5000)
            return False
        ems = dlg.result_endmembers()  # 仅勾选的端元（含各自 i/e）
        if not ems:
            QMessageBox.warning(self, "Hapke", "未选用任何端元。")
            return False

        # 先保存物理参数，避免后续 k 反演异常时状态丢失
        self.hapke_excel_path = path
        # 兼容旧状态字段：记录首个选用端元的实验室几何
        self.hapke_lab_incidence = float(ems[0].lab_incidence_deg)
        self.hapke_lab_emission = float(ems[0].lab_emission_deg)
        self.hapke_endmembers_raw = list(ems)
        self.hapke_endmembers = None
        self.hapke_band_mask_cached = None

        try:
            # 每个矿物用各自的 REFF 测量入射角/发射角反演 k
            ems_k = prepare_endmembers_k(ems)
        except Exception as exc:
            QMessageBox.critical(
                self, "k(λ) 反演失败",
                f"{exc}\n\n已保留 Excel 与物理参数，可稍后重试单光谱计算。",
            )
            self.statusBar().showMessage(
                f"Hapke 端元参数已保存但 k 未反演 ({os.path.basename(path)})", 10000
            )
            return True

        self.hapke_endmembers_raw = ems_k
        lines = []
        for em in ems_k:
            kmin = float(np.nanmin(em.k)) if em.k is not None else float("nan")
            kmax = float(np.nanmax(em.k)) if em.k is not None else float("nan")
            lines.append(
                f"  {em.name}: ρ={em.density:g}, n={em.n:g}, D={em.grain_size_um:g} μm, "
                f"i={em.lab_incidence_deg:g}°, e={em.lab_emission_deg:g}°, "
                f"g={getattr(em, 'lab_phase_deg', 30.0):g}°, "
                f"k∈[{kmin:.3e}, {kmax:.3e}]"
            )
        QMessageBox.information(
            self, "Hapke 端元已就绪",
            f"文件：{os.path.basename(path)}\n"
            f"已选用端元数：{len(ems_k)}\n\n"
            "已反演各矿物折射率虚部 k(λ)：\n" + "\n".join(lines) +
            "\n\n仅勾选的端元将用于单光谱计算 / 图像处理。",
        )
        self.statusBar().showMessage(
            f"Hapke 已选用端元：{len(ems_k)} 个 ({os.path.basename(path)})", 0
        )
        return True

    def _ensure_hapke_endmembers(self):
        """
        复用「加载端元反射率 Excel」已得到的端元（含 k）；
        未加载时提示，并可选择立即加载（不再在每次单光谱时强制重选）。
        """
        if not self.hapke_endmembers_raw:
            reply = QMessageBox.warning(
                self, "Hapke",
                "尚未加载端元矿物反射率 Excel（或上次未点「确定」保存参数）。\n\n"
                "请先执行：Unmixing → Hapke model → 加载端元反射率 Excel…\n"
                "在参数对话框中输入 ρ / n / D 后必须点击「确定」。\n\n"
                "是否现在加载？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.load_hapke_excel_endmembers()
            if not self.hapke_endmembers_raw:
                return False
        if self.wavelengths is None:
            QMessageBox.warning(self, "Hapke", "请先打开高光谱图像。")
            return False

        # 若仅有物理参数、尚无 k，则按各矿物自己的 i/e 补做反演
        if any(em.k is None for em in self.hapke_endmembers_raw):
            from unmixing.hapke_rt import prepare_endmembers_k
            try:
                self.hapke_endmembers_raw = prepare_endmembers_k(
                    self.hapke_endmembers_raw
                )
                self.hapke_endmembers = None
            except Exception as exc:
                QMessageBox.critical(self, "k(λ) 反演失败", str(exc))
                return False

        # 已按当前波长准备好则直接复用，避免重复重采样
        if self.hapke_endmembers is not None:
            return True
        self.hapke_endmembers = [
            em.resample(self.wavelengths) for em in self.hapke_endmembers_raw
        ]
        from unmixing.hapke_rt import ssa_from_k
        for em in self.hapke_endmembers:
            if em.k is not None:
                em.ssa = ssa_from_k(em.k, em.wavelengths, em.n, em.grain_size_um)
        return True

    def export_hapke_k(self):
        ems = self.hapke_endmembers_raw
        if not ems:
            QMessageBox.information(self, "导出 k(λ)", "请先加载端元 Excel 并完成 k 反演。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Hapke k(λ)", "hapke_k_spectra.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            import pandas as pd
            data = {"wavelength_um": ems[0].wavelengths}
            for em in ems:
                data[f"{em.name}_k"] = em.k
                data[f"{em.name}_ssa"] = em.ssa
                data[f"{em.name}_R"] = em.reflectance
            pd.DataFrame(data).to_excel(path, index=False)
            QMessageBox.information(self, "导出完成", path)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    # ================= Sparse unmixing (SUNSAL in SSA space) =================
    def open_sparse_hyperspectral(self):
        """Sparse：加载高光谱图像（I/F）。"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Sparse — 打开高光谱图像（I/F）",
            "",
            "ENVI Header (*.hdr);;All Files (*)",
        )
        if not filename:
            return
        try:
            self.open_file_path(filename, is_radiance=False)
            self.sparse_endmember_ssa = None
            self.sparse_band_mask_cached = None
            self.sparse_mode = None
            QMessageBox.information(
                self, "Sparse",
                "高光谱图像（I/F）已加载。\n\n"
                "请继续：加载辅助立方体（推荐，提供入射角）→ "
                "加载端元 Excel → 单光谱/图像处理。\n"
                "解混在单次散射反照率（SSA）空间用 SUNSAL 进行。",
            )
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    def open_sparse_aux(self):
        """Sparse：加载辅助立方体（入射角/观测角）。"""
        # 复用 Hapke 辅助立方体加载
        self.open_hapke_aux()

    def load_sparse_excel_endmembers(self):
        """加载 Sparse 端元 Excel（第1列波长，其后各列为反射率），并转为 SSA。"""
        from unmixing.excel_endmembers import load_sparse_endmembers_excel
        from unmixing.sparse_ssa import endmember_reff_to_ssa

        if self.current_data is None or self.wavelengths is None:
            QMessageBox.information(self, "Sparse", "请先加载高光谱图像。")
            self.open_sparse_hyperspectral()
            if self.current_data is None:
                return False

        lib_dir = os.path.join(os.path.dirname(_APP_DIR), "data", "libraries")
        os.makedirs(lib_dir, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Sparse 端元反射率 Excel",
            lib_dir,
            "Excel (*.xlsx *.xls);;All Files (*)",
        )
        if not path:
            return False
        try:
            lib = load_sparse_endmembers_excel(path)
        except Exception as exc:
            QMessageBox.critical(self, "读取 Excel 失败", str(exc))
            return False

        lab_i, ok = QInputDialog.getDouble(
            self, "端元测量几何",
            "端元反射率测量入射角 i (°)：",
            value=float(self.sparse_lab_incidence),
            minValue=0.0, maxValue=89.0, decimals=2,
        )
        if not ok:
            return False
        lab_e, ok = QInputDialog.getDouble(
            self, "端元测量几何",
            "端元反射率测量发射角 e (°)：",
            value=float(self.sparse_lab_emission),
            minValue=0.0, maxValue=89.0, decimals=2,
        )
        if not ok:
            return False

        lib_cube = lib.resample(self.wavelengths)
        try:
            A_ssa = endmember_reff_to_ssa(
                lib_cube.spectra,
                incidence_deg=float(lab_i),
                emission_deg=float(lab_e),
            )
        except Exception as exc:
            QMessageBox.critical(self, "SSA 转换失败", str(exc))
            return False

        self.sparse_library_raw = lib
        self.unmix_library_raw = lib
        self.unmix_library = lib_cube
        self.sparse_endmember_ssa = A_ssa
        self.sparse_excel_path = path
        self.sparse_lab_incidence = float(lab_i)
        self.sparse_lab_emission = float(lab_e)
        self.sparse_band_mask_cached = None

        QMessageBox.information(
            self, "Sparse 端元已就绪",
            f"文件：{os.path.basename(path)}\n"
            f"端元数：{lib.n_endmembers}\n"
            f"实验室几何：i={lab_i:.1f}°, e={lab_e:.1f}°\n"
            f"已重采样到图像波长并转换为 SSA。\n\n"
            f"示例：{', '.join(lib.names[:6])}"
            f"{'…' if lib.n_endmembers > 6 else ''}\n\n"
            "请继续：单光谱计算 或 图像处理。",
        )
        self.statusBar().showMessage(
            f"Sparse 端元 {lib.n_endmembers} 个，已转 SSA ({os.path.basename(path)})", 0
        )
        return True

    def _ensure_sparse_ready(self):
        if self.current_data is None or self.wavelengths is None:
            QMessageBox.information(self, "Sparse", "请先加载高光谱图像。")
            self.open_sparse_hyperspectral()
            if self.current_data is None:
                return False
        if self.sparse_endmember_ssa is None:
            reply = QMessageBox.warning(
                self, "Sparse",
                "尚未加载端元 Excel / 未完成 SSA 转换。\n\n是否现在加载？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.load_sparse_excel_endmembers()
            if self.sparse_endmember_ssa is None:
                return False
        if self.aux_data is not None:
            if (self.aux_data.shape[0] != self.current_data.shape[0] or
                    self.aux_data.shape[1] != self.current_data.shape[1]):
                QMessageBox.critical(self, "Sparse", "高光谱与辅助立方体像元尺寸不一致。")
                return False
        return True

    def _sparse_band_mask(self, ask=True):
        if self.sparse_band_mask_cached is not None and not ask:
            return self.sparse_band_mask_cached
        wmin, wmax = 1.0, 2.6
        use_window = True
        if ask:
            reply = QMessageBox.question(
                self, "Sparse",
                f"是否限制波长范围到 {wmin:.1f}–{wmax:.1f} μm？\n是：限制  否：全部波段",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            use_window = reply == QMessageBox.Yes
        band_mask = np.isfinite(self.wavelengths)
        if use_window:
            band_mask &= (self.wavelengths >= wmin) & (self.wavelengths <= wmax)
        A = self.sparse_endmember_ssa
        if A is not None:
            band_mask &= np.all(np.isfinite(A), axis=1)
        if band_mask.sum() < 5:
            QMessageBox.warning(self, "Sparse", "有效波段过少，请检查端元与波长范围。")
            return None
        self.sparse_band_mask_cached = band_mask
        return band_mask

    def _sparse_ask_params(self):
        """弹出 SUNSAL 参数对话框，成功则写回实例属性并返回参数字典。"""
        from unmixing.dialogs import SunsalParamDialog

        dlg = SunsalParamDialog(
            self,
            lambda_=float(self.sparse_lambda),
            positivity=bool(self.sparse_positivity),
            addone=bool(self.sparse_addone),
            al_iters=int(self.sparse_al_iters),
            tol=float(self.sparse_tol),
        )
        if not _dialog_accepted(dlg.exec()):
            return None
        params = dlg.params()
        self.sparse_lambda = float(params["lambda"])
        self.sparse_positivity = bool(params["positivity"])
        self.sparse_addone = bool(params["addone"])
        self.sparse_al_iters = int(params["al_iters"])
        self.sparse_tol = float(params["tol"])
        return params

    def _sparse_params_summary(self, params=None):
        p = params or {
            "lambda": self.sparse_lambda,
            "positivity": self.sparse_positivity,
            "addone": self.sparse_addone,
            "al_iters": self.sparse_al_iters,
            "tol": self.sparse_tol,
        }
        constraints = []
        constraints.append("非负" if p["positivity"] else "允许负值")
        constraints.append("和为1" if p["addone"] else "不约束和为1")
        return (
            f"λ={float(p['lambda']):g}，{ '，'.join(constraints) }，"
            f"最大迭代={int(p['al_iters'])}，TOL={float(p['tol']):g}"
        )

    def sparse_single_spectrum(self):
        """进入 Sparse 单光谱模式：点击像元 → I/F→SSA → SUNSAL。"""
        if not self._ensure_sparse_ready():
            return
        band_mask = self._sparse_band_mask(ask=True)
        if band_mask is None:
            return
        params = self._sparse_ask_params()
        if params is None:
            return

        self.sparse_mode = "single"
        self.hapke_mode = None
        self.ratio_mode = None
        self.disort_mode = None
        n_em = self.sparse_endmember_ssa.shape[1]
        excel_name = os.path.basename(self.sparse_excel_path or "") or "(已加载)"
        geom = (
            "辅助立方体逐像元入射角/观测角"
            if self.aux_data is not None
            else "默认 i=30°, e=0°（未加载辅助立方体）"
        )
        summary = self._sparse_params_summary(params)
        QMessageBox.information(
            self, "Sparse 单光谱计算",
            "已进入 Sparse 单光谱模式。\n\n"
            f"端元：{excel_name}（{n_em} 个，已转 SSA）\n"
            f"参数：{summary}\n"
            f"几何：{geom}\n\n"
            "点击左侧假彩色图像元：\n"
            "• 图像 I/F → REFF → SSA\n"
            "• SUNSAL 稀疏解混\n"
            "• 右上显示原始 I/F 与重建 I/F\n\n"
            "双击图像或菜单「退出 Sparse 单光谱模式」可退出。",
        )
        self.statusBar().showMessage(
            f"Sparse 单光谱模式：{summary}，点击左侧图像计算", 0
        )

    def _sparse_run_single_pixel(self, row, col):
        from unmixing.sparse_ssa import sparse_unmix_ssa

        if not self._ensure_sparse_ready():
            return
        band_mask = self._sparse_band_mask(ask=False)
        if band_mask is None:
            return

        self.selected_pos = (row, col)
        self.update_image_markers(row, col)

        if self.aux_data is not None:
            soz, voz, phase, *_ = self._aux_geometry_at(row, col)
            if not (np.isfinite(soz) and np.isfinite(voz)):
                QMessageBox.warning(
                    self, "Sparse",
                    f"像元 ({row},{col}) 辅助几何无效。",
                )
                return
            inc, emi = float(soz), float(voz)
        else:
            inc, emi = 30.0, 0.0
            phase = float("nan")

        spectrum, w_size = self._extract_window_spectrum(row, col)
        names = list(self.unmix_library.names) if self.unmix_library else [
            f"EM{i+1}" for i in range(self.sparse_endmember_ssa.shape[1])
        ]

        progress = QProgressDialog(
            f"Sparse SUNSAL 计算中…\n像元 ({row}, {col})",
            None, 0, 0, self,
        )
        progress.setWindowTitle("计算进度")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            res = sparse_unmix_ssa(
                spectrum,
                self.sparse_endmember_ssa,
                incidence_deg=inc,
                emission_deg=emi,
                lambda_=float(self.sparse_lambda),
                positivity=bool(self.sparse_positivity),
                addone=bool(self.sparse_addone),
                al_iters=int(self.sparse_al_iters),
                tol=float(self.sparse_tol),
                band_mask=band_mask,
            )
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Sparse 解算失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()

        self.unmix_last_result = res
        self.unmix_method = "sunsal_ssa"
        title = (
            f"Sparse SUNSAL ({row},{col})  窗口{w_size}×{w_size}  "
            f"RMSE={float(res['rmse']):.4g}"
        )
        self._plot_unmix_fit(
            spectrum, res["reconstructed_iff"], title,
            observed_label="原始 I/F", fitted_label="拟合 I/F", ylabel="I/F",
        )
        ab_txt = self._format_abundance_text(res["abundance"], names)
        phase_txt = f", g={phase:.2f}°" if np.isfinite(phase) else ""
        self.statusBar().showMessage(
            f"Sparse ({row},{col}) RMSE={float(res['rmse']):.4g} | "
            + ab_txt.replace("\n", "；"),
            15000,
        )
        QMessageBox.information(
            self, "矿物比例",
            f"像元 ({row}, {col})，窗口 {w_size}×{w_size}\n"
            f"几何：i={inc:.2f}°, e={emi:.2f}°{phase_txt}\n"
            f"{self._sparse_params_summary()}\n"
            f"SUNSAL iters={int(res.get('n_iter', 0))}  "
            f"res_p={float(res.get('res_p', np.nan)):.4g}  "
            f"res_d={float(res.get('res_d', np.nan)):.4g}\n"
            f"RMSE(I/F)={float(res['rmse']):.6f}\n"
            f"端元：{os.path.basename(self.sparse_excel_path or '')}\n\n"
            f"丰度：\n{ab_txt}\n\n"
            "右侧上方：原始 I/F（黑）与拟合 I/F（红）。",
        )

    def exit_sparse_mode(self):
        was = self.sparse_mode
        self.sparse_mode = None
        self.sparse_band_mask_cached = None
        if was:
            self.statusBar().showMessage("已退出 Sparse 单光谱模式", 5000)

    def sparse_image_mode(self):
        """整图 Sparse SUNSAL（SSA 空间）。"""
        self.sparse_mode = None
        if not self._ensure_sparse_ready():
            return
        from unmixing.sparse_ssa import sparse_unmix_cube_ssa

        band_mask = self._sparse_band_mask(ask=True)
        if band_mask is None:
            return
        params = self._sparse_ask_params()
        if params is None:
            return
        stride, ok = QInputDialog.getInt(
            self, "Sparse 图像处理",
            "空间步长（每隔 N 像元计算 1 个，其余置空）：",
            value=4, minValue=1, maxValue=50,
        )
        if not ok:
            return

        names = list(self.unmix_library.names) if self.unmix_library else [
            f"EM{i+1}" for i in range(self.sparse_endmember_ssa.shape[1])
        ]
        progress = QProgressDialog("Sparse 图像处理中…", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        cancelled = {"v": False}

        def cb(done, total):
            if progress.wasCanceled():
                cancelled["v"] = True
                return
            progress.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        per_pix = None
        if self.aux_data is not None:
            def per_pix(r, c):
                soz, voz, *_ = self._aux_geometry_at(r, c)
                if not (np.isfinite(soz) and np.isfinite(voz)):
                    return np.nan, np.nan
                return float(soz), float(voz)

        try:
            out = sparse_unmix_cube_ssa(
                self.current_data,
                self.sparse_endmember_ssa,
                incidence_deg=30.0,
                emission_deg=0.0,
                lambda_=float(params["lambda"]),
                positivity=bool(params["positivity"]),
                addone=bool(params["addone"]),
                al_iters=int(params["al_iters"]),
                tol=float(params["tol"]),
                spatial_stride=int(stride),
                band_mask=band_mask,
                progress_cb=cb,
                per_pixel_geometry=per_pix,
            )
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Sparse 图像处理失败", str(exc))
            return
        progress.close()
        if cancelled["v"]:
            QMessageBox.information(self, "Sparse", "已取消。")
            return

        self.unmix_abundance_cube = out["abundance"]
        self.unmix_rmse_map = out["rmse"]
        self.unmix_method = "sunsal_ssa"
        self.unmix_library = type("L", (), {"names": names})()
        self.show_unmix_abundance_map(default_index=0)
        geom = "逐像元辅助立方体" if per_pix is not None else "默认 i=30°, e=0°"
        QMessageBox.information(
            self, "Sparse 图像处理完成",
            f"端元数：{len(names)}\n"
            f"空间步长：{stride}\n"
            f"参数：{self._sparse_params_summary(params)}\n"
            f"几何：{geom}\n"
            f"算法：SUNSAL（SSA 空间）\n\n"
            "可用「显示丰度图… / 显示 RMSE 图」切换。",
        )

    def run_sparse_unmixing(self):
        """兼容旧入口：转到 Sparse 单光谱模式。"""
        self.sparse_single_spectrum()

    def show_unmix_abundance_map(self, default_index=0):
        if self.unmix_abundance_cube is None:
            QMessageBox.information(
                self, "丰度图",
                "尚无整图解混结果。请先运行 Hapke / Sparse 图像处理。",
            )
            return
        names = []
        if self.hapke_endmembers_raw:
            names = [em.name for em in self.hapke_endmembers_raw]
        elif self.unmix_library is not None and getattr(self.unmix_library, "names", None):
            names = list(self.unmix_library.names)
        if not names:
            names = [f"EM{i+1}" for i in range(self.unmix_abundance_cube.shape[2])]
        items = [f"{i}: {names[i] if i < len(names) else f'EM{i+1}'}" for i in range(len(names))]
        items.append("【主导端元编号图】")
        label, ok = QInputDialog.getItem(
            self, "丰度图", "选择要显示的端元：", items,
            current=min(int(default_index), max(0, len(items) - 2)), editable=False,
        )
        if not ok:
            return
        if label.startswith("【主导"):
            ab = self.unmix_abundance_cube
            with np.errstate(all="ignore"):
                dom = np.nanargmax(np.nan_to_num(ab, nan=-np.inf), axis=2).astype(float)
                dom[~np.any(np.isfinite(ab), axis=2)] = np.nan
            self.show_parameter_result(dom, "Unmixing dominant endmember index")
            return
        idx = int(label.split(":", 1)[0])
        name = names[idx] if idx < len(names) else f"EM{idx+1}"
        self.show_parameter_result(
            self.unmix_abundance_cube[:, :, idx],
            f"Abundance: {name}",
        )

    def show_unmix_rmse_map(self):
        if self.unmix_rmse_map is None:
            QMessageBox.information(self, "RMSE 图", "尚无整图解混结果。")
            return
        self.show_parameter_result(self.unmix_rmse_map, f"Unmixing RMSE ({self.unmix_method})")

    # ================= DISORT：双文件 + 单光谱 / 图像处理 =================
    # 辅助立方体波段（用户说明为 1-based）：
    # 1 太阳入射角, 2 观测角, 3 相位角, 4 纬度, 5 经度, 13 当地时间(小时)
    AUX_BAND_SOZ = 0
    AUX_BAND_VOZ = 1
    AUX_BAND_PHASE = 2
    AUX_BAND_LAT = 3
    AUX_BAND_LON = 4
    AUX_BAND_LOCAL_TIME = 12

    def show_mcd_status(self):
        from disort.mcd_paths import resolve_mcd_data, mcd_install_dir

        data = resolve_mcd_data()
        install = mcd_install_dir()
        fmcd_ok = False
        mcdpy_ok = False
        try:
            import fmcd  # noqa: F401
            fmcd_ok = True
        except Exception:
            pass
        try:
            import mcd  # noqa: F401
            mcdpy_ok = True
        except Exception:
            pass
        msg = (
            f"安装目录：{install}\n"
            f"MCD_DATA：{data or '（未找到本地数据）'}\n"
            f"环境变量 MCD_DATA：{os.environ.get('MCD_DATA') or '（未设置）'}\n"
            f"fmcd 模块：{'可用' if fmcd_ok else '未安装'}\n"
            f"mcd-python：{'可用' if mcdpy_ok else '未安装'}\n\n"
            "完整版需向 LMD 登记获取：\n"
            "https://www-mars.lmd.jussieu.fr/MCD_pro/mcd_pro.html\n"
            "联系：millour@lmd.jussieu.fr\n\n"
            "拿到压缩包或下载链接后，用菜单「配置/安装本地 MCD…」。"
        )
        QMessageBox.information(self, "本地 MCD 状态", msg)

    def setup_local_mcd(self):
        """Install MCD from a local archive or URL into data/mcd/."""
        from disort.mcd_paths import resolve_mcd_data, mcd_install_dir

        choice = QMessageBox.question(
            self,
            "安装本地 MCD",
            "Mars Climate Database 完整版需先在 LMD 网站登记，邮件获取下载链接。\n\n"
            "是：选择本地已下载的 .tar.gz / .zip 安装\n"
            "否：粘贴 LMD 提供的下载 URL\n"
            "取消：仅打开说明",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if choice == QMessageBox.Cancel:
            readme = mcd_install_dir() / "README.md"
            QMessageBox.information(
                self,
                "MCD 说明",
                (readme.read_text(encoding="utf-8") if readme.is_file() else "")
                + "\n登记页：https://www-mars.lmd.jussieu.fr/MCD_pro/mcd_pro.html",
            )
            return

        # Ensure repo scripts importable
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(repo_root, "scripts", "install_mcd.py")
        if not os.path.isfile(script):
            QMessageBox.critical(self, "MCD", f"找不到安装脚本：{script}")
            return

        import runpy

        try:
            if choice == QMessageBox.Yes:
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "选择 MCD 压缩包",
                    "",
                    "Archives (*.tar.gz *.tgz *.tar *.zip);;All Files (*)",
                )
                if not path:
                    return
                rc = runpy.run_path(script, run_name="__not_main__")
                main_fn = rc.get("main")
                code = main_fn(["--archive", path])
            else:
                url, ok = QInputDialog.getText(
                    self,
                    "MCD 下载 URL",
                    "粘贴 LMD 邮件中的下载链接：",
                )
                if not ok or not str(url).strip():
                    return
                rc = runpy.run_path(script, run_name="__not_main__")
                main_fn = rc.get("main")
                code = main_fn(["--url", str(url).strip(), "--keep-download"])
        except Exception as exc:
            QMessageBox.critical(self, "MCD 安装失败", str(exc))
            return

        data = resolve_mcd_data()
        if code == 0 and data:
            os.environ["MCD_DATA"] = data
            QMessageBox.information(
                self,
                "MCD 安装完成",
                f"已安装到本地。\nMCD_DATA={data}\n\n"
                "若尚未编译 fmcd，请按 data/mcd/README.md 与 "
                "third_party/mcd-python/README.md 编译 Fortran/Python 接口。",
            )
        else:
            QMessageBox.warning(
                self,
                "MCD 安装",
                f"安装脚本退出码 {code}。请查看终端输出或 data/mcd/README.md。",
            )

    def open_disort_radiance(self):
        """打开辐亮度高光谱图像，显示在左侧上方。"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开辐亮度高光谱图像", "", "ENVI Header (*.hdr);;All Files (*)"
        )
        if not filename:
            return
        try:
            self.open_file_path(filename, is_radiance=True)
            QMessageBox.information(self, "DISORT", "辐亮度图像已加载到左侧上方。")
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    def open_disort_aux(self):
        """打开 CRISM DDR 辅助立方体（ODE: *.lbl + *.img），band13 当地时间显示在左下。"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "打开辅助信息图像（CRISM DDR）",
            "",
            "CRISM DDR / PDS (*.lbl *.LBL *.img *.IMG);;ENVI Header (*.hdr);;All Files (*)",
        )
        if not filename:
            return
        try:
            aux, meta, lbl_path, img_path = self._load_aux_cube(filename)
            if aux.ndim != 3 or aux.shape[2] < 13:
                raise ValueError("辅助立方体至少需要 13 个波段（band13=当地时间）。")
            if self.current_data is not None:
                if aux.shape[0] != self.current_data.shape[0] or aux.shape[1] != self.current_data.shape[1]:
                    raise ValueError(
                        f"辅助图像尺寸 {aux.shape[:2]} 与辐亮度图像 "
                        f"{self.current_data.shape[:2]} 不一致。"
                    )
            self.aux_data = aux
            self.aux_path = lbl_path
            self.aux_img_path = img_path
            self.aux_metadata = meta
            self._update_ls_from_aux_header()
            self._show_aux_local_time()
            if self.disort_ls_deg is not None:
                src = self.disort_ls_source or "label"
                QMessageBox.information(
                    self, "DISORT",
                    "辅助信息已加载（CRISM DDR）。\n"
                    "左侧下方显示 band13（当地太阳时，小时）。\n\n"
                    f"标签文件：{os.path.basename(lbl_path)}\n"
                    f"观测 UTC：{self.disort_utc_iso or '（标签未提供）'}\n"
                    f"太阳经度 Ls：{self.disort_ls_deg:.3f}°\n"
                    f"来源：{src}\n"
                    "（将用于 MCD 大气查询）",
                )
            else:
                QMessageBox.warning(
                    self, "DISORT",
                    "辅助信息已加载；左侧下方显示 band13（当地时间，小时）。\n\n"
                    "DDR 标签中未找到 SOLAR_LONGITUDE / START_TIME，"
                    "运行 DISORT 前需手动输入 Ls。",
                )
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    def _load_aux_cube(self, filename):
        """Load auxiliary cube from PDS .lbl/.img or ENVI .hdr."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".lbl", ".img"):
            from disort.pds_label import load_pds_cube

            cube, meta, lbl_path, img_path = load_pds_cube(filename)
            return cube, meta, lbl_path, img_path

        # ENVI fallback (some users convert DDR to ENVI)
        img = envi.open(filename)
        cube = np.array(img.load(), dtype=np.float32)
        meta = dict(img.metadata) if getattr(img, "metadata", None) else {}
        return cube, meta, filename, filename

    def _update_ls_from_aux_header(self):
        """从 DDR .lbl 的 SOLAR_LONGITUDE（优先）或 START_TIME 得到 Ls。"""
        from disort.mars_time import ls_from_label_source

        info = ls_from_label_source(self.aux_metadata, self.aux_path)
        if info.get("ok"):
            self.disort_ls_deg = float(info["ls_deg"])
            self.disort_utc_iso = info.get("utc_iso")
            self.disort_ls_source = info.get("ls_source") or info.get("source_key")
            self.statusBar().showMessage(info.get("message", ""), 10000)
            return True
        self.disort_ls_deg = None
        self.disort_utc_iso = None
        self.disort_ls_source = None
        return False

    def open_file_path(self, filename, is_radiance=False):
        """内部：按路径打开 ENVI 立方体（复用 open_file 逻辑）。"""
        # 直接调用现有 open_file 流程：临时简化为复制核心加载
        self.marker_rgb = None
        self.marker_result = None
        self.current_param_img = None
        self.current_param_title = None
        self.auto_featureless_mask = None
        self.auto_col_denominators = None

        img = envi.open(filename)
        self.current_data = np.array(img.load(), dtype=np.float32)

        base_name = os.path.basename(filename).lower()
        if 'fr' in base_name:
            self.current_data[0, :, :] = np.nan
            self.current_data[-1, :, :] = np.nan
            self.current_data[:, 0:31, :] = np.nan
            self.current_data[:, -9:, :] = np.nan
        elif 'hr' in base_name:
            self.current_data[0, :, :] = np.nan
            self.current_data[-1, :, :] = np.nan
            self.current_data[:, 0:17, :] = np.nan
            self.current_data[:, -6:, :] = np.nan

        if img.bands.centers:
            self.wavelengths = np.array(img.bands.centers)
        elif 'wavelength' in img.metadata:
            self.wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
        else:
            bands_count = self.current_data.shape[2]
            self.wavelengths = np.arange(1, bands_count + 1)

        if np.any(self.wavelengths > 100):
            self.wavelengths = self.wavelengths / 1000.0

        r_band = np.argmin(np.abs(self.wavelengths - 2.53))
        g_band = np.argmin(np.abs(self.wavelengths - 1.51))
        b_band = np.argmin(np.abs(self.wavelengths - 1.08))
        rgb = self.current_data[:, :, [r_band, g_band, b_band]]
        rgb_min = np.nanpercentile(rgb, 2)
        rgb_max = np.nanpercentile(rgb, 98)
        self.rgb_image = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-8), 0, 1)
        self.rgb_image[np.isnan(self.rgb_image)] = 0.0

        self.fig_rgb.clf()
        self.ax_rgb = self.fig_rgb.add_subplot(111)
        self.ax_rgb.imshow(self.rgb_image)
        title = "辐亮度假彩色图" if is_radiance else (
            f"假彩色图 (R {self.wavelengths[r_band]:.2f} $\\mu$m, "
            f"G {self.wavelengths[g_band]:.2f} $\\mu$m, "
            f"B {self.wavelengths[b_band]:.2f} $\\mu$m)"
        )
        self.ax_rgb.set_title(title)
        self.ax_rgb.axis('off')
        self.marker_rgb = None
        self._apply_image_layout(self.fig_rgb, self.ax_rgb, hide_cbar=True)
        self.canvas_rgb.draw()
        self._sync_spectrum_axes()
        self.canvas_raw_spec.draw()
        self.canvas_ratio_spec.draw()

    def _show_aux_local_time(self):
        if self.aux_data is None:
            return
        band = self.aux_data[:, :, self.AUX_BAND_LOCAL_TIME]
        self.fig_result.clf()
        self.ax_result = self.fig_result.add_subplot(111)
        im = self.ax_result.imshow(band, cmap='viridis')
        self.ax_result.set_title("辅助信息 band13：当地时间 (hour)")
        self.ax_result.axis('off')
        self.marker_result = None
        self.current_param_img = band
        self.current_param_title = "Local time (h)"
        self._apply_image_layout(self.fig_result, self.ax_result, colorbar_mappable=im)
        self.canvas_result.draw()

    def _ensure_disort_cubes_and_root(self):
        if self.current_data is None:
            QMessageBox.information(self, "DISORT", "请先加载辐亮度高光谱图像。")
            self.open_disort_radiance()
            if self.current_data is None:
                return False
        if self.aux_data is None:
            QMessageBox.information(self, "DISORT", "请先加载辅助信息图像。")
            self.open_disort_aux()
            if self.aux_data is None:
                return False
        if self.aux_data.shape[0] != self.current_data.shape[0] or \
           self.aux_data.shape[1] != self.current_data.shape[1]:
            QMessageBox.critical(self, "DISORT", "辐亮度与辅助图像像元尺寸不一致。")
            return False

        data_root = QFileDialog.getExistingDirectory(
            self, "选择 DISORT 数据根目录（含 input/ 与 optical/；optical 必需）"
        )
        if not data_root:
            return False
        self.disort_data_root = data_root

        # 优先用 DDR 标签 SOLAR_LONGITUDE / START_TIME；仅在缺失时手动输入
        if self.disort_ls_deg is None:
            self._update_ls_from_aux_header()
        if self.disort_ls_deg is not None:
            QMessageBox.information(
                self, "太阳经度 Ls",
                f"观测 UTC：{self.disort_utc_iso or '（标签未提供）'}\n"
                f"太阳经度 Ls = {self.disort_ls_deg:.3f}°\n"
                f"来源：{self.disort_ls_source or 'label'}\n\n"
                "将用于 MCD 大气查询。",
            )
        else:
            ls, ok = QInputDialog.getDouble(
                self, "太阳经度 Ls",
                "DDR 标签中未找到 SOLAR_LONGITUDE / START_TIME。\n"
                "请手动输入火星太阳经度 Ls（度，0–360）：",
                value=90.0,
                minValue=0.0, maxValue=360.0, decimals=2,
            )
            if not ok:
                return False
            self.disort_ls_deg = float(ls)
            self.disort_ls_source = "manual"

        reply = QMessageBox.question(
            self, "计算范围",
            "完整波段 DISORT 较慢。\n是：每 5 个波段计算一个（推荐）\n否：全部波段",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        self.disort_band_step = 5 if reply == QMessageBox.Yes else 1
        return True

    def _aux_geometry_at(self, row, col):
        a = self.aux_data
        soz = float(a[row, col, self.AUX_BAND_SOZ])
        voz = float(a[row, col, self.AUX_BAND_VOZ])
        phase = float(a[row, col, self.AUX_BAND_PHASE])
        lat = float(a[row, col, self.AUX_BAND_LAT])
        lon = float(a[row, col, self.AUX_BAND_LON])
        loct = float(a[row, col, self.AUX_BAND_LOCAL_TIME])
        return soz, voz, phase, lat, lon, loct

    def disort_single_spectrum_mode(self):
        if not self._ensure_disort_cubes_and_root():
            return
        from disort.mcd_client import MCDProfileCache
        self.disort_mcd_cache = MCDProfileCache()
        self.disort_mode = 'single'
        self.ratio_mode = None
        QMessageBox.information(
            self, "单光谱计算",
            "已进入 DISORT 单光谱模式。\n"
            "请在左侧上方辐亮度图像上点击一个像元开始校正。\n"
            "将按该像元经纬度和当地时间从 MCD 读取大气廓线。\n"
            "双击图像可退出。"
        )

    def disort_image_mode(self):
        if not self._ensure_disort_cubes_and_root():
            return
        spat, ok = QInputDialog.getInt(
            self, "空间抽样",
            "整图处理很慢。空间步长（每隔 N 个像元计算 1 个，其余插值/置空）：",
            value=10, minValue=1, maxValue=100,
        )
        if not ok:
            return
        self.disort_mode = 'image'
        self.ratio_mode = None
        from disort.mcd_client import MCDProfileCache
        self.disort_mcd_cache = MCDProfileCache()
        self._start_disort_image_worker(spatial_step=int(spat))

    def exit_disort_mode(self):
        self.disort_mode = None
        QMessageBox.information(self, "DISORT", "已退出 DISORT 点击/处理模式。")

    def _disort_run_single_pixel(self, row, col):
        # 先显示原始光谱
        self.process_click_logic(row, col)
        try:
            soz, voz, phase, lat, lon, loct = self._aux_geometry_at(row, col)
        except Exception as e:
            QMessageBox.critical(self, "辅助信息", f"读取辅助波段失败：{e}")
            return
        if not np.isfinite([soz, voz, phase, lat, lon, loct]).all():
            QMessageBox.warning(self, "辅助信息", "该像元辅助信息含无效值。")
            return

        spectrum, _ = self._extract_window_spectrum(row, col)
        self._start_disort_worker(
            data_root=self.disort_data_root,
            observed_radiance=np.asarray(spectrum, dtype=np.float64),
            wavelengths_um=np.asarray(self.wavelengths, dtype=np.float64),
            band_step=self.disort_band_step,
            soz_deg=soz,
            voz_deg=voz,
            pa_deg=phase,
            lat=lat,
            lon=lon,
            local_time_h=loct,
            ls_deg=self.disort_ls_deg,
            use_mcd=True,
        )

    def _start_disort_image_worker(self, spatial_step=10):
        try:
            run_disort_correction = _import_run_disort_correction()
        except ModuleNotFoundError as e:
            QMessageBox.critical(self, "DISORT 模块缺失", str(e))
            return

        rows, cols, bands = self.current_data.shape
        self._disort_progress = QProgressDialog(
            "正在整图 DISORT 大气校正…", "取消", 0, rows * cols, self
        )
        self._disort_progress.setWindowModality(Qt.WindowModal)
        self._disort_progress.setMinimumDuration(0)
        self._disort_progress.setValue(0)

        class _ImgWorker(QObject):
            finished = Signal(object)
            failed = Signal(str)
            progress = Signal(int, int, str)

            def __init__(self, app_ref, step, run_fn):
                super().__init__()
                self.app_ref = app_ref
                self.step = step
                self.run_fn = run_fn
                self._cancel = False

            def run(self):
                try:
                    from disort.mcd_client import MCDProfileCache, fetch_mcd_profile
                    cache = MCDProfileCache()
                    rows, cols, _ = self.app_ref.current_data.shape
                    # 输出：每个像元一条反照率光谱（抽样波段上有值）
                    # 为控制内存，只存与表波长对齐后的反照率；先用第一像素探测长度
                    albedo_map = None
                    wave_out = None
                    total = ((rows + self.step - 1) // self.step) * ((cols + self.step - 1) // self.step)
                    done = 0
                    for r in range(0, rows, self.step):
                        for c in range(0, cols, self.step):
                            if self._disort_progress_canceled():
                                self.failed.emit("用户取消")
                                return
                            spec = self.app_ref.current_data[r, c, :]
                            if not np.any(np.isfinite(spec)):
                                done += 1
                                self.progress.emit(done, total, f"skip ({c},{r})")
                                continue
                            soz, voz, phase, lat, lon, loct = self.app_ref._aux_geometry_at(r, c)
                            if not np.isfinite([soz, voz, phase, lat, lon, loct]).all():
                                done += 1
                                self.progress.emit(done, total, f"bad aux ({c},{r})")
                                continue
                            try:
                                prof = cache.get(
                                    lat, lon, loct, self.app_ref.disort_ls_deg,
                                    fallback_input_root=self.app_ref.disort_data_root,
                                )
                            except Exception as ex:
                                done += 1
                                self.progress.emit(done, total, f"MCD失败 ({c},{r})")
                                continue
                            result = _call_run_disort_correction(
                                self.run_fn,
                                data_root=self.app_ref.disort_data_root,
                                observed_radiance=np.asarray(spec, dtype=np.float64),
                                wavelengths_um=np.asarray(self.app_ref.wavelengths, dtype=np.float64),
                                band_step=self.app_ref.disort_band_step,
                                soz_deg=float(soz),
                                voz_deg=float(voz),
                                pa_deg=float(phase),
                                atm_profile=prof,
                                allow_partial_input=True,
                            )
                            alb = np.asarray(result["albedo"], dtype=np.float64)
                            if albedo_map is None:
                                wave_out = np.asarray(result["wavelength"], dtype=np.float64)
                                albedo_map = np.full((rows, cols, alb.size), np.nan, dtype=np.float32)
                            albedo_map[r, c, :] = alb.astype(np.float32)
                            done += 1
                            self.progress.emit(done, total, f"({c},{r})")
                    self.finished.emit({"albedo_cube": albedo_map, "wavelength": wave_out})
                except Exception as e:
                    self.failed.emit(str(e))

            def _disort_progress_canceled(self):
                return False

        self._disort_img_thread = QThread(self)
        self._disort_img_worker = _ImgWorker(self, spatial_step, run_disort_correction)
        # bind cancel
        def _canceled():
            self._disort_img_worker._cancel = True
        # monkey-patch cancel check
        def _check():
            return bool(getattr(self._disort_img_worker, "_cancel", False) or
                        (self._disort_progress is not None and self._disort_progress.wasCanceled()))
        self._disort_img_worker._disort_progress_canceled = _check

        self._disort_img_worker.moveToThread(self._disort_img_thread)
        self._disort_img_thread.started.connect(self._disort_img_worker.run)
        self._disort_img_worker.progress.connect(self._on_disort_progress)
        self._disort_img_worker.finished.connect(self._on_disort_image_finished)
        self._disort_img_worker.failed.connect(self._on_disort_failed)
        self._disort_img_worker.finished.connect(self._disort_img_thread.quit)
        self._disort_img_worker.failed.connect(self._disort_img_thread.quit)
        self._disort_progress.canceled.connect(_canceled)
        self._disort_img_thread.start()

    def _on_disort_image_finished(self, result):
        if hasattr(self, "_disort_progress") and self._disort_progress is not None:
            self._disort_progress.close()
        cube = result.get("albedo_cube")
        wave = result.get("wavelength")
        self.disort_albedo_cube = cube
        if cube is None:
            QMessageBox.warning(self, "DISORT", "未得到有效反照率结果。")
            return
        # 结果显示：取 ~1.5 μm 反照率切片
        if wave is not None and np.any(np.isfinite(wave)):
            idx = int(np.nanargmin(np.abs(wave - 1.5)))
        else:
            idx = cube.shape[2] // 2
        band = cube[:, :, idx]
        self.fig_result.clf()
        self.ax_result = self.fig_result.add_subplot(111)
        im = self.ax_result.imshow(band, cmap='gray')
        self.ax_result.set_title(f"DISORT 地表反照率 (~{wave[idx]:.3f} μm)" if wave is not None else "DISORT 地表反照率")
        self.ax_result.axis('off')
        self.marker_result = None
        self.current_param_img = band
        self.current_param_title = "DISORT albedo"
        self._apply_image_layout(self.fig_result, self.ax_result, colorbar_mappable=im)
        self.canvas_result.draw()
        n_ok = int(np.count_nonzero(np.isfinite(band)))
        QMessageBox.information(
            self, "DISORT 图像处理完成",
            f"完成。有效像元约 {n_ok} 个（受空间抽样影响）。\n"
            "左侧下方显示某一波段地表反照率切片。"
        )

    def run_disort(self):
        """兼容旧入口：转到单光谱模式准备。"""
        self.disort_single_spectrum_mode()

    def _start_disort_worker(
        self, data_root, observed_radiance, wavelengths_um, band_step=1,
        soz_deg=None, voz_deg=7.878, pa_deg=65.657,
        lat=None, lon=None, local_time_h=None, ls_deg=None, use_mcd=False,
    ):
        try:
            run_disort_correction = _import_run_disort_correction()
        except ModuleNotFoundError as e:
            QMessageBox.critical(self, "DISORT 模块缺失", str(e))
            return

        self._disort_progress = QProgressDialog(
            "正在运行 DISORT 大气校正…", "取消", 0, 100, self
        )
        self._disort_progress.setWindowModality(Qt.WindowModal)
        self._disort_progress.setMinimumDuration(0)
        self._disort_progress.setValue(0)

        class _Worker(QObject):
            finished = Signal(dict)
            failed = Signal(str)
            progress = Signal(int, int, str)

            def __init__(self, kwargs):
                super().__init__()
                self.kwargs = kwargs

            def run(self):
                try:
                    def cb(cur, tot, msg):
                        self.progress.emit(cur, tot, msg)

                    kw = dict(self.kwargs)
                    if kw.pop("use_mcd", False):
                        from disort.mcd_client import fetch_mcd_profile
                        prof = fetch_mcd_profile(
                            kw.pop("lat"), kw.pop("lon"), kw.pop("local_time_h"),
                            kw.pop("ls_deg"),
                            fallback_input_root=kw.get("data_root"),
                        )
                        kw["atm_profile"] = prof
                        kw["allow_partial_input"] = True
                        warn = prof.get("warning")
                        if warn:
                            self.progress.emit(0, 1, warn)
                    else:
                        for k in ("lat", "lon", "local_time_h", "ls_deg", "use_mcd"):
                            kw.pop(k, None)
                    result = _call_run_disort_correction(
                        run_disort_correction, progress_cb=cb, **kw
                    )
                    if "atm_profile" in self.kwargs or self.kwargs.get("use_mcd"):
                        result = dict(result)
                        result["_mcd_source"] = (
                            self.kwargs.get("_mcd_source")
                            or (kw.get("atm_profile") or {}).get("source")
                        )
                        result["_mcd_warning"] = (kw.get("atm_profile") or {}).get("warning")
                    self.finished.emit(result)
                except Exception as e:
                    self.failed.emit(str(e))

        kwargs = dict(
            data_root=data_root,
            observed_radiance=observed_radiance,
            observed_if=observed_radiance,
            wavelengths_um=wavelengths_um,
            band_step=max(int(band_step), 1),
            soz_deg=soz_deg,
            voz_deg=voz_deg,
            pa_deg=pa_deg,
            lat=lat,
            lon=lon,
            local_time_h=local_time_h,
            ls_deg=ls_deg,
            use_mcd=use_mcd,
        )
        self._disort_thread = QThread(self)
        self._disort_worker = _Worker(kwargs)
        self._disort_worker.moveToThread(self._disort_thread)
        self._disort_thread.started.connect(self._disort_worker.run)
        self._disort_worker.progress.connect(self._on_disort_progress)
        self._disort_worker.finished.connect(self._on_disort_finished)
        self._disort_worker.failed.connect(self._on_disort_failed)
        self._disort_worker.finished.connect(self._disort_thread.quit)
        self._disort_worker.failed.connect(self._disort_thread.quit)
        self._disort_progress.canceled.connect(self._disort_thread.quit)
        self._disort_thread.start()

    def _on_disort_progress(self, cur, tot, msg):
        if hasattr(self, "_disort_progress") and self._disort_progress is not None:
            self._disort_progress.setMaximum(max(tot, 1))
            self._disort_progress.setValue(cur)
            self._disort_progress.setLabelText(f"DISORT: {msg} ({cur}/{tot})")

    def _on_disort_failed(self, err):
        if hasattr(self, "_disort_progress") and self._disort_progress is not None:
            self._disort_progress.close()
        QMessageBox.critical(self, "DISORT 失败", err)

    def _plot_disort_on_raw(self):
        """原始光谱区显示：观测 I/F、模型 I/F、地表反照率（由辐亮度换算）。"""
        wave = self.disort_wavelength
        albedo = self.disort_albedo
        if wave is None or albedo is None:
            return

        self.raw_crosshair_vline = None
        self.raw_crosshair_hline = None
        self.raw_crosshair_text = None

        self.ax_raw_spec.clear()
        y_for_lim = []

        obs_if = self.disort_observed_if
        if obs_if is not None and np.any(np.isfinite(obs_if)):
            self.ax_raw_spec.plot(
                wave, obs_if, color="navy", linestyle="--",
                label="Observed I/F", linewidth=1.0, alpha=0.85
            )
            y_for_lim.append(np.asarray(obs_if, dtype=float).ravel())

        model_if = self.disort_model_if
        if model_if is not None and np.any(np.isfinite(model_if)):
            self.ax_raw_spec.plot(
                wave, model_if, color="gray", linestyle=":",
                label="Modeled I/F", linewidth=1.1, alpha=0.9
            )
            y_for_lim.append(np.asarray(model_if, dtype=float).ravel())

        valid = np.isfinite(albedo) & np.isfinite(wave)
        if np.any(valid):
            self.ax_raw_spec.plot(
                wave[valid], albedo[valid], color="crimson",
                label="Surface albedo", linewidth=1.3
            )
            y_for_lim.append(np.asarray(albedo[valid], dtype=float).ravel())

        pos_tip = ""
        if self.selected_pos is not None:
            r, c = self.selected_pos
            pos_tip = f" | 选中像元 X:{c}, Y:{r}"
        self.ax_raw_spec.set_title(f"DISORT 结果（I/F 与地表反照率）{pos_tip}")
        self.ax_raw_spec.set_xlabel(r"Wavelength ($\mu$m)")
        self.ax_raw_spec.set_ylabel("I/F / Albedo")
        self.ax_raw_spec.legend(fontsize=8)
        self.ax_raw_spec.grid(True, linestyle="--", alpha=0.5)
        if y_for_lim:
            self._apply_raw_spec_ylim(np.concatenate(y_for_lim))

    def _on_disort_finished(self, result):
        if hasattr(self, "_disort_progress") and self._disort_progress is not None:
            self._disort_progress.close()

        wave = np.asarray(result["wavelength"], dtype=np.float64)
        albedo = np.asarray(result["albedo"], dtype=np.float64)
        model_rad = np.asarray(result.get("model_radiance"), dtype=np.float64)
        obs_rad = np.asarray(result.get("observed_radiance"), dtype=np.float64)
        s0 = np.asarray(result.get("s0"), dtype=np.float64)
        obs_if = np.asarray(result.get("observed_if"), dtype=np.float64)
        model_if = np.asarray(result.get("model_if"), dtype=np.float64)

        if (not np.any(np.isfinite(obs_if))) and np.any(np.isfinite(obs_rad)):
            from disort.correction import radiance_to_if
            obs_if = radiance_to_if(obs_rad, s0)
        if (not np.any(np.isfinite(model_if))) and np.any(np.isfinite(model_rad)):
            from disort.correction import radiance_to_if
            model_if = radiance_to_if(model_rad, s0)

        self.ratio_mode = "disort"
        self.disort_wavelength = wave
        self.disort_albedo = albedo
        self.disort_observed_radiance = obs_rad
        self.disort_model_radiance = model_rad
        self.disort_observed_if = obs_if
        self.disort_model_if = model_if
        self.disort_s0 = s0
        self.current_raw_spectrum = albedo.copy()

        self._plot_disort_on_raw()
        self._sync_spectrum_axes()
        self.canvas_raw_spec.draw()

        n_ok = int(np.count_nonzero(np.isfinite(albedo)))
        max_tau = float(np.asarray(result.get("max_tau_co2_2um", [0.0])).ravel()[0])
        mcd_src = result.get("_mcd_source") or ""
        mcd_warn = result.get("_mcd_warning") or ""
        tip = (
            f"大气校正完成，成功反演 {n_ok} 个波段的地表反照率。\n\n"
            "请看红线 Surface albedo。蓝虚线 Observed I/F 仍含大气。\n"
            f"诊断：2 μm 附近 CO₂ 光学厚度峰值 ≈ {max_tau:.3g}\n"
        )
        if mcd_src:
            tip += f"大气廓线来源：{mcd_src}\n"
        if mcd_warn:
            tip += f"注意：{mcd_warn}\n"
        QMessageBox.information(self, "DISORT 完成", tip)

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
        if mode == 'auto':
            if self.current_data is None:
                QMessageBox.warning(self, "警告", "请先打开高光谱图像数据！")
                return
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                QApplication.processEvents()
                featureless, denoms, usable_cols = self._prepare_auto_ratio_denominators()
            except Exception as e:
                QMessageBox.critical(self, "自动提取失败", str(e))
                return
            finally:
                QApplication.restoreOverrideCursor()

            # 在结果图显示无光谱特征区域（1=无特征，作分母候选）
            mask_img = featureless.astype(np.float32)
            self.show_parameter_result(mask_img, "无光谱特征区域(自动比值分母)")

            self.ratio_mode = mode
            self.click_coords = []
            self.click_positions = []
            self.manual_ratio_first_pos = None
            self._clear_manual_lines()

            n_pix = int(np.count_nonzero(featureless))
            QMessageBox.information(
                self, "模式切换",
                "已切换为: 自动提取 模式\n"
                "判定：每个参数计算整幅中值与标准差，"
                "参数-(中值+标准差) 全部 < 0 为无光谱特征区域。\n"
                f"无特征像元 {n_pix} 个，可用分母列 {usable_cols} 列。\n"
                "点击某列像元：该点光谱为分子，该列无特征均值光谱为分母。\n"
                "提示：在左侧图像上双击可退出比值模式。"
            )
            return

        self.ratio_mode = mode
        self.click_coords = []
        self.click_positions = []
        self.manual_ratio_first_pos = None
        self._clear_manual_lines()
        tip = '手动提取'
        QMessageBox.information(
            self, "模式切换",
            f"已切换为: {tip} 模式\n提示：在左侧图像上双击可退出比值模式。"
        )

    def exit_ratio_mode(self):
        """退出 Ratio spectra（自动/手动）或 DISORT 显示模式。"""
        if self.ratio_mode is None:
            QMessageBox.information(self, "提示", "当前不在比值光谱 / DISORT 模式中。")
            return

        was_disort = self.ratio_mode == "disort"
        self.ratio_mode = None
        self.click_coords = []
        self.click_positions = []
        self.manual_ratio_first_pos = None
        self._clear_manual_lines()

        self.current_ratio_spectrum = None
        self._clear_ratio_plot("比值光谱", show_y_labels=True)
        self._sync_spectrum_axes()
        self.canvas_ratio_spec.draw()

        if was_disort:
            self.disort_wavelength = None
            self.disort_albedo = None
            self.disort_observed_radiance = None
            self.disort_model_radiance = None
            self.disort_observed_if = None
            self.disort_model_if = None
            self.disort_s0 = None
            self.raw_crosshair_vline = None
            self.raw_crosshair_hline = None
            self.raw_crosshair_text = None
            self.ax_raw_spec.clear()
            self.ax_raw_spec.set_title("原始光谱")
            self.ax_raw_spec.set_xlabel(r"Wavelength ($\mu$m)")
            self.ax_raw_spec.set_ylabel("Reflectance")
            self.canvas_raw_spec.draw()
            QMessageBox.information(self, "模式切换", "已退出 DISORT 显示，原始光谱区已清空。")
        else:
            QMessageBox.information(self, "模式切换", "已退出比值光谱模式，比值光谱图已清空。")

    def save_spectrum_figure(self, fig, default_stem="spectrum"):
        """
        保存光谱图：支持常见图片格式 (png/jpg/tif) 与矢量格式 (pdf/svg/eps)。
        """
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存光谱图",
            f"{default_stem}.png",
            "PNG Image (*.png);;"
            "JPEG Image (*.jpg *.jpeg);;"
            "TIFF Image (*.tif *.tiff);;"
            "PDF Vector (*.pdf);;"
            "SVG Vector (*.svg);;"
            "EPS Vector (*.eps);;"
            "All Files (*)"
        )
        if not filename:
            return

        # 若用户未写扩展名，按所选过滤器补全
        root, ext = os.path.splitext(filename)
        if not ext:
            filt = (selected_filter or "").lower()
            if "jpeg" in filt or "jpg" in filt:
                filename = root + ".jpg"
            elif "tif" in filt:
                filename = root + ".tif"
            elif "pdf" in filt:
                filename = root + ".pdf"
            elif "svg" in filt:
                filename = root + ".svg"
            elif "eps" in filt:
                filename = root + ".eps"
            else:
                filename = root + ".png"

        try:
            fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "保存成功", f"已保存到:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存光谱图:\n{str(e)}")

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
        - 图中尚无光谱：直接绘制，Y 轴按该谱自身范围
        - 图中已有比值光谱：双 Y 轴叠加，各自按自身数值范围显示谱形；
          对比时隐藏 Y 轴刻度数值（不做连续统去除、不做幅值比例缩放）
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开RELAB库文件", "", "Text Files (*.txt);;All Files (*)"
        )
        if not filename:
            return

        try:
            relab_wave, relab_refl = self._load_relab_txt(filename)
            label = os.path.basename(filename)
            self.relab_overlay = (relab_wave, relab_refl, label)

            if self.current_ratio_spectrum is not None:
                # 加入 RELAB 前先清掉旧十字线，避免残留叠加
                self._clear_ratio_crosshair_artists()
                # 双轴：左轴=现有比值谱，右轴=RELAB，各自用自身范围，隐藏 Y 数值
                self._remove_ratio_twin()
                self._set_ylim_from_data(self.ax_ratio_spec, self.current_ratio_spectrum)

                self.ax_ratio_twin = self.ax_ratio_spec.twinx()
                self.ax_ratio_twin.plot(
                    relab_wave,
                    relab_refl,
                    label=label,
                    linestyle='-.',
                    color='darkorange',
                    linewidth=1.3,
                )
                self._set_ylim_from_data(self.ax_ratio_twin, relab_refl)

                self.ax_ratio_spec.tick_params(axis='y', labelleft=False)
                self.ax_ratio_twin.tick_params(axis='y', labelright=False)
                self.ax_ratio_spec.set_ylabel("")
                self.ax_ratio_twin.set_ylabel("")

                lines1, labels1 = self.ax_ratio_spec.get_legend_handles_labels()
                lines2, labels2 = self.ax_ratio_twin.get_legend_handles_labels()
                self.ax_ratio_spec.legend(
                    lines1 + lines2, labels1 + labels2, fontsize=8
                )
            else:
                # 仅 RELAB：单轴，按自身范围显示，保留 Y 刻度
                self._clear_ratio_crosshair_artists()
                self._remove_ratio_twin()
                self.ax_ratio_spec.plot(
                    relab_wave,
                    relab_refl,
                    label=label,
                    linestyle='-.',
                    color='darkorange',
                    linewidth=1.3,
                )
                self._set_ylim_from_data(self.ax_ratio_spec, relab_refl)
                self.ax_ratio_spec.set_ylabel("Scaled Reflectance")
                self.ax_ratio_spec.tick_params(axis='y', labelleft=True)
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
