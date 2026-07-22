# 火星高光谱图像分析系统

基于 PySide6 + Matplotlib 的 CRISM 高光谱图像分析软件框架。

## 运行

```bash
pip install -r requirements.txt
python app/spectral_app.py
```

## 当前框架能力

- 打开 ENVI 高光谱数据，显示假彩色 RGB
- 点击像元查看原始光谱 / 比值光谱
- 计算 CRISM 光谱参数（BD1400、BD1900 等）
- Identification / Unmixing / DISORT / RELAB 等菜单占位，待填算法
