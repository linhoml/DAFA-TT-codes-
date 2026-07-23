# 火星高光谱图像分析系统

基于 PySide6 + Matplotlib 的 CRISM 高光谱图像分析软件框架。

## 运行

```bash
pip install -r requirements.txt
python app/spectral_app.py
```

## 本地 Mars Climate Database (MCD)

完整版 MCD 需向 LMD 登记获取（不可公开直链）：

1. <https://www-mars.lmd.jussieu.fr/MCD_pro/mcd_pro.html>
2. 收到压缩包/链接后：

```bash
python scripts/install_mcd.py --archive /path/to/MCD.tar.gz
# 或
python scripts/install_mcd.py --url 'https://...'
source data/mcd/env.sh
```

也可在软件中：Tools → DISORT correction → **配置/安装本地 MCD…**

详见 `data/mcd/README.md`。

## 当前框架能力

- 打开 ENVI / CRISM DDR（`.lbl`+`.img`）高光谱与辅助数据
- 点击像元查看原始光谱 / 比值光谱
- 计算 CRISM 光谱参数（BD1400、BD1900 等）
- DISORT 大气校正（本地 MCD 优先）
