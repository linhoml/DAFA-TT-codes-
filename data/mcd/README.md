# Mars Climate Database (local install)

完整版 MCD **不能公开直链下载**，需向 LMD 登记后获得下载链接：

1. 打开登记页：<https://www-mars.lmd.jussieu.fr/MCD_pro/mcd_pro.html>  
   （若网页表单报错，可直接邮件联系 `millour@lmd.jussieu.fr` / `forget@lmd.jussieu.fr`）
2. 收到邮件中的下载链接或 `.tar.gz` / `.zip` 后，在本仓库执行：

```bash
# 从本地压缩包安装
python scripts/install_mcd.py --archive /path/to/MCD_xxxx.tar.gz

# 或从 LMD 邮件里的 URL 安装
python scripts/install_mcd.py --url 'https://...'
```

3. 加载环境并编译 Python 接口（需 `gfortran` + NetCDF）：

```bash
source data/mcd/env.sh
# 按 third_party/mcd-python/README.md 与 MCD 自带 mcd/python 说明编译 fmcd
```

安装成功后会出现：

```
data/mcd/MCD/          # 解压后的官方目录（含 data/ 与 mcd/）
data/mcd/MCD_DATA.path # 指向 data/ 的路径（供程序读取）
data/mcd/env.sh        # export MCD_DATA / PYTHONPATH
```

SpectralApp 的 DISORT 会按顺序尝试：本地 `fmcd`/`mcd-python`（`MCD_DATA`）→ 在线 Web → `input/` 回退。

> 注意：按 LMD 许可，**请勿把 MCD 数据二次分发**到公开仓库。`data/mcd/MCD/` 已在 `.gitignore` 中忽略。
