# Unmixing

## Hapke model（物理辐射传输）

流程：

1. **加载 Hapke Excel 端元…**（或运行 Hapke model 时自动提示）
   - Excel：第 1 列波长 (μm/nm)，其后每列一个矿物反射率（表头=矿物名）
   - 也可多 sheet，每 sheet 一个矿物
2. 弹出表格，输入每个矿物的：
   - 密度 ρ (g/cm³)
   - 折射率实部 **n**（全波段一个常数）
   - 平均粒径 **D** (μm)
   - 以及实验室测量几何 (i, e)，用于从反射率反演 k
3. Hapke 模型反演各矿物 **k(λ)**（折射率虚部）
4. 对观测光谱做 **非线性最小二乘**，解算亲密混合质量丰度
5. 模式：
   - **单光谱计算**：当前点击像元（可窗口平均）
   - **图像处理**：整图（可空间抽样；可选辅助立方体逐像元几何）

菜单还可 **导出 Hapke k(λ)…** 到 Excel。

模板生成：取消选文件时可选生成 `data/libraries/hapke_endmembers_template.xlsx`。

## Sparse unmixing（线性库）

加载 `.mat` / `.txt` 端元库，用 NNLS / OMP / FCLS / UCLS。详见原说明。
