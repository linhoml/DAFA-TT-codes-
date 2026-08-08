# Endmember spectral libraries

## Hapke Excel
- `hapke_endmembers_template.xlsx` — 模板：第1列 `wavelength_um`，其后每列一个矿物反射率
- 加载后需输入各矿物密度 ρ、折射率实部 n、平均粒径 D，程序反演 k(λ)

## Sparse unmixing Excel
- 第1列：波长（μm 或 nm）
- 第2列起：各端元反射率（表头为名称）
- 加载后按实验室几何转为 SSA，再与图像 I/F→SSA 一起做 SUNSAL 稀疏解混

## Sparse / DAFA-TT（旧库，可选）
- `TargetLibrary_paper.mat` — 蛇纹石 + 碳酸盐实验室光谱库
