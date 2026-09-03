# Unmixing

## 菜单结构

```
Unmixing
 ├─ Hapke model
 │   ├─ 加载高光谱图像
 │   ├─ 加载辅助立方体
 │   ├─ 加载端元反射率 Excel…
 │   ├─ 单光谱计算 / 图像处理 …
 │   └─ …
 └─ Sparse unmixing
     ├─ 加载高光谱图像
     ├─ 加载辅助立方体
     ├─ 加载端元反射率 Excel…
     ├─ 单光谱计算
     ├─ 图像处理
     ├─ 退出 Sparse 单光谱模式
     └─ 显示丰度图… / 显示 RMSE 图
```

## Sparse unmixing（SUNSAL + SSA）

核心算法来自 Bioucas-Dias 的 SUNSAL（`matlab/sunsal.m` + `matlab/soft.m`），Python 实现见 `sunsal.py`。

### 流程

1. **加载高光谱图像**（I/F）
2. **加载辅助立方体**（PDS `.lbl` 或普通 ENVI `.hdr` / `.img` / `.dat`；band1 入射角 i，band2 观测角 e）
3. **加载端元 Excel**（第1列波长，第2列起反射率 REFF）→ 按实验室几何转为 **SSA**
4. **单光谱 / 图像处理**（弹窗设置 SUNSAL 参数）
   - 稀疏正则参数 λ
   - 是否和为 1（addone）
   - 是否非负（positivity）
   - 最大迭代次数（AL_ITERS）
   - 原始/对偶残差容差 TOL
   - 图像 I/F → REFF = I/F/cos(i) → SSA → SUNSAL
   - 重建 SSA → REFF → I/F，与观测对比显示

### Sparse 端元 Excel

| 列 | 内容 |
|----|------|
| 第1列 | 波长（μm 或 nm） |
| 第2列起 | 各端元反射率，表头为名称 |

### 量纲

| 数据 | 物理量 |
|------|--------|
| 图像 | I/F |
| Excel 端元 | REFF |
| 解混空间 | SSA |
| `I/F = REFF × cos(i)` | 入射角来自辅助立方体（缺省 i=30°, e=0°） |

---

## Hapke model

见上文历史说明：Excel 含 ρ/n/D 元数据；非线性 Hapke 质量丰度解算；图像背景端元等。

**加载辅助立方体**支持：
- CRISM DDR / PDS：`.lbl`（配合同目录 `.img`）
- 普通 ENVI：`.hdr`，或同名 `.img` / `.dat` / `.bsq` / `.bil` / `.bip`
