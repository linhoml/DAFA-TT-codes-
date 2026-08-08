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
2. **加载辅助立方体**（推荐：band1 入射角 i，band2 观测角 e）
3. **加载端元 Excel**（第1列波长，第2列起反射率 REFF）→ 按实验室几何转为 **SSA**
4. **单光谱 / 图像处理**
   - 图像 I/F → REFF = I/F/cos(i) → SSA
   - SUNSAL 稀疏解混（默认非负 + 和为 1）
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
