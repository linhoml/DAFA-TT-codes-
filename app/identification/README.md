# Identification
软件菜单 **Identification** 下有两个子菜单：

| 子菜单 | 菜单 | 作用 |
|------|------|------|
| **LSGA** | 模型训练 / 测试 / 应用 | 截取 1.02–2.6 μm 后训练/测试/应用 LSGA |
| **HBM** | 模型训练 / 测试 / 应用 | 嵌入 [Banus/crism_ml](https://github.com/Banus/crism_ml) 的 Hierarchical Bayesian Model（bland-pixel ratio + 矿物分类） |

HBM 训练数据来自 Zenodo [13338091](https://zenodo.org/records/13338091)，放到 `data/identification/hbm/datasets/`。源码在 `third_party/crism_ml/`（Apache-2.0）。

训练/测试对话框会显示数据格式说明。立方体和标签支持 **.mat / .img / .dat / .hdr / .lbl** 等（以及 .npy、.tif）。LSGA 训练可选：

- 单个立方体 + 单个标签图
- **立方体文件夹 + 标签文件夹**：每个立方体配对文件名最接近的标签（去掉 `_label` / `_gt` 等后缀后比较，一对一）

LSGA 模型应用可选：

- 默认内置模型：`data/identification/builtin/model_best.pth`
- 本次训练的新模型：读取 `data/identification/last_trained.json`

---

## 命令行模块（嵌入自原工程）

原 `main.py` 现为 `train.py`。其余文件：


```text
project/
├── preprocess.py
├── main.py
├── test.py
├── test_full_image.py
├── crism_common.py
├── lsga.py
├── merged_train.json
├── merged_test.json
└── merged_test_full.json
```

| 文件 | 作用 |
|---|---|
| `preprocess.py` | CRISM 预处理；拟合/读取 `process_model.pkl` |
| `train.py` | 训练入口（原 main.py）；训练和内部测试 |
| `test.py` | 外部场景定量测试，也支持无标签预测 |
| `test_full_image.py` | 整图推理、分类图和详细指标 |
| `merged_*.json` | 训练和测试配置示例 |


## 2. 数据格式

输入为三维高光谱数组，文件可以是：

```text
.mat   MATLAB
.img / .dat / .hdr / .bsq / .bil / .bip   ENVI（.img/.dat 自动找同名头文件）
.lbl   PDS 标签（配合同目录 .img）
.npy / .npz / .tif
```

立方体排布支持：

```text
HWB = Height × Width × Bands
BHW = Bands × Height × Width
```

使用 `--data_layout HWB` 或 `--data_layout BHW` 指定。

预处理支持：
- 438 波段：自动保留原始 1-based 第 4–243 波段，共 240 波段；
- 240 波段：保持不变。

原始 `.mat` 中数组 key 可用 `--data_key` 指定。`preprocess.py` 输出的预处理文件统一使用：
```text
data
```

所以训练/测试 JSON 一般设置：
```json
"data_key": "data"
```


## 3. 完整流程

```text
训练 CRISM 数据
    ↓
preprocess.py --mode train
    ↓
process_model.pkl + preprocessed training data
    ↓
main.py --config merged_train.json
    ↓
*_best.pth
    ↓
外部 CRISM 数据
    ↓
preprocess.py --mode test/transform
复用训练阶段 process_model.pkl
    ↓
preprocessed external data
    ↓
test.py 或 test_full_image.py
```

---


# 4. 外部场景预处理

外部数据复用训练阶段保存的 `process_model.pkl`：

```bash
python preprocess.py \
  --mode test \
  --tile_dir /data/CRISM数据2/HRL000040FF \
  --save_dir /data/CRISM数据2/HRL000040FF/preprocess \
  --model_path ./process_model.pkl \
  --input_pattern "*_corr.mat" \
  --data_layout HWB \
```

---

# 5. 外部场景定量测试 `test.py`

执行：

```bash
python test.py \
  --config merged_test.json \
  --checkpoint_path ./model_best.pth
```

### single 模式

有标签时，只在标签 `1..K` 的像元上计算：

```text
OA
AA
Kappa
per-class accuracy
```

无标签时：

```json
"test_label": ""
```

程序会预测，但不计算指标。

### tile 模式

```json
{
  "test_mode": "tile",
  "tile_dir": "./preprocessed_tiles",
  "label_path": "./label.mat",
  "tile_pattern": "tile_*.mat",
  "tile_w": 600,
  "tile_position_mode": "tile_id"
}
```

如需在原 merged image 上排除训练坐标：

```json
"exclude_train_points": true,
"split_path": "/path/to/split_seed0.npz"
```

---

# 6. 外部场景整图推理 `test_full_image.py`

执行：

```bash
python test_full_image.py --config merged_test_full.json
```

有标签时会计算指标。

---

# 7. 最简运行顺序

```bash
# 1. 在训练数据上拟合预处理模型，并生成训练预处理数据
python preprocess.py --mode train ...

# 2. 训练分类模型
python main.py --config merged_train.json

# 3. 外部数据复用训练 process_model.pkl
python preprocess.py --mode test ...

# 4A. 外部数据有标签定量测试
python test.py --config merged_test.json

# 4B. 外部数据进行整图测试
python test_full_image.py --config merged_test_full.json
```

---

