# Identification 模型文件

## 内置模型（模型应用 → 默认内置已训练模型）

将训练好的分类权重放到本目录：

```text
data/identification/builtin/model_best.pth
```

软件菜单 **Identification → 模型应用** 会优先使用该文件。
读入影像后自动截取 1.02–2.6 μm，并做无效值填充 / 去尖峰 / 空间修补 / SG / L2，不再使用 `preprocess_model.pkl`。

## 本次训练的新模型

**Identification → 模型训练** 结束后，会在输出目录写入：

- `result/crism/crism_seed0_best.pth`

并把路径记录到 `data/identification/last_trained.json`。
**模型应用** 中选择「使用模型训练/测试得到的新模型」即可。

## 输入数据格式

详见 `app/identification/README.md`。训练/测试的立方体与标签支持 `.mat`、`.img`、`.dat`、`.hdr`、`.lbl` 等。
对话框内也有格式说明。
