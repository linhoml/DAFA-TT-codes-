# Identification 模型文件

## 内置模型（模型应用 → 默认内置已训练模型）

将训练好的文件放到本目录：

```text
data/identification/builtin/preprocess_model.pkl
data/identification/builtin/model_best.pth
```

软件菜单 **Identification → 模型应用** 会优先使用这两份文件。

## 本次训练的新模型

**Identification → 模型训练** 结束后，会在输出目录写入：

- `preprocess_model.pkl`
- `result/crism/crism_seed0_best.pth`

并把路径记录到 `data/identification/last_trained.json`。
**模型应用** 中选择「使用模型训练/测试得到的新模型」即可。

## 输入数据格式

详见 `app/identification/README.md`。训练/测试对话框也会显示同样的说明。
