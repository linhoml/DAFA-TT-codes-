将分类模型放在此目录：

- `model_best.pth`        LSGA 分类权重（含 args / model_state_dict）

Identification → 模型应用 →「默认内置已训练模型」读取该文件。
读取影像后会自动截取 1.02–2.6 μm 并做与训练相同的预处理，不再需要 `.pkl`。
