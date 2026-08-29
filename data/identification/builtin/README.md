将内置（已训练）模型放在此目录：

- `preprocess_model.pkl`  训练阶段拟合的预处理模型
- `model_best.pth`        LSGA 分类权重（含 args / model_state_dict）

Identification → 模型应用 →「默认内置已训练模型」读取这两个文件。
