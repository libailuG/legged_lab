# G1 外骨骼 Sim2Real

本目录按策略版本隔离保存部署代码，避免新旧观测定义和控制周期混用。

- `g1_assist_exoskeleton_v1/`：原有 v1（100 维观测、50 Hz）完整归档。
- `g1_assist_exoskeleton_v2/`：当前 v2（150 维观测、100 Hz）部署实现。

各版本目录互相独立。新实验和生成文件只写入对应版本目录。

- `g1_assist_exoskeleton_v3/`：v3 成对助力（150 输入、单动作、100 Hz、±10 N·m），NumPy/C99 独立部署核心、离线演示及可选 G1 PID 接口，详见该目录 README。

- `g1_assist_exoskeleton_v4/`：v4 单峰助力（400输入、单动作峰值、100 Hz），包含NumPy/C99脉冲状态机和model_499部署权重。

- `g1_assist_exoskeleton_v5/`：v5 低速与末端保持助力（450输入、峰值/时长双动作、100 Hz），包含NumPy/C99控制器及本次 `pulse_credit_fix_1024/model_499.pt` 权重，详见 [v5部署说明](g1_assist_exoskeleton_v5/README.md)。
