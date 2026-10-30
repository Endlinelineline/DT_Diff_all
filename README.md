# 广宁 16 台风机 DiffDiff 预测项目

本项目基于论文 **Differencing the Diffusion Trajectory toward Uncertain Components for Time Series Forecasting** 的 DiffDiff 源码，用于广宁风电场 16 台风机的联合功率预测。

项目根目录固定为：

```text
F:\DeepLearning\projects\DiffDiff_Guangning_Wind
```

`run.py` 是唯一命令行入口。模型、数据管线、实验流程和共享组件分别放在顶层的 `models/`、`data_provider/`、`exp/`、`layers/` 和 `utils/` 中。

## 任务定义

- 模型：DiffDiff
- 输入粒度：1 分钟
- 输入窗口：过去 1440 分钟
- 预测窗口：未来 15、30、45、60 或 120 分钟；默认 60 分钟
- 建模方式：16 台风机功率作为 16 个通道，一次训练一个共享模型
- 时间字段：`时间`
- 功率字段：`风机-P`
- 默认公共数据范围：2026-05-01 00:00 至 2026-09-09 23:59
- 数据划分：按时间顺序 70% / 10% / 20%，标准化器只使用训练段拟合

默认配置位于 `configs/dataset/guangning.yaml` 和 `configs/model/DiffDiff_Guangning.yaml`。

## 目录

```text
DiffDiff_Guangning_Wind/
├── configs/                         # 数据集与模型配置
├── data_provider/                   # 读取、清洗、缓存、切分和 Dataset
├── exp/                             # 训练、验证、测试和滚动评估
├── layers/                          # 可复用网络组件
├── models/
│   ├── DiffDiff.py                  # --model DiffDiff，暴露 class Model
│   └── MATSD.py                     # --model MATSD，暴露 class Model
├── scripts/
│   ├── data_preparation/guangning/
│   ├── long_term_forecast/guangning/
│   └── wind_rolling/guangning/
├── utils/                           # 指标、调度和通用工具
├── run.py                           # 统一入口与顶层调度
└── requirements.txt
```

`exp/exp_basic.py` 动态扫描 `models/` 顶层的 Python 文件。新增模型时，文件名必须与 `--model` 完全一致，并暴露 `class Model`；通常无需修改 `models/__init__.py`。

## 数据与缓存

默认原始数据目录：

```text
F:/DeepLearning/datasets/electic/风电数据集/guangningshuju
```

程序通过 `python-calamine` 递归读取 16 台风机的逐日 `.xls` 文件中的 `时间` 和 `风机-P`，并在数据根目录生成缓存：

```text
processed/
├── guangning_power_1min.csv
└── guangning_power_1min.metadata.json
```

预处理会按 `configs/dataset/guangning.yaml` 的规则自动识别疑似停机日：默认至少 12 台风机各自有不少于 60% 的有效分钟功率为 0，且各自当天数据覆盖率不少于 80%。抽查发现 2026-05-09 至 05-11 可能触发该规则，但最终排除日期以实际运行后的 metadata 为准。缓存不会包含判定出的整天；普通缺失分钟只在各连续保留区间内插值，训练和滚动评估的窗口也不会跨越缺口。metadata 会记录每日满足条件的风机数、排除日期和行数。此规则只能识别持续且同步的零功率，不能仅凭功率确认停机原因；需要调整灵敏度时修改三个 `shutdown_*` 参数，并用 `--rebuild_cache` 重建。

原始文件不会被复制、移动或覆盖。若源文件或其他预处理参数变化，可显式传入 `--rebuild_cache`。仓库内的普通数据应放在 `dataset/<DatasetName>/`；大型数据建议继续使用仓库外目录并通过 `--root_path` 引用。

## 环境

本机已配置 Python 3.11 环境 `F:\DeepLearning\envs\diffdiff-wind`，其中的 GPU 版 PyTorch 可识别 GTX 1650。PowerShell 中使用该解释器的完整路径：

```powershell
Set-Location -LiteralPath 'F:\DeepLearning\projects\DiffDiff_Guangning_Wind'
$projectPython = 'F:\DeepLearning\envs\diffdiff-wind\python.exe'
& $projectPython -s -m pip install -r requirements.txt
```

`requirements.txt` 使用已验证版本的最低要求，不会把现有较新版本降级。本机用户级 Python 目录有一份不完整的 pandas，因此命令使用 `-s` 忽略用户级包。若另建环境，先安装适配本机 GPU 的 PyTorch 和 torchvision，再安装其余依赖。运行前可检查：

```powershell
& $projectPython -s --version
& $projectPython -s -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GTX 1650 只有 4 GB 显存，默认批量为 32；若显存不足可降为 16 或 8。测试默认采样 20 条扩散轨迹，可通过 `--n_sample 100` 调整。

## PowerShell 运行

PowerShell 不能直接执行 Bash 脚本，建议在项目根目录直接运行等价命令。

构建数据缓存：

```powershell
& $projectPython -s run.py prepare --root_path 'F:/DeepLearning/datasets/electic/风电数据集/guangningshuju'
```

训练：

```powershell
& $projectPython -s run.py train --model DiffDiff --model_id guangning_16turbine --des joint_forecast --gpu 0 --batch_size 32 --epochs 100
```

训练其他预测长度时加 `--pred_len 15|30|45|60|120`，并为每次实验指定不同的 `--setting`，例如 `--pred_len 30 --setting guangning_fullorders_H30_001`。每个预测长度需要独立训练；测试命令会从对应 checkpoint 的 `run_config.json` 读取预测长度，续训也会沿用原 checkpoint 的长度。请按顺序运行单卡实验。

若需要在指定墙钟时间内结束本阶段（时间包含首次预处理），可加 `--max_wall_minutes 90`。程序会在训练批次边界停止，保留 `checkpoints/<setting>/last.ckpt` 和训练配置，并暂缓全量测评。之后用新 setting 从该 checkpoint 续训，避免覆盖上一阶段的权重：

```powershell
& $projectPython -s run.py train --model DiffDiff --model_id guangning_16turbine --des resumed --resume_checkpoint 'checkpoints/<上阶段setting>/last.ckpt'
```

不传 `--max_wall_minutes` 的续训会按配置继续训练，完成后自动生成分层测评指标和图表。新训练阶段在每轮验证集上反归一化到 kW 后计算 Strict Acc30，并优先选择 Acc30 最高的 checkpoint；Acc30 完全相同时选择 MSE 更低者。`best.ckpt`、`last.ckpt` 和 `final.ckpt` 分别表示最佳验证指标、最近训练状态和最终训练状态。每个阶段的具体 `setting` 以启动时的命令或输出目录为准。
若需先比较多个阶段的验证集表现，再选择 checkpoint 做测试，可在续训命令加 `--defer_eval`。已有的旧 checkpoint 不会因新增选模规则而改变；规则只适用于此后新启动的训练。

未显式传入 `--setting` 时，训练会生成包含时间戳的唯一 setting，并写入 `checkpoints/<setting>/` 和 `results/<setting>/`。

统一测试（将路径替换为实际 checkpoint）：

```powershell
& $projectPython -s run.py test --model DiffDiff --checkpoint 'checkpoints/<setting>' --n_sample 20
```

如果已经有反归一化到 kW 的预测和真实值 `.npy`，可跳过耗时的模型推理，直接生成新报告。原始数组若是 `[样本, 预测步, 风机]`，使用默认 `NHT`；若已经是 `[样本, 风机, 预测步]`，传 `--source_layout NTH`。报告会抽查真实值与原始 kW 数据一致，防止误把标准化数值当功率指标：

```powershell
& $projectPython -s run.py report --model DiffDiff --checkpoint 'checkpoints/<setting>/best.ckpt' --predictions 'results/<旧测试setting>/predictions.npy' --truth 'results/<旧测试setting>/truth.npy' --source_layout NHT
```

`test` 与 `report` 默认自动生成带时间戳的新 setting；指定 `--setting` 时请选用未使用过的名字，程序会拒绝覆盖已有报告。

每 15 分钟发布点滚动测试：

```powershell
& $projectPython -s run.py rolling15 --model DiffDiff --checkpoint 'checkpoints/<setting>' --release_interval 15 --n_sample 20
```

## Bash 脚本

在 Git Bash、WSL 或其他 Bash 环境中运行：

```bash
export PYTHON_BIN='F:/DeepLearning/envs/diffdiff-wind/python.exe'
bash scripts/data_preparation/guangning/prepare_guangning.sh
bash scripts/long_term_forecast/guangning/DiffDiff_guangning_train.sh
CHECKPOINT='checkpoints/<setting>' bash scripts/long_term_forecast/guangning/DiffDiff_guangning_test.sh
CHECKPOINT='checkpoints/<setting>/best.ckpt' PREDICTIONS='results/<旧测试setting>/predictions.npy' TRUTH='results/<旧测试setting>/truth.npy' bash scripts/long_term_forecast/guangning/DiffDiff_guangning_report.sh
CHECKPOINT='checkpoints/<setting>' bash scripts/wind_rolling/guangning/DiffDiff_guangning.sh
```

脚本只设置参数、环境变量并调用 `python run.py`，不会在 `scripts/` 下写日志或结果。

## 指标

报告统一使用 `[样本, 风机, 预测步]`，功率单位 kW，并在反归一化后计算。Strict Acc30 是真实值**严格大于** 100 kW 且绝对误差不超过真实值 30% 的点所占百分比；MAPE 同样只统计真实值严格大于 100 kW 的点。MAE、MSE、RMSE 使用全部成对有限数值，单位分别为 kW、kW²、kW。没有有效功率点时 Acc30/MAPE 标为 `null`，并保存有效点数、总点数和占比。

`metrics.json` 给出所有点、仅第 60 步终点、逐风机、风机等权宏平均、场站总功率和逐预测步指标。场站指标先对 16 台风机功率求和，再按相同公式计算；因此场站 Acc30 与逐风机 Acc30 不可直接等同。表格文件为 `per_turbine_metrics.csv`、`per_horizon_metrics.csv`、`site_metrics.csv`；`predictions.npz` 保存 kW 预测、真实值、风机编号、目标时间戳和历史结束时间戳。另有各图对应的数据表及 `figure_data_manifest.json`。DTW/TDI 属可选指标，当前没有计算。

`test_results/<setting>/` 包含 16 风机 4×4 历史—未来曲线、分风机 Acc30、MAE/RMSE/MAPE、逐预测步总体与 Macro 曲线、场站总功率时序、风机×预测步绝对误差热力图、真实/预测散点图和分风机误差箱线图，共八张高清 PNG。

需要对照 Time-Series-Library 的 `utils/metrics.py` 时，在 `run.py test` 命令加 `--tslib_metrics`。程序会在新的 `results/<setting>/` 中保存完整预测/真实值 `.npy`、MAE/MSE/RMSE/MAPE/MSPE/RSE/CORR 指标 JSON，并固定默认测试随机种子 2021。原版 MAPE/MSPE 在真实值为 0 时除以零，JSON 会记为 `null`，同时提供明确标注的“仅非零真实值”补充值；原版 CORR 公式并非标准 Pearson 相关系数。

## 输出

```text
checkpoints/<setting>/                 # 模型权重、训练配置和噪声调度
results/<setting>/                     # 指标与标准测试结果
test_results/<setting>/                # 标准测试与 report 的高清图表
results/wind_rolling/<setting>/        # 风电滚动预测、指标和图表
m4_results/<model>/                    # M4 预测（若使用 M4 任务）
```

滚动评估可通过 `--rolling_output` 显式改写输出目录。不要把权重、日志、图片或预测结果写入源码目录或 `scripts/`。

## 15 分钟滚动评估

滚动评估在测试段每个 `00/15/30/45` 分钟发布点，仅使用该时刻及以前的真实历史，预测未来 60 分钟。它会输出全部未来分钟与 60 分钟端点的指标、端点预测 CSV、三个示例窗口的完整预测 CSV，以及完整的分层指标、预测数组和八类新图表；原有风场总功率、分风机和误差诊断图也保留。图表位于 `results/wind_rolling/<setting>/`。
