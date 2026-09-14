# Angioparse2 (FusionEt2)

基于 **SAM3 (Segment Anything Model 3)** 视觉编码器的多结构**脑血管分割 + 病灶检测**联合模型。

模型在冻结的 SAM3 视觉编码器上注入轻量 AdapterBank/MLPAdapter 进行参数高效微调，并联合 UNet 分支、PixelShuffle 解码器、不确定性引导的迭代残差精化器（IterativeResidualRefiner）与无锚框检测头，实现多任务输出。

## 任务与架构

- **分割**：6 个血管结构类别（类别 0 为背景），按类别独立的分割头 + 全局前景头 + 前景结构一致性损失。
- **检测**：CenterNet 风格的单类无锚框检测头（中心 heatmap + 偏移回归 + 尺寸回归，tanh 参数化），推理时置信度阈值 + NMS。
- **损失**：Dice/CE 分割损失、结构一致性损失、平衡 focal + dense-L1 + GIoU 检测损失、router presence 辅助损失；权重可按结构类别调节。
- **训练策略**：EMA 权重验证与保存、WeightedRandomSampler 类别均衡采样、一致性权重随 epoch 线性退火、可选早停（保持 LR 调度不变）。

## 目录结构

```text
├── fusionet2.py          # 模型定义：FusionModel、SAM3VisionEncoder、AdapterBank、
│                         # UNetBranch、PixelShuffleDecoder、DetectionHead、精化器等
├── dataset.py            # VesselDataset：分割掩码 + 检测框（归一化 cxcywh）联合数据加载与增强
├── train.py              # 训练脚本（全部超参可通过 SD2_* 环境变量覆盖）
├── test.py               # 验证/推理：分割掩码导出、检测框解码 + NMS + 可视化
├── calculate_metrics.py  # 分割指标计算（按 label.json 逐类别统计）
└── README.md
```

> 注意：`test.py` 中还引用了 `detection_metrics.py`（检测指标评估，AP50 等），当前仓库未包含该文件，如需运行检测评估请自行补充或联系作者。

## 数据集格式（DSCA_new）

```text
DSCA_new/
├── train/
│   ├── images/           # 原始图像 (png/jpg)
│   └── masks/            # RGB 调色板掩码
├── val/
│   ├── images/
│   └── masks/
├── train_detect/
│   └── annotations/      # COCO 风格边界框标注（内部转为归一化 cxcywh）
├── val_detect/
│   └── annotations/
└── label.json            # 颜色 -> 类别映射配置
```

掩码颜色到类别 ID 的映射（见 `dataset.py::COLOR_TO_ID`）：

| ID | 名称 | RGB |
| ---- | ------ | ----- |
| 0 | background | (0, 0, 0) |
| 1 | noise (ICA bulb 检测目标区) | (0, 162, 232) |
| 2 | carotid_artery 颈动脉 | (134, 0, 21) |
| 3 | vertebral_artery 椎动脉 | (185, 122, 87) |
| 4 | anterior_cerebral_artery 大脑前动脉 | (255, 242, 0) |
| 5 | middle_cerebral_artery 大脑中动脉 | (200, 191, 231) |
| 6 | posterior_cerebral_artery 大脑后动脉 | (239, 228, 176) |

## 环境依赖

- Python 3.10+
- PyTorch (CUDA)
- HuggingFace `transformers`（需包含 `Sam3VideoConfig` / `modeling_sam3` 的版本）
- torchvision、numpy、Pillow、tqdm

## 快速开始

```bash
# 训练（默认 220 epochs，验证每 5 个 epoch，输出 best_fusion_model_{seg,det}.pth）
python train.py

# 验证 / 推理（需 DSCA_new 数据与训练得到的 checkpoint）
python test.py

# 计算分割指标
python calculate_metrics.py \
    --pred_dir results/overall \
    --gt_dir DSCA_new/val/masks \
    --label_path DSCA_new/label.json
```

数据集根目录默认为 `DSCA_new/`（见 `train.py` / `test.py` 中的 `DATA_ROOT`），请按上述结构放置数据。

## 主要可配置项（环境变量）

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SD2_EPOCHS` | 220 | 训练轮数 |
| `SD2_STOP_EPOCH` | 0 | 早停轮数（保留完整 LR/一致性调度），0 为关闭 |
| `SD2_LR` | 1e-4 | 学习率 |
| `SD2_ADAPTER_LR_MULT` | 5.0 | Adapter 参数学习率倍数 |
| `SD2_WORKERS` | 8 | DataLoader 工作进程数 |
| `SD2_VAL_EVERY` | 5 | 验证间隔（epoch） |
| `SD2_ITERATIONS` | 2 | 迭代精化次数（1 = 关闭 memory-bank 精化，更快） |
| `SD2_EMA` | 0.999 | EMA 衰减系数 |
| `SD2_DET_WEIGHT` | 1.0 | 检测损失权重 |
| `SD2_CONSIST_W0` / `SD2_CONSIST_W1` | 0.5 / 0.15 | 一致性损失权重（起始 / 末尾） |
| `SD2_DILATIONS` | `1,2,4,8` | 检测主干空洞卷积栈，置空可关闭（换取分割精度） |
| `SD2_EXTRA_UP` | 1 | SAM 解码器额外 PixelShuffle 上采样级数（会改变 checkpoint 兼容性） |
| `SD2_SIZE_PRIOR` | 0.10 | 检测尺寸通道的偏置初始化（平均归一化 GT 框尺寸） |
| `SD2_OFFSET_SCALE` | 4.0 | 中心偏移回归的 tanh 半幅（heatmap 像素） |
| `SD2_OUT_PREFIX` | `best_fusion_model` | checkpoint 文件名前缀 |
| `SD2_LOG` | `train.log` | 训练日志文件 |
| `SD2_SAVE_VIZ` | 0 | 训练验证时是否保存可视化 PNG |
| `SD2_SAVE_FULL` | 0 | checkpoint 是否保存完整模型状态（默认精简，避免 1.9GB 大文件） |

示例：

```bash
SD2_EPOCHS=120 SD2_LR=5e-5 SD2_ITERATIONS=1 python train.py
```

## 输出

- `best_fusion_model_seg.pth`：验证 Dice 最优的分割 checkpoint（EMA 权重）
- `best_fusion_model_det.pth`：验证检测指标最优的 checkpoint（EMA 权重）
- `train.log`：完整训练日志（含每次验证的逐类别指标）

## License

仅供研究使用。
