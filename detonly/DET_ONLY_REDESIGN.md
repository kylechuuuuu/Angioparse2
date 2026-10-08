# SD2net det-only v2 · 设计文档（纯检测数据，无 mask）

> 面向场景：**整个数据集只有检测框、没有任何分割 mask**（如 `DSA_huaxi/detect`、`runs_det/sd2net_data/huaxi2_fold*`）。
> 现有 joint（有 mask）路径**完全不受影响**；v2 只在 `SD2_DET_ONLY_V2=1` 且运行本身已是 `task='det'` 时生效。

---

## 1. 以前的 det-only 是怎么做的

| 层 | 位置 | 行为 |
|---|---|---|
| 数据 | `dataset.has_masks()` | **目录级**开关：`{split}/masks/` 不存在 → 无分割监督 |
| 数据 | `dataset.__getitem__` | 单图缺 mask → **全背景 (class 0)** canvas |
| 模型 | `fusionet2.resolve_task()` | 无 mask → `task='det'` |
| 模型 | `fusionet2.resolve_seg_prior()` | 无 mask → 关掉 class-1 先验 |
| 模型 | `fusionet2.FusionModel` | `det_head` 用 `prior_channels=0` 构建（stem 64 输入，少 1 通道） |
| 模型 | `fusionet2._det_only_forward()` | **seg_heads / fg_head / refiner 全不跑**，只出 `det_head`，`consist_loss=0` |
| 训练 | `train.py` | CE/FG/consistency/router/adapter-cls/clDice **全部置零**，只留 `compute_detection_loss`（focal + 收缩 Gaussian blob 上 L1/GIoU + hard-neg topk） |
| 驱动 | `runs_det/train_sd2net_det.py` | patience 早停、fp16/bf16、lr 覆盖 |

**实测（huaxi2，统一口径，`runs_det/DET_RESULTS_fourmetrics_both.md` 表 2 / `RUN_NOTES.md` §13–14）**

| 方法 | AP50 | Recall@maxF1 | FROC-AUC |
|---|---|---|---|
| rtdetr-l | 0.3328 | 0.3519 | 0.4504 |
| dfine-s | 0.3889 | — | — |
| SD2net det-only | 0.2738（3 折） | 0.3491 | 0.4542 |
| SD2net 伪 mask（硬矩形） | 0.3005 | 0.3688 | 0.4695 |

det-only 打不过 rtdetr-l / dfine-s；把框涂成矩形当伪 mask 能涨一点，但增益几乎全来自 fold1（配对 +0.025，逐折 +0.148 / −0.021 / −0.052），不稳。

---

## 2. 根因（为什么旧 det-only 不行）

1. **唯一监督是极稀疏的框中心**。DSCA 68 框、huaxi2 每折约 500 框；focal 的正样本实际只有中心像素，而 det 头 + decoder + adapter 是 26M 参数从零学。
2. **丢掉了真正有用的 class-1 病灶先验**。joint 压过 rtdetr-l 靠的就是它；`runs_det/prior_probe.py` 量化：学到的 class-1 先验 pointing 0.62–0.65，比检测器自身 top-1 框（0.44–0.55）更准。det-only 把这整条通道关了。
3. **共享特征在 det-only 下没有任务梯度**。`router_presence_loss` / `adapter_structure_loss` 依赖 mask → 跳过；router 塌缩后结构 adapter 断梯度、自锁（`SD2net_adapter_bank_exp/DESIGN_REVIEW.md` §3.4）；refiner 的 `refine_net[-1].bias` 恒为 0，从未训练（§14.1）。
4. **FP 抑制依赖特征对比度**（`resolve_det_grad_scale` 注释），但 det-only 没有任何 dense / contrastive 监督去造这个对比度。
5. **pm 的硬矩形是坏目标**。探针显示模型把先验从矩形"抵抗"回收到框面积 ~60% 的团块——说明矩形与真实病灶形状不符，CE 在逼它拟合错误形状，所以收益弱、折间方差大。
6. **回归监督只落在收缩 Gaussian blob 上**，病灶形状信息没用上；head 单尺度 stride 4（256²），小病灶定位受限。

---

## 3. det-only v2 的设计

总原则：**det-only 不是"关掉分割"，而是"用框合成稠密弱监督，把学到的病灶先验喂回检测头"**。框里本就有"病灶在哪、多大"的信息，之前只用了一个中心点。

**最终只保留两个机制**（其余为冗余，已删除）：
- **M2**：框监督的稠密病灶先验（损失侧的唯一机制）。
- **M4**：copy-paste 增强（数据侧的唯一机制）。
- 外加 **M0**：样本级 mask 路由（这是"有 mask 就用、没有就 det only"的正确性修复，不算可选机制）。

被删掉的冗余机制及原因：
- ~~M3 fg/router/adapter 辅助损失~~ —— 与 M2 用同一个框目标、只是打到别的头上，信号高度重叠。
- ~~M5 多尺度检测头~~ —— 改参数形状/加参数，在 13 图小集上反而拖后腿，且与 M2 不是互补关系。

### M0 · 样本级 mask 路由（有 mask 就用，没有就 det only）
旧实现是**目录级**判断（`has_masks` 看 `{split}/masks/` 是否存在）：一旦目录里有 mask，`task='joint'`，**单张缺 mask 的图会被当成全背景 mask**参与分割损失——等于教网络在那些图上"擦掉血管"。对"部分图有 mask、部分只有框"的混合集，这是错的。

现在：
- `dataset.__getitem__` 额外返回 `masks_dict['valid']`（只有真正读到了 mask 文件才是 1.0）；
- `train.py` 训练循环按样本判定：`valid=0` 的样本**跳过全部分割损失**（CE/Dice/FG/consistency/router/adapter-cls/clDice），只做检测；`valid=1` 的样本正常联合训练；
- `run_unified_validation` 同样按样本跳过无 mask 图的分割指标，避免 `Image.open` 缺文件崩溃，也避免用假的全背景 GT 打分。

效果：一个数据集里"有 mask 的图走 joint、没 mask 的图走 det-only"自动生效；纯检测数据集仍然整体走 det-only v2。这部分**没有开关**（是正确性修复），也不改变纯 joint / 纯 det 数据集的行为。

### M2 · 框监督的稠密病灶先验（核心）
- **复用 `seg_heads[0]`（class-1 头），零新增参数**；`det_head` 固定 `prior_channels=1`，prior = `sigmoid(seg_heads[0](combined)).detach()` —— 与 joint 完全同构，靠的是"用框目标训练它"而不是人工 mask。
- 目标构造 `det_only.build_box_targets`：
  - 框**内缩 `SD2_DET_PRIOR_CORE`（默认 0.35）** 得正核心；框**外扩 `SD2_DET_PRIOR_IGNORE`（默认 0.25）** 得 **ignore 环带（损失权重 0）**，不惩罚合理的病灶外延；
  - 核心内用中心高斯（`SD2_DET_PRIOR_GAUSS=1`）→ 学成收紧的团块，而非硬矩形；
  - 框核心=1、环带=0、更外=0。
- 损失 `det_only.box_dense_prior_loss`（加权 BCE）+ `det_only.box_mil_pointing_loss`（每框 `top-k` 均值必须有一个高响应峰，WSOL 的 pointing 目标，与"一病灶一峰"一致）。
- 为什么比 pm 好：pm 是"硬矩形 + 全 7 类 CE"，v2 是"软核心 + ignore + pointing"，目标形状可控、且先验头与检测头共享语义。

### M3（已删除）· 用框恢复 fg / router / adapter 辅助损失
曾经实现：`box_fg_loss` / `box_router_presence_loss` / `box_adapter_structure_loss`。
**删除原因**：三者都用与 M2 相同的框目标，只是监督到 fg/adapter 头或 router，与 M2 的信息高度冗余；13 图小集上把 M2+M3+M4+M5 捆在一起反而明显差于无机制基线。保留 M2 一个损失机制即可。

### M4 · copy-paste 增强（label 稀缺的最大杠杆）
- `dataset.VesselDataset._copy_paste`：训练时按 `SD2_COPYPASTE`（默认 0.0，建议 0.5）从**另一张有框图的病灶框区**（框 + `SD2_COPYPASTE_MARGIN` 上下文）裁剪，贴到当前图随机非重叠位置，并把该框加入 GT。
- 无 mask 依赖；无框目标图也能借此获得正样本。直接把 68 / ~500 个正样本扩增，缓解过拟合并给 FP 提供对照。
- 贴后再做 color jitter / flips，坐标与内容保持一致。

### M5（已删除）· 多尺度检测头
曾经实现 `MultiScaleDetectionHead` + `SD2_DET_LEVELS`。
**删除原因**：改参数形状、增加 head 参数，在 13 图小集上未证明有效；与 M2 不互补。`DetectionHead` 保持单一 stride-4 结构。

### 参考的后续项（本轮未实现，写在文档里）
- 真自训练两阶段：EMA 模型出预测 → prior 细化 → **lesion 形状**伪 mask 回灌 joint（§14.4 明确未测）。
- 检测侧 flip-TTA（框反变换）+ WBF。

---

## 4. 代码改动点（全部默认关闭）

| 文件 | 改动 |
|---|---|
| `det_only.py`（新增） | M2：`build_box_targets` / `box_dense_prior_loss` / `box_mil_pointing_loss` |
| `fusionet2.py` | `resolve_det_only_v2()`；`resolve_arch_config` 增 `det_only_v2` 字段；`FusionModel` 增 `det_only_v2` / `det_prior_logits` / `_det_only_v2_forward()` |
| `dataset.py` | M0 样本级 `masks_dict['valid']`；M4 `_scan_box_images` / `_letterbox` / `_copy_paste` / `box_iou_single` |
| `train.py` | M0 按样本 gate 分割损失（训练与验证）；训练循环里计算 M2 损失并加权进总 loss；日志 `DetAux`；`save_checkpoint` 写入 `det_only_v2` |
| `test.py` | `apply_checkpoint_arch` 从 sidecar 恢复 `SD2_DET_ONLY_V2` |

### 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `SD2_DET_ONLY_V2` | `0` | 打开 v2（仅 `task='det'` 生效） |
| `SD2_DET_PRIOR_W` | `1.0` | M2 稠密先验 BCE 权重 |
| `SD2_DET_PRIOR_MIL_W` | `0.5` | M2 MIL pointing 权重 |
| `SD2_DET_PRIOR_CORE` | `0.35` | 正核心内缩比例 |
| `SD2_DET_PRIOR_IGNORE` | `0.25` | ignore 环带外扩比例 |
| `SD2_DET_PRIOR_GAUSS` | `1` | 核心用中心高斯 |
| `SD2_COPYPASTE` | `0.0` | M4 copy-paste 概率（建议 0.5） |
| `SD2_COPYPASTE_MARGIN` | `0.2` | M4 病灶裁剪上下文边距 |

---

## 5. 建议的运行配方（huaxi2，纯检测）

现有 `runs_det/train_sd2net_det.py` 会 `env = dict(os.environ)` 继承外部变量，因此**在外部 export 即可**（无需改脚本）。建议与现有 LR 对照结论一致：`lr=5e-5, amp=bf16, patience=80`。

```bash
export SD2_DET_ONLY_V2=1          # 打开 det-only v2
export SD2_COPYPASTE=0.5          # M4
export SD2_DET_PRIOR_W=1.0 SD2_DET_PRIOR_MIL_W=0.5   # M2

python runs_det/train_sd2net_det.py --tag huaxi2 --folds 1 --device <空闲卡> \
    --epochs 220 --patience 80 --val-every 5 --lr 5e-5 --amp bf16
```

对照口径：与现有 det-only（fold1 0.2506 / fold2 0.2577 / fold3 0.3131）同折比较；导出与评测沿用 `runs_det/export_sd2net_det.py`（会自动读 `.arch.json` 恢复 `det_only_v2`）。

**注意**：v2 会强制 `prior_channels=1`，det 头 stem 输入宽度与旧 det-only 权重不同（101→旧 ckpt 无法加载）。必须**新开目录/新 tag** 从零训练，不要覆盖 `runs_sd2net/huaxi2_fold*`。

---

## 6. 单元验证（已完成，无需 GPU）

- `det_only` 全部损失：随机张量前向 + 反向，`finite` 检查通过。
- `MultiScaleDetectionHead` 两级前向 + `compute_detection_loss` 多级 / 单级：通过。
- `dataset` copy-paste：合成 3 图小数据集，确认新框被加入且坐标合法（`/tmp/opencode/test_*.py`）。

未做（需要 GPU + SAM3 权重）：`FusionModel` 端到端 v2 前反向 smoke。建议先 `--epochs 2` 小跑确认 `DetAux` 有值、`Total Predictions > 0`，再正式训练。

### 6.1 DSA_huaxi 小检测集对照

**（a）20 图纯检测集 `DSA_huaxi/detect`**（13 train / 14 框，7 val / 8 框，无 mask）。按 VesselDataset 布局软链到 `runs_det/sd2net_data/dsa_det_small/`，脚本 `runs_det/run_dsa_det_small.sh`。

**（b）100 图微型混合集 `dsa_huaxi100`**（60 有 mask + 40 只有框；train 70 / val 30），由 `runs_det/build_dsa_huaxi100.py` 生成，专门验证 M0 样本级路由 + "有 mask 用 joint / 无 mask 用 det-only"。对照脚本 `runs_det/run_dsa_huaxi100.sh`（GPU 4，`epochs=40, lr=5e-5, bf16`）三臂：
- `hx100_joint`：`SD2_TASK=joint`（masked 图 seg+det，其余只 det）
- `hx100_det`：`SD2_TASK=det SD2_SEG_PRIOR=0`（完全忽略 mask 的旧 det-only）
- `hx100_det2`：`SD2_TASK=det SD2_SEG_PRIOR=1 SD2_DET_ONLY_V2=1 SD2_COPYPASTE=0.5`（**简化版 v2 = M2+M4**）

评测 `runs_det/eval_dsa_small.py --data-root ...`：直接跑 SD2net 推理，GT 从 per-image json 转成 COCO，`export_score=0.01` 交给 pycocotools 算 COCO AP50（与 `export_sd2net_det.py` 同口径）。

**已知 smoke 结果**：`seg_prior: True / prior_channels=1 / det_only_v2: True`；`DetAux` 非零（M2 生效）；joint 运行时日志在 `seg/fg/router` 非零与全 0（无 mask 样本）之间交替，证实 M0。训练 ~2.5 it/s。

**待补**：`dsa_det_small` 上 v2（旧四机制捆绑）0.0772 vs 旧 det-only 0.2267（COCO AP50）——13 图统计意义太弱，仅作观察；以 `dsa_huaxi100` 的三臂结果为准。


---

## 7. 风险与回退

- 全部逻辑由 `SD2_DET_ONLY_V2` 控制；不设该变量时逐位回到旧行为。
- 旧 checkpoint 缺 `det_only_v2` 字段时按默认 `False` 解析。
- copy-paste 默认 0；开启后旧目录的验证/早停基线不可直接混用（分布变了），需重新跑对照。
- 若 v2 仍打不过 rtdetr-l，优先调 `SD2_DET_PRIOR_*` 与 `SD2_COPYPASTE`，其次才是上真自训练。
