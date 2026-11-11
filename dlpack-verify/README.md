# DLPack 混合过渡态验证: MindSpore → PyTorch 渐进迁移 (零拷贝交换 + 梯度桥接)

**业务背景**:客户模型当前为 MindSpore,规划迁移至 PyTorch;因迁移人力有限且模型持续演进,选择 DLPack 混合编程作为**迁移过渡态**——未迁移模块留在 MS,已迁移模块改用 PT,边界两侧零拷贝交换 + sens 梯度桥接联合训练。在 **昇腾 910B(Atlas A2)**上验证过渡态的"敢切(等价)、选对(机制)、全程适用(对称)",最终由一个脚本生成验证报告。

**工作流**:在服务器上跑实验(自动落盘机器可读摘要与逐步日志)→ 把 `results/` 目录拷回 Windows → 运行**一键报告脚本**生成最终交付报告。

**对照设计**(三要素):

- **敢切(切换等价性)**:每次运行同跑 **MS 参照臂**(纯 MindSpore 端到端 = 唯一基线 B0)与混合臂,判收敛 / held-out 指标 / 训练验证曲线(loss 曲线与过拟合监测为必须项)统计等价——混合态 vs 现状"训练效果一致";
- **选对(交换机制主对照 E2 vs E3)**:同为混合场景,唯一变量 = 交换机制(E2 = DLPack / E3 = host 拷贝)——混合前段的调度开销两跑完全相同,差值纯粹来自交换机制,归因天然干净;
- **全程适用(形态对称性 E4)**:PT前段+MS后段(迁移过半形态)实测,证明切分方向不影响机制。

## 1. 目录结构

```
dlpack-verify/
├── scripts/                # 可执行脚本
│   ├── run_all.py          #   一键全跑 E0~E5 (tee 时间戳日志, 末尾自动导 xlsx)
│   ├── check_env.py        #   环境自检 (CPU/910B 通用)
│   ├── verify_dlpack.py    #   E1 机制验证 (11 个 STEP / 17 检查项, 含 GNN C1/C3)
│   ├── train_hybrid.py     #   MS 参照臂 + E2/E3 主对照 + E4 对称形态 + E5 --bench
│   ├── export_xlsx.py      #   数据结果整合: 多 sheet Excel 工作簿 (时间戳命名)
│   └── make_report.py      #   一键报告: 读 results/ 摘要 → 绘 5 图 + 渲染验证报告
├── docs/                   # 文档
│   ├── 验证报告.md         #   最终交付报告 (npu 数据齐备时由 make_report.py 生成)
│   ├── 验证报告_draft.md   #   草稿 (数据不全/cpu 调试时生成, 保护正式报告)
│   └── 代码解读.md         #   verify_dlpack.py 代码解读 (Mermaid 图)
├── charts/                 # 报告引用的证据图 ×5 (make_report.py 生成)
├── results/                # 运行产物 (脚本自动落盘, 按设备分目录)
│   ├── npu/                #   910B: 摘要 JSON + 逐步 CSV + run_all_<时间戳>.log
│   │                       #        + results_<时间戳>.xlsx (多 sheet 数据工作簿)
│   └── cpu/                #   本地调试产物
├── requirements-cpu.txt    # Windows CPU 环境依赖 (已实测锁定)
├── requirements-npu.txt    # 910B 环境依赖 (已按 CANN 8.3.RC1 锁定 torch/torch_npu 2.7.1)
├── .gitignore
└── README.md               # 本文件
```

## 2. 环境要求

| 项 | 910B(验证对象) | Windows(开发调试, 非验证对象) |
|---|---|---|
| 硬件 | Atlas A2, 昇腾 910B3 ×1 | x64 CPU |
| CANN | **8.3.RC1**(version.cfg 确认) | — |
| Python | 3.10.21(项目内 venv) | 3.10.11 官方构建(`D:\Python\Python310`) |
| torch / torch_npu | 2.7.1 / 2.7.1(官方配套表锁定) | 2.5.1+cpu |
| MindSpore | 2.10.0 | 2.11.0(2.7.0 起恢复 Windows 包) |
| numpy | 1.26.4(**必须 <2.0**,见 §6 踩坑) | 1.26.4 |

依赖清单:`requirements-npu.txt`(服务器)/ `requirements-cpu.txt`(Windows),版本已实测锁定,直接 `pip install -r` 即可。

## 3. 命令速查与实验矩阵(控制变量法)

> **原则**:所有对照实验共用同一组固定参数(下表"固定参数"列),每个实验**只改变
> 一个变量**(表中"变化量"列),命令里显式写出全部参数,保证可复现、可直接对照。

### 3.1 固定参数(所有实验一致,勿随意改动)

| 参数 | 值 | 说明 |
|---|---|---|
| `--seed` | 7 | 两跑同初始化、同数据 |
| `--steps` | 300 | 训练步数 |
| `--batch` | 128 | 样本数(全批) |
| `--hidden` | 64 | 隐藏层宽度(计算密度敏感性实验 E4 中才允许改) |
| `--lr` | 0.05 | 学习率(hidden=1024 组配套 0.005, 见 E4 说明) |
| `--result-every` | 50 | 控制台打印间隔(逐步 CSV 不受影响, 每步都记) |
| `--eval-every` | 10 | held-out 验证评估间隔(计时段外, 首末步必评; 验证曲线进逐步 CSV) |
| `--rtol` | 1e-3(CPU)/ **5e-3(NPU)** | M4 数值容差 |

> **数据落盘密度**:训练 loss 每步全量写逐步 CSV(300 点完整可审计);图表按**每 5 步**
> 取样绘制(约 60 点,业界常用密度,不密不疏);held-out 验证点每 10 步。

### 3.2 实验矩阵(控制变量法: 每行只改一个变量)

**一键全跑(推荐, 服务器一条命令出全部结果与日志)**:

```bash
python scripts/run_all.py
# 顺序执行 E0 自检 → E1 机制验证 → E2/E3/E4 训练(每次运行同跑 MS 参照臂) → E5 bench;
# 实时回显并 tee 到 results/npu/run_all_<时间戳>.log(每阶段有横幅标注做了什么);
# 任一阶段失败立即停止; 全部通过后自动导出 results_<时间戳>.xlsx 多 sheet 数据工作簿。
# 本地快速冒烟: python scripts/run_all.py --device cpu --steps 20 --rtol 1e-3
```

以下为分阶段命令(单独重跑某一项时使用; 参数显式写出保证可复现):

| 编号 | 实验 | 命令(在项目根目录, 已激活 venv 与 CANN 环境) | 变化量 | 回答的问题 |
|---|---|---|---|---|
| E0 | 环境自检 | `python scripts/check_env.py` | — | torch_npu/NPU/CANN 全 OK |
| E1 | 机制验证(11 STEP / 17 检查项, 含 GNN C1/C3) | `python scripts/verify_dlpack.py --device npu --rtol 5e-3` | — | 零拷贝/桥接数学正确 |
| **E2** | **混合训练(dlpack)** | `python scripts/train_hybrid.py --device npu --rtol 5e-3` | —(基准) | 混合训练收敛/底线/时延 |
| **E3** | **混合训练(copy 对照)** | `python scripts/train_hybrid.py --device npu --exchange copy --rtol 5e-3` | **仅交换方式: dlpack→host 拷贝** | **DLPack 交换机制的直接收益(主对照)** |
| **E4** | **形态对称性(PT前段+MS后段)** | `python scripts/train_hybrid.py --device npu --topology front_pt --rtol 5e-3` | **仅混合形态: MS前段→PT前段** | 迁移过半形态机制成立(切分方向不影响) |
| E5 | 纯交换吞吐(尺寸扫描) | `python scripts/train_hybrid.py --device npu --bench`(默认 batch 128/4096/16384,`--bench-sizes` 自定) | 仅 bench 模式 + 边界张量尺寸档 | 收益随边界张量放大趋势(机制级上界) |

**对照关系与归因原则(重要)**:

- **主对照 = E2 vs E3**:同为混合场景(MS 前段 + PT 后段、同切分点、同数据同种子),唯一变量是交换机制——这是"DLPack 本身价值"的直接证据;
- **归因原则**:混合前段的 MindSpore 调度开销在 E2/E3 两跑中完全相同,因此E2 vs E3 的差值纯粹来自交换机制,归因天然干净;
- 混合训练的**绝对**步时延主体为 MS PYNATIVE 单算子调度常数,与交换无关,该常数随计算密度增大被摊薄,生产规模按占比评估(接入真实模型时回归);
- 本地 Windows 冒烟:同命令换 `--device cpu`(E2/E3 建议先在本地跑通再上真机)。

### 3.3 一键生成最终报告(NPU 跑完 → 拷回 → Windows 执行)

```bash
# 服务器上一条命令跑完全部实验后(run_all 已自动生成 xlsx 工作簿),
# 把整个 results/ 目录拷回 Windows 项目根目录, 然后在 Windows 上:
.venv\Scripts\activate.bat
python scripts\make_report.py
# 产出: charts/ 五张证据图 + docs\验证报告.md (实测数字自动填充)
#       + results\<时间戳>.xlsx (多 sheet 数据工作簿, 避免零散同名文件搞混)
```

`make_report.py` 读取 `results/{npu|cpu}/` 下的 `verify_summary.json`、`train_summary_h{hidden}_{dlpack|copy}.json`、`bench_summary.json`(尺寸扫描格式, v1 单尺寸扁平格式自动兼容)与逐步训练 CSV,**数据优先级 npu > cpu**(有真机摘要时绝不使用 cpu 数据)。

### 3.4 判读要点(判据体系:正确性 C1~C4 / 性能 P1~P3,对齐行业咨询口径)

**基线**:B0 = MS 参照臂(纯 MindSpore,每次运行同跑)为**唯一基线**; B1(纯 PyTorch)不做——那是迁移终点的实现工作,不在验证范围。

正确性(以 B0 为锚):

- **C1 图语义聚合一致(必过)**:交换边界两侧各做一次 scatter 聚合,结果一致≤1e-5(E1 STEP 10)——不只比数值,要证明聚合语义未被交换破坏;
- **C2 梯度等价性(双锚)**:解析解/torch 参考(M4 torch 锚)+ **B0 纯 MS 端到端** (M4 B0 锚),相对误差 < rtol(真机实测双锚逐位一致 0.0);
- **C3 稀疏参数完整性**:交换路径上未访问 Embedding 行梯度恒零、稀疏更新后零变化(E1 STEP 11);
- **C4 收敛曲线一致性(必须项)**:E2/E3 的训练+验证曲线**逐位一致**(同框架同机制,交换不影响数值——真机 final loss 逐位相同);混合 vs B0 为**统计等价** (loss 比 ∈[0.5,2.0]、held-out ≤1.01、曲线形态一致;跨框架算子 ulp 差异经训练动力学混沌放大属正常,不要求逐点相等);过拟合检查同步进行;

性能(先卡回退,再谈快慢):

- **P1 算子回退检测**:run_all 在 E2/E3/E4 阶段自动并行采样 npu-smi(AI Core/HBM, 1 秒间隔,`results/npu/smi_<阶段>_<时间戳>.csv`)存证;真实 GNN 接入时"AI Core长时间近零 + HBM 非预期回落"为**一票否决**门禁(玩具模型算子小、AI Core 本征偏低,不自动判定);
- **P2 交换信噪比**:交换段占单步训练 <1% 为佳(摘要 `exchange_ratio_pct`);玩具模型单步毫秒级占比天然偏高,真实模型单 batch 数十~数百 ms 时将 <1%;
- **P3 epoch 耗时总账**:接入真实模型后 ≥3 epoch 耗时 ≤B0×1.2 为 PASS;玩具代理按交换占比摊薄披露;

收益证据(非门禁):

- E2 vs E3 机制选型对照:真机 final loss 逐位一致、训练内交换段快 6.1x/4.4x、bench 13.6x/9.4x(机制级上界);混合 vs B0 的步时延差为**范式代价披露项** (MS 调度常数主导,生产规模按占比摊薄);
- **形态对称性(全程适用)**:E4(PT前段+MS后段,迁移过半形态)全底线通过即成立——切分方向不影响机制,迁移任何阶段都适用;
- 本地 Windows 冒烟:同命令换 `--device cpu`;数据齐全(verify+E2+E3)→ 正式报告,缺项 → 草稿 + "待复测"占位,控制台列出补跑命令。

## 4. verify_dlpack.py 的验证项 ↔ 验收标准映射

| STEP | 验证项 | 对应验收标准 |
|---|---|---|
| 1 | MS→PT 交换 (dtype/shape/device) | 第一层-2 |
| 2 | PT→MS 交换 | 第一层-3 |
| 3 | **零拷贝语义** (一侧原地修改另一侧可见,双向) | 第一层-4(核心) |
| 4 | DLPack 叶子 `requires_grad_` → PT 局部图 backward | 第二层-6 |
| 5 | `pt_input.grad` → sens → `GradOperation(sens_param=True)` | 第二层-7 |
| 6 | **float64 解析解逐项对照**(loss、pt_input.grad、MS 参数梯度) | 第二层-7(数值≤容差,不止"非零") |
| 7 | 负对照:ones_like 假 sens 产生不同梯度 | 第二层-8 |
| 8 | 两侧参数均被更新 | 第二层-9 |
| 9 | 交换耗时:dlpack 路径 vs 拷贝路径 | NFR-1 |
| 10 | **图语义聚合对照**:交换边界两侧各做一次 scatter 聚合(unsorted_segment_sum / index_add_),结果一致 ≤1e-5 | GNN C1(必过) |
| 11 | **稀疏 Embedding 完整性**:交换路径上未访问行梯度恒零、稀疏更新后零变化 | GNN C3 |

## 5. 训练主对照设计与真机结果(train_hybrid.py)

设计:每次运行同跑两臂——**MS 参照臂**(纯 MindSpore 端到端 = 唯一基线 B0)与混合臂(`--topology` 选形态)。**模型为固定小图(128 节点、每节点 4 邻居)上的单层消息传递图网络**:聚合 = 均值消息传递 `segment_sum(gather(X,src),dst)/k`(真实包含scatter/gather 前向与梯度散布,与真实 GNN 算子形态一致),教师同构、权重固定(输出按 1/√h 缩放);切分点在聚合之后(边界 = 节点表示),各臂权重值同源拷贝; E2/E3/E4 同种子同初始化同数据同超参;held-out 验证集(独立验证图,512 节点)训练前/每 10 步(`--eval-every`)/训练后各评一次(计时段外,验证曲线落盘逐步 CSV);每次运行内置底线 = 收敛性 + 泛化(过拟合检查)+ 切换等价性(混合 vs 参照:loss 比 ∈[0.5,2.0]、held-out ≤1.01)+ M4 边界梯度标定(训练前同权重单次对照,按拓扑取 fc1/fc2 边界)+初始一致性。

真机结果:待 910B 重跑后由 `make_report.py` 自动填充(报告 §4.4~§4.9:机制选型对照、切换等价性、形态对称性、梯度双锚、性能门禁)。E5 bench 与任务无关,已有真机基准值:基准尺寸(边界张量 32KB)正向 copy 慢 **13.6x**、回传 **9.4x**,1MB/4MB 档重跑后刷新。

机制级补充:PYNATIVE GradOperation 反向会重执行一次前向(每步 2 次 MS 前向,固有行为)。已实装 `mindspore.vjp` 模式消除冗余前向(注意官方用法是 `vjp(fn, *inputs)`;标量输出可用标量种子 `bwd_fn(ms.Tensor(1.0))` 一次得全部输入梯度)。

## 6. 910B (Atlas A2) 移植指南

### 6.1 版本配套

1. 服务器查 CANN:`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
2. 查官方配套表:github.com/Ascend/pytorch 仓库 `COMPATIBILITY.md` (实测环境 CANN 8.3.RC1 → torch_npu 2.8.0/2.7.1/2.6.0.post3/2.1.0.post17,本验证锁定 torch 2.7.1 + torch_npu 2.7.1)
3. torch 与 torch_npu **主版本必须一致**;变更 CANN 版本用本套脚本回归

### 6.2 关键行为差异(CPU → NPU)

| 项 | CPU 本机 | 910B 上 |
|---|---|---|
| 设备 | `--device cpu` | `--device npu`(自动 `import torch_npu`) |
| MindSpore | `device_target="CPU"` | `device_target="Ascend"`(脚本自适应) |
| 原生 torch DLPack | 可用 | **A2 标记 Not Supported,必须走 torch_npu 补丁** |
| sens 传递 | CPU 内存共享 | NPU 显存共享;导出报错时回退 `.cpu().numpy()` 中转 |
| 流同步 | 单流,无感 | 数值偶发不一致时加 `torch.npu.synchronize()` |

### 6.3 服务器操作流程(JupyterLab 终端, 纯 python/venv)

```bash
# 1. 整目录上传后, 建独立 venv (不动客户已有环境):
python -m venv .venv && source .venv/bin/activate
# 2. 安装锁定版本依赖:
pip install -r requirements-npu.txt
# 3. 每次使用前激活 CANN 运行时 (容易漏!):
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 4. 按 §3 实验矩阵执行 (E0→E1→E2→E3→E4→E5), 输出与 results/ 即交付物
```

### 6.4 已知风险与无害告警

1. **torch_npu 的 DLPack 走 `from_blob` 补丁**(310 NZ TODO);A2 常规 ND 格式已验证, FRACTAL_NZ 格式需先转 ND。
2. **torch 导出带梯度张量必须 `.detach()`**;MindSpore 侧 PYNATIVE 无梯度上下文时直接导出即可。
3. **NNAL/ATB 已知 issue**(Ascend/pytorch#99):vLLM/ATB 场景限定,本验证不受影响。
4. **无害告警**(脚本已内置精确过滤):torch_npu `owner does not match`(CANN root安装 vs 普通用户)、numpy `smallest subnormal is zero`(ARM 容器关闭 denormal,与验证数值量级差 30 倍)。
5. **真机结论**(2026-11-10):机制验证 14/14 PASS;PT→MS 共享 **NPU device 显存** (torch_npu 补丁通道),MS→PT 共享 host 内存(MS PYNATIVE 张量驻留 host,算子消费时才上 device)——两者均为真零拷贝。
6. **行业图网络模型(GNN)接入**:消息传递类算子(scatter/gather/变长邻接结构)的边界张量可能为动态 shape/非常规布局,按首个真实切分点单独回归;图网络的采样batch 可调,等价性/对称性判据可直接复用。

## 7. 常见坑速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 提示符显示 venv 名但 import 全报缺包 | 幽灵激活(激活的 venv 已删除),回落到 base | 关终端重开重新激活;`where python` 确认路径 |
| `pip install -r` 报 GBK 解码错误 | requirements 含非 ASCII 注释 | 保持 requirements 纯 ASCII(本仓库已是) |
| `torch.utils.dlpack` 对 NPU 张量报错 | 忘了 `import torch_npu` | 任何 torch 调用前先 import |
| torch 导出报 `requires gradient` | 带 grad 张量直接导出 | `.detach()` 后导出 |
| MS 梯度"对但不对" | sens 用了默认 ones | STEP 7 负对照抓这个 |
| 数值偶发不一致(910B) | 跨框架流未同步 | 消费前 `torch.npu.synchronize()` |
| pip 找不到 mindspore Windows 包 | 2.4~2.6 无 Windows wheel | 选 2.11.0(或 ≥2.7)+ py3.10 |
| `ms.jit` 包装 Cell 后权重不更新 | jit 编译图固化编译时权重值 | 函数式前端 + 权重作参数传入(本仓库 train_hybrid 已如此实现) |
| `ms.jit` 编译梯度函数后梯度数值有偏差 | jit 编译图与 PYNATIVE 数值路径不一致(CPU 实测 M2 恶化至 1.27) | 梯度路径保持 PYNATIVE(本仓库已按此实现,实验开关已移除) |
| `mindspore.vjp` 多参数输入报缺参 | vjp(fn) 未注册 inputs | 官方用法:`vjp(fn, x, w, b)` 返回 (out, bwd_fn),bwd_fn 只收种子向量 |

## 8. 交付物与报告生成

```bash
# NPU 实验全部跑完后(服务器推荐直接 python scripts/run_all.py 一条命令),
# results/ 拷回 Windows, 一键生成最终报告:
python scripts/make_report.py
# 产出: charts/ 五张证据图(实测数据重绘) + docs/验证报告.md(实测数字自动填充)
#       + results/<时间戳>.xlsx(多 sheet 数据工作簿: 摘要总览/机制验证/各跑曲线/bench)
# 正式条件: npu 的 verify_summary.json + train_summary_h64_dlpack.json +
#           train_summary_h64_copy.json 三份齐备
# 保护机制: 数据不全时生成 docs/验证报告_draft.md(不覆盖正式报告),
#           控制台列出缺失项与补跑命令
```

验证报告的完整内容(结论/证据表/环境清单/已知限制/收益评估框架 附录 A)由模板自动渲染,实测数字来自 `results/` 下的机器可读摘要——**报告即数据,可复现、可审计**。
