# DLPack 混合编程验证: MindSpore ↔ PyTorch 零拷贝交换 + 梯度桥接

在 **昇腾 910B(Atlas A2)** 上验证 MindSpore 与 PyTorch(torch_npu) 通过 DLPack 进行
Tensor 零拷贝交换,并通过 **sens 梯度桥接** 让两侧参数都能更新——机制验证 + 训练收敛
三臂对照 + 收益量化,最终由一个脚本生成验证报告。

**工作流**:在服务器上跑实验(自动落盘机器可读摘要与逐步日志)→ 把 `results/` 目录
拷回 Windows → 运行**一键报告脚本**生成最终交付报告。

## 1. 目录结构

```
dlpack-verify/
├── scripts/                # 可执行脚本
│   ├── check_env.py        #   环境自检 (CPU/910B 通用)
│   ├── verify_dlpack.py    #   E1 机制验证 (9 个 STEP, 14 检查项)
│   ├── train_hybrid.py     #   E2~E5 训练收敛三臂对照 (M1~M4) + --bench 纯交换收益
│   └── make_report.py      #   一键报告: 读 results/ 摘要 → 重绘图 → 渲染验证报告
├── make_report_charts.py   # 图表绘制函数库 (由 make_report.py 调用)
├── docs/                   # 文档
│   ├── 验证报告.md         #   最终交付报告 (由 make_report.py 生成)
│   ├── 验证报告_draft.md   #   草稿 (仅 cpu 调试数据时生成, 保护正式报告)
│   └── 代码解读.md         #   新人版代码解读 (Mermaid 图)
├── charts/                 # 报告引用的证据图 (make_report.py 生成)
├── results/                # 运行产物 (脚本自动落盘, 按设备分目录)
│   ├── npu/                #   910B: verify/train/bench 摘要 JSON + 逐步训练 CSV + 运行日志
│   └── cpu/                #   本地调试产物
├── requirements-cpu.txt    # Windows CPU 环境依赖 (已实测锁定)
├── requirements-npu.txt    # 910B 环境依赖 (已按 CANN 8.3.RC1 锁定 torch/torch_npu 2.7.1)
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

依赖清单:`requirements-npu.txt`(服务器)/ `requirements-cpu.txt`(Windows),
版本已实测锁定,直接 `pip install -r` 即可。

## 3. 命令速查与实验矩阵(控制变量法)

> **原则**:所有对照实验共用同一组固定参数(下表"固定参数"列),每个实验**只改变
> 一个变量**(表中"变化量"列),命令里显式写出全部参数,保证可复现、可直接对照。

### 3.1 固定参数(所有实验一致,勿随意改动)

| 参数 | 值 | 说明 |
|---|---|---|
| `--seed` | 7 | 两臂同初始化、同数据 |
| `--steps` | 300 | 训练步数 |
| `--batch` | 128 | 样本数(全批) |
| `--hidden` | 64 | 隐藏层宽度(计算密度敏感性实验中才允许改) |
| `--lr` | 0.05 | 学习率(hidden=1024 组配套 0.005, 见 E4 说明) |
| `--result-every` | 50 | 控制台打印间隔(逐步 CSV 不受影响, 每步都记) |
| `--rtol` | 1e-3(CPU)/ **5e-3(NPU)** | M4 数值容差 |

### 3.2 实验矩阵(控制变量法: 每行只改一个变量)

| 编号 | 实验 | 命令(在项目根目录, 已激活 venv 与 CANN 环境) | 变化量 | 回答的问题 |
|---|---|---|---|---|
| E0 | 环境自检 | `python scripts/check_env.py` | — | torch_npu/NPU/CANN 全 OK |
| E1 | 机制验证(14 检查项) | `python scripts/verify_dlpack.py --device npu --rtol 5e-3` | — | 零拷贝/桥接数学正确 |
| **E2** | **训练三臂对照(dlpack 交换)** | `python scripts/train_hybrid.py --device npu --rtol 5e-3` | —(基准) | 混合训练收敛/指标/时延 |
| **E3** | **训练三臂对照(copy 交换)** | `python scripts/train_hybrid.py --device npu --exchange copy --rtol 5e-3` | **仅交换方式: dlpack→host 拷贝** | **DLPack 交换机制的直接收益(G2 主对照)** |
| E4 | 计算密度敏感性 | `python scripts/train_hybrid.py --device npu --hidden 1024 --batch 1024 --lr 0.005 --rtol 5e-3` | hidden/batch ×16(lr 组内配套) | M3 摊薄趋势 |
| E5 | 纯交换收益 | `python scripts/train_hybrid.py --device npu --bench` | 仅 bench 模式 | G2 交换开销(纯交换段) |

**对照关系与归因原则(重要)**:

- **主对照 = E2 vs E3**:同为混合场景(MS 前段 + PT 后段、同切分点、同数据),唯一变量是
  交换机制——这是"DLPack 本身收益"的直接证据。
- E4 单独看:与 E2 比,变化的是计算密度(框架调度常数被摊薄的比例),不是交换机制。
- **归因纪律**:M3 的 B/A 比值(混合 vs 单框架)**不能归因给 dlpack**——真机实测 B−A ≈ 1.5 ms
  中 dlpack 交换仅占 ~0.3 ms,其余为 MS PYNATIVE 单算子调度常数(纯 MS 训练同样存在)。
  DLPack 交换机制的收益只能用 **E2 vs E3 的交换段对比 / E5 bench** 度量。
- 本地 Windows 冒烟:同命令换 `--device cpu`(E2/E3 建议先在本地跑通再上真机)。

### 3.3 一键生成最终报告(NPU 跑完 → 拷回 → Windows 执行)

```bash
# 服务器跑完后, 把整个 results/ 目录拷回 Windows 项目根目录, 然后在 Windows 上:
.venv\Scripts\activate.bat
python scripts\make_report.py
# 产出: charts/ 三张证据图 + docs\验证报告.md (实测数字自动填充)
```

`make_report.py` 读取 `results/{npu|cpu}/` 下的 `verify_summary.json`、
`train_summary_h{hidden}_{exchange}.json`、`bench_summary.json` 与逐步训练 CSV,
**数据优先级 npu > cpu**(有真机摘要时绝不使用 cpu 数据)。

### 3.4 E2/E3 判读要点(三臂对照的预期)

- E2 与 E3 的 **loss 曲线逐位一致**(交换方式不影响数值正确性)——差异只在交换耗时;
- M1 收敛等价性:B/A 最终 loss 比 ∈ [0.5, 2.0](真机实测 0.9976 / 0.9980);
- M2 指标不恶化:held-out 评估比 ≤ 1.01(真机实测 0.9989 / 0.9979);
- M4 边界数值:桥接梯度 vs 单框架梯度(jit 修复后真机实测**逐位一致 0.0**);
- M3 玩具规模未达属预期(MS 单算子调度常数主导,非交换开销),
  加大 `--hidden/--batch` 观察比值摊薄,生产规模复测为最终判据(详见 §4.5);
- **归因纪律**:M3 的 B/A 比值是"混合 vs 单框架",其中 dlpack 交换仅占 ~0.3 ms,
  主体为 MS 执行模式常数——**该比值不能归因给 dlpack**;dlpack 的收益只看 E2 vs E3
  的交换段对比与 E5 bench。

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

## 5. 训练收敛验证的设计与真机结果(train_hybrid.py)

三臂设计:A 臂 = 同构模型整体 PyTorch 端到端(基线);B 臂 = MS 前段 ─DLPack─ PT 后段
(每步完整梯度桥接);C 臂 = 同 B 用 host 拷贝交换。教师为固定随机权重的同构非线性映射
(输出按 1/√h 缩放),B 臂初始化从 A 臂拷贝,held-out 评估集 512 条独立采样。

真机结果(hidden=64 与 hidden=1024 两轮,M1/M2/M4 全过):

| 底线 | 判据 | h=64 | h=1024 |
|---|---|---|---|
| M1 收敛等价性 | B/A 最终 loss 比 ∈ [0.5,2.0] | **0.9976** ✅ | **0.9980** ✅ |
| M2 指标不恶化 | held-out 评估比 ≤ 1.01 | **0.9989** ✅ | **0.9979** ✅ |
| M3 步时延不恶化 | B/A ≤ 1 + m3_tol | 2.38x ❌(见下) | 2.81x ❌(见下) |
| M4 边界数值正确 | 桥接梯度 vs 单框架梯度 | **0.0(逐位一致)** ✅ | **0.0(逐位一致)** ✅ |

**M3 诚实解读**:玩具规模下 B 臂每步的 MindSpore PYNATIVE 单算子调度开销(毫秒级
框架常数)占主导,与交换本身无关(交换仅 ~0.3ms/步);该常数随计算密度增大被摊薄
(hidden 64→1024 已可见改善),**生产规模(计算密集)按占比复测为最终判据**;
混合范式净增成本仅为交换段(µs 级),MS 侧前段的调度开销无论是否混合都存在。
**归因纪律**:该 2.38x 是"混合 vs 单框架"的比值,**不能归因给 dlpack 交换**;
dlpack 交换机制的收益只看 E2 vs E3 交换段对比(bench 13.9x / 8.7x)。

机制级补充:PYNATIVE GradOperation 反向会重执行一次前向(每步 2 次 MS 前向,固有
行为)。已实装 `mindspore.vjp` 模式消除冗余前向(注意官方用法是 `vjp(fn, *inputs)`);
`ms.jit(grad_fn)` 编译版梯度数值有偏差,保留为实验开关 `--ms-jit`(默认关闭)。

## 6. 910B (Atlas A2) 移植指南

### 6.1 版本配套

1. 服务器查 CANN:`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
2. 查官方配套表:github.com/Ascend/pytorch 仓库 `COMPATIBILITY.md`
   (实测环境 CANN 8.3.RC1 → torch_npu 2.8.0/2.7.1/2.6.0.post3/2.1.0.post17,
   本验证锁定 torch 2.7.1 + torch_npu 2.7.1)
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
# 4. 按 §3 实验矩阵执行 (E0→E2→E3→E4→E5), 输出与 results/ 即交付物
```

### 6.4 已知风险与无害告警

1. **torch_npu 的 DLPack 走 `from_blob` 补丁**(310 NZ TODO);A2 常规 ND 格式已验证,
   FRACTAL_NZ 格式需先转 ND。
2. **torch 导出带梯度张量必须 `.detach()`**;MindSpore 侧 PYNATIVE 无梯度上下文时
   直接导出即可。
3. **NNAL/ATB 已知 issue**(Ascend/pytorch#99):vLLM/ATB 场景限定,本验证不受影响。
4. **无害告警**(脚本已内置精确过滤):torch_npu `owner does not match`(CANN root
   安装 vs 普通用户)、numpy `smallest subnormal is zero`(ARM 容器关闭 denormal,
   与验证数值量级差 30 倍)。
5. **真机结论**(2026-11-10):机制验证 14/14 PASS;PT→MS 共享 **NPU device 显存**
   (torch_npu 补丁通道),MS→PT 共享 host 内存(MS PYNATIVE 张量驻留 host,
   算子消费时才上 device)——两者均为真零拷贝。

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
| `mindspore.vjp` 多参数输入报缺参 | vjp(fn) 未注册 inputs | 官方用法:`vjp(fn, x, w, b)` 返回 (out, bwd_fn),bwd_fn 只收种子向量 |

## 8. 交付物与报告生成

```bash
# NPU 实验全部跑完后, results/ 拷回 Windows, 一键生成最终报告:
python scripts/make_report.py
# 产出: charts/ 三张证据图(实测数据重绘) + docs/验证报告.md(实测数字自动填充)
# 保护机制: 仅 cpu 调试数据时生成 docs/验证报告_draft.md, 不覆盖正式报告
```

验证报告的完整内容(结论/证据表/环境清单/已知限制/收益评估框架 附录 A)由模板
自动渲染,实测数字来自 `results/` 下的机器可读摘要——**报告即数据,可复现、可审计**。
