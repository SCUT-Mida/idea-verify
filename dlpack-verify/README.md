# DLPack 混合编程验证: MindSpore ↔ PyTorch 零拷贝交换 + 梯度桥接

在 **昇腾 910B(Atlas A2)** 上验证 MindSpore 与 PyTorch(torch_npu) 通过 DLPack 进行
Tensor 零拷贝交换,并通过 **sens 梯度桥接** 让两侧参数都能更新——机制验证 + 训练
**主对照(E2 vs E3)** + 收益量化,最终由一个脚本生成验证报告。

**工作流**:在服务器上跑实验(自动落盘机器可读摘要与逐步日志)→ 把 `results/` 目录
拷回 Windows → 运行**一键报告脚本**生成最终交付报告。

**主对照设计**:验证目标是"DLPack 交换机制本身"的价值,因此对照双方**同为混合场景**
(MS 前段 + PT 后段、同切分点、同数据同种子),唯一变量 = 交换机制
(E2 = DLPack 零拷贝 / E3 = host 拷贝)。混合前段的框架调度开销在两跑中完全相同,
故 E2 vs E3 的任何差异都纯粹来自交换机制——归因天然干净。

## 1. 目录结构

```
dlpack-verify/
├── scripts/                # 可执行脚本
│   ├── check_env.py        #   环境自检 (CPU/910B 通用)
│   ├── verify_dlpack.py    #   E1 机制验证 (9 个 STEP, 14 检查项)
│   ├── train_hybrid.py     #   E2/E3/E4/E5 混合训练主对照 (收敛/M4) + --bench
│   └── make_report.py      #   一键报告: 读 results/ 摘要 → 绘 4 图 → 渲染验证报告
├── docs/                   # 文档
│   ├── 验证报告.md         #   最终交付报告 (npu 数据齐备时由 make_report.py 生成)
│   ├── 验证报告_draft.md   #   草稿 (数据不全/cpu 调试时生成, 保护正式报告)
│   └── 代码解读.md         #   verify_dlpack.py 代码解读 (Mermaid 图)
├── charts/                 # 报告引用的证据图 ×4 (make_report.py 生成)
├── results/                # 运行产物 (脚本自动落盘, 按设备分目录)
│   ├── npu/                #   910B: 摘要 JSON + training_log_{exchange}_h*.csv + 运行日志
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

依赖清单:`requirements-npu.txt`(服务器)/ `requirements-cpu.txt`(Windows),
版本已实测锁定,直接 `pip install -r` 即可。

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

### 3.2 实验矩阵(控制变量法: 每行只改一个变量)

| 编号 | 实验 | 命令(在项目根目录, 已激活 venv 与 CANN 环境) | 变化量 | 回答的问题 |
|---|---|---|---|---|
| E0 | 环境自检 | `python scripts/check_env.py` | — | torch_npu/NPU/CANN 全 OK |
| E1 | 机制验证(14 检查项) | `python scripts/verify_dlpack.py --device npu --rtol 5e-3` | — | 零拷贝/桥接数学正确 |
| **E2** | **混合训练(dlpack)** | `python scripts/train_hybrid.py --device npu --rtol 5e-3` | —(基准) | 混合训练收敛/底线/时延 |
| **E3** | **混合训练(copy 对照)** | `python scripts/train_hybrid.py --device npu --exchange copy --rtol 5e-3` | **仅交换方式: dlpack→host 拷贝** | **DLPack 交换机制的直接收益(主对照)** |
| E5 | 纯交换吞吐(尺寸扫描) | `python scripts/train_hybrid.py --device npu --bench`(默认 batch 128/4096/16384,`--bench-sizes` 自定) | 仅 bench 模式 + 边界张量尺寸档 | 收益随边界张量放大趋势(机制级上界) |

**对照关系与归因原则(重要)**:

- **主对照 = E2 vs E3**:同为混合场景(MS 前段 + PT 后段、同切分点、同数据同种子),
  唯一变量是交换机制——这是"DLPack 本身价值"的直接证据;
- **归因原则**:混合前段的 MindSpore 调度开销在 E2/E3 两跑中完全相同,因此
  E2 vs E3 的差值纯粹来自交换机制,归因天然干净;
- 混合训练的**绝对**步时延(≈2 ms)主体为 MS PYNATIVE 单算子调度常数,与交换无关,
  该常数随计算密度增大被摊薄,生产规模按占比评估(接入真实模型时回归);
- 本地 Windows 冒烟:同命令换 `--device cpu`(E2/E3 建议先在本地跑通再上真机)。

### 3.3 一键生成最终报告(NPU 跑完 → 拷回 → Windows 执行)

```bash
# 服务器跑完后, 把整个 results/ 目录拷回 Windows 项目根目录, 然后在 Windows 上:
.venv\Scripts\activate.bat
python scripts\make_report.py
# 产出: charts/ 四张证据图 + docs\验证报告.md (实测数字自动填充)
```

`make_report.py` 读取 `results/{npu|cpu}/` 下的 `verify_summary.json`、
`train_summary_h{hidden}_{dlpack|copy}.json`、`bench_summary.json`(尺寸扫描格式,
v1 单尺寸扁平格式自动兼容)与逐步训练 CSV,**数据优先级 npu > cpu**(有真机摘要时
绝不使用 cpu 数据)。旧版(v1)摘要键名自动兼容,不必重跑旧实验。

### 3.4 E2/E3 判读要点(主对照的预期)

- **C1 数值一致性**:E2 与 E3 的 loss 曲线**逐位一致**(零拷贝与 host 拷贝传递的
  数值完全相同,交换机制不影响数值正确性)——真机实测 final loss 与 held-out
  评估均逐位相同;
- **泛化(过拟合检查)**:held-out 验证损失随训练同步下降(图 1 虚线),终点与
  训练损失同量级——E2/E3 的验证曲线同样逐位一致;
- **C2 步时延**:dlpack ≤ copy × 1.02(真机实测 copy 反而慢 1.36x);
- **C3 交换段**:dlpack ≤ copy(E5 bench 为机制级上界,训练内实测同向);
- 运行内底线:收敛性(loss 下降、无 NaN)+ **M4 边界梯度 vs torch 参考逐位一致
  (rel_err=0.0)** + 初始一致性;
- **归因原则**:混合训练的绝对步时延主体是 MS PYNATIVE 调度常数,在两种交换
  机制下完全相同——不要把它当成交换开销,主对照差值才是;
- 数据齐全(verify + E2 + E3)→ 生成正式报告;缺项 → 草稿 + "待复测"占位,
  控制台会列出补跑命令。

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

## 5. 训练主对照设计与真机结果(train_hybrid.py)

设计:混合学生 = MS(fc1 4→h + ReLU) ──交换──▶ torch(fc2 h→2),教师为固定随机权重
的同构非线性映射(输出按 1/√h 缩放);E2/E3 同种子同初始化同数据同超参;held-out
验证集 512 条独立采样,训练前/每 10 步(`--eval-every`)/训练后各评一次(计时段外,
验证曲线落盘逐步 CSV);每次运行内置底线 = 收敛性 + 泛化(验证曲线同步下降,过拟合
检查)+ M4 边界梯度标定(训练前同权重单次对照,torch 参考模型不参与训练)+ 初始一致性。

真机结果(h=64,E2 vs E3,2026-11-10):

| 指标 | E2 (dlpack) | E3 (copy) | 判定 |
|---|---|---|---|
| final loss | 0.004477296024560928 | 0.004477296024560928 | **逐位一致** ✅ |
| held-out 评估 MSE | 0.006813785061240196 | 0.006813785061240196 | **逐位一致** ✅ |
| 步时延(去预热) | 2.02 ms | 2.75 ms | **copy 慢 1.36x** ✅ |
| 交换段 正向 | 54.9 µs/步 | 332.7 µs/步 | **copy 慢 6.1x** ✅ |
| 交换段 回传 | 52.4 µs/步 | 233.2 µs/步 | **copy 慢 4.4x** ✅ |
| M4 边界梯度 | rel_err = 0.0(逐位一致) | rel_err = 0.0(逐位一致) | ✅ |

过拟合检查:held-out 验证损失与训练损失同步下降、终点同量级(0.0068 vs 0.0045),
无病态过拟合分歧——且该验证曲线本身在 E2/E3 间也逐位一致(泛化行为不受交换机制
影响;新脚本重跑后训练/验证双曲线进图 1)。

E5 bench(尺寸扫描,同进程 N=200):基准尺寸(边界张量 32KB)正向 copy 慢 **13.6x**、
回传 **9.4x**;扫描档位 batch 128/4096/16384(边界张量 32KB/1MB/4MB)待重跑刷新——
收益随边界张量增大线性放大,真实模型边界张量更大,收益只会更显著。

> 表中数字由 `results/npu/` 摘要自动渲染进报告(报告即数据,可复现、可审计);
> 混合臂训练逻辑未变,重跑数值可复现。绝对步时延 ≈2 ms 的主体是 MS PYNATIVE
> 单算子调度常数(两种交换机制下相同),生产规模按占比评估(见 §3.4 归因原则)。

机制级补充:PYNATIVE GradOperation 反向会重执行一次前向(每步 2 次 MS 前向,固有
行为)。已实装 `mindspore.vjp` 模式消除冗余前向(注意官方用法是 `vjp(fn, *inputs)`)。

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
# 4. 按 §3 实验矩阵执行 (E0→E1→E2→E3→E4→E5), 输出与 results/ 即交付物
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
| `ms.jit` 编译梯度函数后梯度数值有偏差 | jit 编译图与 PYNATIVE 数值路径不一致(CPU 实测 M2 恶化至 1.27) | 梯度路径保持 PYNATIVE(本仓库已按此实现,实验开关已移除) |
| `mindspore.vjp` 多参数输入报缺参 | vjp(fn) 未注册 inputs | 官方用法:`vjp(fn, x, w, b)` 返回 (out, bwd_fn),bwd_fn 只收种子向量 |

## 8. 交付物与报告生成

```bash
# NPU 实验全部跑完后, results/ 拷回 Windows, 一键生成最终报告:
python scripts/make_report.py
# 产出: charts/ 四张证据图(实测数据重绘) + docs/验证报告.md(实测数字自动填充)
# 正式条件: npu 的 verify_summary.json + train_summary_h64_dlpack.json +
#           train_summary_h64_copy.json 三份齐备
# 保护机制: 数据不全时生成 docs/验证报告_draft.md(不覆盖正式报告),
#           控制台列出缺失项与补跑命令
```

验证报告的完整内容(结论/证据表/环境清单/已知限制/收益评估框架 附录 A)由模板
自动渲染,实测数字来自 `results/` 下的机器可读摘要——**报告即数据,可复现、可审计**。
