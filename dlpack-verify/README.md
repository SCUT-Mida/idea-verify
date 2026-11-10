# DLPack 验证: MindSpore ↔ PyTorch 零拷贝交换 + 梯度桥接

对应验证方案(与 DeepSeek 对话确认):在 **昇腾 910B (Atlas A2)** 上验证 MindSpore 与
PyTorch(torch_npu) 通过 DLPack 进行 Tensor 零拷贝交换,并在交换后通过
**sens 梯度桥接** 让两侧参数都能更新。

**本次(Windows CPU)**:在本机用 CPU 版框架把整条链路跑通、把代码固化;
**后续(910B)**:按 §4 移植到服务器,只改设备参数。

## 1. 目录结构

```
dlpack-verify/
├── scripts/                # 可执行脚本
│   ├── check_env.py        #   环境自检 (CPU/910B 通用, 上服务器先跑这个)
│   ├── verify_dlpack.py    #   机制验证 (9 个 STEP, PASS/FAIL 报告)
│   ├── train_hybrid.py     #   训练收敛 + 三臂基线对照 (M1~M4, --bench)
│   └── make_report_charts.py  # 报告图表生成 (读 results/{cpu,npu}, 写 charts/)
├── docs/                   # 文档
│   ├── 验证报告.md         #   交付报告 (结论/证据/环境/已知限制 + 附录 A 收益框架)
│   └── 代码解读.md         #   新人版代码解读 (Mermaid 图)
├── charts/                 # 报告引用的证据图 (make_report_charts.py 生成)
├── results/                # 运行产物 (按设备分目录)
│   ├── npu/                #   910B: 逐步训练日志 + 运行日志
│   └── cpu/                #   本地调试日志
├── requirements-cpu.txt    # Windows CPU 环境依赖 (已实测锁定)
├── requirements-npu.txt    # 910B 环境依赖 (已按 CANN 8.3.RC1 锁定 torch/torch_npu 2.7.1)
└── README.md               # 本文件 (使用说明 + 移植指南)
```

## 2. 本机 CPU 环境信息

纯 **python + venv** 方案,不依赖 conda:

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | **3.10.11** (官方构建) | 安装于 `D:\Python\Python310`,已入 PATH |
| venv | `dlpack-verify\.venv` | **项目内 venv**(环境自包含,PyCharm 自动识别),由上面官方 Python 创建,验证依赖装在这里 |
| numpy | 1.26.4 | **实测必须 <2.0**(见下方"踩坑实录") |
| torch | 2.5.1 (CPU) | PyPI Windows wheel 即 CPU 版 |
| mindspore | 2.11.0 (CPU) | 2.7.0 起 PyPI 恢复 Windows 包;选最新以贴近 910B 目标版本 |

> 版本说明:3.10.12 起 Python 3.10 进入"仅安全修复"阶段,**官方不再发 Windows 安装包**
> (3.10.11 是最后一个),所以本机用 3.10.11。它与服务器的 3.10.21 同属 cp310 ABI,
> 所有 wheel 完全一致,安全补丁差异对本验证无影响。

> **踩坑实录(重要)**:mindspore 2.11.0 的安装元数据声明 `numpy>=1.25`,pip 会装出
> numpy 2.x;但其 Windows wheel 实际按 **numpy 1.x ABI** 编译,导入虽不报错,
> 一旦走 Tensor 算术/梯度路径就抛 `_ARRAY_API not found`。**必须手动锁 numpy<2**。
> 910B 上同样注意:装完先用 verify 脚本全跑一遍,别只看 import 成功。

常用命令(在项目根目录执行;脚本都在 scripts/ 下):

```bat
:: 激活项目 venv (PowerShell 用 .venv\Scripts\Activate.ps1)
.venv\Scripts\activate.bat

:: 环境自检 + 运行验证
python scripts\check_env.py
python scripts\verify_dlpack.py --device cpu
```

> PyCharm 调试:解释器选 `Virtualenv Environment → Existing` →
> `E:\Code\Python\dlpack-verify\.venv\Scripts\python.exe`(项目内 `.venv` 通常会被自动识别)。

## 3. verify_dlpack.py 的验证项 ↔ 验收标准映射

| STEP | 验证项 | 对应验收标准 |
|---|---|---|
| 1 | MS→PT 交换 (dtype/shape/device) | 第一层-2 |
| 2 | PT→MS 交换 | 第一层-3 |
| 3 | 零拷贝语义 (一侧原地修改另一侧可见,双向) | 第一层-4(核心) |
| 4 | DLPack 叶子 `requires_grad_` → PT 局部图 backward | 第二层-6 |
| 5 | `pt_input.grad` → sens → `GradOperation(sens_param=True)` | 第二层-7 |
| 6 | **float64 解析解对照**(loss、pt_input.grad、MS 参数梯度逐一比对) | 第二层-7(数值≤容差,不止"非零") |
| 7 | 负对照:ones_like 假 sens 产生不同梯度 | 第二层-8 |
| 8 | 两侧参数均被更新 | 第二层-9 |
| 9 | 交换耗时:dlpack 路径 vs 拷贝路径 | NFR-1 |

容差:CPU 默认 `--rtol 1e-3`;910B 上如遇浮点误差可放宽 `--rtol 5e-3`。

## 3.5 训练收敛 + 基线对照验证(train_hybrid.py)—— M1~M4 底线全覆盖

`verify_dlpack.py` 证明的是**机制**(单步交换/梯度桥接数学正确);`train_hybrid.py`
用**三臂对照**证明"训练可用性 + 范式不恶化"(对应收益评估的 M1~M4 底线,
见 `验证报告.md` 附录 A):

- **A 臂(基线)**:同构模型整体在 PyTorch 内端到端训练(不跨框架)
- **B 臂(目标方案)**:前段 MindSpore(Dense+ReLU) ──DLPack──▶ 后段 PyTorch(Linear),
  每步完整梯度桥接
- **C 臂(对照,`--exchange copy`)**:同 B 但用 host 拷贝交换

模型:非线性小网 Dense(4→h)+ReLU ‖ Linear(h→2),教师为固定随机权重的同构非线性映射
(输出按 1/√h 缩放,任意宽度下 loss 保持 O(1));B 臂初始化从 A 臂权重值拷贝,
数据/超参/更新完全一致。

**四条底线判据与 CPU 实测(hidden=64 / hidden=4096,lr=0.005)**:

| 底线 | 判据 | hidden=64 | hidden=4096 |
|---|---|---|---|
| M1 收敛等价性 | B/A 最终 loss 比 ∈ [0.5,2.0] 且两臂均收敛(逐点轨迹在非线性+大宽度下会因跨框架浮点核 ulp 差异混沌放大,同框架换硬件重跑亦然,故只判收敛结果) | 0.9976 ✅ | 0.9977 ✅ |
| M2 指标不恶化 | B 臂 held-out 评估 MSE ≤ A 臂 × 1.01(512 条独立样本) | 0.9989 ✅ | 0.9981 ✅ |
| M3 步时延不恶化 | B/A 步时延 ≤ 1 + m3_tol(默认 2%) | 7.7x ❌ | 3.0x ❌ |
| M4 边界数值正确 | 桥接梯度 vs 单框架同边界梯度(h=4096 时 1.79e-05) | ✅ | ✅ |

**M3 的诚实解读(重要)**:玩具规模下 B 臂每步的 MindSpore PYNATIVE 单算子调度开销
(毫秒级框架常数)占绝对主导,与交换本身无关(交换仅 ~0.3ms/步);加大计算密度
(hidden 64→4096)比值从 7.7 收敛到 3.0,验证了"占比随计算密度下降"的趋势。
**生产规模(计算密集,每步百 ms~秒级)才是 M3 的最终判据场景**,届时框架常数被摊薄;
若 100M 模型实测仍超阈,则按方案附录 A 的结论如实报告"时延不恶化不成立"。

```bash
# 主实验 (A 臂基线 + B 臂 dlpack)
python scripts/train_hybrid.py --device npu
# 收益对照 (C 臂: host 拷贝交换)
python scripts/train_hybrid.py --device npu --exchange copy
# 纯交换吞吐 microbenchmark (同进程对比 dlpack/copy, 剥离框架算子开销)
python scripts/train_hybrid.py --device npu --bench
```

`--bench` 实测对比:Windows CPU 正向 dlpack 28.6 vs copy 30.2 µs/次、回传 5.9 vs
52.6 µs/次(慢 9x);**910B 真机(2026-11-10):正向 copy 慢 13.9x(206.3 vs 14.8 µs/次)、
回传 copy 慢 8.7x(120.5 vs 13.8 µs/次),每步交换节省 298.2 µs**——PCIe 往返 + 流同步
的代价在真机上被显著拉开。

参数:`--steps/--batch/--hidden/--lr/--seed/--rtol/--m3-tol` 均可调;接真实模型改
`build_everything` 中 `MsFront` 与 `pt_head`,保持"MS 输出维度 == PT 输入维度"即可。

逐步日志:每次运行落盘 `results/{cpu|npu}/training_log_{exchange}_h{hidden}.csv`
(每步 loss/耗时),`scripts/make_report_charts.py` 优先读取它绘制逐步 dense 曲线。
`--ms-jit`(实验性):ms.jit 编译 GradOperation 反向 —— 本版本实测编译梯度数值有偏差,
默认关闭;消除"反向时重执行前向"的正解是升级 MindSpore 后使用 `mindspore.vjp`
(当前版本 vjp 多参数输入有缺陷,详见验证报告 §4.5)。

## 4. 910B (Atlas A2) 移植指南

### 4.1 移植前必查:版本配套

脚本代码**不需要改**(设备自动/参数切换),要换的只有依赖版本:

1. 服务器上查 CANN:`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
2. 按 CANN 查官方配套表:github.com/Ascend/pytorch 仓库的 `COMPATIBILITY.md`
3. torch 与 torch_npu **主版本必须一致**(如都 2.7.1)
4. MindSpore 用支持 A2 的版本(客户环境已有就沿用,拿版本号跑本脚本回归)

**实测记录(2026-11,客户环境已确认)**:CANN **8.3.RC1**(8.3.0.1.200)/ aarch64 /
Python 3.10.21 / 客户 MindSpore 2.10.0。官方配套表 8.3.RC1 行:torch_npu
**2.8.0 / 2.7.1** / 2.6.0.post3 / 2.1.0.post17。本验证选用 **torch 2.7.1 +
torch_npu 2.7.1**(requirements-npu.txt 已按此锁定,直接装即可)。

### 4.2 关键行为差异(CPU → NPU)

| 项 | CPU 本机 | 910B 上 |
|---|---|---|
| 设备 | `--device cpu` | `--device npu`(自动 `import torch_npu`) |
| MindSpore | `device_target="CPU"` | `device_target="Ascend"`(脚本自适应) |
| 原生 torch DLPack | 可用 | **A2 标记 Not Supported,必须走 torch_npu 补丁**(导入即生效) |
| sens 传递 | CPU 内存共享 | NPU 内存共享;若导出报错,回退 `pt_input.grad.cpu().numpy()` 转 ms.Tensor(多一次拷贝,梯度张量小,可接受) |
| 流同步 | 单流,无感 | 两框架异流时消费前需同步;若数值偶发不一致,加 `torch.npu.synchronize()` |

### 4.3 A2 已知风险(验证时重点关注)

1. **torch_npu 的 DLPack 走 `from_blob` 补丁**,源码留有"310 NZ 格式"TODO —— A2 常规 ND 格式预期可用;若张量处于 FRACTAL_NZ 内部格式导出失败,先转 ND 再导出,并记为已知限制。
2. **detach 的差异**:torch 侧导出带梯度张量会直接报错(`Can't export tensors that require gradient`),必须 `.detach()`;MindSpore 侧在 PYNATIVE 模式、无梯度上下文时 Tensor 本身不带图,**直接导出即可**(脚本即如此,调用 `ms_to_dlpack` 前没有 detach)。
3. **消费方持有内存**:DLPack 约定消费方负责释放;不要让导出方提前释放,也不要两侧同时管理同一块内存。
4. **性能基线**:`from_dlpack` 单次应为 **µs 级**。Windows CPU 实测:dlpack 交换 13.5 µs/次 vs 拷贝路径 27.5 µs/次(本机参考值);外部资料中 A2 上 from_dlpack 约 128.6 µs 量级,以 910B 实测为准。若实测毫秒级,大概率走了隐式拷贝,查 STEP 3 零拷贝是否 PASS。
5. **torch_npu 2.7.1 + CANN 8.3.RC1 的已知 issue**:vLLM/ATB 场景存在 NNAL 符号不匹配的未决问题(github.com/Ascend/pytorch/issues/99);本验证只用基础 torch 算子 + DLPack,不加载 ATB,**不受影响**。但若 `import torch_npu` 时报 NNAL/ATB 相关符号错误,先确认服务器没装版本不匹配的 `Ascend-cann-nnal` 包。
6. **CANN 运行时必须激活**:导入 torch_npu 前必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`,否则报 so 找不到(libasccl 等)。
7. **已知无害告警(两个脚本已内置精确过滤)**:①torch_npu 的 `owner does not match`——CANN 由 root 安装而以普通用户运行,属主检查提示,零影响;②numpy 的 `smallest subnormal is zero`——ARM 容器内核关闭非规格化浮点(denormal ~1e-38 刷零),与验证数值量级(≥1e-7)相差 30 个数量级,零影响。过滤按消息匹配,其他真实警告仍会显示。
8. **2026-11-10 真机验证结论**:910B3 上 `verify_dlpack.py --device npu` **14/14 PASS**,梯度桥接数值误差 ~1e-7(远优于 5e-3 容差);dlpack 交换 18.5 µs/次 vs 拷贝 32.4 µs/次。技术细节:STEP 3 双向共享内存的层次不同——MS→PT 共享 host 内存(MindSpore PYNATIVE 下由 numpy 创建的 Tensor 驻留 host,被算子消费时才上 device),PT→MS 共享 **NPU device 显存**(torch_npu 补丁通道);两者均为真零拷贝,deliverable 里建议如实区分表述。

### 4.4 在客户服务器上的操作流程(JupyterLab 终端, 纯 python/venv, 不用 conda)

```bash
# 1. 传文件: 把 dlpack-verify/ 整个目录传上去 (scp / lrzsz / jupyter 上传)
# 2. 建独立 venv (依赖全是 pip wheel; 不动客户已有环境):
python -m venv .venv
source .venv/bin/activate
# 3. 安装 (requirements-npu.txt 已按 CANN 8.3.RC1 锁定 torch/torch_npu 2.7.1):
pip install -r requirements-npu.txt
# 4. 每次使用前激活 CANN 运行时 (容易漏! 否则 import torch_npu 报 so 找不到):
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 5. 自检 → 验证 (脚本在 scripts/ 下):
python scripts/check_env.py         # torch_npu/NPU/CANN 三项必须 OK
python scripts/verify_dlpack.py --device npu --rtol 5e-3
python scripts/train_hybrid.py --device npu --rtol 5e-3     # 训练收敛三臂对照
# 6. 输出截图留档; 训练逐步日志自动落盘 results/npu/, 即验证交付物
```

## 5. 常见坑速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 提示符显示 `(.venv)` 但 import 全部报缺包 | 激活的是**已被删除的旧 venv**(幽灵激活),python/pip 实际回落到 base | 关掉终端重开,重新激活正确的 venv;`where python` 确认路径 |
| `pip install -r` 报 `UnicodeDecodeError: 'gbk' codec` | requirements 文件含非 ASCII 注释,pip 用系统 locale 解码 | requirements 文件保持纯 ASCII(本仓库已改);或用 `python -m pip` 且确认 venv 激活 |
| `pip` 装的包跑去别的解释器里 | 激活失效后 `pip` 落到 base | 用绝对路径最稳:`<venv>\Scripts\python.exe -m pip install ...` |
| `torch.utils.dlpack` 对 NPU 张量报错 | 忘了 `import torch_npu`,补丁未生效 | 任何 torch 调用前先 import |
| torch 导出报 `Can't export tensors that require gradient` | 带 grad 的张量直接导出 | `.detach()` 后再导出 |
| MS 侧梯度看起来"对但不对" | sens 用了默认 ones 而非真实上游梯度 | 脚本 STEP 7 负对照就是抓这个 |
| 数值偶发不一致(910B) | 跨框架执行流未同步 | 消费前 `torch.npu.synchronize()` |
| pip 装 mindspore 找不到 Windows 包 | 2.4~2.6 无 Windows wheel(2.7.0 起恢复) | Windows 选 2.11.0(或 ≥2.7)+ py3.10 |
