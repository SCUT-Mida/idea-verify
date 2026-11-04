# DLPack 验证: MindSpore ↔ PyTorch 零拷贝交换 + 梯度桥接

对应验证方案(与 DeepSeek 对话确认):在 **昇腾 910B (Atlas A2)** 上验证 MindSpore 与
PyTorch(torch_npu) 通过 DLPack 进行 Tensor 零拷贝交换,并在交换后通过
**sens 梯度桥接** 让两侧参数都能更新。

**本次(Windows CPU)**:在本机用 CPU 版框架把整条链路跑通、把代码固化;
**后续(910B)**:按 §4 移植到服务器,只改设备参数。

## 1. 目录结构

```
dlpack-verify/
├── check_env.py            # 环境自检 (CPU/910B 通用, 上服务器先跑这个)
├── verify_dlpack.py        # 主验证脚本 (9 个 STEP, PASS/FAIL 报告)
├── requirements-cpu.txt    # Windows CPU 环境依赖 (已实测锁定)
├── requirements-npu.txt    # 910B 环境依赖模板 (版本需按客户 CANN 配套)
└── README.md
```

## 2. 本机 CPU 环境信息

| 组件 | 版本 | 说明 |
|---|---|---|
| Miniconda | C:\Miniconda3 | conda 已配清华镜像 |
| conda env | `dlpack-cpu` | Python 3.10.22 |
| numpy | 1.26.4 | **实测必须 <2.0**(见下方"踩坑实录") |
| torch | 2.5.1 (CPU) | PyPI Windows wheel 即 CPU 版 |
| mindspore | 2.11.0 (CPU) | 2.7.0 起 PyPI 恢复 Windows 包;选最新以贴近 910B 目标版本 |

> **踩坑实录(重要)**:mindspore 2.11.0 的安装元数据声明 `numpy>=1.25`,pip 会装出
> numpy 2.x;但其 Windows wheel 实际按 **numpy 1.x ABI** 编译,导入虽不报错,
> 一旦走 Tensor 算术/梯度路径就抛 `_ARRAY_API not found`。**必须手动锁 numpy<2**。
> 910B 上同样注意:装完先用 verify 脚本全跑一遍,别只看 import 成功。

> 说明:MindSpore 的 Windows wheel 在 2.4~2.6 缺失、2.7.0 起恢复。本地选 2.11.0
> 是为了与 910B 上的目标版本(客户环境 ≥2.10)尽量对齐;若客户版本不同,
> 在服务器上用同一脚本回归即可。

常用命令:

```bat
:: 激活环境 (或直接用绝对路径 C:\Miniconda3\envs\dlpack-cpu\python.exe)
C:\Miniconda3\Scripts\activate.bat dlpack-cpu

:: 环境自检
python check_env.py

:: 运行验证
python verify_dlpack.py            --device cpu
:: 910B 上:
python verify_dlpack.py            --device npu
```

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

## 4. 910B (Atlas A2) 移植指南

### 4.1 移植前必查:版本配套

脚本代码**不需要改**(设备自动/参数切换),要换的只有依赖版本:

1. 服务器上查 CANN:`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
2. 按 CANN 查 torch_npu 配套表:gitee.com/ascend/pytorch 的"版本配套表"
3. torch 与 torch_npu **主版本必须一致**(如都 2.5.1)
4. MindSpore 用支持 A2 的版本(客户环境已有就沿用,拿版本号跑本脚本回归)

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

### 4.4 在客户服务器上的操作流程(JupyterLab 终端)

```bash
# 1. 传文件: 把 dlpack-verify/ 整个目录传上去 (scp / lrzsz / jupyter 上传)
# 2. 确认客户已有 python 环境 (conda/venv), 装好配套依赖后:
python check_env.py                 # 先自检, 看 torch_npu/NPU/CANN 三项必须 OK
python verify_dlpack.py --device npu --rtol 5e-3
# 3. 输出截图留档, PASS/FAIL 报告即验证交付物
```

## 5. 常见坑速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `torch.utils.dlpack` 对 NPU 张量报错 | 忘了 `import torch_npu`,补丁未生效 | 任何 torch 调用前先 import |
| torch 导出报 `Can't export tensors that require gradient` | 带 grad 的张量直接导出 | `.detach()` 后再导出 |
| MS 侧梯度看起来"对但不对" | sens 用了默认 ones 而非真实上游梯度 | 脚本 STEP 7 负对照就是抓这个 |
| 数值偶发不一致(910B) | 跨框架执行流未同步 | 消费前 `torch.npu.synchronize()` |
| pip 装 mindspore 找不到 Windows 包 | 2.4~2.6 无 Windows wheel(2.7.0 起恢复) | Windows 选 2.11.0(或 ≥2.7)+ py3.10 |
