# MiniMax H3 on a Single AMD R9700 32GB with Only 48GB System RAM

**AMD R9700 32GB + 48GB RAM 的 MiniMax H3 15 秒生成实测与低内存部署排障记录**

*Deployment & OOM-troubleshooting field report — Ubuntu / ROCm / Docker / ComfyUI — 2026-10-31 → 2026-11-03*

> **English TL;DR** — A single AMD Radeon AI PRO R9700 (32GB VRAM) workstation with only **48GB of system RAM** completed a full-length MiniMax H3 generation: FL2VA FP8 + Qwen3-VL 32B FP8 text encoder, 608×352, 24 FPS, **362 frames / 15.083s with a native audio track**, 20 steps, 442.26s end-to-end, no Docker OOM kill. The hard part was not the GPU — it was host/container memory management while the 32B text encoder and the H3 DiT swap in and out: Docker cgroup RAM/Swap budgeting, `--cache-none`, dropping `--fast-disk`, and adding 16GiB of host SSD swap. This is an engineering field report, **not** a first-run record and **not** a new inference algorithm. Full log: [FULL-REPORT.md](FULL-REPORT.md). Community post drafts: [POST.md](POST.md).

## 关键结果

| 项目 | 实测 |
|---|---|
| GPU | AMD Radeon AI PRO R9700 32GB（`gfx1201`） |
| 系统内存 | **48GB**（本案例的核心约束） |
| 模型 | MiniMax H3 FL2VA **FP8** + Qwen3-VL **32B FP8** 文本编码器 |
| 输出 | 608×352 / 24 FPS / **362 帧 / 15.083 秒** / 原生音频轨道 ×1 |
| 采样 | 20 Steps |
| 完整工作流耗时 | **442.26 秒** |
| 容器内存峰值 | RAM 34.00GiB（达到容器上限）+ Swap 5.71GiB |
| GPU 热点峰值 | 98°C（监控采样） |
| Docker OOM Kill | **否** |
| 核验方式 | MP4 逐帧解码：362 帧 / 24 FPS / 音频流 1 条 |

## 这份报告是什么、不是什么

**是**：一份资源受限环境下的工程实践与故障排查记录——在 48GB 物理内存下，如何通过容器内存管理与 Swap 扩容，让完整 FP8 工作流（不降精度、不减步数、不用 Turbo）完成 15 秒生成。

**不是**：

- 不是 R9700 运行 H3 的首例（社区今年已有 Windows 与双卡案例，见下方对照）；
- 不是新的推理算法或 ROCm 算子（模型、权重、工作流均为官方与社区现成成果）；
- 不是性能基准（n=1，未做连续稳定性与更高分辨率验收）。

它的受众：拥有 R9700 或类似 32GB AMD 显卡、系统 RAM 只有 32–64GB、想在 Linux 上跑完整大型视频模型的用户。读完最可能的收获不是新算法，而是少走几天弯路。

## 环境

| 项目 | 版本 / 配置 |
|---|---|
| CPU | AMD Ryzen 9 9950X |
| 系统 | Ubuntu 26.04.1 LTS |
| 基础镜像 | `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0` |
| PyTorch | 2.14.0+rocm10.1.0 |
| ComfyUI | v0.33.2（commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa`） |
| 模型文件 | DiT `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`（19.52GiB）；编码器 `qwen3vl_32b_minimax_h3_fp8.safetensors`（24.71GiB）；Video VAE FP16（4.85GiB）；Audio VAE FP32（577MiB） |
| 权重来源 | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)；编码器 [rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui](https://huggingface.co/rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui) |

> 复现限制：本机下载时的 HF revision 与模型 SHA-256 未完整留档；本地构建镜像的完整 Dockerfile 未公开。因此本文是**实测记录**，不是一键复现包。补齐清单见文末。

## 最有价值的三个发现

**1. 48GB 内存也能跑完完整工作流。** 主流指南通常默认 64GB 起步。本案例证明：48GB 物理 RAM + 合理容器限额 + SSD Swap + 模型卸载流程，可以完成 362 帧满时长生成，不必先加钱买内存。

**2. 三种「内存不足」不是一回事，必须分开看。**
- GPU 显存不足（VRAM）：本次 32GB 全程不是瓶颈；
- Docker cgroup 限额：容器 RAM 与容器 Swap 是两个独立额度，且受 `RAM+Swap 总限额` 约束；
- 宿主机 Swap 耗尽：会在 Qwen 编码器加载阶段先爆。
一个可教学的反面案例：把容器 RAM 从 34GiB 提到 38GiB、总限额仍锁 42GiB 时，容器可用 Swap 反而从 8GiB 被压缩到 4GiB，问题更严重——最终恢复 34GiB 基线才对。

**3. 分阶段数据比一张成功截图有用。** 39 → 124 → 209 → 294 → 362 帧逐级推进，每阶段保留耗时、内存峰值、失败条件；其中 294 帧的两次中止经查是安全监控因整机 Swap 不足主动停机（`OOMKilled=false`），不是模型 OOM——误判会把人引向完全错误的修复方向。

## 实验数据（相同模型与参数：608×352 / 24 FPS / 20 Steps）

| 阶段 | 帧数 | 时长 | 工作流耗时 | 结果 |
|---|---:|---:|---:|---|
| `--cache-none` + `--fast-disk` | 39 | 1.625s | 97.14s | 成功；Swap 峰值 ≈5.74GiB |
| 去掉 `--fast-disk` | 39 | 1.625s | 86.74s | 成功；Swap 峰值 ≈5.22GiB |
| 逐级放大 | 124 | 5.167s | 138.25s | 成功；解码 124 帧 + 音频 1 条 |
| 逐级放大 | 209 | 8.708s | 217.74s | 成功；解码 209 帧 + 音频 1 条 |
| Swap 扩容前 | 294（目标） | — | — | 两次在 Qwen 加载阶段被安全监控主动中止（整机 Swap 见底），非生成失败 |
| 宿主机 +16GiB Swap 后 | 294 | 12.25s | 332.99s | 成功 |
| **最终** | **362** | **15.083s** | **442.26s** | **成功；MP4 逐帧核验通过** |

失败路径摘要：原生 venv 先卡在 ROCm 动态库（`libroctracer64.so` 缺失）→ 改走 Docker ROCm 镜像；30GiB 容器跑完整流程 OOM → H3-only 单独加载成功，锁定问题在 **Qwen 32B 编码器 → H3 DiT 的切换峰值**，而非单个模型装不下。

## 最终成功配置

**宿主机**：物理 RAM 48GB；系统 Swap 合计 ≈23.5GiB（原 ≈8GiB + 新增 16GiB SSD swapfile）。

**Docker 基线**（容器名 `comfyui-h3-34g-nofastdisk`）：

```text
镜像（本地构建）：ai-stack/comfyui-h3:rocm10.1-v1
--memory=34g            # 容器 RAM 上限
--memory-swap=42g       # RAM+Swap 总限额 → 容器可用 Swap 8GiB
--shm-size=8g
--device=/dev/kfd --device=/dev/dri   # GPU 透传；设备组号按本机实际
端口：127.0.0.1:8190 → 8190（仅本机映射）
模型目录只读 bind mount；缓存 / 输入 / 输出独立挂载
```

**ComfyUI 启动关键参数**：

```bash
python3 main.py --listen 0.0.0.0 --port 8190 --disable-auto-launch \
  --disable-pinned-memory --cache-none
# 不使用 --fast-disk
```

> 注意：不能把成功完全归因于某一个旗标——各轮测试之间缓存与驻留状态并不完全一致。这是经实测成功的**组合**，不是严格单因素的因果证明。

## 诊断速查

```bash
free -h && swapon --show                                   # 宿主机 RAM / Swap
sudo docker inspect <容器> --format 'RAM={{.HostConfig.Memory}} RAM+Swap={{.HostConfig.MemorySwap}} OOM={{.State.OOMKilled}}'
rocm-smi --showtemp --showfan --showpower --showuse        # GPU 状态
# 容器 cgroup 实际用量（含 swap.current / memory.events）见 FULL-REPORT.md 第 6 节
```

判断顺序建议：先看 `OOMKilled` 与 cgroup 事件，再看宿主机 Swap 余量，最后才怀疑显存与驱动——本次内核日志出现过 AMDGPU HMM/SVM 调用栈，但证据不支持把驱动定为根因。

## 与已有社区工作的关系

本案例建立在社区已有工作之上，主要区别在约束条件与问题域：

| 维度 | 已有社区案例 | 本案例 |
|---|---|---|
| R9700 运行 H3 | 已有实现 | 非首例 |
| 15 秒 / 362 帧 | 已有公开记录（如 AMD-Pro-Bench，864×480、W4A8 Turbo、8 步等不同条件） | 608×352、FP8、20 步，条件不同，不直接比速度 |
| 系统内存 | 有案例 128GB | **48GB** |
| ROCm 环境 | 多为 ROCm 7.2 / Windows / 双卡 | **ROCm 10.1 + Docker / Ubuntu / 单卡** |
| 主要方向 | GPU 推理性能、量化、算子补丁 | **主机与容器内存管理、Swap 调度** |
| 资料形态 | 侧重性能结果 | **完整失败与修复路径** |

致谢与来源说明（哪些来自已有资料，哪些是本机实验）：

- 模型权重与 FP8 适配：[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)、[rzgar 的 Qwen3-VL 32B FP8 编码器](https://huggingface.co/rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui)；
- 单卡 R9700 的 H3 优化与编码器卸载研究：[boxwrench/R9700](https://github.com/boxwrench/R9700)；
- R9700 视频推理补丁（128GB 内存环境）：[charlie12345/R9700AIProComfyUIPatch](https://github.com/charlie12345/R9700AIProComfyUIPatch)；
- `--cache-none` / `--fast-disk` 与编码器宿主内存驻留问题：Hugging Face `Comfy-Org/MiniMax-H3` 讨论区已有讨论在先；
- 本机独立完成的部分：48GB 约束下的容器 RAM/Swap 配比实验、38GiB 反例、宿主机 Swap 扩容决策、分阶段帧数推进与全部实测数据。

## 局限（发布时不删）

- 362 帧仅成功验收 **1 次**；未做连续多轮稳定性测试；
- 分辨率 608×352 低于 H3 原生 768 短边；未测更高分辨率；
- 模型 revision / SHA-256、完整 Dockerfile、可直接运行的工作流 JSON 尚未补齐——可复现程度目前为**中等**；
- 温度与功耗各有一处未决读数（GUI 103°C vs 终端 97°C；`power1_average` 曾采到 389W 而 cap 为 300W），口径未统一，不据此下结论；
- 音频仅核验轨道存在，未做可听性与同步的主观验收。

## 后续补齐清单

- [ ] 完整 Dockerfile 与依赖锁定
- [ ] 模型文件 SHA-256 与 HF revision 记录
- [ ] 完整工作流 JSON
- [ ] 内存时序日志（cgroup / rocm-smi 采样）
- [ ] 连续多轮稳定性复测

## 样片

`sample-r9700-h3-15s.mp4`：608×352 / 24 FPS / 362 帧 / 15.083 秒 / 含原生音频（约 0.9MB）。可通过本仓库的 **Releases** 下载，或在 Hugging Face 讨论帖中直接观看。

## 文件

- [FULL-REPORT.md](FULL-REPORT.md) — 完整复盘原文：时间线、全部实验、诊断命令、问题速查表（333 行）
- [POST.md](POST.md) — 社区发帖文案（Hugging Face 讨论版 / 简介版）
