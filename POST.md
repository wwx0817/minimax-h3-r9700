# 社区发帖文案

## A. Hugging Face 讨论帖（Comfy-Org/MiniMax-H3 Discussions）

**标题**：Field report: MiniMax H3 full 15s generation on a single R9700 32GB with only 48GB system RAM (Ubuntu / ROCm / Docker) — OOM debugging notes

**正文**：

Sharing a field report from a memory-constrained AMD workstation, in case it saves someone a few days:

**Setup**
- AMD Radeon AI PRO R9700 32GB (gfx1201), Ryzen 9 9950X, **48GB system RAM**
- Ubuntu 26.04.1, Docker, base image `rocm/pytorch:rocm10.1.0_..._pytorch_release_2.14.0`, ComfyUI v0.33.2
- MiniMax H3 FL2VA FP8 + Qwen3-VL 32B FP8 encoder (Comfy-Org weights + rzgar encoder), no Turbo, no step reduction

**Result**
- 608×352, 24 FPS, **362 frames = 15.083s, with 1 native audio track**, 20 steps
- 442.26s end-to-end, Docker `OOMKilled=false`, verified by decoding the MP4 frame by frame
- Container peaks: 34.00GiB RAM (at limit) + 5.71GiB swap; GPU hotspot 98°C

**What actually broke, and what fixed it**
The GPU was never the problem. The failures were all in host/container memory during the Qwen-encoder → H3-DiT switch:
1. 30GiB container OOM on the full workflow, while H3-only loading succeeded — the peak is in the *transition*, not in either model alone.
2. Raising container RAM 34→38GiB **made it worse**: with `--memory-swap` total still capped at 42GiB, container swap shrank 8→4GiB.
3. Two 294-frame attempts were stopped by my own safety monitor when *host* swap ran out — not model OOMs.
4. Fix: +16GiB host SSD swap (≈23.5GiB total), back to the 34GiB container baseline, `--cache-none`, and dropping `--fast-disk`.

Staged runs with full data (39→124→209→294→362 frames), the working Docker/ComfyUI config, and cgroup diagnostic commands are in the repo: <填仓库链接>

To be clear about scope: this is **not** a first R9700+H3 run (Windows and dual-GPU reports exist, and boxwrench/R9700's single-card work predates this), and it's n=1 at 608×352 — an engineering/debugging record, not a benchmark. Model revisions/SHA-256 and the full Dockerfile are still missing; listed as open items in the repo.

Sample video attached (608×352, 15.083s).

---

## B. 简介版（Reddit / 知乎 / 论坛通用）

**AMD R9700 32GB 本地部署 MiniMax H3：48GB 内存下从 OOM 到生成 15 秒原生音视频**

经过几天调试，在单张 AMD Radeon AI PRO R9700 32GB 上跑通了 MiniMax H3 的完整工作流。最难的不是显卡，而是只有 48GB 系统内存时，大型文本编码器与 H3 主模型切换期间的内存管理。

- 硬件：Ryzen 9 9950X + R9700 32GB + **48GB RAM**
- 软件：Ubuntu 26.04.1 / ROCm 10.1 / PyTorch 2.14 / Docker + ComfyUI 0.33.2
- 模型：MiniMax H3 FL2VA FP8 + Qwen3-VL 32B FP8（不降精度、不用 Turbo）
- 结果：362 帧 / 24 FPS / 15.083 秒 / 608×352 / 20 Steps，442.26 秒完成，带原生音频，Docker 未 OOM Kill

关键经验：GPU 显存、Docker 容器限额、宿主机 Swap 是三种不同的「内存不足」，要分开监控——我曾把容器 RAM 从 34GiB 提到 38GiB，却因总限额不变把容器 Swap 压到 4GiB，越改越糟；最终靠恢复基线配比 + 给宿主机扩 16GiB Swap 跑通。

完整报告（分阶段实验数据、失败路径、最终配置、诊断命令）：<填仓库链接>

说明：这不是 R9700 跑 H3 的首例，社区已有优秀工作在先（见报告致谢）；本报告的定位是低内存条件下的部署排障记录，362 帧目前仅验收 1 次，未做稳定性与高分辨率测试。
