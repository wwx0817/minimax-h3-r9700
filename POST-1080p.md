# Hugging Face 讨论帖文案（1080p 更新）

## A. Hugging Face 讨论帖（Comfy-Org/MiniMax-H3 Discussions，新帖）

**标题**：Follow-up: MiniMax H3 at 1080p on a single R9700 32GB — 5/7/8s succeeded, 10s hits the VRAM ceiling (Ubuntu / ROCm / Docker)

**正文**：

Following up on #74 (15s at 608×352 on the same machine): same workstation, now at 1080p.

**Setup**

- AMD Radeon AI PRO R9700 32GB (gfx1201), Ryzen 9 9950X, **48GB system RAM**
- Ubuntu 26.04.1, Docker, base image `rocm/pytorch:rocm10.1.0_..._pytorch_release_2.14.0`, ComfyUI v0.33.2
- MiniMax H3 FL2VA FP8 + Qwen3-VL 32B FP8 text encoder, 20 steps, no Turbo

**Results** (1920×1088 native, delivered as 1920×1080, 24 FPS)

- 5s (124 frames): 2600.5s, ~27.8G VRAM — success
- 7s (175 frames): 4759s, 27.96G — success
- 8s (192 frames): 5628s (93.8 min), 28.97G — success
- 10s (243 frames): CUDA OOM at 31.3G / 31.86G — fails; the GNOME desktop crashed with it (framebuffer starvation) and the machine rebooted
- All three successes verified frame by frame with ffprobe plus visual spot-checks; native AAC stereo audio on all clips

**What I learned this round**

1. Before a long generation, stop gdm3 — the desktop competes for VRAM, and when VRAM fills up the desktop crashing can take the whole machine down.
2. After a run, `/free` (unload_models + free_memory) does NOT release the ~24G of anonymous memory left behind; restarting the container is the only reliable cleanup (28.5G → 0.6G).
3. Don't run `swapoff -a` while memory is tight — it triggered a container restart.
4. Set timeouts by measured frames-per-second: my first 8s attempt was killed by my own 90-minute script timeout about 2 minutes before it would have finished; 120 minutes worked.
5. Frame counts must be 17k+5; width/height must be multiples of 32 (1080 is not legal, 1920×1088 is). VAEDecodeTiled must be on — it handles H3's NestedTensor.

Full report (timeline, data tables, diagnostics, Docker baseline): https://github.com/wwx0817/minimax-h3-r9700 — see FULL-REPORT-1080p.md

Scope, as before: n=1, single card, single model — an engineering field report, not a benchmark, and not a first R9700+H3 run. The 10s ceiling is specific to 32GB VRAM at these settings.

Samples attached: 8s clip (1920×1080, 8.000s) and 7s clip (1920×1080, 7.292s), both with native audio.

---

## B. 简介版（Reddit / 知乎 / 论坛通用）

**AMD R9700 32GB 跑 MiniMax H3：1080p 8 秒出片，10 秒撞上显存上限**

同一台机器（R9700 32GB + 48GB 内存），这次把分辨率推到了 1080p：1920×1088 原生、24FPS、20 步，5 秒、7 秒、8 秒三档全部成功，8 秒那档跑了 93.8 分钟、峰值显存 28.97G，带原生立体声音频。10 秒是这张卡的硬上限：显存冲到 31.3G 触发 CUDA OOM，还连带 GNOME 桌面崩溃、整机重启。

这次最有用的两个教训：长片段生成前先停掉桌面（gdm3），别让桌面跟你抢显存；跑完一轮别指望 `/free` 清内存，24G 匿名内存它清不掉，重启容器才是可靠办法。

完整报告（时间线、数据表、诊断命令、Docker 配置）：https://github.com/wwx0817/minimax-h3-r9700

说明：单卡单模型的工程实测（n=1），不是性能基准；也不是 R9700 跑 H3 的首例。
