# AMD Radeon AI PRO R9700 32GB 本地运行 MiniMax H3：从 720p 到 1080p 8 秒、显存死机根因与清内存方案

> **社区技术复盘｜2026-11-23—2026-11-24｜单卡 AMD / Ubuntu / ROCm / Docker / ComfyUI**
> **结论：** 一台 R9700 32GB 显存、48GB 系统内存的工作站，使用 MiniMax H3 FL2VA FP8 + Qwen3-VL 32B FP8，在 **1920×1088（原生，成片缩放到 1920×1080）/ 24FPS / 20 Steps** 下，成功生成 **5 秒（124 帧）、7 秒（175 帧）、8 秒（192 帧）** 三档视频 + 原生 AAC 立体声；8 秒档完整耗时 **5628 秒（93.8 分钟）**，峰值显存 28.97GiB。10 秒档（243 帧）因显存 31.3G/31.86G 触发 CUDA OOM，并连带 GNOME 桌面崩溃导致整机重启。
> **证据范围：** 三档均成功出片并对 MP4 做了 ffprobe 逐帧/时长/音轨结构核验，抽帧做了主观画质检查（全部 PASS）；这是连续多次成功生成，不是单次巧合；但仍是单卡单模型的工程实测，不是跨硬件 benchmark，也不是画质盲测。

## 0. 摘要：什么问题最终解决了？

一开始的问题不是"能不能跑 H3"——这台机器早就能跑 15 秒低分辨率视频。真正要解决的是三个叠加问题：

1. **1080p 长视频能不能出片**——能，5/7/8 秒全部成功；10 秒是显存硬上限。
2. **为什么 10 秒会把整台机器搞死机**——不是内存 OOM、不是温度，是 GPU 显存被采样占满后 GNOME 桌面分不到 framebuffer，崩溃循环。
3. **为什么连续跑第二轮会卡死**——生成后模型残留 24G 匿名内存不释放；`/free` 无效，必须重启容器。

**最值得复用的经验：把 GPU 显存、容器 cgroup 内存/Swap、宿主 RAM/Swap 分开监控；长片段生成前停 gdm3 释放桌面显存，生成后重启容器清内存；超时上限要按实测帧数比例给足，否则会误杀本可成功的任务。**

## 1. 我的硬件与系统

| 项目 | 本机情况 |
|---|---|
| GPU | **AMD Radeon AI PRO R9700，32GB VRAM**，`gfx1201`，实际约 31.86GiB |
| CPU | AMD Ryzen 9 9950X |
| 物理 RAM | 48GB，`free -h` 约 45GiB |
| SSD | Linux 独立约 1TB SSD；Swap 用 Linux SSD 存储 |
| 系统 | Ubuntu **26.04.1 LTS**（内核 7.0.0-44-generic） |
| Docker | Ubuntu `docker.io` 29.x |
| 基础镜像 | `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0` |
| 本地构建镜像 | `ai-stack/comfyui-h3:rocm10.1-v1`（个人构建，非公开可拉取） |
| 容器 Python | 3.12.3 |
| PyTorch | `2.14.0+rocm10.1.0` |
| HIP | 7.16.26385 |
| ComfyUI | **v0.33.2**，commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa` |
| comfy-kitchen / comfy-aimdo | 0.2.31 / 0.4.13 |
| 操作界面 | ComfyUI localhost:8190；个人封装 AI Studio Desktop v0.2.1 |

这里的 ROCm 10.1、PyTorch 2.14 和 HIP 版本是**已运行容器**中的实际版本，不等于宿主机每个 ROCm 库都处于同一版本。

## 2. 模型、文件与路径

| 角色 | 文件 | 本地实际大小（约） |
|---|---|---:|
| H3 FL2VA DiT（生成使用） | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 19.52GiB |
| Qwen3-VL 32B 文本编码器 | `qwen3vl_32b_minimax_h3_fp8.safetensors` | 24.71GiB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 4.85GiB |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 577MiB |
| 备用 R2V 模型（本次 T2V 不使用） | `minimax_h3_ref2va_pruned_fp8_scaled.safetensors` | 19.52GiB |

目录结构（`$HOME` 替换本地用户名）：

```text
~/AI-Stack/
├── apps/ComfyUI/
├── models/minimax-h3/
│   ├── diffusion_models/
│   │   ├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors
│   │   └── minimax_h3_ref2va_pruned_fp8_scaled.safetensors
│   ├── vae/
│   │   ├── minimax_h3_video_vae_fp16.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
├── models/qwen-vision/text_encoders/
│   └── qwen3vl_32b_minimax_h3_fp8.safetensors
├── workflows/minimax-h3/
│   ├── h3_t2v_720p_{5s,10s}.json
│   └── h3_t2v_1080p_{5s,7s,8s,10s,probe}.json
├── scripts/
│   ├── h3-720p-test.py          # 提交+监控+核验一条龙
│   ├── h3-monitor.py            # 前台实时监控
│   ├── h3-monitor-bg.py         # 后台监控
│   ├── h3-cleanup.py            # 清内存并验证
│   └── h3-1080p-7s8s.py         # 7s→8s 自动流程
├── cache/{huggingface,torch,triton,torchinductor,miopen}/
├── inputs/minimax-h3/
├── outputs/videos/minimax-h3/
├── runtime/comfyui-user/
└── backups/
```

公开下载参考地址（供读者核对模型版本/许可证）：
- H3 FP8 与 VAE 文件：[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- Qwen3-VL 32B FP8 编码器：[rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui](https://huggingface.co/rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui)
- ComfyUI 源码：[Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)

**复现时**应记录 Hugging Face 仓库 revision、模型 SHA-256、ComfyUI commit 和自定义节点版本；不能只凭文件名认定与本机完全一致。

## 3. 从 720p 到 1080p：时间线

### 第一轮：720p 打通（11月23日晚）
沿用 11 月已验证的 FP8 + Qwen 卸载配置，先做 720p（1280×704，32 对齐下最接近 720p 的合法档）：
- 5 秒档（124 帧）成功，664.6 秒，无 OOM，温度峰值 95°C
- 10 秒档（243 帧）成功，1962.1 秒，Swap 稳定 5.43G——**证明这套配置大面积扩展仍稳**

### 第二轮：1080p 探针与一次死机（11月24日凌晨 00:38）
- 第一次 1080p 探针卡死：宿主 RAM 被压到 4.6G、Swap 7.3G，系统 thrashing 假死，最终人工重启
- 根因是前一轮 720p 任务结束后模型/页缓存没释放，加上 1080p 要加载 25GB Qwen——**内存残留问题第一次暴露**
- 重启容器释放 34G 后，干净状态重跑探针成功（39 帧 / 1.6 秒 / 455.5 秒），画质核验无崩坏

### 第三轮：1080p 10 秒——CUDA OOM + 整机死机（约 00:57）
- 10 秒档（243 帧）采样阶段 VRAM 冲到 31.3G/31.86G，先报 `CUDA out of memory`（PyTorch 占 29.54G，仅余 1.63G，需分配 1.48G 失败）
- 随后整机重启。查内核日志发现：GNOME 桌面合成器分不到 framebuffer（`error -12` / `Not enough memory for command submission`），gnome-shell 每 4-6 秒崩溃一次，gdm 反复重建会话 → 桌面死循环

### 第四轮：确认"清内存"的正确方法（约 01:10-01:15）
- 用 `/free`（unload_models + free_memory）清理：90 秒后容器 RAM 纹丝不动 28.5G（其中 anon 匿名内存 24.4G）
- 改为**重启容器**：RAM 28.5G → 0.62G，VRAM → 0.96G——确认重启是唯一可靠手段
- 期间一次 `swapoff -a` 在内存吃紧时反而触发容器重启，列为禁忌

### 第五轮：1080p 5/7/8 秒三档全成功（11月24日凌晨—清晨）
固定配方：**停 gdm3 释放桌面显存 → 重启容器清内存 → 生成 → 重启容器清内存 → 恢复 gdm3**
- 5 秒档：2600.5 秒（43.3 分），VRAM 27.8G，成功
- 7 秒档：4759 秒（79.3 分），VRAM 27.96G，成功
- 8 秒档：首轮 5341 秒被脚本 90 分钟超时切断（差约 2 分钟完成），超时放宽到 120 分钟后重跑，**5628 秒（93.8 分）成功**，VRAM 28.97G

## 4. 实验数据表（1920×1088 / 24FPS / 20 Steps / res_multistep-simple）

| 阶段 | 帧数 | 时长 | 完整工作流耗时 | 显存峰值 | 结果 |
|---|---:|---:|---:|---:|---|
| 720p 5 秒 | 124 | 5.167s | 664.6s | 容器Swap峰值5.85G | 成功 |
| 720p 10 秒 | 243 | 10.125s | 1962.1s | 容器Swap峰值5.43G | 成功 |
| 1080p 探针 | 39 | 1.625s | 455.5s | — | 成功，画质无崩坏 |
| 1080p 5 秒 | 124 | 5.167s | 2600.5s | ~27.8G | 成功，画质 PASS |
| 1080p 7 秒 | 175 | 7.292s | 4759s | 27.96G | 成功，画质 PASS |
| 1080p 8 秒 | 192 | 8.000s | 5628s | 28.97G | 成功，画质 PASS |
| **1080p 10 秒** | 243 | 10.125s | ~75s 中断 | **31.3G** | **失败：CUDA OOM + 桌面崩溃重启** |

**8 秒"首轮失败"的说明**：首轮在 5341s 被脚本 90 分钟超时上限切断。按 7 秒（4759s）帧数比例推算 8 秒仅需约 5221s，实际已超过推算时间；放宽超时到 120 分钟后重跑成功（5628s）。**失败是脚本超时设置问题，不是模型能力问题。**

## 5. 最终成功运行方案

### 主机端
```text
Ubuntu 26.04.1 LTS
物理 RAM：48GB
Swap：/swap.img（8GB） + /swapfile-h3-16g（16GB） = 约 23GB
生成大分辨率长视频前：sudo systemctl stop gdm3
生成完成后：sudo systemctl start gdm3
```

### Docker 成功基线
```text
镜像：ai-stack/comfyui-h3:rocm10.1-v1（本地构建）
容器：comfyui-h3-34g-nofastdisk
端口：127.0.0.1:8190 → 容器8190
RAM限制：--memory=34g
RAM+Swap总限制：--memory-swap=42g
共享内存：--shm-size=8g
GPU透传：--device=/dev/kfd --device=/dev/dri
设备组：--group-add 44 --group-add 990
安全参数：--security-opt seccomp=unconfined（受控环境评估）
显卡选择：HIP_VISIBLE_DEVICES=0
启动命令：python3 main.py ... --disable-pinned-memory --cache-none
不使用：--fast-disk
清内存：sudo docker restart comfyui-h3-34g-nofastdisk（每轮前后）
功耗上限：300W（rocm-smi power1_cap = 300000000）
```

后期成功配置的主要运行参数：

```bash
python3 main.py \
  --listen 0.0.0.0 --port 8190 \
  --disable-auto-launch \
  --extra-model-paths-config /opt/ComfyUI/extra_model_paths.yaml \
  --output-directory /ai/outputs/videos/minimax-h3 \
  --input-directory /ai/inputs/minimax-h3 \
  --user-directory /ai/runtime/comfyui-user \
  --database-url sqlite:////ai/runtime/comfyui-user/comfyui.db \
  --disable-pinned-memory --cache-none
```

### H3 视频参数

```text
Model: MiniMax H3 FL2VA pruned FP8 scaled
Text encoder: Qwen3-VL 32B FP8
Video VAE: FP16（VAEDecodeTiled tile=512, temporal=64）
Audio VAE: FP32
Width: 1920  Height: 1088（成片 lanczos 缩放到 1920×1080）
Frames: 124 / 175 / 192（对应 5 / 7 / 8 秒）
FPS: 24
Steps: 20  res_multistep / simple
任务：文本生成视频 + 原生音频分支
```

**帧数与分辨率的两个硬约束**（读节点源码得出）：
1. 帧数必须满足 `17k+5`：124=17×7+5，175=17×10+5，192=17×11+5，243=17×14+5，362=17×21+5
2. 宽高必须是 32 的倍数：1080 不合法，1920×1088 是 1080p 合法档

**VAEDecodeTiled 必须启用**：它原生支持 H3 的 NestedTensor（`if latent.is_nested: latent = latent.unbind()[0]`），能显著降低高位宽的解码显存。

## 6. 可复用的诊断方式

```bash
# 三类资源分开看（关键！）
free -h                                    # 宿主 RAM/Swap
rocm-smi --showmeminfo vram --json         # GPU 显存
sudo docker inspect comfyui-h3-34g-nofastdisk --format \
  'RAM={{.HostConfig.Memory}} RAM+Swap={{.HostConfig.MemorySwap}} OOM={{.State.OOMKilled}}'

# 容器 cgroup 实时内存（无需 docker 权限，从 /proc 找主进程）
PID=$(pgrep -f 'main.py --listen 0.0.0.0 --port 8190' | head -1)
REL=$(awk -F: '$1=="0"{print $3}' /proc/$PID/cgroup)
CG=/sys/fs/cgroup$REL
cat $CG/memory.current $CG/memory.swap.current $CG/memory.events

# 区分 anon 与 file cache（判断 /free 是否真的释放）
grep -E '^(anon|file)' $CG/memory.stat

# 显存/温度/功耗
rocm-smi --showtemp --showpower --showuse --showmeminfo vram

# 死机后查内核日志（上一 boot）
journalctl -k -b -1 --no-pager | grep -iE 'pin framebuffer|Not enough memory|gnome-shell|oom'
journalctl --list-boots        # 确认重启窗口

# 清内存（每轮前后）
curl -s -X POST http://127.0.0.1:8190/free -H 'Content-Type: application/json' \
  -d '{"unload_models":true,"free_memory":true}'   # 辅助，对残留 anon 无效
sudo docker restart comfyui-h3-34g-nofastdisk       # 真正有效的手段

# 核验成片
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,nb_read_frames -of csv=p=0 output.mp4
```

## 7. 成果、脚本与自动化

| 文件 | 说明 |
|---|---|
| `~/H3-调试记录/t2v-1080p-5s_1920x1080.mp4` | 5 秒严格 1080p 成片（124 帧 / 5.167s / H.264 + AAC） |
| `~/H3-调试记录/t2v-1080p-7s_1920x1080.mp4` | 7 秒严格 1080p 成片（175 帧 / 7.292s） |
| `~/H3-调试记录/t2v-1080p-8s_1920x1080.mp4` | 8 秒严格 1080p 成片（192 帧 / 8.000s） |
| `~/AI-Stack/outputs/videos/minimax-h3/minimax-h3/t2v-1080p-{5s,7s,8s}_00001_.mp4` | 原生 1920×1088 输出 |
| `~/AI-Stack/workflows/minimax-h3/h3_t2v_1080p_{5s,7s,8s,10s,probe}.json` | API 工作流 |
| `~/AI-Stack/scripts/h3-720p-test.py` | 提交+逐秒监控+安全线中断+ffprobe 核验 |
| `~/AI-Stack/scripts/h3-cleanup.py` | 清内存并验证归零 |
| `~/AI-Stack/scripts/h3-1080p-7s8s.py` | 7s→8s 全流程（每轮强制清内存） |
| `~/AI-Stack/scripts/h3-monitor.py` / `h3-monitor-bg.py` | 前台/后台实时监控 |

## 8. 问题—原因—解决办法速查

| 问题 | 证据与边界 | 采用的解决/判断 |
|---|---|---|
| 原生 venv `torch` 找不到 ROCm 库 | Python/ROCm 动态库依赖 | 改用验证过的 ROCm PyTorch Docker 镜像 |
| 容器 30GiB 跑完整工作流 OOM | Qwen→H3 切换时内存峰值 | 34GiB 基线 + 16GiB 系统 Swap + `--cache-none` |
| 第一次 1080p 探针卡死 | 宿主 RAM 4.6G、Swap 7.3G，系统 thrashing | 重启容器释放 34G；不在残留状态下跑新一轮 |
| `/free` 清不掉内存 | 90 秒后 anon 仍 24.4G | 改为重启容器：28.5G → 0.6G |
| `swapoff -a` 后容器重启 | 内存吃紧时清空 Swap 反噬 | 禁用 swapoff，Swap 交给系统自管 |
| 10 秒档 CUDA OOM | `Tried to allocate 1.48 GiB, 1.63 GiB free`，PyTorch 占 29.54G | 判定单卡 32GB 上限；降到 8 秒 |
| 10 秒时整机死机 | `pin framebuffer error -12`，gnome-shell 每 4-6 秒崩溃 | 生成前停 gdm3，桌面不抢显存 |
| 8 秒首轮"失败" | 5341s 被 90min 超时切断，推算仅需 5221s | 超时放宽到 120min，重跑成功 |
| 连续多次成功是否偶然 | 5/7/8 三档独立工作流全部成功且逐帧核验 | 确认为稳定可复现，非单次巧合 |

## 9. 成果、局限与后续可做的事

**已证明：**
- 单卡 R9700 32GB，FP8 精度、20 步、带原生音频，1080p（1920×1088）5/7/8 秒三档全部成功；8 秒档耗时 5628 秒、峰值 28.97G、无 OOM
- 三档 MP4 均经 ffprobe 逐帧/时长/音轨核验，抽帧主观画质 PASS，1080p 分辨率外推无崩坏
- 清内存的正确方法（重启容器）与桌面抢显存的死机根因均已定位并验证

**尚未证明：**
- 1080p 10 秒及以上（受 32GB 显存硬限）
- 其它 GPU / 更低精度 / 更低步数下的表现
- 主观画质盲测（本次抽帧为操作者目视 PASS，非 A/B 盲测）

**对类似 AMD 单卡用户的建议：**
- 先把 GPU VRAM、容器 cgroup、宿主 RAM/Swap 三类资源分开监控，再判断瓶颈
- 长片段生成前停桌面（gdm3），否则显存满载时桌面崩溃会拖垮整机
- 每轮生成后重启容器清内存，不要相信 `/free` 能释放 anon 内存
- 超时上限按实测帧数比例给足，8 秒档实测要 95 分钟以上
- 不要执行 `swapoff -a`

## 10. 公开参考和社区对照

- [ComfyUI 官方源码](https://github.com/Comfy-Org/ComfyUI)
- [Comfy-Org H3 FP8 权重 / VAE](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Qwen3-VL 32B H3 FP8 适配编码器](https://huggingface.co/rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui)
- [boxwrench/R9700：单 R9700 H3 优化记录](https://github.com/boxwrench/R9700)——其公开视频均为 864×480/124 帧量级
- [HF Comfy-Org/MiniMax-H3 讨论区](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions)——可见记录同样以 864×480 为主；官方说明 H3-Base 输出 768p，2K 依赖未开源的 H3-Regenerate-2K
- 本机 15 秒低分辨率复盘（前作）：`~/下载/R9700_MiniMax_H3_15s_Field_Report_2026-11-03.md`（英文版）与同目录 `%E7%A4%BE%E5%8C%BA...` 编码文件名的中文社区版

关于"领先"的边界说明：在**单卡 32GB、原生精度、直接生成 1080p** 这一窄维度，本次 5/7/8 秒是公开渠道未见的结果；但这不是整体领先——社区有 12GB 卡分段拼接 39 秒（低分辨率）、有 14.96GiB 的 4-trit 量化方案，属于不同维度。

### 发布前需要做的事

- [ ] 裁剪或遮挡截图中的真实账户、主机名、目录用户、远程 IP、Token、SSH/Tailscale 信息
- [ ] 若公开 `docker inspect`、环境变量、日志或工作流 JSON，先检查是否包含用户路径、凭证和未公开项目资料
- [ ] 样片提供单独预览，清楚标注 1920×1088 原生 / 1920×1080 成片、帧数、时长
- [ ] 公开分享模型下载链接而非未经确认许可的大文件；遵守相应模型许可证
- [ ] 明确本报告是 n=1 硬件的工程实测，不在标题里声称"benchmark"

---

*记录结束：2026-11-24 07:05。以上实测值来自该工作站的终端输出、环境审计、运行日志、内核日志与 MP4 结构核验。外部链接用于读者进一步检索，不作为本机实测的替代证据。*
