# AMD Radeon AI PRO R9700 32GB 本地运行 MiniMax H3：从下载、ROCm/Docker 报错、内存 OOM 到成功生成 15 秒视频

> **社区技术复盘｜2026-10-31—2026-11-03｜单卡 AMD / Ubuntu / ROCm / Docker / ComfyUI**  
> **结论：** 一台 R9700 32GB 显存、48GB 系统内存的工作站，使用 MiniMax H3 FL2VA FP8 + Qwen3-VL 32B FP8，在 **608×352 / 24FPS / 362 帧 / 20 Steps** 下完成 **15.083 秒视频 + 1 条音频轨道**，完整执行用时 **442.26 秒（7 分 22 秒）**，未发生 Docker OOM Kill。  
> **证据范围：** 成功生成一次 362 帧任务并对 MP4 结构进行了逐帧解码核验；这不是 1080P 测试，也不是连续多次生成或主观画质验收。

## 0. 摘要：什么问题最终解决了？

最初不是 GPU 无法识别，而是 **模型加载与主机/容器内存管理** 的组合问题。ComfyUI 在加载大型 Qwen 文本编码器、再加载 H3 DiT 时，存在高峰值系统内存和换页压力。经过独立的 H3 模型加载测试、关闭 ComfyUI 的执行缓存（`--cache-none`）、去掉 `--fast-disk`、合理分配 Docker RAM 与 Swap，以及给 Ubuntu **新增 16 GiB SSD Swap** 后，同一套完整精度配置最终跑通 15 秒。

**最值得复用的经验：把主机 RAM、主机 Swap、Docker cgroup RAM、Docker cgroup Swap、GPU VRAM 分开监控；不要只看 GPU 的 32GB 显存，也不要把一次因安全监控停止的任务误判为模型 OOM。**

## 1. 我的硬件与系统

| 项目 | 本机情况 |
|---|---|
| GPU | **AMD Radeon AI PRO R9700，32GB VRAM**，`gfx1201`，实际约 31.86GiB |
| CPU | AMD Ryzen 9 9950X |
| 物理 RAM | 48GB，Ubuntu `free -h` 显示约 45GiB |
| SSD | Linux 独立约 1TB SSD；另有 Windows SSD；以下模型及 Swap 均使用 Linux 存储环境 |
| 系统 | Ubuntu **26.04.1 LTS**（内核记录为 7.0.0-38-generic） |
| Docker | Ubuntu `docker.io`，安装时记录版本 **29.1.3-0ubuntu4.1** |
| 基础镜像 | `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0` |
| 本地构建镜像 | `ai-stack/comfyui-h3:rocm10.1-v1`（**个人构建镜像名，并非可直接公开拉取的镜像**） |
| 容器 Python | 3.12.3 |
| PyTorch | `2.14.0+rocm10.1.0` |
| HIP（容器日志） | `7.16.26385` |
| Triton | `3.8.0+git669b31ac.rocm10.1.0` |
| ComfyUI | **v0.33.2**，commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa` |
| comfy-kitchen / comfy-aimdo | 0.2.31 / 0.4.13 |
| ComfyUI 前端 | `comfyui-frontend-package 1.49.6` |
| 操作界面 | ComfyUI localhost:8190；另使用个人封装的 **AI Studio Desktop v0.2.1** |

这里的 ROCm 10.1、PyTorch 2.14 和 HIP 版本是**已运行容器**中的实际版本，不等于宿主机每个 ROCm 库都处于同一个版本。个人工程中还有其他 AI 任务，但这次 H3 运行时不与其他大模型同时运行。

## 2. 模型、文件与路径

这轮**没有为省内存更换成 4B 文本编码器，也没有把 H3 改成 W4A8 或低步数 Turbo**。核心工作流如下：

| 角色 | 文件 | 本地实际文件大小（约） |
|---|---|---:|
| H3 FL2VA DiT（本次生成使用） | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 19.52GiB |
| Qwen3-VL 32B 文本编码器 | `qwen3vl_32b_minimax_h3_fp8.safetensors` | 24.71GiB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 4.85GiB |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 577MiB |
| 备用 R2V 模型（本次 T2V 不使用） | `minimax_h3_ref2va_pruned_fp8_scaled.safetensors` | 19.52GiB |

文件大小来自环境审计中的 `stat` 字节数，不是 GPU 加载日志中的 `MB loaded`；两者不是同一个概念。以上未提供本地完整 SHA-256，因此**不能把第三方公布的模型哈希当作本机已验证哈希**。

原目录结构（把本地用户名替换成 `$HOME`）：

```text
~/AI-Stack/
├── apps/ComfyUI/
├── models/minimax-h3/
│   ├── diffusion_models/
│   │   ├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors
│   │   └── minimax_h3_ref2va_pruned_fp8_scaled.safetensors
│   └── vae/
│       ├── minimax_h3_video_vae_fp16.safetensors
│       └── minimax_h3_audio_vae_fp32.safetensors
├── models/qwen-vision/text_encoders/
│   └── qwen3vl_32b_minimax_h3_fp8.safetensors
├── cache/{huggingface,torch,triton,torchinductor,miopen}/
├── inputs/minimax-h3/
├── outputs/videos/minimax-h3/
├── runtime/comfyui-user/
├── scripts/
└── backups/
```

公开下载参考地址（**供读者自行核对模型版本/许可证；我本机文件的精确下载 revision 未完整留档**）：

- H3 FP8 与 VAE 文件：[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)；其中 [FL2VA FP8 单文件](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors)。
- Qwen3-VL 32B FP8 编码器：[rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui](https://huggingface.co/rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui)。本机保存时改成了上表中的短文件名。
- ComfyUI 源码：[Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)。

**复现时**应记录 Hugging Face 仓库 revision、模型 SHA-256、ComfyUI commit 和自定义节点版本；不能只凭文件名认定与本文完全一致。阅读模型许可证后再决定如何分发样例和权重。

## 3. 从下载到安装：时间线

### 10月31日：先排查 ROCm/PyTorch，原生 venv 没有直接跑通

起初尝试本机 Python 虚拟环境中的 ComfyUI，执行 `import torch` 时出现缺少 `libroctracer64.so` / `libroctx64.so` 等 ROCm 动态库的问题。此时不应直接归因于显卡不兼容：它首先暴露的是运行时库加载问题。

后来安装 Docker，并拉取与 PyTorch 2.14 对应的 ROCm 基础镜像：

```bash
sudo apt install docker.io
sudo systemctl enable --now docker
sudo docker pull rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0
```

下载初期曾出现网络连接被重置（`connection reset by peer`）；重新尝试后成功拉取。使用 `/dev/kfd` 与 `/dev/dri` 的 GPU 设备透传后，容器内 `torch.cuda` 能识别 R9700，并完成矩阵乘法测试。这里的 `torch.cuda` 是 PyTorch 对 HIP 设备沿用的 API 名称，不代表在使用 NVIDIA CUDA。

**这一阶段的意义：先证明 Docker + ROCm + PyTorch 可以实际做 GPU 计算，再安装完整 H3。**

### 11月1日：安装 ComfyUI 与模型、构建独立镜像

将 ComfyUI 源码、模型、缓存、输入输出分别放在宿主机目录，通过 Docker bind mount 使用。容器端口映射为 `127.0.0.1:8190:8190`，避免默认暴露到所有网络接口；GPU 设备使用 `/dev/kfd`、`/dev/dri`，以及该机器对应的设备组（44、990）。

使用本地镜像 `ai-stack/comfyui-h3:rocm10.1-v1`，补齐 Python 依赖、ComfyUI 和 H3 所需节点/模型路径。这里**没有保留完整可公开验证的 Dockerfile 与所有包的 hash 锁文件**，所以本文提供的是真实运行配置与排障记录，而非声称一条命令即可在所有机器从零安装。

界面方面，后续使用个人封装的 AI Studio Desktop v0.2.1 操作 ComfyUI，便于调整参数、查看状态及生成视频。前端属于操作层，不是H3的计算后端。

### 11月2日：定位 OOM，做隔离对照实验

早期 Docker 给了 **30GiB RAM**，运行H3完整工作流出现过 OOM/异常中止。部分内核日志还出现 AMDGPU HMM/SVM 相关调用栈，但仅凭这个调用栈不能证明 GPU 驱动就是根因。后来将容器RAM提高到 **34GiB**、使用 Swap，也曾失败。

当时没有直接升级整套 ROCm、没有降低模型精度，而是先分阶段验证：

1. **H3-only 单模型加载实验：成功。** 在受限容器中，单独加载 H3 FP8 可以完成，说明不是 H3 权重根本无法加载。
2. **Qwen → H3 过渡成为重点。** 完整流程必须先加载约25GB量级的文本编码器，再进入约20GB量级的 DiT，内存加载峰值与卸载时机比空闲RAM快照更关键。
3. **关闭 ComfyUI 执行缓存：增加 `--cache-none`。** 39帧测试成功。
4. **去掉 `--fast-disk`，保持 `--cache-none`。** 同一39帧工作流再次成功，容器峰值/Swap状况有所改善。
5. **保留 Qwen 条件编码后显式卸载路径。** 当前源码在条件编码后调用模型管理卸载函数 `comfy.model_management.unload_model_and_clones(clip.patcher)`；这并不意味着 Python 对象和进程匿名内存一定立刻全部释放。

**不能把成功完全归因于一个旗标**：测试之间的缓存、驻留状态和进程生命周期可能不同。可以说这是一套经实测成功的组合，但不是经过严格单因素研究的因果证明。

### 11月2日晚—11月3日凌晨：由 39 帧推进到 362 帧

通过39帧后不再改模型精度，沿用相同分辨率和20步参数，逐级测试124、209、294、362帧。294帧期间，最初两次停止不是 H3 长时序显存OOM，而是Qwen加载时**整机 Swap 不足触发人为设置的安全停止条件**。

- 第一次：34GiB RAM + 8GiB容器Swap，整机Swap剩余降到约0.96GiB，监控主动 `docker stop`。
- 第二次：临时将容器改为38GiB RAM、但 RAM+Swap 总额仍为42GiB，导致容器Swap额度反而只有4GiB；在Qwen阶段碰到该上限，并且整机Swap仅余约0.04GiB，监控再次中止。
- 修复：给 **Ubuntu宿主机增加16GiB磁盘Swap**，原有约8GiB保持不变，整机Swap合计约23.5GiB；然后**恢复已经成功的34GiB基线容器**，不再使用38GiB测试容器。
- 结果：294帧和362帧完整工作流均成功。

## 4. 实验数据表（相同模型、608×352、24FPS、20 Steps）

| 阶段 | 帧数 | 视频时长 | 完整工作流耗时 | 结果与说明 |
|---|---:|---:|---:|---|
| P2：`--cache-none`，仍有 `--fast-disk` | 39 | 1.625s | 97.14s | 成功；Swap峰值约5.74GiB |
| P3：去掉 `--fast-disk` | 39 | 1.625s | 86.74s | 成功；Swap峰值约5.22GiB |
| P4 | 124 | 5.167s | 138.25s | 成功；实际MP4解码124帧、音频轨道1条 |
| P5.1 | 209 | 8.708s | 217.74s | 成功；实际MP4解码209帧、音频轨道1条 |
| P5.2：Swap扩容前 | 294（目标） | 12.25s（目标） | — | 两次在Qwen加载阶段被**安全监控主动中止**，不能算生成失败或GPU OOM |
| P5.2：Swap扩容后 | 294 | 12.25s | **332.99s** | 工作流成功；监控热点峰值97°C |
| **P5.3：最终** | **362** | **15.083s** | **442.26s** | **成功，MP4逐帧结构核验通过** |

**最终362帧的监控实测：**

| 项目 | 峰值 / 最低值 |
|---|---:|
| 容器RAM峰值 | 34.00GiB（达到容器上限） |
| 容器Swap峰值 | 5.71GiB（容器额度8GiB） |
| 宿主机最低可用RAM | 10.54GiB |
| 整机最低剩余Swap | 14.33GiB |
| GPU热点最高监控值 | **98°C** |
| GPU核心最高监控值 | 70°C |
| GPU风扇最高转速 | 4250 RPM |
| 功耗传感器最高采样 | **389W（须核实测量口径）** |
| `Docker OOMKilled` | `false` |

有两个观察不能隐瞒：其一，先前另一个图形界面显示过103°C，但对应终端采样最高仅97°C，不同采样时点或显示路径可能导致差异，不能认定哪一方有故障；其二，`power1_average`出现过389W采样，而 `power1_cap` 设置为300W，可能涉及传感器/功耗统计口径或瞬时行为，**这不是已经证实的389W持续板卡功耗**。

## 5. 最终成功运行方案

### 主机端

```text
Ubuntu 26.04.1 LTS
物理 RAM：48GB（Linux 约45GiB）
新增系统 Swap：/swapfile-h3-16g，16GiB
旧系统 Swap：/swap.img，约8GB
合计系统 Swap：约23.5GiB（free -h 显示约23GiB）
```

新增 Swap 使用的是 **SSD 存储空间**，不是新增RAM，也不是扩展显卡VRAM。创建前应确认磁盘空间和文件系统，本机为ext4。原创建方法如下（**仅供有相应磁盘空间、且已确认文件不存在的人参考，不要无检查地重复运行**）：

```bash
# 本机最终增加的是一个16GiB文件；已有文件不可覆盖
sudo install -m 600 /dev/null /swapfile-h3-16g
sudo dd if=/dev/zero of=/swapfile-h3-16g bs=1M count=16384 status=progress conv=fsync
sudo chmod 600 /swapfile-h3-16g
sudo mkswap /swapfile-h3-16g
sudo swapon /swapfile-h3-16g
# 经检查后写入 /etc/fstab 持久化：
# /swapfile-h3-16g none swap sw 0 0
```

### Docker 成功基线

```text
镜像：ai-stack/comfyui-h3:rocm10.1-v1（本地构建）
容器：comfyui-h3-34g-nofastdisk
端口：127.0.0.1:8190 → 容器8190
RAM限制：--memory=34g
RAM+Swap总限制：--memory-swap=42g
允许容器使用的Swap：8GiB
共享内存：--shm-size=8g
GPU透传：--device=/dev/kfd --device=/dev/dri
设备组：--group-add 44 --group-add 990（具体组号要按本机检查）
安全参数：--security-opt seccomp=unconfined（只建议在受控环境评估）
显卡选择：HIP_VISIBLE_DEVICES=0
启动命令：python3 main.py ... --disable-pinned-memory --cache-none
不使用：--fast-disk
模型目录：只读bind mount；缓存、数据库、输入和输出独立挂载
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

注意：容器内监听 `0.0.0.0`，但**宿主机只将8190映射到127.0.0.1**。Docker本地镜像是单独构建的，以上参数**不是完整的从零部署脚本**；复现还需要实际构建依赖与完整挂载配置。

### H3 视频参数

```text
Model: MiniMax H3 FL2VA pruned FP8 scaled
Text encoder: Qwen3-VL 32B FP8
Video VAE: FP16
Audio VAE: FP32
Width: 608
Height: 352
Frames: 362
FPS: 24
Steps: 20
任务：文本生成视频 + 原生音频分支
```

设置为362帧，输出为15.083秒；不等于“15秒视频在任意分辨率都能成功”。

## 6. 可复用的诊断方式

**区分主机资源与Docker cgroup资源：**

```bash
free -h
swapon --show
sudo docker inspect comfyui-h3-34g-nofastdisk --format \
  'RAM={{.HostConfig.Memory}} RAM+Swap={{.HostConfig.MemorySwap}} OOM={{.State.OOMKilled}}'
sudo docker logs --tail 100 comfyui-h3-34g-nofastdisk
rocm-smi --showtemp --showfan --showpower --showuse
```

**检查容器是否真正发生OOM、目前Swap多少：**

```bash
C=comfyui-h3-34g-nofastdisk
PID=$(sudo docker inspect -f '{{.State.Pid}}' "$C")
CG_REL=$(awk -F: '$1=="0"{print $3}' "/proc/$PID/cgroup")
CG="/sys/fs/cgroup$CG_REL"
cat "$CG/memory.events" "$CG/memory.current" "$CG/memory.swap.current"
```

**检查最新MP4的实际帧数和音频流：**

```bash
sudo docker exec -i comfyui-h3-34g-nofastdisk python3 - <<'PY'
from pathlib import Path
import av
folder = Path('/ai/outputs/videos/minimax-h3')
p = max(folder.rglob('*.mp4'), key=lambda f:f.stat().st_mtime)
with av.open(str(p)) as c:
    video = c.streams.video[0]
    frames = sum(1 for _ in c.decode(video=0))
    print('file:', p)
    print('size:', video.width, video.height)
    print('fps:', video.average_rate)
    print('decoded frames:', frames)
    print('audio streams:', len(c.streams.audio))
PY
```

这条代码核验的是**最新MP4的结构**，不检查画质、音频是否真正可听或音画主观同步；请先确认文件时间与所测任务一致。

## 7. 阶段结束后：保存成果与手动启动

完成15秒测试后，另部署了一个本机手动启动脚本（用户反馈安装完成）：

- 不设置容器开机自启动：`docker update --restart=no`。
- 备份362帧MP4、实验日志、已有工作流、关键启动脚本和Docker inspect配置；**不重复复制几十GB模型文件，也不是完整离线镜像备份**。
- 在Ubuntu应用程序列表中建立 **“AI Studio H3（手动启动）”** 图标；按需启动原容器和界面。
- 保留300W原厂GPU功耗上限。监控采用**103°C日志预警、106°C停止容器**，加上宿主机RAM/Swap保护。
- 106°C距离此前报告的110°C临界值仅4°C：**这是用户选择的软件停止线，不是被证明安全的最高持续温度，也不替代显卡固件热保护。**

这次记录确认了安装脚本执行完成，**尚未记录“重启Ubuntu后点击桌面图标并成功打开界面”的独立验收结果**。不应在社区帖子里把该项写成实测通过。

## 8. 问题—原因—解决办法速查

| 问题 | 证据与边界 | 我采用的解决/判断 |
|---|---|---|
| 本机 venv `torch` 找不到 `libroctracer64.so` 等 | Python/ROCm动态库依赖问题 | 改用验证过的ROCm PyTorch Docker基础镜像；并非证明原生venv已被修复 |
| Docker镜像下载连接中断 | `connection reset by peer` | 检查网络、重新拉取，最终成功 |
| 30GiB / 34GiB容器跑完整工作流 OOM | 模型加载峰值，部分内核日志有HMM/SVM信息 | 分阶段独立加载、监控cgroup、对照缓存与内存配置 |
| H3-only能加载，完整流程加载失败 | 单模型可用，不代表Qwen→H3切换可用 | 重点观察文本编码器卸载及内存缓存 |
| 加载期间触及RAM上限 | 内存和文件页缓存共同占用cgroup | `--cache-none`、移除 `--fast-disk` 的成功组合；不把相关性说成唯一原因 |
| 294帧连续两次被“中止” | Qwen加载时整机Swap触发自设阈值；`OOM=false` | 新增16GiB宿主机Swap、恢复34GiB成功容器，不降低模型精度 |
| 容器改38GiB反而不解决 | 总上限仍42GiB，容器Swap额度被压缩为4GiB | 保留原34GiB + 8GiB容器Swap配置 |
| GUI显示与终端峰值不一致 | 界面103°C、日志97°C，采样口径未知 | 以传感器和原始时序日志交叉核验，不盲目归因 |
| `power1_average`采到389W而cap为300W | 两种读数口径/时间窗口未统一 | 列为未决事项，不能用单个数推断持续超功耗 |
| 15秒是否真的成功输出？ | `Prompt executed` 不能替代文件检查 | 使用PyAV逐帧解码、核对362帧/24FPS/音频轨道 |

## 9. 成果、局限与后续可做的事

**已证明：** 本机原样 FP8 模型、Qwen3-VL 32B条件编码、20步、608×352、362帧音视频工作流成功，耗时442.26秒；MP4确实362帧、24FPS且有一个音频轨道；容器没有OOM Kill。

**尚未证明：** 362帧连续多轮稳定性、音频可听且完全同步、1080P/更高原生分辨率、其他GPU的兼容性、重启后桌面启动入口的可靠性、389W传感器读数含义。连续长时间高负载时仍需关注散热，98°C是本次监控峰值而非目标日常温度。

**对类似AMD单卡用户的建议：** 先跑最小GPU矩阵测试，再做 H3-only 加载；固定模型与参数，逐级扩大帧数；留存每轮 `docker inspect`、cgroup监控、`rocm-smi`、容器日志和实际输出MP4。不要因为别人的低显存工作流成功，就默认同样精度、帧数、音频与ROCm后端可直接复刻。不要用减少模型精度掩盖未知的加载问题。

## 10. 公开参考和社区对照

- [ComfyUI 官方源码](https://github.com/Comfy-Org/ComfyUI)
- [Comfy-Org H3 FP8 权重 / VAE](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Qwen3-VL 32B H3 FP8 适配编码器](https://huggingface.co/rzgar/qwen3-vl-32b-minimax-h3-fp8-comfyui)
- [boxwrench/R9700：单R9700 H3优化记录](https://github.com/boxwrench/R9700/blob/main/docs/minimax-h3-r9700-optimization-20260818.md)
- [AMD-Pro-Bench：R9700视频生成公开记录](https://amd-pro-bench.com/video-benchmarks/gpus/r9700)

注意：公开社区中存在更高分辨率的362帧测试，但可能使用**不同量化方法、不同采样步数、其他版本ROCm或另一套实现**，不能直接把它们的速度和显存占用与本次FP8/20步完整工作流排名比较。

---

### 发布前需要做的事

- [ ] 先裁剪或遮挡截图中的真实账户、主机名、目录用户、远程IP、Token、SSH/Tailscale信息。
- [ ] 若公开 `docker inspect`、环境变量、日志或工作流JSON，先检查是否包含用户路径、凭证和未公开项目资料。
- [ ] 样片最好提供单独预览（清楚标注608×352、15.083秒）；帖子中的“结构通过”不等于承诺主观质量。
- [ ] 公开分享模型下载链接而非未经确认许可的大文件；遵守相应模型许可证。
- [ ] 如果将来补充 Dockerfile、模型hash、完整工作流 JSON、不同分辨率测试，应作为新增版本记录，不要回填成当时已经验证的事实。

*记录结束：2026年11月3日。以上实测值来自该工作站的终端输出、环境审计、运行日志和 MP4 结构核验；外部链接用于读者进一步检索，不作为本机实测的替代证据。*
