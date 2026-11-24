# MiniMax H3 on AMD Radeon AI PRO R9700

**Native 1080p-tier audio–video generation, GPU/desktop memory contention, Turbo LoRA, and recovery diagnostics**  
Research snapshot: **2026-11-25** · Single workstation · ComfyUI / PyTorch / ROCm / Docker

This repository documents a local MiniMax H3 FL2VA deployment on a single R9700 with limited host memory. It is an empirical engineering report, not a new model, a general performance guarantee, or a claim to the world record. The historical fastest completed native 1920×1088 / 192-frame / 24 fps workflow took **1,281 s (21m 21s)** using Turbo 4-step and FP8-scaled weights. Current production uses an additional 3 GiB application-level VRAM budget, and its equivalent four-step completion time has not yet been established.

- Community: [Hugging Face discussion #76](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/76)
- Upstream models: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- Research manuscript: `paper.md` / `paper.pdf` in the proposed release bundle (publish and verify before linking).

## 1. Experimental setup

| Item | Value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 32 GB, gfx1201, 31.86 GiB reported |
| Host | Ryzen 9 9950X, 48 GB RAM, ~23 GiB swap reported |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-44 in campaign records |
| Runtime | PyTorch 2.14.0+rocm10.1.0 / HIP 7.16.26385 |
| Historical ComfyUI | v0.33.2, recorded commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa` |
| Current environment | Python 3.12.3, Triton 3.8.0; live ComfyUI checkout reports `nogit` |
| Container | 34 GiB RAM / 42 GiB total memory+swap limit / 8 GiB shm |
| GPU policy | 300 W cap, GNOME retained, current `--reserve-vram 3` |
| Models | FL2VA pruned FP8-scaled, Qwen3-VL FP8, video VAE FP16, audio VAE FP32 |

The historical image `ai-stack/comfyui-h3:rocm10.1-v1` is no longer the active production image. The current locally derived image is `ai-stack/comfyui-h3:with-extras`; these are local image tags, not a published image distribution.

## 2. Benchmark results

### Generation (historical)

| Native canvas | Frames | FPS | Steps / mode | Wall time (s) | Sampled peak VRAM (GiB) | Outcome |
|---|---:|---:|---|---:|---:|---|
| 1280×704 | 124 | 24 | 20 | 664.6 | — | PASS |
| 1280×704 | 243 | 24 | 20 | 1962.1 | — | PASS |
| 1920×1088 | 39 | 24 | 20 | 455.5 | — | PASS |
| 1920×1088 | 124 | 24 | 20 | 2600.5 | ~27.8 | PASS |
| 1920×1088 | 175 | 24 | 20 | 4759 | 27.96 | PASS |
| 1920×1088 | 192 | 24 | 20 | 5628 | 28.97 | PASS |
| 1920×1088 | 192 | 24 | Turbo 8 | 2358 | 28.97 | PASS |
| 1920×1088 | 192 | 24 | Turbo 8 + EasyCache | 2364 | 28.91 | PASS; no observed speed benefit |
| 1920×1088 | 192 | 24 | Turbo 4 | **1281** | 28.91 | PASS |
| 1920×1088 | 243 | 24 | 20 | — | 31.3 | FAIL: allocation error + desktop disruption |

The historical tests span more than one desktop/configuration state. `PASS` means a completed reported output, not a multi-run reliability guarantee. The 192-frame result is the largest successful point observed in this campaign, not the hardware's proven maximum. Intermediate legal 209 and 226 frames are not tested. The 20-to-4-step result is 4.39× faster, but changes the sampler and LoRA as well as steps. Turbo 8-to-4 is 1.84×. FP8 arithmetic and actual kernel dispatch are not yet confirmed.

### Current production reservation

| Configuration | Result |
|---|---|
| 1920×1088 / 124 frames / 20 steps / GNOME on / reserve3 | PASS, 2613 s, desktop retained, no observed reboot |
| 1920×1088 / 192 frames / Turbo 4 / reserve3 | No valid completed benchmark in reviewed P0 evidence |
| P0 experimental instance with production still running | ABORTED; not eligible for performance comparison |

The 2,613 s result is only 12.5 s above the historical 2,600.5 s result (~0.48% raw difference); the controlled overhead is unknown. A separate internal note quotes 11% without a matching denominator. Do not extrapolate either value to the 192-frame workload.

## 3. Checkpoints and workflow

| Component | Filename |
|---|---|
| H3 | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` |
| Qwen | `qwen3vl_32b_minimax_h3_fp8.safetensors` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| Turbo LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` |
| RIFE (optional) | `rife426.pkl` |
| ESRGAN (optional) | `RealESRGAN_x4.pth` |

Do not download these again if already available; obtain checkpoints only from the corresponding upstream distribution under its applicable terms. The project does not redistribute model weights.

The P0 log captured the following four-step workflow:

```text
UNETLoader + CLIPLoader(type=minimax)
  -> MiniMaxH3ImageToVideo(width=1920, height=1088, length=192)
  -> MiniMaxH3TurboLoRA(strength=1.0, low_vram=True)
  -> BasicGuider + BasicScheduler(simple, steps=4, denoise=1.0)
  -> MiniMaxH3TurboSampler + RandomNoise(seed=8112026)
  -> SamplerCustomAdvanced
  -> VAEDecodeTiled(tile=512, overlap=64, temporal=64, temporal_overlap=8)
  + VAEDecodeAudio -> CreateVideo(fps=24, bit_depth=8)
  -> SaveVideo(format=mp4, codec=auto)
```

This is a logical dependency summary, not executable JSON. Use the archived workflow JSON for exact node connections. Captured prompt:

> A tiny brass robot carefully waters a glowing mushroom garden at night. Gentle rain, soft mechanical movements, warm lantern light. Audio: quiet rain, small servo sounds, and a soft bell chime. No text or logos.

The chosen native canvases are 1280×704 and 1920×1088. Tested frame counts follow `17k+5`. Resize to standard 720p/1080p only after generation; do not confuse a resized delivery file with exact native sampling dimensions.

## 4. Operational guidance (existing deployment only)

These commands inspect the author's existing production setup. They do not install a new environment or prove that an arbitrary host supports the workflow. Adjust the container name and paths for your own authorized instance.

```bash
# Read-only checks; run before any submission.
docker ps --format '{{.Names}} {{.Status}}'
curl -sS -m 30 -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8190/system_stats
free -h
rocm-smi --showmeminfo vram --json
```

Keep GNOME active, reserve3 unchanged, the power cap at 300 W and the current software versions frozen. Do not start a second inference container while production is running. Do not stop/mask the display manager, change the GPU selection, use `swapoff` under pressure, or retry the known failure workload merely for benchmarking. If available host memory approaches the project's 6 GiB stop gate, stop safely before continuing. Temperature and frequency must be monitored with the sensor type identified; the campaign's 94–96 °C measurements do not establish a universal safe temperature limit.

The existing local automation uses `h3-cleanup.py` to restart the idle container and `h3-720p-test.py <workflow.json> <name>` to submit jobs. Before executing either, inspect the script and confirm there are no active jobs, verify the correct container name and confirm the monitoring gates. These scripts are not embedded in this README or asserted to be available in the public repository. Avoid short liveness checks (3 s previously gave false failure indications); the documented production workflow uses 30 s.

### Verify an output

```bash
ffprobe -v error -count_frames \
  -show_entries stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames,start_time,duration,channels,sample_rate \
  -show_entries format=duration -of json output.mp4
```

For frame alignment, additionally export video frame timestamps (`ffprobe -select_streams v:0 -show_frames ...`) and inspect audio start/end time, not just nominal FPS and frame counts. A successful MP4 container is not a proof of audiovisual synchronization or visual quality equivalence.

## 5. Known failures and workarounds

| Observation | Evidence / interpretation | Current response |
|---|---|---|
| GPU allocation failure at 243 frames | ~31.3 GiB sampled VRAM; PyTorch OOM | Do not intentionally reproduce |
| GNOME/GDM disruption | AMDGPU framebuffer pin ENOMEM, repeated sessions, reboot | Keep desktop; fixed application-level reserve3; tested at 124 frames |
| Residual anonymous memory | `/free` unchanged over 90 s; restart 28.5→0.62 GiB | Restart only after active jobs end; verify memory and API |
| Dynamic LoRA injection | gfxhub page fault + exit 139 | `low_vram=True` merge mode |
| AAC/PyAV failure | `avcodec_send_frame() returned 22` | `codec=auto` in tested workflow |
| RIFE checkpoint mismatch | missing `encode.cnn3.weight` | v4.26 + IFNet key remapping + corrected forward signature |
| Incorrect timing parser | whole prompt dict treated as node ID | No phase timing; repair and validate collector before future authorized P0 |
| Concurrent test/production instances | host available RAM fell to 736 MiB; safety abort | One active inference instance; no new performance record |

## 6. Post-processing (separate from native diffusion)

All branches use `turbo-8s-easycache_00001_.mp4` as the source. A2 contains A1; B2 contains B1. The reported times must not be represented as a measured four-step-plus-postprocessing pipeline.

| Branch | Raster / FPS / frames | Time (s) |
|---|---|---:|
| A1 — FFmpeg interpolation + unsharp/CAS | 1920×1088 / 48 / 381 | 62 |
| S1 — Real-ESRGAN GPU | 2560×1451 / 24 / 192 | 653 |
| A2 — A1 + ESRGAN | 2560×1451 / 48 / 381 | 1302 |
| B1 — RIFE v4.26 | 1920×1088 / 48 / 383 | 138 |
| B2 — B1 + ESRGAN | 2560×1451 / 48 / 379 | 1359 |

A1 is the documented classical-filter path:

```bash
ffmpeg -i source.mp4 -vf \
  'unsharp=5:5:0.25:5:5:0.05,minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,cas=strength=0.25' \
  -c:v libx264 -crf 17 -preset medium -c:a copy A1-ffmpeg-48fps.mp4
```

RIFE and ESRGAN use the author's custom `rife_batch.py` and `sr_batch.py` in the existing container; release the actual inspected scripts and dependency versions before claiming turnkey reproduction. The RIFE replacement had 198 keys, `head_ch=4`, `channels=[192,128,96,64,32]`; key mapping strips `module.`/`flownet.` and maps `blockN.` to `blocks.N.`. The call is `forward(img0, img1, timestep=0.5)`.

Known limitation: A1/B1/B2 differ in frame count. The four-frame B1→B2 change is unresolved; inspect frame timestamps and audio duration. The reported odd output height (1451) also requires pixel-format/encoding verification. Operator inspection of selected frames is qualitative only. ESRGAN GPU (~3.4 s/frame) vs CPU ONNX Runtime (~169 s/frame) is a cross-implementation comparison, not an isolated hardware speedup.

## 7. P0 diagnostic status

**P0 incomplete; P1 not started.** The current freeze records reserve3 and the image/workflow configuration, but H3 and Qwen hash lookups failed. `rocprofv3` was unavailable. FP8 storage and source quantization code were identified; actual native FP8 arithmetic, SDPA dispatch and hotspot rankings remain unknown. TunableOp was not established as active. The experimental run ended without a usable benchmark: host MemAvailable 14 GiB → 736 MiB, swap 5.7→15 GiB, sampled GPU VRAM 30.5 GiB; after stopping/removing the test instance, host available memory returned to 19 GiB and the production API to HTTP 200. The timing log records the whole workflow dictionary instead of separate nodes. No new speedup is claimed.

## 8. Evidence and release layout

The following is a **proposed publication layout**, not a claim that all files are already public. Preserve the original evidence and build a separate redacted public copy.

```text
README.md
paper.md
paper.pdf
benchmarks.csv
720p-data/              # existing workflow JSON, TSVs, SHA256SUMS, sample MP4s
figures/                # historical native/delivered sample frames and clips
postprocessing/         # selected source and A1/S1/A2/B1/B2 samples
p0/                     # redacted freeze, audit, invalid timing log, abort report
workflows/              # exact reviewed 1080p API JSONs, if recovered
scripts/                # reviewed automation and post-processing scripts, if recovered
SHA256SUMS.txt           # generated for the released files, after redaction
```

The source archive includes the first four evidence categories and P0 reports, but the full 1080p workflow/script bundle and all main checkpoint hashes still need verification. Do not publish personal account names, authentication prompts, private service identifiers, private absolute paths, unrelated control-plane source, or local credentials. Keep internal storage locations out of the public README. Do not claim that the new manuscript, videos or raw logs are already public until the repository release has been verified.

## 9. Related work, scope and transparency

Relevant upstream/community projects include [Comfy-Org H3](https://huggingface.co/Comfy-Org/MiniMax-H3), [boxwrench/R9700](https://github.com/boxwrench/R9700) and [SyncTile H3](https://github.com/alaintural/synctile-h3). Their tasks and pipelines may differ. This repository is a single-system engineering record, not a comprehensive literature survey. No global-first or fastest-GPU claim is made.

Generative AI tools assisted with experimental planning, scripting and agent-driven execution, log interpretation and writing. Measurements come from the reported physical workstation; the human author must verify the evidence and approve final publication. No model weights are included in this proposed release. Apply an appropriate license to original code/documentation only after checking ownership and upstream terms; third-party artifacts retain their respective licenses.
