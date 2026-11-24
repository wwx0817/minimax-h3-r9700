# When the Desktop Eats the Framebuffer

## Memory contention, reclamation and inference feasibility for MiniMax H3 on a single AMD GPU workstation

**Technical manuscript — revision 2026-11-25**  
**Anonymous author(s)** — affiliation withheld; author approval required before submission.

### Abstract

Local audio–video diffusion inference can approach the memory limits of a workstation even when model weights are compressed. We report a single-system engineering case study of MiniMax H3 on an AMD Radeon AI PRO R9700 (32 GB, 31.86 GiB reported) with 48 GB host RAM, using ComfyUI, PyTorch/ROCm and Docker. With pruned FP8-scaled diffusion weights, 20-step generation at a native 1920×1088 canvas succeeded at 124, 175 and 192 frames (24 fps), with the 192-frame run taking 5,628 s and reaching 28.97 GiB sampled GPU memory usage. A 243-frame attempt failed under the historical configuration, accompanied by a GPU allocation error, AMDGPU framebuffer-pinning failures, repeated graphical-session reconstruction and a subsequent reboot. The evidence supports a desktop–inference memory-contention explanation, but does not establish the complete reboot mechanism. A separate idle-process reclamation observation found no reduction in container memory over 90 s after a `/free` request; restarting the container reduced its reported memory from 28.5 to 0.62 GiB. A 3 GiB application-level VRAM reservation was subsequently validated in one 124-frame, 20-step generation with the graphical session active. Historical Turbo LoRA runs at eight and four steps completed in 2,358 and 1,281 s, respectively (1.84×), with nearly identical sampled VRAM peaks. These times do not establish performance under the later reservation configuration. Runtime LoRA injection was associated with a GPU page fault and process segmentation fault; the merge path completed successfully. We additionally report five post-processing chains and an aborted performance-diagnostic attempt: concurrent container residency coincided with host memory pressure, invalid phase timing and a safety abort. The findings emphasize separating GPU, cgroup and host memory, maintaining a single inference instance, distinguishing sampling efficiency from memory capacity, and treating failed measurements as evidence limitations rather than new performance records.

**Keywords:** video diffusion; RDNA4; GPU memory management; ROCm; containerized inference; desktop contention; empirical systems evaluation.

## 1. Introduction

Video diffusion transformers jointly process many spatial and temporal latent elements. The resulting workload can exceed the memory budget of a single GPU workstation even when model weight storage is reduced. In desktop deployment, inference must also coexist with graphical applications and display buffers. A model-level out-of-memory error, a host-memory pressure event and a failure in the display subsystem are different phenomena, and an end-to-end time alone cannot distinguish them.

This study examines the MiniMax H3 FL2VA ComfyUI workflow on one commercially available AMD workstation. The objective is not to propose a new diffusion architecture or claim a world record. It is to characterize observed workload feasibility, failure signatures and operational mitigations under a documented resource configuration. The contributions are:

1. A seven-configuration historical feasibility matrix and an additional three-configuration Turbo/cache matrix, with recorded wall time and available memory measurements.
2. Evidence of GPU memory contention involving the graphical session, together with an application-level VRAM-budget mitigation tested on a smaller successful workload.
3. A bounded negative observation on `/free` reclamation and a separate GPU virtual-address failure associated with dynamic LoRA injection.
4. A post-processing comparison distinguishing native generation from interpolation and super-resolution, and a documented failed profiling attempt that defines the remaining measurement gaps.

The study covers the project period from 2 November to 25 November 2026. It is a single-machine, limited-repetition study; reported successful configurations are observed outcomes, not hardware limits or general guarantees.

## 2. Experimental environment and task definition

### 2.1 Hardware and software

| Component | Recorded configuration |
|---|---|
| GPU | AMD Radeon AI PRO R9700, RDNA4 / gfx1201, 32 GB; 31.86 GiB reported |
| CPU / RAM | AMD Ryzen 9 9950X / 48 GB installed; approximately 45 GiB reported usable by OS |
| Host swap | Approximately 23 GiB reported (two SSD-backed swap files) |
| Host OS | Ubuntu 26.04.1 LTS; kernel 7.0.0-44 in campaign records |
| Docker | 29.x; GPU device access via `/dev/kfd` and `/dev/dri` |
| Historical image | `ai-stack/comfyui-h3:rocm10.1-v1` |
| Current image | `ai-stack/comfyui-h3:with-extras` |
| PyTorch / HIP | 2.14.0+rocm10.1.0 / 7.16.26385 (reported runtime identifiers) |
| Current Python / Triton | 3.12.3 / 3.8.0, as recorded by P0 |
| ComfyUI | Historical report: v0.33.2, commit `7cee3ceb1a35503172e0dfb8dbdbdedee2aba8aa`; P0 checkout reported `nogit` |
| Resource limits | 34 GiB container RAM; 42 GiB combined memory+swap limit; 8 GiB shared memory; 300 W GPU power cap |
| Launch options | `--disable-pinned-memory --cache-none`, no `--fast-disk`; later production adds `--reserve-vram 3` |

The historical image and current derived image must not be conflated. The production freeze recorded image ID `sha256:5bb466f58157f527e66262ddb74e821e4f2e97af0c714c1f1805c24b04dfc131`. Current ComfyUI revision provenance was not independently established by the freeze, because Git metadata was absent. The software versions above are reported by the archived environment, not a claim about current upstream releases.

### 2.2 Model components and numerical precision

| Component | Checkpoint | Reported file size |
|---|---|---:|
| H3 diffusion transformer | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 19.52 GiB |
| Qwen3-VL text encoder | `qwen3vl_32b_minimax_h3_fp8.safetensors` | 24.71 GiB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 577 MiB |
| Turbo LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` | 779,849,816 bytes |

H3 is described in the project documentation as a 33B-class audio–video model; the deployed pruned checkpoint is not the unpruned BF16 model. FP8 refers to the compressed weight format of the deployed main checkpoint and text encoder, not necessarily the precision of every arithmetic operation. The VAE components use different stored precisions. P0 did not establish the actual matrix multiplication precision or the proportion of native FP8 instructions. We therefore refer to **FP8-weight, mixed-precision inference**, not full-precision inference or verified native FP8 GEMM.

### 2.3 Workload and generation pipeline

The examined nodes use frame counts on the grid `17k + 5` and spatial dimensions divisible by 32. Our chosen near-720p and near-1080p native canvases were 1280×704 and 1920×1088. These are the specific canvases tested, not the only possible aligned dimensions. Standard-resolution delivery files were Lanczos-resized to 1280×720 or 1920×1080 after generation. This final resize is separate from diffusion sampling and does not constitute generation at the exact delivered pixel dimensions.

The standard workflow used a `res_multistep` sampler, `simple` schedule and 20 steps. H3's audio and video latents were decoded using `VAEDecodeTiled` and `VAEDecodeAudio`, respectively. The documented video tile settings are 512 spatial and 64 temporal; the captured four-step workflow additionally specifies spatial overlap 64 and temporal overlap 8. Output is MP4 at 24 fps with H.264 video and AAC stereo audio in the reported successful runs.

The four-step diagnostic workflow captured in the P0 log specifies `MiniMaxH3TurboSampler`, `MiniMaxH3TurboLoRA` at strength 1.0 with `low_vram=True`, `BasicScheduler` with `simple` and denoise 1.0, `BasicGuider`, seed 8112026, 192 frames, and `SaveVideo(codec=auto)`. The prompt describes a small brass robot watering luminous mushrooms in a rainy nighttime garden, with rain, servo sounds and a bell. These exact P0 settings are evidence for the attempted current configuration; historical runs must be matched to their own archived workflows before strict paired comparisons.

### 2.4 Measurement protocol and evidence classification

Three memory domains were monitored separately: GPU-global VRAM (`rocm-smi`), container cgroup memory and swap (`memory.current`, `memory.swap.current`, `memory.stat`, `memory.events`), and host memory and swap (`/proc/meminfo`). Campaign records describe GPU sampling every 2–5 s and host/cgroup sampling at approximately 1 s where available. Samples are not necessarily synchronized and observed peaks may miss transient allocations. Reported wall time is submission-to-saved-video time for completed generation runs; failed/aborted runs do not have a completion time. GPU temperature and power were recorded, but sensor type and clock traces are incomplete. `ffprobe` and operator frame inspection were used for output checks; neither is a blinded quality evaluation.

We distinguish completed generation, workload failure, harness timeout, safety abort and not-tested configurations. Each historical table entry is a reported observation rather than a repeated-run median. Current source review used the campaign reports, production manual, P0 evidence and archive inventory; it did not independently re-run all workloads, re-hash every artifact or decode every video.

## 3. Historical feasibility and scaling

| Native canvas | Frames | Seconds at 24 fps | Steps | Wall time (s) | Sampled VRAM / other evidence | Outcome |
|---|---:|---:|---:|---:|---|---|
| 1280×704 | 124 | 5.167 | 20 | 664.6 | cgroup swap 5.85 GiB | Completed |
| 1280×704 | 243 | 10.125 | 20 | 1962.1 | cgroup swap 5.43 GiB | Completed |
| 1920×1088 | 39 | 1.625 | 20 | 455.5 | Not reported | Completed |
| 1920×1088 | 124 | 5.167 | 20 | 2600.5 | approx. 27.8 GiB | Completed |
| 1920×1088 | 175 | 7.292 | 20 | 4759 | 27.96 GiB | Completed |
| 1920×1088 | 192 | 8.000 | 20 | 5628 | 28.97 GiB | Completed |
| 1920×1088 | 243 | 10.125 | 20 | Not completed | 31.3 GiB | GPU allocation failure; desktop disruption |

*Table 1. Historical feasibility matrix. Desktop handling changed during the campaign; later successful 1080p runs were reported under the earlier display-manager mitigation. These are not a homogeneous reserve3 benchmark. The 243-frame interrupted run has no valid completion time.*

The 720p archive contains native and resized videos, API workflows, telemetry TSVs and SHA-256 manifest. Its README reports minimum host available RAM of 11.24 GiB (124 frames) and 10.94 GiB (243 frames), minimum host free swap of 16.31 and 14.97 GiB, respectively, and `oom_kill=0` in both runs. This indicates that those particular 720p runs completed without recorded cgroup OOM kills; it does not imply that all higher-resolution workloads fit host memory.

At 124 frames, increasing the native canvas from 1280×704 to 1920×1088 increases pixel area by approximately 2.32× and reported wall time by 3.91×. At 1920×1088, 124→192 frames increases the frame count by 1.55× and wall time by 2.16×. At 1280×704, 124→243 frames increases wall time by 2.95×. These comparisons are descriptive: attention computation, tensor dimensions, weight movement and decode cost may contribute, and the separate effects have not been measured. We do not infer an attention-complexity exponent from these few points.

The 192-frame workload is the largest successful native 1080p-tier workload in this historical matrix. No outcome is established for the intermediate legal counts 209 and 226, or for 243 frames under the current production configuration. Accordingly, 192 frames is an observed successful point, not an established absolute capacity ceiling.

### 3.1 Timeout and monitoring artifacts

The initial 192-frame 20-step attempt was interrupted at a reported 5,341 s by the harness's nominal 90-minute limit. A subsequent attempt allowed 120 minutes and completed in 5,628 s. The first attempt is therefore classified as a harness timeout; the later success demonstrates workload feasibility, but does not prove the first run would otherwise have finished. A separate 3 s `/system_stats` liveness timeout produced false failure indications during model loading. The harness subsequently used a 30 s tolerance. Timeout should not be treated as proof of GPU failure; evidence from independent monitoring is required.

## 4. Desktop–inference memory contention

### 4.1 Evidence from the failed 243-frame run

The campaign recorded GPU memory usage reaching approximately 31.3 GiB of 31.86 GiB. The inference error reported by PyTorch on the ROCm backend was:

```
CUDA out of memory. Tried to allocate 1.48 GiB.
GPU 0 has a total capacity of 31.86 GiB of which 1.63 GiB is free.
Of the allocated memory 29.54 GiB is allocated by PyTorch.
```

The CUDA-style diagnostic string is preserved as logged; the hardware and runtime are AMD/ROCm. The difference between reported free memory and the requested allocation is notable, but fragmentation, allocator reservations, concurrent changes and driver constraints are not distinguished by these figures alone.

The previous-boot kernel log excerpt records:

```
amdgpu: Failed to pin framebuffer with error -12
amdgpu: Not enough memory for command submission!
amdgpu: VM memory stats for proc gnome-shell(...) is non-zero when fini
```

The archive identifies framebuffer failures at approximately 01:12:50 local time, followed by recurring graphical-session stop/reconstruction events approximately every 4–6 seconds. The old boot ended at 01:13:17 and the next began at 01:14:10. This temporal association supports the interpretation that inference-related GPU pressure and display-buffer allocation failures contributed to desktop unresponsiveness. It does not establish whether a user action, power-state transition or another mechanism initiated the reboot. The archived search found no corresponding kernel OOM-killer, GPU reset/ring-timeout or watchdog entries. Absence of these log entries is not proof of their impossibility. Campaign temperatures reached approximately 94–96 °C, but the available evidence does not establish sensor identity or exclude frequency throttling.

### 4.2 Mitigation with the desktop retained

Earlier experiments stopped the display manager to reduce contention, but this terminated the operator's graphical session and was rejected as a production practice. Attempts to redirect rendering using environment variables did not demonstrate reliable migration of the user compositor away from the discrete GPU. The adopted configuration instead applies ComfyUI's `--reserve-vram 3` while keeping GNOME active. This is an application-level budget, not a physical memory partition or a guaranteed reservation enforced for every client by the driver.

The reserve3 validation generated 124 frames at 1920×1088, 24 fps and 20 steps in 2,613 s. The project report records an unchanged boot time, an active Wayland session and valid output. The historical 124-frame run took 2,600.5 s: their raw time difference is 12.5 s (0.48% relative to the historical value). The project summary separately describes an approximately 11% penalty, but does not supply a compatible denominator for that figure. Neither value establishes the causal overhead of reservation without matched and repeated runs. The 124-frame observation does not establish stable 192-frame generation or universal prevention of reboot under reserve3. Current production policy retains the graphical desktop and the 3 GiB budget; headless or integrated-GPU experiments are not authorized.

## 5. Memory reclamation and lifecycle

A stalled subsequent job was associated with substantial residual anonymous process memory. The archived measurements report cgroup `anon=24.37 GiB`, `file=6.03 GiB` and `memory.current=28.5 GiB`. Because these values are not documented as one synchronized snapshot, they must not be summed as an exact accounting identity.

The operator sent `POST /free` with `{"unload_models":true,"free_memory":true}`. During a 90 s observation interval the reported container memory remained about 28.5 GiB and GPU usage about 2.1 GiB. Restarting the container reduced these to 0.62 and 0.96 GiB, respectively, and the API became healthy after about 10 s. This supports restarting the process as an effective recovery action in the tested state. It does not prove a permanent memory leak, universal failure of `/free`, or that no other reclamation mechanism exists. The request-processing state, allocator caches, retained references and page residency remain potential contributors.

The current operational rule is to finish or safely stop an active job, restart the inference container between long runs when residual memory requires it, and verify host memory, GPU occupancy and API health before resubmission. A previous attempt to run `swapoff` under pressure was followed by a container restart; it is not part of the approved recovery procedure.

## 6. Generation-side acceleration and fault observations

| Historical native 1920×1088, 192 frames | Steps | Wall time (s) | Sampled peak VRAM (GiB) | Speed relative to Turbo 8 |
|---|---:|---:|---:|---:|
| Standard sampler | 20 | 5628 | 28.97 | 0.42× |
| Turbo LoRA | 8 | 2358 | 28.97 | 1.00× |
| Turbo + EasyCache (reuse threshold 0.2) | 8 | 2364 | 28.91 | 1.00× (rounded) |
| Turbo LoRA | 4 | 1281 | 28.91 | 1.84× |

*Table 2. Historical generation observations, not measurements of the current reserve3 production baseline.*

The standard-to-eight-step comparison gives an observed 2.39× speedup and the standard-to-four-step comparison gives 4.39×, but changes in LoRA and sampler also distinguish the standard workflow from Turbo. The eight-to-four-step comparison gives 1.84× (45.7% lower elapsed time). Sampled GPU memory differed by only 0.06 GiB. This supports a hypothesis that reducing steps may decrease elapsed computation without appreciably changing peak memory for these configurations; it does not demonstrate statistical equivalence, universal step-count independence or a bound on memory demand.

EasyCache at threshold 0.2 changed the recorded time from 2,358 to 2,364 s (0.25% longer). No meaningful benefit was observed in this single comparison; actual skip counts were not independently recorded in the reviewed evidence. The result does not disprove other cache thresholds or calibrated algorithms. Operator visual checks found no obvious structural collapse in the four-step output; no formal quality-equivalence claim is justified.

### 6.1 Dynamic LoRA injection

The historical runtime injection configuration (`low_vram=False`) was associated with an AMDGPU `[gfxhub]` page-fault report containing `MAPPING_ERROR RW:0x0` and container exit 139 (SIGSEGV). The corresponding merge configuration (`low_vram=True`) completed. This is a distinct observed failure signature from the desktop allocation error. The merge mode is the current workaround. The evidence does not locate the erroneous address operation, attribute fault exclusively to the driver or prove that runtime injection fails on all ROCm systems.

### 6.2 Output encoding

A Turbo run using explicit `codec=h264` produced a PyAV AAC encoding/mux error (`avcodec_send_frame() returned 22`); the upstream-style `codec=auto` setting produced a valid reported H.264/AAC file. The latter is retained in the production workflow. This is an output-stage compatibility observation, not a diffusion-model failure.

## 7. Post-processing and output integrity

The five post-processing outputs were derived from the same historical eight-step EasyCache source (G1), not from the four-step video. A2 and B2 are two-stage branches that include A1 and B1, respectively; the branches are separate from each other. Their reported times are post-processing times only and must not be added to the four-step historical time and presented as a measured complete pipeline. Whether all combined-chain timers include the intermediate processing is not established by the reviewed report; retain the reported values without recomputing totals.

| Output | Processing chain | Delivered raster | FPS | Video frames | Reported wall time (s) |
|---|---|---|---:|---:|---:|
| A1 | Unsharp + FFmpeg `minterpolate` + CAS | 1920×1088 | 48 | 381 | 62 |
| S1 | Real-ESRGAN x4, output resized to width 2560 | 2560×1451 | 24 | 192 | 653 |
| A2 | A1 + Real-ESRGAN | 2560×1451 | 48 | 381 | 1302 |
| B1 | RIFE v4.26, 2× interpolation | 1920×1088 | 48 | 383 | 138 |
| B2 | B1 + Real-ESRGAN | 2560×1451 | 48 | 379 | 1359 |

*Table 3. Post-processing observations. Raster, FPS and frame count are taken from the archived reports. The resulting video streams are not necessarily duration-identical.*

The 381/383/379 frame counts are significant. Interpolating between 192 source frames yields 383 frames under a convention that retains each original frame and inserts one between adjacent frames, but this alone does not establish the implementation used or explain A1/B2. At nominal 48 fps, the three counts correspond to 7.9375, 7.9792 and 7.8958 s by count divided by rate, not necessarily container duration or the last frame timestamp. The B1-to-B2 four-frame difference has not been explained by the archived reports; PTS, frame duplication/drop, encoder behavior, odd-height pixel format and audio endpoint alignment require separate verification. Do not silently change or round these counts to 384 or describe the outputs as exactly matched in duration. Reported AAC packets/frames are not equivalent to video frame counts.

Operator inspection found enhanced textures after ESRGAN and no obvious tearing/ghosting in selected interpolated frames. A1's 62 s versus B1's 138 s gives a 2.23× time ratio, but A1 also applies sharpening, and endpoint handling differs. These are pipeline observations, not a matched interpolation-only benchmark or proof of equal perceptual quality. The GPU ESRGAN path used PyTorch/ROCm and `spandrel` (~3.4 s/frame); a CPU ONNX Runtime attempt was reported at ~169 s/frame. Their approximately 50× ratio is cross-implementation, not a controlled GPU-versus-CPU hardware speedup.

### 7.1 RIFE compatibility

The checkpoint named `rife_v4.pkl` contained 160 keys under four block families and lacked the encode head expected by the built-in IFNet. Its size and key structure were consistent with the archived report's identification as an ECCV2022-RIFE v3.6 checkpoint, but size alone does not establish identity. Loading failed at `encode.cnn3.weight`. The replacement `rife426.pkl` (24,636,301 bytes, 198 keys) was obtained from the recorded RIFE v4.26 distribution, then remapped by stripping `module.`/`flownet.` prefixes and converting `blockN.` to `blocks.N.`. IFNet configuration detection returned `head=4`, `channels=[192,128,96,64,32]`, and loading succeeded. The script also changed its invocation to `forward(img0, img1, timestep=0.5)`. The replacement produced B1 and B2. This establishes an operational compatibility fix in the tested code, not a new interpolation algorithm.

## 8. Production freeze and aborted P0 diagnostic

### 8.1 Frozen baseline and source audit

The current production instance is `comfyui-h3-34g-reserve3`, with the desktop retained, 3 GiB reserve, 34 GiB memory cap, 8 GiB additional swap allowance and 300 W cap. The current configuration excludes changes to the ROCm/PyTorch versions, desktop GPU selection and the ComfyUI/Docker architecture. Qwen and H3 are intended to be staged rather than fully resident together. No separate inference client should run concurrently. P0 did not dynamically verify this residency policy at the node level.

The freeze captured the current image ID, startup command, model mounts, Python/PyTorch/HIP/Triton identifiers and workflow MD5s. The H3 and Qwen checkpoint hash lookups failed because the queried file paths were not found; the full model provenance is incomplete. The logged four-step workflow MD5 is `c14e3168f5a04479285008cfe37cc19e`; Turbo LoRA MD5 is `8bb5391bc99b98427e09d6d3f9868220`. Hashes are transcribed from the freeze, not independently recomputed here.

| Diagnostic | Result | Evidence limit |
|---|---|---|
| T0.0 baseline freeze | Configuration captured | Main checkpoint hashes missing; current ComfyUI commit not verified |
| T0.1 phase timing | Failed | `/queue` parser treated whole workflow as node ID |
| T0.2 residency and transfer | Partial | GPU/system samples, no valid phase-aligned H2D/D2H attribution |
| T0.3 FP8 path | Source-level audit | FP8 dtype and code path present; actual GEMM precision unknown |
| T0.4 compatibility | Source/API audit | SDPA interface present; actual dispatch unknown; TunableOp not established as active; MXFP8 path incomplete |

Neither `rocprofv3` nor `rocprof` was available in the inspected host/container. Source analysis identified ComfyUI FP8 quantization code invoking `comfy_kitchen`; this proves neither actual kernel dispatch nor FP8 arithmetic on the hardware. The timing log contains only a whole-workflow dictionary recorded as an alleged node start/end, not usable per-phase durations. The T0.2 JSON mentioned in the P0 report was not present in the inspected P0 folder inventory. Consequently, no validated phase share, GPU hotspot ranking or measured FP8 speedup can be reported.

### 8.2 Safety abort

| Indicator | Before | At reported abort | After cleanup |
|---|---:|---:|---:|
| Host MemAvailable | 14 GiB | 736 MiB | 19 GiB |
| Host RAM used | 31 GiB | 44 GiB | 26 GiB |
| Host swap used | 5.7 GiB | 15 GiB | approx. 7 GiB |
| GPU-global VRAM | 2.5 GiB | 30.5 GiB | 2.5 GiB |
| Production `/system_stats` | HTTP 200 | Timed out | HTTP 200 |

*Table 4. P0 operational observations as recorded by the diagnostic report.*

An experimental container was launched while the production instance remained running, contrary to the single-inference-instance requirement. The 4-step, 192-frame attempt had not completed at the reported 52-minute safety abort; the faulty polling trace ended at 3,252.8 s (about 54.2 minutes), not a completed generation time. Host available memory fell far below the predeclared 6 GiB gate, and production health checks timed out. The experimental instance was stopped and removed; production health and memory availability recovered, with the graphical session retained. The exit status 137 after stopping is not by itself evidence of an OOM kill. Concurrent residency and paging are plausible contributors to the pressure; per-process GPU occupancy and a controlled single-instance comparison were not obtained, so the precise causal contribution of each instance is unmeasured. The run is classified **ABORTED / INVALID FOR PERFORMANCE**, not a successful benchmark or evidence of reserve3 overhead.

The historical 1,281 s value was measured without `--reserve-vram 3`; the P0 freeze explicitly distinguishes it from current production. No valid four-step reserve3 completion time was obtained in the reviewed diagnostic campaign. P1 was not started, and no additional speedup is claimed. A future separately authorized P0 attempt would need a single inference instance, a corrected event/timing collector validated on a small test, preflight checks of model hashes and measured memory margins, independent safety monitoring and comparable cache state. It would report normal unprofiled wall time separately from profiling overhead. No new experiment is proposed as part of this manuscript revision.

### 8.3 Operational reproducibility notes

The cleanup campaign removed transient artifacts, one duplicate video and retired containers/image tags. Reported physical disk recovery was 4.26 GiB because image layers were shared; retained-file manifests were compared. This is ancillary maintenance, not a generation-side speedup. The desktop-client assessment inspected but did not execute or install a Linux AppImage. The separately managed environment would require revalidation, so the current container architecture was retained. Local control-plane configuration was updated to point to the current container name after stale state was found; that administrative change is not evidence of improved inference. Private account identifiers and unrelated project implementation details are excluded from public reporting.

## 9. Related work and interpretation

The Comfy-Org H3 repository provides the reference checkpoint and workflow context [1]. The boxwrench R9700 engineering notes describe text-encoder offload and alternative acceleration methods [2]. These are directly relevant precedent: our deployment is not the first R9700 H3 deployment and our use of pre-sampling offload is not a claim of algorithmic novelty. The SyncTile project explores latent-space refinement, a different route to high-resolution deliverables [3]. Such results are not directly comparable to native sampling at 1920×1088. Independent low-bit and sparse-attention efforts are relevant candidate work; quoted performance multipliers have been omitted because their workload, precision, and execution paths are not matched to the present experiment. The earlier community record [4] is a project disclosure, not independent validation.

The system-level interpretation is intentionally narrower than a general GPU-memory theory. Three observations motivate further investigation: near-capacity GPU inference can coincide with display allocation failures; ending a job does not guarantee immediate host-process memory reclamation; fewer sampling steps may reduce elapsed time with little change in sampled peak VRAM. The P0 abort illustrates a fourth issue: an experiment intended to characterize one inference instance can instead measure the resource interactions of multiple resident processes. Taken together, the data favor resource-aware deployment and rigorous measurement design rather than unsupported speed or priority claims.

## 10. Limitations and conclusion

This is an n=1 system study with limited repetitions, multiple campaign configurations, incomplete synchronization of telemetry, and largely operator-assessed quality. Peak GPU-global readings are not process-exclusive; they cannot directly establish allocator-level working sets. No controlled reserve0/reserve3 comparison at the 192-frame Turbo workload, complete arithmetic-path trace, matched post-processing duration analysis or statistically supported quality-equivalence study is available. The 209/226-frame gap remains untested. Current checkpoint hashes and the live source revision must be recovered before claiming full reproducibility. No claim is made about global priority, the fastest platform, universal driver failure, or physical capacity beyond the observed configurations.

The strongest established results are the historical 192-frame native-canvas completion (5,628 s at 20 steps, 1,281 s in the four-step Turbo configuration), a documented 243-frame failure signature, a bounded memory-reclamation observation and a successful 124-frame desktop-on reservation test. The later P0 attempt did not yield a comparable current benchmark or valid per-node timing. Keeping these evidence classes separate protects both reproducibility and operational safety. Future work should first repair measurement collection and recover missing provenance; further optimization requires explicit authorization and empirical quality, performance and stability checks.

### Data and artifact availability

The author-maintained archive contains `720p-data/`, `figures/`, `优化矩阵-视频/`, `全部调试资料/`, and `P0-诊断证据包/`. The 720p directory contains API JSON workflows, telemetry TSVs and a SHA-256 manifest; other folders include historical outputs, reports and selected logs. The public repository [5] and community discussion [4] are existing project entry points. Availability of the revised archive on those platforms, complete executable scripts, raw GPU traces and public redistribution permissions has not yet been established. Do not interpret this draft as a release of those artifacts. A stable public dataset identifier and validated file manifest may be added after author approval and redaction.

### AI assistance and author responsibility

Generative AI tools assisted in planning and refining experiments, scripting and operational execution through agents, interpreting logs and drafting/revising the manuscript. Measurements described here originate from the reported physical workstation rather than simulated benchmark values. AI is not an author. Before submission, the human author must verify the reported evidence and conclusions, confirm the actual tools and extent of their use, and approve the final manuscript and applicable venue-specific disclosure. This draft does not claim that independent human verification of every raw artifact has already occurred.

### References and provenance

[1] Comfy-Org. MiniMax H3 model and workflow repository. https://huggingface.co/Comfy-Org/MiniMax-H3

[2] boxwrench. R9700 engineering repository; MiniMax H3 optimization notes. https://github.com/boxwrench/R9700

[3] alaintural. SyncTile H3 repository. https://github.com/alaintural/synctile-h3

[4] Project community report, Comfy-Org/MiniMax-H3, discussion #76. https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/76

[5] Project technical repository. https://github.com/wwx0817/minimax-h3-r9700

Internal evidence reviewed: `paper.md` (latest Drive revision dated 2026-11-24 16:27 UTC); `H3-项目总汇总报告-20261125.md`; `1080p-7s8s-实测报告.md`; `MiniMax-H3-R9700-1080p-实测报告.md`; `死机根因分析.md`; `720p-data/README.md`; `稳定性方案最终结论.md`; `优化矩阵-完整报告.md`; `RIFE-B1B2-完整报告.md`; `H3-正常生产流程-生成插帧超分.md`; `H3-可行工作流程-运行手册.md`; P0 diagnostic report, freeze, timing log and FP8/compatibility audit. External project references are retained as related-work pointers; verify current bibliographic metadata and claims before journal submission.
