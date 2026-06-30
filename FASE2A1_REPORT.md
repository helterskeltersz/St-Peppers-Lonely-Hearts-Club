# FASE 2a-1 — Laporan: Recipe + Config Generator + Classifier (Full-CPU)

Branch: `fase2a-recipe` · Acuan: `MATRIKS_RECIPE_FINAL_v2.md` · **Tanpa training** (CPU; training butuh GPU = Fase 2a-2).
**Verdict:** ✅ recipe keisi & complete, config TOML/YAML ke-generate, classifier 4 sinyal jalan (3-way split benar), 3 watch-item ke-handle. Semua divalidasi di CPU.

---

## 1. Recipe per-arch × bucket (status tiap angka)

Komposisi: §1 baseline (✅ VERIFIED dari template) → §3a unconditional override (📐) → §3b/c bucket delta (📐).

| arch | bucket | rank | alpha | LR | optimizer | snr_γ | caption_dropout | conv | catatan kunci |
|---|---|---|---|---|---|---|---|---|---|
| SDXL | aggressive | 32 | 16 | **2e-5** | AdamW8Bit | **1** | 0.30 | off | 📐 snr↓1, LR↑ dari 1e-5 |
| SDXL | prior | 32 | 16 | 1e-5 | AdamW8Bit | 5 | **0.40** | **16** | 📐 conv ON (LoCon), dropout↑ |
| SDXL | logo | 32 | 16 | 1e-5 | AdamW8Bit | 5 | **0.10** | **16** | 📐 exact caption, trigger kept |
| Flux | aggressive | 128 | 128 | 5e-5 | Adafactor | — | 0.30 | — | flow_shift 3.1582, sigmoid |
| Flux | prior | 128 | 128 | 5e-5 | Adafactor | — | 0.40 | — | TE-cache OFF (watch-item) |
| Flux | logo | 128 | 128 | 5e-5 | Adafactor | — | 0.10 | — | exact caption |
| Qwen | aggressive | 32 | 32 | 1e-4 | adamw8bit | — | 0.30 | — | weighted; **flow_shift None** |
| Qwen | prior | 32 | 32 | 1e-4 | adamw8bit | — | 0.40 | — | toolkit derive shift internal |
| Qwen | logo | 32 | 32 | 1e-4 | adamw8bit | — | 0.10 | — | andelin native text |
| Z-Image | aggressive | 32 | 32 | 1e-4 | adamw8bit | — | 0.30 | **16** | weighted, adapter v2 |
| Z-Image | prior | 32 | 32 | 1e-4 | adamw8bit | — | 0.40 | **16** | conv baseline 16 |
| Z-Image | logo | 32 | 32 | 1e-4 | adamw8bit | — | 0.10 | **16** | exact caption, res 1024 |

**Status angka (di komentar `runtime/recipe.py`):**
- ✅ VERIFIED (baseline §1, path:baris di komentar): rank/alpha/LR/optimizer/scheduler/min_snr_gamma/flow_shift/timestep_type/noise_scheduler/repeats/steps/arch/quant tiap arch.
- 📐 INFERRED (keputusan kita, sweepable): caption_dropout_rate (default 0.30, range 0.30–0.50), drop_trigger_word, ema=off, high_noise_bias, bucket delta (snr↓1 aggressive, conv ON prior/logo, LR bump aggressive SDXL, dropout 0.40 prior / 0.10 logo).

`is_complete()` → **True** (semua field kritis keisi). `caption_dropout` jadi parameter (`resolve_recipe(..., caption_dropout=x)`) biar gampang di-sweep 2a-2.

---

## 2. Contoh config ke-generate

### SDXL aggressive → `t1.toml` (sd-scripts)
```toml
network_module = "networks.lora"
network_dim = 32
network_alpha = 16
learning_rate = 2e-05
unet_lr = 2e-05
text_encoder_lr = 1e-05
optimizer_type = "AdamW8Bit"
lr_scheduler = "constant"
resolution = "1024,1024"
train_batch_size = 4
max_train_steps = 1600
caption_dropout_rate = 0.3
min_snr_gamma = 1
loss_type = "l2"
# (+ pretrained_model_name_or_path, train_data_dir, output_dir, cache_latents, ...)
```

### Qwen prior → `t1.yaml` (ai-toolkit, dipangkas)
```yaml
config:
  process:
  - type: diffusion_trainer
    network: {type: lora, linear: 32, linear_alpha: 32}
    datasets:
    - {folder_path: ..., caption_ext: txt, resolution: [512,768,1024], caption_dropout_rate: 0.4}
    train:
      steps: 3000
      lr: 0.0001
      optimizer: adamw8bit
      noise_scheduler: flowmatch
      timestep_type: weighted          # NOT sigmoid
      ema_config: {use_ema: false}     # §3a EMA off
      # NO flow_shift key (ai-toolkit derives it)
    model: {arch: qwen_image, quantize: true, qtype: uint3, ...}
```
(Z-Image YAML identik + `network.conv/conv_alpha: 16` dan `model.assistant_lora_path` ke adapter v2.)

---

## 3. Classifier — 4 sinyal jalan, 3-way split benar

Divalidasi di 3 dataset sintetis (12 img each), CPU:

| dataset | bucket | conf | image_count | trigger | subject_variance (src) | ocr_area_ratio |
|---|---|---|---|---|---|---|
| logo (teks "ACME") | **logo-specialist** | 0.95 | 12 | None | 0.0 (proxy) | **0.175** |
| subject (mirip, trigger) | **aggressive-fit** | 0.807 | 12 | tomx | 0.0 (proxy) | 0.0 |
| style (warna beragam) | **prior-preserving** | 0.482 | 12 | None | **0.081** (proxy) | 0.0 |

**4 sinyal:** `image_count` ✅, `caption_entropy` ✅, `trigger_present` ✅, `subject_variance` ✅ (CLIP kalau `open_clip`+weights ada, else **CPU proxy** PIL/numpy — di test pakai proxy), `ocr_area_ratio` ✅ (pytesseract). **Cap confidence 0.4 dilepas** — sekarang confidence dari margin sinyal.

**Bug nyata yang ke-fix saat validasi (bukan fixture):** `ocr_area_ratio` awalnya nge-sum box tesseract **lintas semua level** (page/block/line/word) → overcount parah, gambar abstrak ke-flag logo (0.46). Fix: hitung **word-level (level==5)** doang, conf≥60, teks alfanumerik. Hasil: gambar non-teks → 0.0.

**Threshold per-source (📐):** skala jarak proxy ≠ CLIP. `SUBJECT_VAR_LOW_PROXY=0.04`, `SUBJECT_VAR_LOW_CLIP=0.30`. Kalibrasi beneran butuh dataset real + CLIP (2a-2).

---

## 4. Tiga watch-item — ke-handle ✅

| Watch-item | Status | Bukti |
|---|---|---|
| (1) `--hours-to-complete` = sumber kebenaran step solver; JANGAN +0.5h lagi | ✅ | Entrypoint pakai `args.hours_to_complete` buat `build_plan`. `predict_window_hours` cuma di-log sbg **sanity** ("NOT used for steps"). Qwen run hours=1.25 → dipakai apa adanya, nggak ditambah lagi. |
| (2) Flux: matiin TE-cache (konflik caption_dropout) | ✅ | `config_gen` Flux emit `cache_text_encoder_outputs = false` + `network_train_unet_only = false`. Test: substring present = True. |
| (3) Qwen: JANGAN set flow_shift | ✅ | Baseline Qwen `discrete_flow_shift=None`; `config_gen` YAML nggak punya key `flow_shift`. Test: `"flow_shift" not in yaml` = True. |

---

## 5. measure_throughput — wiring (stub, nggak ngarang)

`step_solver.measure_throughput(step_iter, n_steps=50)` udah di-wire struktur: konsumsi 50 step dari step-source trainer → steps/sec. TAPI **raise `RuntimeError` kalau `torch.cuda.is_available()` False** atau `step_iter` None — nolak ngarang it/s di CPU. Pengukuran live = Fase 2a-2 (GPU).

Entrypoint: setelah config-gen, cek `cuda_available()`. CPU → log "training NOT launched (needs GPU)" + exit 0. GPU → `measure_throughput` → `target_steps` → `run_training`.

---

## 6. Build deps (CLIP/OCR) di Dockerfile

Dua Dockerfile ditambah: apt `tesseract-ocr`, pip `pytesseract` + `open_clip_torch`. **Build A di-rebuild & SUKSES** (9.74GB, +90MB) — `pytesseract`, `open_clip`, `tesseract 4.1.1` ke-import di dalam image, build nggak pecah. Image B pakai dep line identik (belum di-rebuild buat hemat resource; risiko minim). CLIP weights BELUM di-bake → classifier fallback ke proxy (terutama image toolkit yang offline). Bake CLIP weights = item 2a-2 kalau mau CLIP aktif.

---

## 7. Validasi CPU — ringkas

| Cek | Hasil |
|---|---|
| compileall semua modul | ✅ OK |
| recipe 12 kombinasi complete | ✅ semua `is_complete=True` |
| watch-item Flux/Qwen | ✅ flow_shift 3.1582 / None; TE-cache false |
| config TOML+YAML ke-generate, field benar, ext benar | ✅ |
| classifier 3-way split | ✅ ALL CORRECT |
| caption trigger-drop | ✅ "portrait of tomx person" → "portrait of person" (12 file) |
| entrypoint 4 arah → guard CUDA → exit 0 | ✅ sdxl/flux→.toml, qwen/z→.yaml |
| pytest resolver dockerfile | ✅ 8/8 |

---

## 8. Daftar nyangkut buat Fase 2a-2 (GPU)

1. **Wire `measure_throughput` ke step-callback trainer real** (sd-scripts + ai-toolkit) → isi `target_steps`, override `max_train_steps`/`steps` di config.
2. **Bake CLIP weights** (ViT-B-32) ke kedua image biar `subject_variance` pakai CLIP (bukan proxy), khususnya image toolkit yang `HF_HUB_OFFLINE=1`.
3. **Kalibrasi threshold classifier** (OCR_LOGO, SUBJECT_VAR_LOW proxy/clip) di dataset turnamen real. Catatan: OCR masih bisa false-positive di tekstur — pertimbang filter tambahan (panjang token, jumlah kata).
4. **A/B caption_dropout** (0.30→0.50) di re-eval harness — lever #1 ke term 0.75.
5. **Validasi DoRA/LoKr load** di inference (LyCORIS 3.x `wd_on_output` breaking, §5b) sebelum andelin conv/LoCon.
6. **Checkpoint cadence + upload** ke `/app/checkpoints/{task_id}/{repo}` (sisain waktu, solver reserve 10%).
7. **Step solver bucket logic** (aggressive cap di knee, prior pakai full window) — sekarang formula dasar, perlu tuning per-arch throughput.
8. **Training run beneran** di VM H100 (Fase 2a-2) — buktiin config valid ke trainer + checkpoint kehasil.
