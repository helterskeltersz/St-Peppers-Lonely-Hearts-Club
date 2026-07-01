# MATRIKS_RECIPE_v3.md — Spec Final Engine Adaptif (Langkah 1)

**Status:** Slot `__` kerangka v3 TERISI dari kode zayden + `model_plans.csv`. Bukan ingatan.
**Tag:** ✅ VERIFIED = kebaca di kode (path:baris). 📐 INFERRED = turunan analisa CSV/logika.
**Ref kode:** repo winner `god-image-tourn-...position-1` (read-only). Tanggal 2026-07-01.

---

## FONDASI F1–F9 (VERIFIED dari kode, kecuali ditandai)

### F1 — Caption mode per kategori ✅ VERIFIED (`auto_caption.py:33-60`)
| Mode | Kategori | Perilaku |
|------|----------|----------|
| `inc` (rewrite penuh, LLaVA) | **style SAJA** (`:60`) | caption diganti, source dilipat masuk |
| `tail` (verbatim + append) | **person, product, logo, social, design** (else branch `:60`) | source di depan utuh + tail pendek |
| `verbatim` | override `ZAYDEN_CAPTION_TAIL=0` / no-budget | source utuh tanpa VLM |
Model rewrite: **LLaVA-1.5-7B** (`auto_caption.py:199`, prebake dockerfile). Budget: SDXL 75 / Flux 220 (`image_trainer.py:264`).

### F2 — Trigger keep + keep_tokens ✅ VERIFIED (`zayden_recipe.py:552-563`)
- Subject & style **dua-duanya** `keep_tokens=1` + `shuffle_caption=True`.
- Trigger di-inject ke DEPAN caption (`auto_caption.py:143 _apply_trigger`); keep_tokens=1 memaku comma-group pertama (trigger) supaya selamat dari shuffle.
- Trigger source: validator `--trigger-word`, else recovery `zayden_detect_trigger` (leading tag ≤4 kata, ≥60% caption; `category_detection.py:114,141`).

### F3 — caption_dropout per shape ✅ VERIFIED (`zayden_recipe.py:555,563`)
- **subject = 0.2** · **style = 0.1**
- `caption_tag_dropout_rate`: tidak di-set (default 0).

### F4 — Perceptual dedup pra-caption ✅ VERIFIED (`zayden_dedup.py:108`)
- Method: **dHash 64-bit + colour signature ganda** (dua-duanya harus cocok; `:142-144`).
- **hamming ≤ 6**, **colour_thresh ≤ 14.0**.
- Gate: **N < 15 → skip total** (`:126`). Keep-floor: **max(15, ceil(0.6·N))** (`:154`).
- Aksi: buang non-representatif (keep lexicographically-first per grup), hapus image + sidecar. Buang dari cluster terbesar dulu.
- Dipanggil SEBELUM caption (`image_trainer.py:238`).

### F5 — Multires noise per kategori ✅ VERIFIED (`zayden_recipe.py:233-252`)
- **iter = 6 · discount = 0.3 · noise_offset = 0.0** (offset dinolkan; mutually exclusive).
- Kategori: **person** (`:242`) + **logo, social, design** (`:239`).
- product/style/unknown → `{}` (inherit noise_offset dari recipe/bucket).

### F6 — Qwen rank adaptif ✅ VERIFIED (`zayden_recipe.py:396-401`)
```
rank  = clamp(round(118 + 0.85·N), 120, 152)
lr    = clamp(9.4e-5 · (134/rank)^0.5, 8.6e-5, 9.9e-5)
wd    = clamp(8.8e-6 · (134/rank),     7.5e-6, 1.05e-5)
alpha = rank    (linear_alpha = rank, scale 1.0)
```
EMA: `use_ema=True, ema_decay=0.995` (`:456`). do_cfg=True, cfg_scale=6.0 (`:457`). train_unet only (`:454`).

### F7 — Step budget sublinear per kategori ✅ VERIFIED (`zayden_recipe.py:356-374`)
Formula: `steps = base · (N/n_ref)^p`, clamp [min,max].
| arch | path | base | n_ref | min | max | p |
|------|------|------|-------|-----|-----|---|
| z-image | non-subject | 1100 | 30 | 800 | 1400 | 0.5 |
| qwen-image | non-subject | 1000 | 30 | 700 | 1400 | 0.5 |
| z-image | **subject** (person/product) | 230 | 14 | 120 | 680 | 0.85 |
| qwen-image | **subject** | 210 | 14 | 110 | 640 | 0.85 |
(SDXL/Flux TIDAK pakai step budget ini — pakai max_train_epochs dari tabel/bucket.)

### F8 — Family split epoch 📐 INFERRED (dari `model_plans.csv`, bukan formula eksplisit di kode)
- Subject talla CH rata-rata: **anime ep≈43.4 vs photoreal ep≈29.3** → **anime ≈ +48% epoch**.
- LR hampir sama (~0.94). d_coef: photoreal ~1.20 (agresif), anime ~1.12.
- ⚠️ zayden encode ini PER-BARIS tabel, bukan sebagai multiplier. Angka +48% = turunan median CSV, dipakai HANYA di generator fallback (base non-tabel).

### F9 — Capacity preset per base ✅ VERIFIED (`zayden_recipe.py:108-137`)
Berlaku SUBJECT path saja. Style path = selalu **(32,32,conv)**.
| Preset | (dim,alpha,conv) | Base model |
|--------|------------------|------------|
| default | (32,32,conv) | 14 base tak-terdaftar-preset |
| RANK64_CONV | (64,64,conv) | RealVisXL_V4.0, albedobase2-xl, ProteusV0.5, RealisticStockPhoto-fp16, blue-pencil-xl-v7 |
| RANK64_PLAIN | (64,64,plain) | TempestV0.1, Mann-E_Dreams, aam-xl-anime-mix |
| RANK96_CONV | (96,96,conv) | animagine-xl-4.0 |
| RANK32_PLAIN | (32,32,plain) | hassaku-illustrious, Fluently-XL-Final, DynaVisionXL, colorfulxl, anima-pencil-v5, anything-xl, mobius, Illustrious-early-v0 |
network_args conv = `["conv_dim=4","conv_alpha=4","dropout=null"]` (`:141`).

---

## SUMBU 1 — SHAPE ✅ VERIFIED (`zayden_recipe.py:39-40, 201-211`)
- `ZAYDEN_SUBJECT_CATEGORIES = {person, product}` → **subject → Prodigy** (path aggressive).
- `ZAYDEN_STYLE_CATEGORIES = {style, logo, social, design}` → **style → AdamW** (path prior).
- Resolusi: kategori authoritative; `is_style` fallback saat kategori unknown (`:211`).
- Optimizer args:
  - Prodigy (`:178`): `decouple=True, d_coef=<dc>, weight_decay=<wd>, use_bias_correction=True, safeguard_warmup=True`.
  - AdamW (`:187`): `betas=(0.9,0.999), weight_decay=<wd>, eps=1e-08`.

## SUMBU 2 — SIZE BUCKET (talla) ✅ VERIFIED (`zayden_recipe.py:49-100`)
Bucket: **ECH ≤10 · CH ≤20 · M ≤30 · G ≤50 · EG >50** (`:49`).
Kurva GLOBAL (fallback generator; per-model tabel override ini):

**Subject buckets** (`:64-80`):
| talla | ep | batch | ulr=telr | sched | snr | dc | wd |
|-------|-----|-------|----------|-------|-----|----|----|
| ECH | 73 | 4 | 0.95 | constant | 5 | 1.2 | 0.012 |
| CH | 33 | 4 | 1.0 | constant | 5 | 1.0 | 0.008 |
| M | 26 | 8 | 1.0 | constant_with_warmup(50) | 6 | 1.1 | 0.01 |
| G | 19 | 10 | 1.1 | constant_with_warmup(75) | 6 | 1.0 | 0.01 |
| EG | 11 | 12 | 1.2 | constant_with_warmup(100) | 6 | 0.9 | 0.01 |
noise_offset: hanya ECH=0.03; bucket lain None (inherit).

**Style buckets** (`:84-100`):
| talla | ep | batch | ulr | telr | sched | warm | snr | no | wd |
|-------|-----|-------|-----|------|-------|------|-----|-----|----|
| ECH | 28 | 4 (gas2) | 3.0e-5 | 1.0e-6 | constant_with_warmup | 25 | 6 | 0.02 | 1.5e-4 |
| CH | 32 | 6 | 2.8e-5 | 1.2e-6 | cosine | 50 | 6 | 0.025 | 1.2e-4 |
| M | 24 | 8 | 3.2e-5 | 1.5e-6 | cosine | 75 | 6 | 0.025 | 1.0e-4 |
| G | 18 | 10 | 3.6e-5 | 1.8e-6 | cosine | 100 | 6 | 0.035 | 1.0e-4 |
| EG | 11 | 12 | 4.0e-5 | 2.0e-6 | cosine | 120 | 6 | 0.04 | 1.0e-4 |

## SUMBU 3 — BASE MODEL: CASCADE TABEL-HEAVY + 1 GUARD ✅ (desain final, audit-backed)
```
base_model:
  ├─ ada di ZAYDEN_MODEL_PLANS[shape][model][talla]?
  │    ├─ lolos sanity-guard → PAKAI BARIS TABEL
  │    └─ kena guard         → generator fallback
  └─ tidak ada               → generator fallback
```
**Sanity-guard universal** (netralkan blue-pencil style/CH + bug LR-scale masa depan):
```python
if optimizer in {"adamw","adamw8bit","lion"} and unet_lr > 1e-2:
    use_generator_fallback()   # blue-pencil style/CH (ulr=1.0 @ adamw) ketangkep
```
**Generator fallback** = kurva global Sumbu 2 (VERIFIED) + F8 family multiplier (📐) + F9 capacity preset (VERIFIED) + Huber flag photoreal (📐).
**Opsional low-prio:** `ep = min(ep, 3×median_talla)` (jaga outlier RealisticStockPhoto subject/EG=36; tak wajib, tak divergen).

---

## BASE TURNAMEN — RECIPE PERSIS (dari `model_plans.csv`, VERIFIED)

### DreamShaper-XL (`Lykon/dreamshaper-xl-1-0`) — 100% BERSIH, salin langsung
**SUBJECT** (Prodigy, rank 32/32 conv):
| talla | ep | batch | ulr=telr | sched(warm) | snr | dc | wd | noise |
|-------|-----|-------|----------|-------------|-----|----|----|-------|
| ECH | 73 | 4 | 0.95 | constant | 5 | 1.2 | 0.012 | 0.03 |
| CH | 33 | 4 | 1.0 | constant | 5 | 1.0 | 0.008 | (multires jika person) |
| M | 26 | 8 | 1.0 | const_warmup(50) | 6 | 1.1 | 0.01 | — |
| G | 19 | 10 | 1.1 | const_warmup(75) | 6 | 1.0 | 0.01 | — |
| EG | 11 | 12 | 1.2 | const_warmup(100) | 6 | 0.9 | 0.01 | — |

**STYLE** (AdamW, rank 32/32 conv):
| talla | ep | batch(gas) | ulr | telr | sched(warm) | snr | no | wd |
|-------|-----|-----------|-----|------|-------------|-----|-----|----|
| ECH | 40 | 4 (2) | 2.2e-5 | 1.0e-6 | const_warmup(25) | 6 | 0.02 | 1.5e-4 |
| CH | 35 | 6 | 2.8e-5 | 1.2e-6 | cosine(50) | 6 | 0.025 | 1.2e-4 |
| M | 30 | 8 | 3.2e-5 | 1.5e-6 | cosine(75) | 6 | 0.03 | 1.0e-4 |
| G | 26 | 8 | 3.8e-5 | 1.9e-6 | cosine(150) | 6 | 0.033 | 1.0e-4 |
| EG | 11 | 12 | 4.0e-5 | 2.0e-6 | cosine(120) | 6 | 0.04 | 1.0e-4 |

### RealVisXL_V4.0 (`SG161222/RealVisXL_V4.0`) — BERSIH, 1 bump (subject/M ep=40)
**SUBJECT** (Prodigy, **rank 64/64 conv** ← RANK64_CONV):
| talla | ep | batch | ulr=telr | sched(warm) | snr | dc | wd |
|-------|-----|-------|----------|-------------|-----|----|----|
| ECH | 43 | 8 | 0.9 | constant | 7 | 1.2 | 0.015 |
| CH | 27 | 6 | 0.9 | constant | 5 | 1.2 | 0.015 |
| M | **40** ⚠️ | 8 | 1.0 | const_warmup(50) | 7 | 1.1 | 0.01 |
| G | 16 | 10 | 1.1 | const_warmup(75) | 5 | 1.0 | 0.01 |
| EG | 9 | 12 | 1.2 | const_warmup(100) | 5 | 0.9 | 0.01 |
⚠️ subject/M ep=40 > CH=27 (non-monoton). Usable (LR sehat), opsional turunkan M → ~24-26.

**STYLE** (AdamW, rank 32/32 conv):
| talla | ep | batch(gas) | ulr | telr | sched(warm) | snr | no | wd |
|-------|-----|-----------|-----|------|-------------|-----|-----|----|
| ECH | 32 | 4 (2) | 2.2e-5 | 1.0e-6 | const_warmup(25) | 5 | 0.02 | 1.5e-4 |
| CH | 27 | 6 | 2.8e-5 | 1.2e-6 | cosine(50) | 5 | 0.025 | 1.2e-4 |
| M | 21 | 8 | 3.2e-5 | 1.5e-6 | cosine(75) | 5 | 0.03 | 1.0e-4 |
| G | 15 | 10 | 3.6e-5 | 1.8e-6 | cosine(100) | 5 | 0.035 | 1.0e-4 |
| EG | 9 | 12 | 4.0e-5 | 2.0e-6 | cosine(120) | 5 | 0.04 | 1.0e-4 |

---

## OUTPUT PER ARSITEKTUR (2.6)

### SDXL ✅ VERIFIED (base_diffusion_sdxl_*.toml + recipe overlay)
- rank/alpha: per F9 capacity (default 32/32; RealVis 64/64; animagine 96/96). network conv default.
- LR: subject Prodigy 0.9–1.2 (ulr=telr) · style AdamW 2.2e-5–4.5e-5 (ulr), telr 1e-6–2e-6.
- optimizer: Prodigy (subject) / AdamW (style). **TE DILATIH** (telr di-set kedua path).
- snr: 5–7 · noise: multires(person/text-heavy) atau offset(style/product).
- dropout: 0.2 subject / 0.1 style · dedup: on · resolution 1024,1024 · loss l2.

### Flux ✅ VERIFIED (`base_diffusion_flux.toml`)
- rank **128** / alpha **64** (`:79-80`) · network lora_flux, train_double+single all, **train_t5xxl=True** (`:81`).
- optimizer **Lion**, args `weight_decay=0.005, betas=(0.9,0.99)` (`:60-61`).
- **guidance_scale 85.0** (`:50`) · timestep_sampling sigmoid · discrete_flow_shift 3.1582 (`:51-52`).
- unet_lr 8e-5, **TE(T5) lr [8e-6,8e-6]** (`:62-63`) · lr_scheduler cosine · max_train_steps 250, batch 4, gas 2.
- Recipe overlay (`zayden_recipe.py:308-324`): unet_lr 5e-5, te_lr [5e-6,5e-6], caption_dropout 0.1. (base TOML pegang epochs/guidance.)

### Qwen ✅ VERIFIED (`zayden_recipe.py:441-463`)
- rank **clamp(118+0.85N, 120, 152)** · LR ~9e-5 (formula F6) · wd ~8.8e-6 · alpha=rank.
- **EMA 0.995** · do_cfg True cfg_scale **6.0** · optimizer adamw8bit · train_unet only.
- quantize float8, low_vram · flowmatch · timestep_type weighted · step budget F7.

### Z-Image ✅ VERIFIED (`zayden_recipe.py:421-439`)
- rank **32 + conv 16** (linear32/alpha32/conv16/conv_alpha16) · adapter **v2** (`assistant_lora_path`).
- timestep_type **weighted** · LR **1e-4** fixed · optimizer adamw8bit · flowmatch · quantize qfloat8 · step budget F7.

---

## SLOT EDGE (2.7) — DEFAULT OFF, A/B pas GPU (📐 magnitudo)
| Flag | Default | Alternatif (Tier) |
|------|---------|-------------------|
| timestep_sampling | uniform/sigmoid | eval_aligned oversample[700,950] (T1) |
| zero_terminal_snr | off | on per-base (T1) |
| noise_extend | person+text-heavy | +product+style (T1) |
| ema | qwen-only | all_paths (T1, hampir selalu keep) |
| early_stop | off | holdout_proxy_L2 (T2) |
| lr_probe | off | range-test buat base anonim (T2) |
| router | keyword-only | hybrid CLIP/face/OCR vote (T3) |
| dedup | dhash+colour | +DINOv2/SSCD (T3) |
| captioner | LLaVA-1.5-7B | Qwen2.5-VL (T3) |
| peft | lora/locon | DoRA (T3) |

---

## STATUS SLOT
- Semua F1–F9, Sumbu 1–3, output per-arch, base turnamen = **TERISI**.
- ✅ VERIFIED: F1-F7, F9, Sumbu 1-2, Sumbu 3 guard, semua base rows, output arch.
- 📐 INFERRED (perlu A/B/hati-hati): F8 family multiplier (+48%), Huber-photoreal flag, semua slot EDGE.
- Slot yang butuh keputusan GPU: urutan A/B edge (BAGIAN 4 kerangka).
