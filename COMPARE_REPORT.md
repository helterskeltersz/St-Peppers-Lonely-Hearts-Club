# COMPARE_REPORT.md — Repo Baru vs f-poseidon (Targeted)

**Repo baru:** `St-Peppers-Lonely-Hearts-Club` @ `fase2a-recipe` — teori (belum pernah di-score).
**Repo lama:** `f-poseidon` (`Keilrock/god-image-one`) — **satu-satunya artifact yang pernah di-score** (lolos R1, kalah R2: SDXL product, base DreamShaper-XL, Wen 0.0532 vs lawan 0.0468, gap ~13.6%).

> **Caveat penting:** repo lama yang gw baca = state **dethrone-tuned POST-R2** (komentar nyebut FASE 2A, T3, "winner 5FW2" — eksperimen Wen SETELAH kalah). Jadi ini bukan config persis yang kalah R2, tapi **akumulasi pembelajaran empiris Wen** = sinyal bernilai. Recipe asli BISA diekstrak (bukan cuma wrapper) — ada di `scripts/image_trainer.py` + `scripts/core/config/*` + `scripts/lrs/*.json`.

---

## Q1 — Hyperparameter SDXL: side-by-side

Repo lama SDXL = **routing per-kategori** (`scripts/image_trainer.py:135-143, 461-490`). R2 loss = **product** → kategori `default`.

| Param | LAMA `default` (person/**product**/social) | LAMA `style` (trigger=None) | LAMA `logo`/`design` | BARU `aggressive` (≈product) | BARU `prior` | BARU `logo` |
|---|---|---|---|---|---|---|
| network_module | `networks.lora` (plain) | **`lycoris.kohya` (DoRA)** | **`lycoris.kohya` (DoRA)** | `networks.lora` | `networks.lora`+conv | `networks.lora`+conv |
| rank / alpha | **64 / 64** | 32 / 32 | 32 / 32 | 32 / 16 | 32 / 16 | 32 / 16 |
| conv | — | **conv4 / α4** | **conv4 / α4** | off | 16 / 16 | 16 / 16 |
| DoRA | ❌ no | **✅ `dora_wd=True`** | **✅ `dora_wd=True`** | ❌ | ❌ | ❌ |
| extra net args | — | **`loraplus_lr_ratio=16`** | — | — | — | — |
| optimizer | **prodigy** (adaptif) | adamw | (default toml) | AdamW8Bit | AdamW8Bit | AdamW8Bit |
| LR (unet) | **prodigy d_coef ~1.0** (lr≈1.0) | 2e-5 (te 2e-6) | (toml) | **2e-5** | 1e-5 | 1e-5 |
| scheduler | constant / constant_w_warmup | **cosine** + warmup25 | constant | constant | constant | constant |
| min_snr_gamma | **6** | **8** | 5 | **1** | 5 | 5 |
| caption_dropout | **0.05** (product) / **0.10** (person) | 0.05 | 0.05 | **0.30** | **0.40** | **0.30** |
| steps | **size-aware epochs** (xs..xl: 35→11 ep) | size-aware | — | hours×it/s solver | solver | solver |
| TE training | prodigy melatih TE (lr=1) | te 2e-6 (dilatih) | dilatih | **UNet-only (TE dibuang)** | UNet-only | UNet-only |
| timestep clamp | **❌ NO clamp** (`:271`) | ❌ | ❌ | ❌ | ❌ | ❌ |
| flip_aug | default | default | default | default | default | **off** |

**Beda paling besar (product ↔ aggressive):**
1. **caption_dropout 0.05 → 0.30 (6×).** Lever utama tesis baru (0.75 no-text). Repo lama nggak exploit ini.
2. **Optimizer prodigy (adaptif, self-tuning LR) → AdamW8Bit fixed 2e-5.** Repo lama pakai prodigy size-aware; repo baru LR flat manual.
3. **rank 64 → 32, snr 6 → 1.** Repo baru rank lebih kecil + snr agresif (overfit cepat).
4. **size-aware epochs (xs/s/m/l/xl, prodigy d_coef turun seiring data gede) → solver hours×it/s.** Dua filosofi step beda total.

> Catatan: repo lama `base_diffusion_sdxl.toml` (LR1e-5/AdamW8Bit/rank32/snr5) = **fallback** kalau LR-JSON nggak match; jalur aktif = prodigy size-aware dari `person_config.json`.

---

## Q2 — Apakah repo baru nutup penyebab kalah R2?

Lawan R2 diduga **DoRA + LyCORIS + timestep clamping**. Cek:

| Faktor | Repo lama (product) | Repo baru (product/aggressive) | Verdict |
|---|---|---|---|
| **DoRA** | ❌ plain LoRA-64 — **sengaja** ("DoRA-default = blunder product **-26%**, bukti T3", `:126`) | ❌ NO DoRA (bahkan style/logo nggak) | ⚠️ Repo baru **nggak nambah DoRA**. TAPI buat **product** itu DEFENSIBLE — bukti Wen DoRA bikin product LEBIH PARAH. Jadi "lawan punya DoRA" ≠ penyebab pasti kalah product. |
| **LyCORIS** | ✅ ada (style/logo/design pakai) | ❌ nggak dipakai sama sekali | ⚠️ repo baru kehilangan jalur DoRA/LyCORIS yg repo lama punya buat style/logo. |
| **timestep clamping** | ❌ NO clamp (`:271` eksplisit) | ❌ nggak ada | ❌ **GAP TERBUKA DI DUA-DUANYA.** Edge lawan ini **nggak di-address siapapun.** |
| **Strategi 0.75 no-text** | cd0.05 (lemah) | **cd0.30 + UNet-only + drop trigger** | ✅ repo baru **bet baru** di sini — lever yg repo lama nggak tau. |

**Verdict Q2:**
- Repo baru **TIDAK** nutup gap secara literal (no DoRA, no timestep clamp). 
- Tapi bet-nya **beda by design**: alih-alih niru DoRA lawan (yg Wen buktikan nyakitin product), repo baru optimasi **term 0.75 no-text** via caption_dropout agresif + UNet-only. Ini **teori solid tapi UNTESTED**.
- **Gap paling mencurigakan yg masih nganga: timestep clamping.** Lawan pakai, repo lama nggak, repo baru nggak. Kalau itu beneran edge lawan R2, **dua repo sama-sama belum jawab.** → **rekomendasi investigasi (R5 bawah).**
- **Risiko:** repo baru buang SEMUA tuning empiris lama (prodigy, size-aware, DoRA-style) demi teori murni. Buat product, plain-LoRA OK (sesuai T3). Tapi belum ada bukti cd0.30 > cd0.05 di product (Wen bilang product = "fortress, cd10 untested").

---

## Q3 — Recipe per-arch lain (Flux / Qwen / Z-Image)

| Param | Flux LAMA | Flux BARU | Qwen LAMA | Qwen BARU | Z LAMA | Z BARU |
|---|---|---|---|---|---|---|
| rank / alpha | 128 / **64** | 128 / **128** | **128 / 128** | **32 / 32** | 32 / 32 +conv16 | 32 / 32 +conv16 |
| LR (unet) | **8e-5** | 5e-5 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| optimizer | **Lion** | Adafactor | adamw8bit | adamw8bit | adamw8bit | adamw8bit |
| scheduler | **cosine** | constant | — | — | — | — |
| steps | **250** (+per-model) | 3000 (fallback) | **2500** / person **12×img** | solver | **2000** / person **12×img** | solver |
| caption_dropout | (per-model) | 0.30+ | **0.05** / person **dibuang** | 0.15-0.40 | **0.05** / person dibuang | 0.20-0.40 |
| EMA | — | off | **use_ema=True (0.995)** / person off | off | none | off |
| CFG training | — | — | **do_cfg=True, cfg_scale 6.0** | ❌ | ❌ | ❌ |
| flow_shift | 3.1582 | 3.1582 | (toolkit derive) | (toolkit derive) | (derive) | (derive) |
| guidance_scale | **85.0** | — | — | — | — | — |
| qtype | — | — | **float8** | uint3 | qfloat8 | qfloat8 |

**Tuning lama yg LEBIH SPESIFIK / mungkin kelewat (verified Wen dari winner HF metadata):**
1. **Qwen rank 128** (lama) vs **32** (baru). Repo baru ambil **default template G.O.D (32)**, repo lama pakai **128 (champion-derived)**. ⚠️ Potensi undertraining Qwen di repo baru.
2. **Qwen/Z person `steps = 12×img`** (winner: Qwen 108=12×9, Z 168=12×14) + **caption_dropout dibuang** + Qwen **EMA-off**. Repo baru pakai solver hours-based → **risiko overtrain person** (nggak ada cap 12×img).
3. **Qwen `do_cfg + cfg_scale 6.0` + `EMA 0.995`** (non-person). Repo baru nggak punya CFG training sama sekali.
4. **Flux steps 250 + Lion + cosine + alpha64** (lama, agresif/low-step) vs **baru alpha128/Adafactor/constant/3000**. ⚠️ Flux overfit cepat; default 3000 baru bisa overtrain (solver bantu, tapi recipe default beda jauh).
5. **SDXL logo: lama = DoRA + conv4 + cd0.05**; baru = LoCon conv16 + cd0.30, **no DoRA**. Taxonomy & nilai beda.
6. **Z qtype `qfloat8`** (lama) vs baru `qfloat8` ✅ match; **Qwen `float8`** (lama) vs `uint3` (baru) — baru lebih agresif quant (hemat VRAM, risiko kualitas).

---

## Q4 — Hal yg PERFORM di repo lama, repo baru NGGAK punya

(Repo lama lolos R1 → ada yg jalan. Di luar hyperparameter:)

1. **🔴 LLaVA-1.5-7B auto-captioning** (`scripts/auto_caption.py`) — repo lama **re-caption SELURUH dataset** pakai LLaVA sebelum training (model di-bake di dockerfile). Repo baru **cuma drop trigger token**, nggak re-caption. **Sinyal besar — caption quality nyetir conditional term.**
2. **BLIP2 caption enhancer** (`scripts/core/caption_enhancer.py`).
3. **Person `12×img` step recipe** (Qwen/Z) — winner empiris, repo baru nggak punya.
4. **Size-aware LR/optimizer/epochs** (`person_config.json`/`style_config.json`, bucket xs/s/m/l/xl, prodigy d_coef scaling) — repo baru flat per-bucket.
5. **Deteksi kategori high-precision** (`category_detector.py`: logo/social/person/art cascade utk ai-toolkit; style/logo/design/default utk SDXL, fraction-threshold 0.6 bias-aman). Repo baru = 3-bucket classifier (CLIP/OCR), **taxonomy beda**.
6. **Person-SDXL `is_person_dataset`** high-precision (PERSON vs PRODUCT_EXCLUDE patterns) → cd0.10 khusus person, product dilindungi cd0.05.
7. **DoRA + `loraplus_lr_ratio=16`** (style), **`dora_wd=True`** (style/logo/design).
8. **Defensive lycoris fallback** (kalau lib nggak ada → plain-64, jangan crash). *(Repo baru udah install open_clip/pytesseract tapi recipe-nya emang nggak pakai DoRA.)*
9. **Qwen CFG training + EMA** (non-person).
10. **Per-base-model tuning** (`network_config_person`/`config_mapping`/flux per-hash) — *kemungkinan LEGACY* (komentar `:462` bilang routing per-kategori "gantikan seleksi per-model lama"), tapi flux.json per-hash masih kepakai.

---

## Q5 — Drift / regresi tanpa penjelasan matriks

Pola umum: **repo baru sistematis ngikut DEFAULT TEMPLATE G.O.D (§1 "verified"), buang nilai champion-tuned repo lama.** Sebagian disengaja (caption_dropout per tesis), sebagian **kehilangan sinyal empiris tanpa justifikasi eksplisit**:

| Nilai | Lama (champion) | Baru (template G.O.D) | Status |
|---|---|---|---|
| Qwen rank | **128** | 32 | ⚠️ **DRIFT** — matriks §1 ambil 32 dari template G.O.D, tapi champion pakai 128. Nggak ada penjelasan kenapa 32 > 128. |
| Flux alpha | **64** | 128 | ⚠️ DRIFT (template 128, champion 64). |
| Flux optimizer/scheduler/steps | **Lion/cosine/250** | Adafactor/constant/3000 | ⚠️ DRIFT — champion Flux agresif low-step; baru pakai default berat. |
| Qwen EMA + CFG | **EMA0.995 + cfg6.0** | off / none | ⚠️ Sebagian disengaja (§3a EMA off), tapi CFG training ilang tanpa bahasan. |
| SDXL caption_dropout | 0.05 / 0.10 | 0.30-0.40 | ✅ **DISENGAJA** (tesis 0.75 no-text, matriks §3a). Bukan drift. |
| SDXL snr (aggressive) | 6 | 1 | ✅ disengaja (§3b). |
| SDXL UNet-only / drop TE | TE dilatih (prodigy) | UNet-only | ✅ disengaja (keputusan Fase 2a-1). |
| person 12×img steps | ada | nggak ada | ⚠️ DRIFT — recipe winner ilang, diganti solver tanpa cap. |

---

## 6. Rekomendasi (ranked by confidence) — keputusan akhir di Wen

### LAYAK di-recover (sinyal empiris valid, nggak konflik sama tesis):

| # | Item | Confidence | Alasan |
|---|---|---|---|
| R1 | **Person `12×img` step cap (Qwen/Z)** + drop caption_dropout utk person | **TINGGI** | Winner empiris eksplisit (108/168). Solver baru bisa overtrain person. Tambah sbg cap, nggak konflik tesis. |
| R2 | **Caption pipeline (LLaVA auto-caption)** atau minimal caption-normalize | **TINGGI** | Repo lama re-caption; lolos R1. Repo baru train caption mentah. ⚠️ tapi selaraskan sama tesis (caption pendek/generic + dropout) — mungkin LLaVA caption pendek, bukan verbose. |
| R3 | **Qwen rank 128 (investigasi)** | **TINGGI** | Champion pakai 128, baru pakai 32 (template). A/B 32 vs 128 wajib sebelum percaya 32. |
| R4 | **Flux empiris: steps rendah (~250) + Lion/cosine** (investigasi) | **SEDANG-TINGGI** | Flux overfit cepat; default 3000 baru ekstrem. Minimal jadiin fallback ceiling lebih rendah utk Flux. |
| R5 | **Timestep clamping (investigasi BARU)** | **SEDANG** | Edge lawan R2 yg **dua repo nggak punya**. Worth eksperimen — bisa jadi penyebab kalah sebenernya. |
| R6 | **Size-aware step/LR** (prodigy adaptif atau bucket xs..xl) | **SEDANG** | Prodigy adaptif ngurangin risiko salah-tuning LR. Repo baru LR flat manual = rapuh ke ukuran dataset. |
| R7 | **DoRA + loraplus utk style/logo** (BUKAN product) | **SEDANG-RENDAH** | Repo lama pakai DoRA buat style/logo. Konsisten sama §5b (DoRA fitur). TAPI Wen: DoRA nyakitin product → batasi ke style/logo aja. |
| R8 | **Qwen CFG training (do_cfg/cfg6.0) + EMA non-person** | **RENDAH** | Champion pakai; tapi konflik parsial sama §3a (EMA off). A/B opsional. |

### BIARIN BEDA (disengaja, riset — jangan recover):
- **caption_dropout 0.30-0.40** (vs lama 0.05) — inti tesis 0.75 no-text. A/B di Fase 2b ({0.10,0.20,0.30,0.40}), JANGAN balik ke 0.05 tanpa data.
- **UNet-only SDXL + drop trigger** — keputusan Fase 2a-1, konsisten tesis.
- **SDXL snr1 aggressive, conv strategy** — §3b.
- **Qwen uint3 quant** (vs float8) — keputusan hemat-VRAM (monitor kualitas).

### ⚠️ Yg PALING kritis disampaikan ke Wen:
1. **Repo baru = teori murni, buang SEMUA empiris champion.** Buat product itu sebagian OK (T3 bukti DoRA nyakitin), tapi Qwen rank32, Flux 3000-step, no-12×img-person = **sinyal winner yg dibuang tanpa A/B.**
2. **timestep clamping** = gap nganga di dua repo. Kalau itu edge lawan R2, **belum kejawab.**
3. **Caption pipeline ilang** — repo lama re-caption (LLaVA), repo baru nggak. Bisa ngaruh ke conditional term (25%).
4. Verdict kalah R2: repo baru **secara teori nge-address term 0.75** (yg lama abaikan), TAPI **belum nutup DoRA/timestep-clamp lawan**, dan **belum tervalidasi** > cd0.05 lama. Net: **bet beda, bukan perbaikan terbukti.**
