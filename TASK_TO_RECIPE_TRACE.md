# TASK_TO_RECIPE_TRACE.md — Trace Alur Task → Recipe

Branch `fase2a-recipe`. **Verifikasi dengan kode aktual** (path:baris), tanpa ubah kode. Contoh dijalanin di CPU.

Pertanyaan Wen: *"Dapet task SDXL logo, dia pakai tuning yang mana, dan gimana kode nentuinnya?"*
**Jawaban singkat:** task → arsitektur (dari `--model-type`) + bucket (di-infer classifier dari dataset) → `resolve_recipe(arch, bucket)` → config. Bucket **BENERAN nyetir recipe** (bukan default doang). TAPI link `dataset → bucket` rawan: **`trigger_present` nge-short-circuit subject_variance**, dan threshold belum dikalibrasi real.

---

## 1. Diagram alur end-to-end (dengan path:baris)

```
validator → run_image_trainer.sh → image_trainer.py main()
  │
  ├─ parse_args(): --model-type, --dataset-zip, --hours-to-complete, [--trigger-word]
  │        scripts/image_trainer.py:38-48
  │
  ├─ [1] extract_dataset(zip) → images_dir
  │        scripts/image_trainer.py:118-119
  │
  ├─ [2] classify(images_dir, trigger_word) → (bucket, confidence, signature)   ◄── INFER SUBTYPE
  │        scripts/image_trainer.py:122
  │        └─ runtime/subtype_classifier.py:218  (cascade: OCR→logo / var+trigger→aggr/prior)
  │
  ├─ [3] resolve_recipe(model_type, bucket) → Recipe                            ◄── BUCKET MASUK SINI
  │        scripts/image_trainer.py:127
  │        └─ runtime/recipe.py:219  = _baseline(model_type)            (arch baseline §1)
  │                                  + _apply_unconditional(...)        (§3a, recipe.py:167)
  │                                  + _apply_bucket(model_type,bucket) (§3b/c, recipe.py:179)
  │
  ├─ [4] prepare_captions(images_dir, recipe, trigger)  (drop trigger / exact)
  │        scripts/image_trainer.py:133  (recipe.drop_trigger_word nyetir, recipe.py)
  │
  ├─ [5] generate_config(recipe, model_type, ctx) → TOML(sdxl/flux) | YAML(qwen/z)
  │        scripts/image_trainer.py:146  → runtime/config_gen.py:generate_config
  │        steps = recipe.max_train_steps (FALLBACK ceiling)  scripts/image_trainer.py:143
  │
  ├─ [6] step plan: hours_to_complete = authoritative; window = sanity-only
  │        scripts/image_trainer.py:150-152
  │
  └─ [7] CUDA gate: CPU → stop (exit 0); GPU → measure_throughput → train
           scripts/image_trainer.py:155-165
```

**Link kritis bucket → recipe = `image_trainer.py:127`** (`resolve_recipe(args.model_type, bucket)`). Bucket dari classifier (line 122) **diteruskan langsung** ke `resolve_recipe`. **NYAMBUNG, bukan putus** — classifier beneran nyetir recipe. Verifikasi: `_apply_bucket` (recipe.py:179-217) ngubah field nyata (caption_dropout, conv, min_snr_gamma, flip_aug) per bucket.

---

## 2. T2 — Contoh konkret: SDXL logo (12 gambar wordmark + trigger)

Dijalanin CPU (`/root/gate-test/trace.py`):

| Tahap | Nilai aktual |
|---|---|
| **Sinyal classifier** | image_count=12, caption_entropy=1.585, trigger_present=**True**, subject_variance=0.0 (**proxy**), **ocr_area_ratio=0.1755** |
| **Keputusan** | ocr 0.1755 ≥ OCR_LOGO_THRESHOLD 0.08 → **LOGO** (`subtype_classifier.py:224-226`) |
| **Bucket** | `logo-specialist`, **confidence=0.95** |
| **Recipe terpilih** | `resolve_recipe("sdxl","logo-specialist")` |

**Nilai aktual yang masuk config (SDXL logo TOML):**

| Field | Nilai | Asal (path:baris) |
|---|---|---|
| network_dim (rank) | **32** | baseline `recipe.py:_baseline sdxl` |
| network_alpha | **16** | baseline |
| network_args (conv) | **`conv_dim=16, conv_alpha=16`** | logo bucket `recipe.py:210` |
| caption_dropout_rate | **0.30** | `LOGO_CAPTION_DROPOUT["sdxl"]` `recipe.py:31,206` |
| learning_rate | **1e-5** | baseline (logo nggak override LR) |
| min_snr_gamma | **5** | baseline (logo nggak override) |
| flip_aug | **false** | logo bucket `recipe.py:208` |
| network_train_unet_only | **true** | SDXL baseline `recipe.py:102` |
| text_encoder_lr | **(absent)** | SDXL UNet-only → None → omit `config_gen.py` |
| trigger handling | **KEPT** (drop_trigger_word=False, use_exact_caption=True) | logo bucket `recipe.py:204-205` |

→ Jadi "SDXL logo" pakai: **plain-LoRA-32 + conv16 + cd0.30 + flip-off + UNet-only + trigger dipertahankan**. (Catatan: **TANPA DoRA** — beda dari repo lama yg logo=DoRA+conv4, lihat COMPARE_REPORT.md.)

---

## 3. T3 — Tabel 4 kasus (nilai aktual CPU)

| Input (model-type + ciri) | Sinyal (trigger / subj_var / ocr) | Bucket | Conf | Recipe kunci (rank/conv/cd/flip/trigger) |
|---|---|---|---|---|
| **SDXL logo** (teks, trigger) | T / 0.0(proxy) / **0.175** | `logo-specialist` | 0.95 | rank32, **conv16**, **cd0.30**, **flip=false**, trigger **KEPT**, unet-only, snr5 |
| **SDXL person** (subjek mirip, trigger) | T / 0.0 / 0.0 | `aggressive-fit` | 0.807 | rank32, **conv off**, cd0.30, flip unset, trigger **DROPPED**, unet-only, **snr1**, **lr2e-5** |
| **Qwen logo** (teks, trigger) | T / 0.0 / **0.175** | `logo-specialist` | 0.95 | rank32, conv none, **cd0.15**, **flip_x=false**, trigger **KEPT**, weighted |
| **Z-Image style** (subjek beragam, NO trigger) | F / **0.081** / 0.0 | `prior-preserving` | 0.482 | rank32, **conv16**, **cd0.40**, flip unset, trigger DROPPED, weighted |

Observasi: bucket bener buat 4 kasus "bersih" ini, dan recipe-nya **beda nyata** per bucket (cd 0.15/0.30/0.40; conv on/off; flip; trigger kept/drop). Step belum task-driven (lihat T5).

---

## 4. T4 — Fallback, threshold, peta risiko misklasifikasi (PALING PENTING)

### Cascade keputusan (`subtype_classifier.py:218-245`)
```
1. ocr_area_ratio ≥ 0.08              → LOGO            (:224-226)
2. else if subject_variance ≤ thr  OR  trigger_present → AGGRESSIVE   (:231-233)
   else                                → PRIOR          (:234-236)
3. else (subject_variance == None)   → trigger? AGGRESSIVE : PRIOR, conf=0.45  (:243-245)
```

### Threshold
| Threshold | Nilai | path:baris | Status kalibrasi |
|---|---|---|---|
| OCR_LOGO_THRESHOLD | **0.08** | `:33` | 📐 **TEBAKAN** (Fase 2a-1, belum real data) |
| SUBJECT_VAR_LOW (proxy) | **0.04** | `:37` | 📐 di-tune ke dataset SINTETIS, bukan nyata |
| SUBJECT_VAR_LOW (clip) | **0.30** | `:36` | 📐 belum pernah dipakai (CLIP weights belum di-bake → selalu proxy) |

### 🔴 Risiko misklasifikasi (DIBUKTIIN di CPU)
**Akar masalah: `or sig.trigger_present` di `:231`** → kalau trigger ADA, `subject_variance` **DIABAIKAN**, langsung AGGRESSIVE (kecuali OCR udah nyetak LOGO duluan).

| Skenario | Sinyal aktual | Harusnya | Kenyataan | Kenapa |
|---|---|---|---|---|
| **Logo teks kecil + trigger** | ocr=**0.0047** (<0.08), trigger=T | logo | **aggressive-fit** (0.807) | OCR di bawah 0.08 → lolos step 1 → trigger short-circuit ke aggressive. **Recipe logo (conv/exact-caption/flip-off/cd0.30) HILANG**, malah trigger DI-DROP. |
| **Style + ada trigger** | subj_var=0.119 (>0.04→harusnya prior), trigger=T | prior-preserving | **aggressive-fit** (0.468) | `:231` trigger menang atas subject_variance. |

**Batas meleset konkret:**
- **Logo selamat HANYA kalau `ocr_area_ratio ≥ 0.08`.** Teks kecil/stylized/low-contrast yang OCR-nya di bawah 0.08 → jatuh ke aggressive (kalau ada trigger) atau prior (kalau nggak). Threshold 0.08 ini **tebakan** — di dataset nyata bisa kebanyakan logo asli kelewat, atau gambar bertekstur false-positive (bug OCR multi-level udah di-fix, tapi threshold belum di-validate real).
- **subject_variance praktis MATI kalau trigger ada.** Karena turnamen sering ngasih trigger buat person/product/object, jalur prior-vs-aggressive lewat variance **cuma aktif buat dataset tanpa trigger.** Style/art yang dikasih trigger → salah ke aggressive.
- **CLIP belum aktif** (weights belum di-bake, `subject_variance_source` selalu `proxy`) → threshold proxy 0.04 dipakai, skalanya sintetis. Confidence proxy di-penalti ×0.85 (`:239-240`) tapi tetap dipakai.

### Confidence TIDAK dipakai buat fallback
`confidence` dihitung (`:226,241,245`) tapi **cuma di-log** (`image_trainer.py:124`) — **nggak ada cabang "kalau conf rendah lakukan X".** Jadi conf 0.45 (ragu) dan 0.95 (yakin) **dapet perlakuan sama**: bucket tetap dipakai apa adanya. Nggak ada safety-fallback pada keraguan.

---

## 5. T5 — Gap desain vs implementasi

| Desain (KONTEKS) | Implementasi aktual | Status |
|---|---|---|
| 1. model-type → arsitektur + baseline | `_baseline(model_type)` `recipe.py:_baseline` | ✅ NYAMBUNG |
| 2. classifier baca dataset → bucket | `classify()` `image_trainer.py:122` | ✅ NYAMBUNG (tapi lihat catatan) |
| 3. resolve_recipe = baseline+override+delta bucket | `image_trainer.py:127` → `recipe.py:219` | ✅ NYAMBUNG (bucket beneran nyetir) |
| 4. recipe → config TOML/YAML | `generate_config` `image_trainer.py:146` | ✅ NYAMBUNG |

**Putus / belum nyambung / hardcode:**
1. **subject_variance "setengah mati":** didesain 4 sinyal, tapi `:231` bikin trigger nge-override variance. 2 dari 4 sinyal (variance + sebagian trigger) jadi sub-ordinat. **Bukan putus total, tapi sinyal ke-dilusi.**
2. **CLIP path belum aktif** → `subject_variance` selalu proxy (CLIP weights belum di-bake). Jalur `SUBJECT_VAR_LOW_CLIP=0.30` (`:36`) **dead** sampai 2a-2.
3. **Step count belum task-driven:** `ctx.steps = recipe.max_train_steps` (`image_trainer.py:143`) = **baseline fallback ceiling**, BUKAN dari dataset/window. Step beneran dari `measure_throughput` (butuh GPU, masih stub `:161`). Jadi sekarang config keluar dengan steps baseline (1600/3000/2500/2000), belum disesuaikan task.
4. **confidence nggak dipakai** buat keputusan apa pun (cuma log).
5. **Threshold semua 📐** (belum kalibrasi real).
6. Taksonomi bucket (aggressive/prior/logo) ≠ taksonomi repo lama (person/product/style/logo/social) — keputusan desain, bukan bug, tapi recipe lama nggak transfer 1:1.

---

## 6. Ringkasan jujur

**Apakah "task → recipe yang bener" jalan otomatis?** 
**YA, untuk happy-path & pipa-nya nyambung:** `--model-type` → arch baseline, dataset → bucket (classifier), bucket → recipe (`resolve_recipe`, link di `image_trainer.py:127` terbukti nyetir field nyata), recipe → config. 4 kasus bersih (T3) ke-resolve dengan recipe beda-beda yang benar.

**Tapi titik PALING RAWAN salah pilih tuning:**
1. **`trigger_present` short-circuit (`subtype_classifier.py:231`)** — sinyal #1 risiko. Karena trigger ada → `subject_variance` diabaikan → **style/art-dengan-trigger salah ke aggressive**, dan **logo yang OCR-nya lemah + trigger → aggressive** (recipe logo hilang). Dibuktiin di CPU.
2. **Logo bergantung 100% pada OCR ≥ 0.08** (`:224`) dengan threshold **tebakan**. Teks kecil/stylized lolos → bukan logo.
3. **Threshold belum dikalibrasi dataset nyata**, dan **CLIP belum aktif** (selalu proxy) → skala variance sintetis.
4. **Confidence dihitung tapi nggak dipakai** → nggak ada jaring pengaman saat ragu.
5. **Step belum task-driven** (masih baseline ceiling) sampai GPU solver (2a-2).

**Verdict:** mekanik "task→recipe" **secara struktur sudah nyambung dan otomatis**, tapi **akurasi pemilihan bucket masih rapuh** — terutama gara-gara trigger nge-dominasi variance dan threshold OCR/variance yang belum tervalidasi. Untuk task turnamen yang hampir selalu ada trigger, praktisnya: **OCR nentuin logo-or-not, sisanya hampir selalu aggressive** — prior/style cuma kepilih kalau NGGAK ada trigger. Ini area yang paling butuh kalibrasi + revisi logika (keputusan di Wen).
