# MATRIKS RECIPE FINAL v2 — Turnamen Image G.O.D (Bittensor SN56)

**Status dokumen:** Acuan tunggal (single source of truth) untuk fase penulisan recipe & kode.
Menggabungkan: matriks riset awal → koreksi Fase 0 (source `rayonlabs/G.O.D` @ `2939bf24`) → temuan unconditional reconstruction.

**Aturan baca status angka:**
- `✅ VERIFIED` — terbaca langsung dari source G.O.D (path:baris di Fase 0).
- `📐 INFERRED` — turunan logis dari mekanik scoring/arsitektur, belum ada di source, perlu A/B.
- `⚠️ UNVERIFIED` — klaim riset lama yang BELUM ketemu di source. Jangan dipakai sampai diverifikasi.
- `❌ KILLED` — klaim riset lama yang TERBUKTI SALAH. Jangan dipakai.

---

## 0. Prinsip Utama (yang ngebalik semua intuisi normal)

**Skor = `0.25 × text_guided_L2 + 0.75 × no_text_L2`** ✅ VERIFIED (`validator/scoring/tasks.py:293-296`)

Konsekuensi yang harus nyetir SEMUA recipe:
1. **75% skor dari output TANPA prompt.** Optimasi utama = bikin unconditional prior model mirip distribusi dataset. Bukan caption, bukan trigger.
2. **Trigger word hampir nggak relevan buat skor** (cuma nyumbang ke 25% term). Buang atau minimalkan.
3. **Overfitting TERKONTROL itu menguntungkan di sini** — karena test set dari distribusi yang sama (split ~80/10), dan metriknya rekonstruksi L2, bukan estetika. Ini kebalikan dari best practice LoRA normal.
4. **Eval = img2img single denoise tinggi**, bukan multi-noise. Denoise per-arch: SDXL 0.9 / Flux 0.75 / Z-Image 0.90 / Qwen 0.93 ✅ VERIFIED (`validator/evaluation/constants.py:19-24`). Makin tinggi denoise → makin banyak digenerate ulang dari noise → makin dominan peran prior.
5. **10 seed (master_seed=42), 2 pass (prompt on/off) per test image** ✅ VERIFIED.

---

## 1. Baseline Dev (angka asli dari source — TITIK MULAI, bukan target akhir)

✅ SEMUA VERIFIED dari `core/training_templates/`:

| Param | SDXL | Flux | Qwen-Image | Z-Image |
|---|---|---|---|---|
| Toolchain | sd-scripts (TOML) | sd-scripts (TOML) | ai-toolkit (YAML) | ai-toolkit (YAML) |
| Dockerfile | `standalone-image-trainer` | `standalone-image-trainer` | `standalone-image-toolkit-trainer` | `standalone-image-toolkit-trainer` |
| rank (network_dim/linear) | 32 | **128** | 32 | 32 |
| alpha | 16 | **128** | 32 | 32 |
| conv / conv_alpha | — | — | — | **16 / 16** |
| learning rate | **1e-5** | 5e-5 (unet) | 1e-4 | 1e-4 |
| optimizer | AdamW8Bit | **Adafactor** | adamw8bit | adamw8bit |
| scheduler | constant | constant | — | — |
| epoch | 10 | 100 | — | — |
| steps (max/target) | 1600 | 3000 | 3000 | 2000 |
| repeats | **10** | **1** | (pakai steps) | (pakai steps) |
| resolution | 1024,1024 | (template) | (template) | (template) |
| batch | 4 | 1 | (template) | (template) |
| noise_scheduler | — | flowmatch | flowmatch | flowmatch |
| timestep_type | — | — | **weighted** | **weighted** |
| min_snr_gamma | **5** (aktif) | absen | — | — |
| discrete_flow_shift | — | **3.1582** | absen | absen |
| base/arch khusus | — | — | qwen_image, quant uint3 | zimage:turbo + adapter v2 |

**Catatan kritis baseline:**
- SDXL LR **1e-5** (bukan 1e-4 spt klaim riset lama). Beda 10×.
- Flux rank **128/128** (besar banget vs SDXL 32/16). Adafactor, bukan AdamW.
- Z-Image satu-satunya yang punya conv (16/16) di baseline.
- Qwen quant uint3 — hemat VRAM, tapi perhatiin efek ke kualitas.

---

## 2. Koreksi Klaim Riset Lama (WAJIB diingat)

| Klaim lama | Status | Koreksi |
|---|---|---|
| SDXL LR 1e-4–2e-4 | ❌ KILLED | Dev pakai **1e-5**. Untuk unconditional push boleh naik, tapi mulai dari 1e-5. |
| Z-Image timestep_type sigmoid | ❌ KILLED | Z-Image = **weighted**. "sigmoid" itu punya Flux (`timestep_sampling`). |
| Qwen flow_shift 2.205 | ⚠️ UNVERIFIED | TIDAK ADA di source G.O.D. Mungkin default internal ai-toolkit — perlu cek source ai-toolkit, jangan hardcode. |
| min-SNR "inert di Flux" | ⚠️ koreksi narasi | Di Flux param-nya **nggak di-set sama sekali**, bukan "inert". Flow-matching nggak pakai min-SNR. |
| Window budget acak | ⚠️ koreksi | Jam **deterministik** dari jumlah pair (acak 10–50). Range 0.5/0.75/1.0h; Qwen +0.5h. BISA diprediksi dari ukuran dataset. |
| Dua Dockerfile ada di repo | ❌/⚠️ | Requirement nyata (const+routing+test), tapi file **ABSEN di base main**. Miner WAJIB sediain di `ops/docker/`. |
| LICENSE byte-match file "LICENSE" | ⚠️ | Nama file `LICENSE.md` (sha256 `6ad6353e...`), NOTICE (sha256 `3e316950...`). |

---

## 3. Matriks Recipe Final (Baseline Dev + Layer Unconditional)

**Filosofi:** mulai dari baseline dev (kolom §1), lalu terapkan "unconditional override" di bawah ke SEMUA arch. Delta per-subtype tetap dipakai tapi sekarang tunduk ke prinsip 75% no-text.

### 3a. UNCONDITIONAL OVERRIDE (berlaku lintas arsitektur) 📐 INFERRED — A/B wajib

| Lever | Nilai | Alasan |
|---|---|---|
| **caption_dropout_rate** | **0.30–0.50** (vs default komunitas 0.05) | Dorong konsep ke null-prompt path. Ini lever #1 buat term 0.75. |
| trigger word | **buang / generic caption** | Trigger nge-gate pengetahuan di balik token yg absen di no-text pass. |
| EMA | **off** | Standar LoRA; izinin overfit terkontrol yg kita mau. |
| caption | pendek/generic, atau single repeated caption | Bikin conditional path overlap sama unconditional. |
| timestep emphasis | **bias ke high-noise** | Eval mulai denoise dari titik noise tinggi (0.75–0.93). |
| overfit posture | **lean-in terkontrol** | Test set sedistribusi; rekonstruksi L2 reward bias-ke-dataset. Monitor mode collapse. |

### 3b. Delta per-subtype (di ATAS baseline + override)

Subtype TIDAK ADA di payload — harus diinfer runtime dari dataset signature. Cluster jadi 3 bucket:

**Bucket A — Aggressive-fit (person/face/object/concept):**
- Overfit cepat → step lebih sedikit, checkpoint di ~50–70% window, save tiap 250 step.
- SDXL: min_snr_gamma boleh turun ke ~1 (📐 INFERRED), conv OFF, rank 16–32.
- LR boleh sedikit naik dari baseline (SDXL 1e-5 → coba 2e-5..5e-5; monitor).

**Bucket B — Prior-preserving (style/scene):**
- Pakai seluruh window, repeats lebih banyak, LR lebih rendah.
- SDXL: min_snr_gamma 5 (baseline), conv ON (LoCon/LoHa) buat tekstur.
- caption_dropout boleh ke ujung atas (0.4–0.5) — style paling diuntungkan unconditional push.

**Bucket C — Logo/text specialist:**
- conv ON (edge tajam) di SDXL; di Qwen andelin native text rendering (jangan over-engineer network).
- caption = exact visible text; resolusi jaga 1024 (jangan downscale).
- Z-Image: conv baseline (16) udah bantu.

### 3c. Tabel ringkas per-arch × bucket (angka = titik mulai, bukan final)

| | SDXL | Flux | Qwen | Z-Image |
|---|---|---|---|---|
| **Aggressive** | rank16-32, α=rank/2, LR 2e-5..5e-5, snr_γ~1, conv off, dropout 0.3 | rank 64-128, LR 5e-5, Adafactor, flow_shift baseline, dropout 0.3 | rank 16-32, LR 1e-4, weighted, dropout 0.3 | rank 32, LR 1e-4, weighted, adapter v2, dropout 0.3 |
| **Prior-preserve** | rank 32-64, LR 1e-5, snr_γ 5, conv on, dropout 0.4-0.5, repeats↑ | rank 128, LR 5e-5, flow_shift↑ (high-noise), dropout 0.4 | rank 32, LR 1e-4, weighted, dropout 0.4, steps↑ | rank 32-64, LR 1e-4, De-Turbo opsi run panjang, dropout 0.4 |
| **Logo/text** | rank 16-32, conv ON, LR 1e-5, caption=exact text | rank 64-128, LoKr, exact text | rank 16-32, **andelin native text**, exact text, res 1024 | rank 32 + conv 16, exact text, res 1024 |

---

## 4. Window-Aware Step Solver (sekarang deterministik)

✅ VERIFIED mekanik (`validator/tasks/synthetics/diffusion.py:594-629`):
- `num_pairs = randint(10, 50)` → map deterministik → `hours = 0.5 + scale×0.5`, kuantisasi `ceil(h×4)/4` (step 15 menit).
- Range: SDXL/Flux/Z-Image = **0.5 / 0.75 / 1.0 h**. Qwen = **+0.5h → 1.0 / 1.25 / 1.5 h**.

**Implikasi:** window bisa diprediksi dari ukuran dataset pas zip dibuka. Solver:
```
1. Baca jumlah pair dari dataset → tentuin window (deterministik).
2. Timing 50 step pertama → ukur it/s aktual di H100.
3. target_steps = floor(it/s × window_detik × safety 0.80-0.85).
4. Bucket A: cap di knee (~1500-2500), save tiap 250.
   Bucket B: pakai semua step.
   Bucket C: prioritas resolusi > jumlah step.
5. Sisain waktu buat upload checkpoint ke /app/checkpoints/{task_id}/{repo_name}.
```

Throughput H100 (📐 ESTIMATE — kalibrasi live wajib): SDXL ~3-5 it/s @1024; Flux ~1.5-2.5; Z-Image ~1-1.5; Qwen ~0.4-0.7. **Qwen = risiko UNDERtraining** (paling lambat, window terpanjang justru karena itu). SDXL/Z/Flux = risiko OVERfitting.

---

## 5. Struktur Repo Wajib (gerbang lolos validator)

| Item | Status | Detail |
|---|---|---|
| `LICENSE.md` | ✅ | sha256 `6ad6353ec71a92944b5e97adc783c01564044eee65e628a04efc8505db9506f8` (11473 bytes). Nama `.md`! |
| `NOTICE` | ✅ | sha256 `3e316950fc1bc25c91a74614fa6ceb57e0ff14e20cf518fcc2628ee5f780469e` (369 bytes). |
| `ops/docker/standalone-image-trainer.dockerfile` | ❌ ABSEN | SDXL/Flux. **Miner wajib bikin.** Entrypoint `${BASE_MODEL}_train_network.py`. |
| `ops/docker/standalone-image-toolkit-trainer.dockerfile` | ❌ ABSEN | Qwen/Z-Image. **Miner wajib bikin.** ai-toolkit path. |
| Entrypoint parse CLI | ❌ ABSEN | Konsumen argparse `--model-type` dst nggak ada di main. **Miner wajib bikin.** |
| Port 7999 | ✅ | `uvicorn ... port=7999` (`miner/asgi.py:53`). |
| Source non-obfuscated | ✅ req | Nggak boleh .pyc/packed/minified. |

**CLI signature** ✅ VERIFIED (`trainer/runtime.py:174-190`):
```
--task-id --model --dataset-zip --model-type --expected-repo-name --hours-to-complete [--trigger-word]
```
Tidak ada `--cache` (mount volume). Output `/app/checkpoints/`. Env opsional `BASELINE_STATS_PATH`.

---

## 5b. Toolchain Pins (kunci Dockerfile) — hasil riset toolchain

**Prinsip:** tetap di jalur dev (kohya buat SDXL/Flux, ai-toolkit buat Qwen/Z-Image). Isi Dockerfile bebas, tapi pin commit SHA (jangan floating branch) biar reproducible pas validator re-clone.

### kohya / sd-scripts (SDXL + Flux) — `standalone-image-trainer.dockerfile`
| Item | Nilai | Status |
|---|---|---|
| Branch | **`sd3`** (BUKAN `main`) | ✅ — README main nyuruh Flux/SD3 user pakai `sd3`. Flux belum di-merge ke main. |
| Commit | pin HEAD `sd3` ~21 Des 2025 | 📐 — verifikasi SHA pas clone |
| torch / CUDA | `torch==2.6.0 torchvision==0.21.0` (cu124) | ✅ |
| LyCORIS | `lycoris-lora==3.3.0` (rilis 4 Okt 2025) | ✅ — fallback 2.2.0 kalau DoRA/LoKr gagal load di inference |
| Entrypoint | `sdxl_train_network.py` / `flux_train_network.py` | ✅ |
| Fitur dikonfirmasi | DoRA (`use_dora`/`dora_wd`), LoKr, LoHa, conv/LoCon, min_snr_gamma, caption_dropout_rate, Adafactor, AdamW8bit, bf16, latent cache, Flux TE cache, discrete_flow_shift, timestep_sampling sigmoid | ✅ |

⚠️ **Peringatan LyCORIS 3.x breaking change:** default `wd_on_output=True` ngubah norm DoRA/LoKr vs config lama. Validasi checkpoint DoRA/LoKr beneran ke-load di inference sebelum commit.
⚠️ **TE cache vs caption_dropout konflik di Flux:** enable `--cache_text_encoder_outputs` nge-disable caption augmentation (shuffle/dropout) + maksa `--network_train_unet_only`. **Nggak bisa dua-duanya bareng di Flux.** Karena strategi kita butuh caption_dropout agresif (term 0.75), pertimbangin MATIIN TE cache di Flux.

### ai-toolkit Ostris (Qwen + Z-Image) — `standalone-image-toolkit-trainer.dockerfile`
| Item | Nilai | Status |
|---|---|---|
| Source | **git clone `main`** (JANGAN PyPI — stale di 0.2.7 Nov 2024) | ✅ — repo live udah ~0.10.x |
| Commit | pin SHA ≥ Des 2025 (postdate Z-Image support) | 📐 |
| torch / CUDA | `torch==2.9.1 torchvision==0.24.1` (cu128) | ✅ |
| diffusers | **build from source** (pin commit) — Z-Image baru di-merge ke diffusers, belum ada di wheel rilis | ✅ |
| quant | optimum-quanto (qfloat8) + ARA uint3 | ✅ |
| Fitur dikonfirmasi | LoKr (`type:"lokr"`, `lokr_full_rank`, `lokr_factor:8`), conv, caption_dropout, timestep_type weighted & sigmoid, flowmatch, flow shift dinamis internal, `arch:zimage`+adapter v2, De-Turbo path | ✅ |

⚠️ **Gotcha offline PALING KRITIS:** container nggak ada internet runtime. ai-toolkit normal-nya download model index + adapter pas job start.
- **Bake `zimage_turbo_training_adapter_v2.safetensors` (~324MB)** ke image (path: `ostris/zimage_turbo_training_adapter/...`).
- Set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`.
- Base weights dari `/cache/models/<org--model>` (HF "/" → "--").
- Bake diffusers-from-source di build time.

### Config Z-Image (ai-toolkit) — verified dari working example
```yaml
arch: "zimage"
name_or_path: "Tongyi-MAI/Z-Image-Turbo"
assistant_lora_path: "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors"
noise_scheduler: "flowmatch"
optimizer: "adamw8bit"
quantize: true
```
Adapter v2 = de-distill Turbo sementara biar LoRA belajar konsep tanpa ngerusak 8-step. Dilepas pas inference (fix "Turbo drift"). De-Turbo = ganti transformer ke `ostris/Z-Image-De-Turbo`, train tanpa adapter.

### Sanity check: kenapa nggak ganti trainer
- Z-Image: ai-toolkit = reference trainer (Ostris yg bikin adapter-nya). Nggak ada keunggulan SimpleTuner.
- Qwen: ai-toolkit + ARA low-bit udah matang. Nggak ada gap decisive.
- SDXL/Flux DoRA: kohya `sd3` cover DoRA + full LyCORIS. OneTrainer nggak nawarin fitur yang hilang.
- **Kesimpulan: jalur dev cukup. Nggak ada alasan swap toolchain.**
- musubi-tuner: cover Qwen/Z-Image TAPI nggak cover SDXL, dan bukan jalur sanctioned. Cuma fallback kalau ai-toolkit rusak.



1. ⚠️ **caption_dropout default di template G.O.D** — belum kepegang. Cek `core/training_templates/*` semua arch. (Catatan: di ai-toolkit, `caption_dropout_rate` ada di blok dataset, default contoh 0.05 — konfirmasi nilai di template G.O.D.)
2. ✅ **TERTUTUP — flow_shift internal ai-toolkit buat Qwen.** ai-toolkit ngitung shift **dinamis internal** via `CustomFlowMatchEulerDiscreteScheduler.calculate_shift()` (resolution-dependent mu dari sequence length). MAKANYA config G.O.D Qwen nggak nyebut `flow_shift` — toolkit nurunin sendiri. **JANGAN set manual buat Qwen.** (Angka 2.205 yg dulu "hantu" = ekuivalen matematis shift native Qwen di resolusi native, cuma relevan kalau lo set manual di sisi kohya — bukan ai-toolkit.)
3. ⚠️ **Dua Dockerfile + entrypoint** — harus dibikin dari pemahaman framework, BUKAN nyontek repo juara (yg tersanitasi: `.gitignore` exclude folder `G.O.D`, requirements cuma docker+typer).
4. 📐 **Semua angka unconditional (dropout 0.3-0.5, snr_γ, LR naik)** — A/B di re-eval harness lokal (`run_evaluation`, task <7 hari, image `diagonalge/tuning_validator_diffusion`).
5. 📐 **Throughput H100 per arch** — ukur live, jangan percaya estimate.

---

## 7. Urutan Eksekusi yang Disarankan

1. **Tutup gap #1** (caption_dropout default di template — gap #2 flow_shift udah TERTUTUP) — Claude Code baca source lagi.
2. **Bikin gerbang lolos validator** (#5: Dockerfile + entrypoint + LICENSE/NOTICE) — pakai toolchain pins §5b. Tanpa ini, score 0.
3. **Runtime: classifier subtype + step solver deterministik.**
4. **Inject recipe** (matriks §3) dengan unconditional override.
5. **A/B di re-eval harness** — mulai dari lever caption_dropout (dampak terbesar ke 75% term).

**Decision rule menyeluruh:** optimasi komposit `0.25×text + 0.75×no_text` di held-out lokal, percaya angka L2, JANGAN nilai dari estetika. 1 unit perbaikan no-text = 3 unit text-guided.
