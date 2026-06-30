# FASE 0 — Verifikasi Source vs Riset Matrix

**Tujuan:** verifikasi klaim matriks recipe (SDXL / Flux / Qwen-Image / Z-Image) terhadap source asli G.O.D. **Tidak** menulis recipe / training code.

**Source yang dibaca:**
- Repo: `rayonlabs/G.O.D` (catatan: `github.com/gradients-ai/G.O.D` redirect ke sini — `gradients-ai` bukan org sebenarnya).
- Lokasi clone reference: `/root/G.O.D` (di luar repo submission — tidak di-commit).
- Branch: `main` @ `2939bf24` (`add execute permissions to autoupdate_auditor_steps (#1253)`).
- Semua kutipan = `path:baris` persis dari working tree, bukan parafrase.

**Legenda Status:**
`✅ cocok` · `❌ beda` · `⚠️ nggak ketemu di source` · `📋 nilai terverifikasi di source, TAPI angka klaim dari matriks lo belum gw pegang → lo wajib cross-check sendiri`

> ⚠️ **Catatan setup penting (baca dulu):** branch `f-poseidon` dan repo submission lo **TIDAK ADA** di mesin ini. Satu-satunya git repo di sistem = `/root/G.O.D` (reference rayonlabs). Jadi `git checkout f-poseidon && git checkout -b fase0-verify-source` **tidak bisa gw jalanin** — nggak ada targetnya. File ini gw taro di `/root/FASE0_VERIFY.md` dulu; nanti lo pindahin ke repo submission lo setelah repo-nya ada di sini. (Detail di bagian "Blocker Setup" paling bawah.)

---

## Tabel Claimed-vs-Verified

| # | Klaim (dari riset) | Arsitektur/Subtype | Status | Nilai asli di source | Path:baris | Catatan |
|---|---|---|---|---|---|---|
| a | SDXL pakai repeats lebih banyak dari Flux | SDXL vs Flux | ✅ | `DIFFUSION_SDXL_REPEATS = 10`; `DIFFUSION_FLUX_REPEATS = 1` | `trainer/constants.py:63-64` | Benar — SDXL 10× vs Flux 1×. Repeats hanya didefinisikan utk SDXL & Flux (sd-scripts). Qwen/Z-Image pakai `steps` langsung, bukan repeats. |
| b1 | default rank/alpha/LR — SDXL | SDXL | 📋 | rank(`network_dim`)=`32`, `network_alpha`=`16`, `learning_rate`=`1e-5`, `unet_lr`=`1e-5`, `text_encoder_lr`=`1e-5`, optimizer=`AdamW8Bit`, scheduler=`constant`, `epoch`=`10`, `max_train_steps`=`1600`, resolution=`1024,1024`, batch=`4` | `core/training_templates/base_diffusion_sdxl.toml:37,35,20,58,54,42,22,10,31,47,55` | Angka source jelas. Cocokkan dgn angka matriks lo. |
| b2 | default rank/alpha/LR — Flux | Flux | 📋 | rank(`network_dim`)=`128`, `network_alpha`=`128`, `unet_lr`=`5e-5`, `text_encoder_lr`=`[5e-5, 5e-5]`, optimizer=`Adafactor`, scheduler=`constant`, `epoch`=`100`, `max_train_steps`=`3000`, batch=`1` | `core/training_templates/base_diffusion_flux.toml:40,38,62,58,44,26,11,33,60` | Rank/alpha jauh lebih besar dari SDXL (128 vs 32/16). Optimizer beda (Adafactor vs AdamW8Bit). |
| b3 | default rank/alpha/LR — Qwen-Image | Qwen-Image | 📋 | rank(`linear`)=`32`, `linear_alpha`=`32`, `lr`=`1e-4`, optimizer=`adamw8bit`, `steps`=`3000`, `noise_scheduler`=`flowmatch`, arch=`qwen_image`, quant=`uint3` | `core/training_templates/base_diffusion_qwen_image.yaml:10,11,26,27,25,29,37,39` | Format ai-toolkit (YAML), bukan sd-scripts. LR 1e-4 (10× SDXL). |
| b4 | default rank/alpha/LR — Z-Image | Z-Image | 📋 | rank(`linear`)=`32`, `linear_alpha`=`32`, `conv`=`16`, `conv_alpha`=`16`, `lr`=`1e-4`, optimizer=`adamw8bit`, `steps`=`2000`, `noise_scheduler`=`flowmatch`, arch=`zimage:turbo` | `core/training_templates/base_diffusion_zimage.yaml:10,11,12,13,28,29,27,31,36` | Satu-satunya template yg punya `conv`/`conv_alpha`. Pakai `assistant_lora_path` (zimage_turbo adapter v2) — `:41`. |
| c | flow shift Qwen 2.205 | Qwen-Image | ⚠️ | **Tidak ada** param `flow_shift`/`discrete_flow_shift` di template Qwen. Nilai `2.205` tidak muncul di mana pun di repo. Qwen pakai `noise_scheduler: flowmatch` + `timestep_type: weighted`. Satu-satunya flow shift di repo = **Flux** `discrete_flow_shift = 3.1582`. | `core/training_templates/base_diffusion_qwen_image.yaml:29-30` (Qwen); `base_diffusion_flux.toml:9` (Flux 3.1582) | **KRITIS** — klaim 2.205 tidak terverifikasi. Kosong di source. Jangan masukin ke recipe sampai lo temuin source aslinya (mungkin default internal ai-toolkit, bukan di config G.O.D). |
| d | timestep_type sigmoid Z-Image | Z-Image | ❌ | Z-Image: `timestep_type: weighted` (BUKAN sigmoid). Kata "sigmoid" cuma muncul sbg `timestep_sampling = "sigmoid"` di **Flux** (sd-scripts), bukan Z-Image. | `core/training_templates/base_diffusion_zimage.yaml:32` (weighted); `base_diffusion_flux.toml:59` (sigmoid milik Flux) | **KRITIS** — klaim ketuker. Z-Image & Qwen sama-sama `timestep_type: weighted`. "sigmoid" itu punya Flux. |
| e | min-SNR gamma kepakai di SDXL tapi inert di Flux | SDXL & Flux | ⚠️ | SDXL: `min_snr_gamma = 5` (ADA & aktif) ✅. Flux: **tidak ada** key `min_snr_gamma` sama sekali di template (cuma `huber_schedule = "snr"`, beda hal). | `core/training_templates/base_diffusion_sdxl.toml:33` (ada); `base_diffusion_flux.toml` (absen; `:19` huber_schedule) | Bagian SDXL ✅. Bagian Flux: klaim "set-tapi-inert" **tidak akurat** — di Flux param-nya nggak di-set sama sekali, jadi nggak ada yg "inert". Secara fungsi min-SNR memang nggak relevan utk flow-matching, tapi source-nya literally absen, bukan inert. |
| f | formula scoring dual-L2 (text-guided + empty-prompt) | semua image | ✅ | `weighted_loss = 0.25 * text_guided_avg + 0.75 * no_text_avg`. L2 = `np.mean((test_norm - gen_norm)**2)` pada RGB ternormalisasi [0,1]. Per image: 10 seed (master_seed=42), 2 pass (use_prompt=True/False). Bobot: `DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT = 0.25`. | `validator/scoring/tasks.py:293-296`; `validator/evaluation/evaluators/diffusion.py:175-181,252-262`; `validator/evaluation/constants.py:17` | Dual-term L2 ✅. **Penting:** empty-prompt (no_text) dibobot **0.75** (lebih berat), text-guided cuma **0.25**. Tidak ada term perceptual/LPIPS (grep kosong). "Noise level" = `denoise` img2img per-model di `EVAL_DEFAULTS` (sdxl 0.9 / flux 0.75 / z-image 0.90 / qwen-image 0.93) — single level, bukan multi-noise. `validator/evaluation/constants.py:19-24`. |
| g | window `--hours-to-complete` dari budget acak | semua image | ⚠️ (nuance) | Hours **bukan** angka acak langsung. Alurnya: `num_pairs = randint(10, 50)` → dipetakan **deterministik** ke jam: `hours = 0.5 + scale*(1.0-0.5)`, dikuantisasi `ceil(h*4)/4` (step 15-menit). Range: **0.5 / 0.75 / 1.0 h** utk SDXL/Flux/Z-Image. Qwen: **+0.5 h** → **1.0 / 1.25 / 1.5 h**. | `validator/tasks/synthetics/diffusion.py:594-607,613,627-629`; `validator/tasks/synthetics/constants.py:61-62,70-72` | Konstanta: `MIN_IMAGE_SYNTH_PAIRS=10`, `MAX=50`, `MIN_IMAGE_COMPETITION_HOURS=0.5`, `MAX=1.0`, `QWEN_IMAGE_EXTRA_COMPETITION_HOURS=0.5`. Randomness ada di **jumlah pair**, bukan di jam-nya. Kalau matriks bilang "budget acak" → koreksi jadi "deterministik dari ukuran dataset yg acak". |
| h1 | LICENSE byte-match | repo struktur | 📋 | File namanya `LICENSE.md` (BUKAN `LICENSE`). `11473` bytes, sha256 `6ad6353ec71a92944b5e97adc783c01564044eee65e628a04efc8505db9506f8` | `/root/G.O.D/LICENSE.md` | Tidak bisa "byte-match" — repo submission lo nggak ada utk dibandingin. Ini hash reference utk lo cocokin nanti. Awas nama file: `.md`. |
| h2 | NOTICE byte-match | repo struktur | 📋 | `NOTICE`, `369` bytes, sha256 `3e316950fc1bc25c91a74614fa6ceb57e0ff14e20cf518fcc2628ee5f780469e` | `/root/G.O.D/NOTICE` | Copyright "Grads LLC" 2025, wajib atribusi ke Gradients.io. Hash reference utk byte-match. |
| h3 | dua Dockerfile (image + toolkit) | repo struktur | ❌/⚠️ | Path WAJIB (dipakai validator): `ops/docker/standalone-image-trainer.dockerfile` (SDXL/Flux) & `ops/docker/standalone-image-toolkit-trainer.dockerfile` (Qwen-Image/Z-Image). Routing dikonfirmasi di kode. **TAPI kedua file ini ABSEN di tree `main`** (yg ada cuma `miner-diffusion`, `validator-diffusion`, dll). | Ref: `trainer/constants.py:2-3`; routing `trainer/runtime.py:837+`; test `tests/trainer/test_runtime_dockerfile_paths.py:29-46`. Absen: `ops/docker/` (lihat ls) | Requirement-nya **nyata** (ada const + routing + test), tapi file fisiknya nggak di-commit di upstream main → besar kemungkinan **miner yg wajib nyediain** dua Dockerfile ini di repo submission. Ada juga fallback path lama `dockerfiles/standalone-*` (LEGACY, `constants.py:6-8`). |
| h4 | port 7999 | repo struktur | ✅ | `uvicorn.run(app, host="127.0.0.1", port=7999)` | `miner/asgi.py:53` | Confirmed. |

---

## Temuan Tambahan (di luar daftar minimal, tapi penting buat fase recipe)

**1. Signature CLI persis yang dipanggil validator ke image trainer** — `run_trainer_container_image()`, `trainer/runtime.py:174-190`:
```
--task-id <id>  --model <model>  --dataset-zip <zip>  --model-type <type>
--expected-repo-name <name>  --hours-to-complete <float>  [--trigger-word <word>]
```
- `--trigger-word` **opsional** (cuma ditambah kalau ada trigger_word) — `runtime.py:189-190`.
- **Tidak ada flag `--cache`.** `/cache` di-mount sbg docker volume read-only; output volume `checkpoints` di-mount rw ke `/app/checkpoints/`. — `runtime.py:217-220`.
- Env opsional `BASELINE_STATS_PATH` di-inject kalau ada baseline_stats — `runtime.py:193-198`.

**2. Path cache & output (authoritative)** — `trainer/constants.py:41-46`:
- `CACHE_ROOT_PATH = /cache`, `CACHE_MODELS_DIR = /cache/models`, `CACHE_DATASETS_DIR = /cache/datasets`, `HUGGINGFACE_CACHE_PATH = /cache/hf_cache`.
- `OUTPUT_CHECKPOINTS_PATH = /app/checkpoints/` (`:43`). Cocok dgn klaim output `/app/checkpoints`.

**3. Routing arsitektur → Dockerfile/toolkit** — `trainer/runtime.py:837+`:
- `Z_IMAGE` & `QWEN_IMAGE` → **toolkit** dockerfile (ai-toolkit, config YAML).
- `SDXL` & `FLUX` → **image** dockerfile (sd-scripts/kohya, config TOML, entrypoint `${BASE_MODEL}_train_network.py` — lihat `ops/docker/miner-diffusion.dockerfile:12`).

**4. Entrypoint image trainer (yg parse `--model-type` dst) TIDAK ADA di `main`.** Konsumen argparse utk flag2 ini nggak ditemukan sbg script standalone di tree (hanya `trainer/containers/downloader.py` yg parse subset arg downloader). Konsisten dgn h3: stack standalone-image-trainer (Dockerfile + entrypoint) belum di-commit di main — kemungkinan disediakan miner / ada di branch belum-merge.

**5. `core/models/task_models.py`** — field image task dikonfirmasi: `TaskType.IMAGETASK = "ImageTask"` (`:39`). Schema task lengkap (model_type, trigger_word, hours_to_complete, image_text_pairs, dll) ada di `validator/tasks/models.py` & `ImageRawTask` (`validator/tasks/synthetics/diffusion.py:632-644`), bukan di `task_models.py` (file ini cuma enum status/type + result models).

---

## Ringkasan

**Hitungan status (8 klaim minimal a–h, h dipecah 4 sub):**
- ✅ cocok: **3** → (a) repeats, (f) dual-L2, (h4) port 7999.
- ❌ beda: **2** → (d) timestep_type Z-Image (`weighted` bukan `sigmoid`), (h3) dua Dockerfile (requirement ada, file absen di main).
- ⚠️ nggak ketemu / nuance: **3** → (c) flow shift Qwen 2.205 (absen total), (e) min-SNR "inert di Flux" (di Flux nggak di-set sama sekali), (g) window hours (deterministik dari ukuran dataset, bukan acak langsung).
- 📋 perlu cross-check angka matriks lo: **6** → b1–b4 (rank/alpha/LR per arch), h1 (LICENSE hash), h2 (NOTICE hash).

**Paling kritis buat dikoreksi SEBELUM nulis recipe (urut prioritas):**
1. **(d) Z-Image `timestep_type` = `weighted`, BUKAN `sigmoid`.** Klaim ketuker sama Flux (`timestep_sampling="sigmoid"`). Salah set ini = recipe Z-Image langsung melenceng.
2. **(c) Flow shift Qwen `2.205` tidak ada di source.** Jangan tulis angka ini ke recipe — unverified. Qwen cuma punya `flowmatch` + `timestep_type: weighted`. Satu-satunya flow shift nyata = Flux `3.1582`.
3. **(h3) Dua standalone-image Dockerfile absen di `main`.** Sangat mungkin lo (miner) yang wajib nyediain `ops/docker/standalone-image-trainer.dockerfile` + `standalone-image-toolkit-trainer.dockerfile` di repo submission. Pastikan ada sebelum submit, atau validator gagal build (lihat test `test_get_dockerfile_path_errors_when_no_supported_path_exists`).
4. **(g) Reframe "budget acak".** Hours = deterministik dari jumlah pair (yg acak 10–50), range 0.5–1.0h (Qwen 1.0–1.5h), step 15 menit. Recipe harus asumsikan window pendek ini.
5. **(e) Koreksi narasi min-SNR Flux** dari "inert" → "tidak di-set di template Flux".

**Blocker Setup (perlu aksi lo):**
- Repo submission + branch `f-poseidon` belum ada di mesin ini → instruksi `git checkout f-poseidon && git checkout -b fase0-verify-source` belum bisa dieksekusi. Begitu lo clone/attach repo submission lo ke sini, kasih tau gw — gw setup branch-nya & pindahin `FASE0_VERIFY.md` ke sana.
- `/root/G.O.D` (reference rayonlabs) sengaja **tidak** di-commit ke mana pun, sesuai instruksi.
