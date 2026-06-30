# FASE 1 — Laporan: Gerbang Validator + Runtime Scaffold

Branch: `fase1-validator-gate` · Acuan: `MATRIKS_RECIPE_FINAL_v2.md` · Source kebenaran: `/root/G.O.D` (rayonlabs @ `2939bf24`) · Oracle struktur: poseidon (`Keilrock/god-image-one@f-poseidon`).

**Aturan ditegakkan:** TIDAK ada recipe/hyperparameter konkret (Fase 2). Tidak nyalin kode training poseidon — cuma validasi struktur. Pin SHA (bukan floating branch).

---

## Langkah 0 — Gap verifikasi (§6)

| Gap | Status | Hasil |
|---|---|---|
| #1 caption_dropout default di template G.O.D | ✅ TERTUTUP | **Tidak di-set di template arch mana pun, dan tidak muncul di seluruh repo G.O.D.** `grep -rni caption_dropout core/training_templates/` → 0 match; grep seluruh repo → 0 match. Yang ada cuma `cache_text_embeddings`/`train_text_encoder`/`text_encoder_lr`. Artinya "default 0.05 komunitas" **bukan** dari G.O.D — G.O.D nyerahin ke default framework (kohya `caption_dropout_rate=0.0`). Jadi naikin ke 0.30–0.50 (§3a) = murni keputusan kita, bukan override angka G.O.D. |
| #2 flow_shift Qwen 2.205 | ✅ (sudah ditutup di matriks) | ai-toolkit ngitung dinamis internal (`calculate_shift`). Dockerfile B kasih komentar: JANGAN set manual buat Qwen. |

> Karena `MATRIKS_RECIPE_FINAL_v2.md` read-only di `/ephemeral`, update §6 gw catat di sini (file matriks di repo = salinan utuh). Pegangan Fase 2: caption_dropout source-default = **unset/0.0**.

---

## Langkah 1 — Gerbang lolos validator (DELIVERABLE UTAMA)

### Hasil test gerbang

| Gerbang | Hasil | Bukti |
|---|---|---|
| Dua Dockerfile ada di path resolver | ✅ LOLOS | `ops/docker/standalone-image-trainer.dockerfile` + `…-toolkit-trainer.dockerfile` (path DEFAULT). |
| Resolver contract (DEFAULT > LEGACY > error) | ✅ LOLOS | `pytest` → **8 passed**. Termasuk `test_errors_when_no_supported_path_exists` (mirror test G.O.D). |
| CLI signature persis | ✅ LOLOS | Smoke test 2 arah jalan: `--task-id --model --dataset-zip --model-type --expected-repo-name --hours-to-complete [--trigger-word]`. |
| Routing by model-type | ✅ LOLOS | sdxl→`accelerate launch …/sd-scripts/sdxl_train_network.py`; qwen-image→`python3 /app/ai-toolkit/run.py`. |
| Path I/O | ✅ LOLOS | baca `/cache/datasets/{task_id}_tourn.zip` + `/cache/models/<org--model>`; tulis `/app/checkpoints/{task_id}/{repo}`. Tidak ada flag `--cache`. |
| Window deterministik | ✅ LOLOS | 12 pair→0.75h (sdxl); 50 pair→1.5h (qwen, +0.5h). Sesuai §4. |
| `LICENSE.md` byte-match | ✅ LOLOS | sha256 `6ad6353e…` (dari base, BUKAN poseidon). |
| `NOTICE` byte-match | ✅ LOLOS | sha256 `3e316950…` (dari base). |
| Port 7999 | ✅ ADA | `miner/asgi.py` bind `127.0.0.1:7999` (mirror `miner/asgi.py:53` G.O.D). |
| Source non-obfuscated | ✅ | Semua `.py`/`.dockerfile` plaintext, no `.pyc`/packed. |
| Build container clone-and-run | ⚠️ BELUM DIUJI DI VM | Tidak ada Docker+GPU di mesin ini. From-source build (kohya sd3 + torch cu124 / ai-toolkit + cu128) **wajib diuji di VM H100**. Sesuai peringatan lo: jangan ekspektasi sekali jalan. |

### Toolchain pins (§5b) — yang dipakai di Dockerfile

| | kohya sd-scripts (A) | ai-toolkit (B) | diffusers (B) |
|---|---|---|---|
| ref | branch **`sd3`** | **`main`** | **`main`** (from source) |
| SHA | `b8d1eb067eba32bb105984678b97f05b11452940` | `4e50535478d59a6e418c4e153c6daa908ad240c5` | `b549ca91ac4a31350511ee5625043e5dc922fe3a` |
| status | 📐 current-HEAD (resolved 2026-06-30) | 📐 | 📐 |
| torch | 2.6.0 / tv 0.21.0 (cu124) | 2.9.1 / tv 0.24.1 (cu128) | — |
| lain | lycoris-lora==3.3.0 | optimum-quanto; adapter v2 di-bake; `HF_HUB_OFFLINE=1`+`TRANSFORMERS_OFFLINE=1` | — |

⚠️ **SHA = current HEAD (Juni 2026), bukan persis "21 Des 2025".** Keduanya postdate merge Flux-on-sd3 / Z-Image, jadi valid. Override gampang: `--build-arg SD_SCRIPTS_SHA=<sha>` kalau mau commit tanggal spesifik. (Matriks sendiri nandain 📐 "verifikasi SHA pas clone".)

### Cross-check struktur vs poseidon (oracle)

| Item | repo-baru | poseidon | Catatan |
|---|---|---|---|
| dockerfile image+toolkit | `ops/docker/` | `dockerfiles/` (legacy) | **Beda folder, sama nama file.** Dua-duanya diterima resolver (DEFAULT vs LEGACY). Gw pilih `ops/docker/` sesuai instruksi lo §5 + path DEFAULT modern. |
| entrypoint | `run_image_trainer.sh` → `image_trainer.py` | sama | ✅ pola identik. |
| argparse CLI | sama persis | sama persis | ✅ `--model-type` choices `[sdxl,flux,qwen-image,z-image]`. |
| LICENSE.md/NOTICE | hash **base** `6ad6/3e31` | hash **beda** `af05/556b` | ⚠️ **Poseidon MODIF LICENSE/NOTICE-nya** (juara tetap lolos → byte-match kemungkinan TIDAK strict di validator). Gw ikut spec lo: pakai hash base. **Keputusan lo:** mau byte-match base (aman per spec) atau niru poseidon? Gw rekomen base. |
| base image | from-source (cu124/cu128) | prebuilt `diagonalge/*` + vendored sd-script | **Beda strategi (sesuai instruksi lo).** Poseidon lebih ringan/cepat-build; from-source lebih auditable tapi berat. |
| import framework | standalone (no `validator.*`) | overlay (import `validator.utils.logging`, `G.O.D` gitignored) | Gw bikin **standalone bersih** sesuai "dari nol". Poseidon numpang G.O.D runtime. |

---

## Langkah 2 — Runtime scaffold (struktur, recipe kosong)

| Modul | Isi | Status |
|---|---|---|
| `runtime/subtype_classifier.py` | dataset signature → bucket (`aggressive-fit`/`prior-preserving`/`logo-specialist`). Implemented: image_count, caption_entropy, trigger_present. Stub (Fase 2): subject_variance (CLIP/face), ocr_area_ratio (OCR). Confidence di-cap 0.4 selama 2 sinyal berat masih stub. | ✅ struktur |
| `runtime/step_solver.py` | `predict_window_hours` (deterministik, mirror G.O.D :594-607), `solve_target_steps = floor(it/s × usable_s × 0.82)`, reserve 10% buat upload, checkpoint tiap 250. `measure_throughput` = stub (raise, jangan ngarang it/s). | ✅ struktur |
| `runtime/recipe.py` | `Recipe` dataclass (semua field `None`), `resolve_recipe()→Recipe()` kosong, `is_complete()`→False di Fase 1 (jadi entrypoint berhenti sebelum training beneran). | ✅ placeholder |

---

## Daftar Nyangkut buat FASE 2

1. **Isi `runtime/recipe.py`** dari matriks §1 baseline + §3a unconditional override + §3b/3c delta bucket. Lever #1 = `caption_dropout_rate` 0.30–0.50.
2. **Wire `measure_throughput`** ke step-callback trainer (50 step pertama) → isi `target_steps`.
3. **Lengkapi 2 sinyal classifier** (subject_variance via CLIP/face-embedding, ocr_area_ratio via OCR) + kalibrasi threshold bucket. Promosi ke `logo-specialist` butuh ocr_area_ratio.
4. **`create_config` beneran:** generate TOML (sd-scripts) / YAML (ai-toolkit) dari recipe. Flux: MATIIN TE-cache (konflik caption_dropout, §5b). Qwen: JANGAN set flow_shift.
5. **Uji build kedua Dockerfile di VM H100** (clone-and-run validator) — bagian paling berisiko, kemungkinan butuh iterasi (lib pin, CUDA, offline adapter resolve).
6. **Keputusan LICENSE/NOTICE**: konfirmasi byte-match base vs niru poseidon.
7. **Keputusan path**: `ops/docker/` (sekarang) vs `dockerfiles/` (poseidon legacy) — dua-duanya lolos resolver.

---

## Ringkasan status

- **Gate code-level: LOLOS** (8/8 test, 2 arah entrypoint jalan, byte-match ✅, port 7999 ✅, dua Dockerfile di path DEFAULT ✅).
- **Gate VM-level: BELUM** (build Docker + GPU run wajib diuji di VM — tidak bisa di sini).
- **Recipe: kosong by design** (Fase 2).
- **Belum di-commit** — nunggu aba-aba lo (push perlu PAT yang udah lo revoke; kasih PAT baru pas mau push).
