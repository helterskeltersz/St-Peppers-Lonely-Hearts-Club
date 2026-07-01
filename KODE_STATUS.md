# KODE_STATUS.md — Baseline Refactor (Langkah 2)

Potret kode repo `St-Peppers-Lonely-Hearts-Club` @ `7ac886e` (Fase 1, 2026-06-30) SEBELUM refactor v3.
Read-only survey. Tanggal 2026-07-01.

## Ringkasan: repo = SCAFFOLD Fase 1 (wiring jalan, recipe logic masih stub)

VM-gate hijau, tapi **tidak ada satu pun hyperparameter/recipe nyata** — semua ditahan untuk "Fase 2". Artinya refactor v3 mayoritas **ADITIF** (isi placeholder), bukan bongkar tesis-lama (tesis lama belum sempat di-code).

## Komponen — ADA vs BELUM

| Komponen v3 | Status | File / bukti |
|-------------|--------|--------------|
| Entrypoint wiring | ✅ ADA | `scripts/image_trainer.py:115` main: dataset→classify→step_plan→config skeleton→(stop krn recipe kosong) |
| Subtype classifier | ⚠️ ADA tapi **3-bucket cacat** | `runtime/subtype_classifier.py` |
| Recipe engine | ❌ PLACEHOLDER kosong | `runtime/recipe.py` semua field None, `resolve_recipe` return `Recipe()` |
| Step solver (window) | ✅ ADA | `runtime/step_solver.py:36` deterministik dari num_pairs |
| Step budget per-kategori | ❌ BELUM | step_solver window-based saja; tak ada budget sublinear F7 |
| Category detection (6-kat) | ❌ BELUM | tak ada; yang ada 3-bucket |
| Dedup (dHash+colour) | ❌ BELUM | tak ada file |
| Caption engine (verbatim/tail/inc) | ❌ BELUM | tak ada file |
| Noise per-kategori (multires) | ❌ BELUM | tak ada file |
| model_plans tabel + sanity-guard | ❌ BELUM | tak ada |
| Docker/ops/trainer/miner/tests | ✅ ADA | `ops/docker/*`, `trainer/*`, `tests/` (VM gate) |

## Detail 3 file engine

### `runtime/subtype_classifier.py` — 3-bucket + short-circuit trigger (CACAT)
- Bucket: `AGGRESSIVE / PRIOR / LOGO` (3, bukan 6-kategori v3).
- **Short-circuit di `:102-105`:** `if not trigger_present → PRIOR else AGGRESSIVE`. Ini rule cacat (pakai kehadiran trigger sbg router utama) + warisan tesis lama ("trigger absent = style-leaning"). LOGO bucket tak pernah kepilih (butuh ocr_area_ratio yg di-stub).
- Sinyal implemented: image_count, caption_entropy, trigger_present. Stub (None): subject_variance, ocr_area_ratio.
- Confidence di-cap 0.4. → **HARUS DIGANTI** oleh 6-kategori keyword detector (fondasi zayden).

### `runtime/recipe.py` — PLACEHOLDER (semua None)
- `resolve_recipe()` return `Recipe()` kosong; `is_complete()` selalu False → entrypoint berhenti sebelum training sampah.
- **Sisa tesis-lama (di KOMENTAR/TODO, bukan kode aktif):**
  - `:6` "unconditional override §3a"
  - `:17` "matriks §1 + §3a unconditional override + §3b/3c"
  - `:27` `caption_dropout_rate ... §3a lever #1 (0.30-0.50)`
  - → hapus/replace saat isi v3. Tidak ada efek runtime (belum dipakai), tapi menyesatkan.

### `runtime/step_solver.py` — window solver JADI, throughput STUB
- `predict_window_hours()` ✅ deterministik (num_pairs→hours, +0.5h Qwen). VERIFIED dari G.O.D.
- `solve_target_steps()` ✅ floor(it/s × usable × safety 0.82).
- `measure_throughput()` = STUB (raise NotImplementedError) — wiring GPU Fase 2.
- **Catatan v3:** ini mengubah step jadi fungsi WINDOW+throughput. Zayden F7 pakai budget sublinear per-kategori (Qwen/Z) yang INDEPENDEN dari window. Perlu diselaraskan: SDXL/Flux pakai epoch (tabel), Qwen/Z pakai F7 budget — window solver jadi CAP atas, bukan penentu tunggal.

## Sisa tesis-lama: status
- **Kode AKTIF tesis-lama: TIDAK ADA** (recipe kosong; tak ada drop-trigger/dropout 0.3-0.5/UNet-only yang ter-eksekusi).
- **Warisan pasif:** (a) komentar §3a di recipe.py, (b) short-circuit trigger di classifier, (c) `MATRIKS_RECIPE_FINAL_v2.md` masih di root repo (matriks lama).
- → refactor v3 = ganti classifier, isi recipe, tambah 5 modul baru, bersihkan komentar, ganti matriks committed ke v3.

## Rencana refactor (Langkah 3) — urutan
1. Bersihkan tesis-lama: hapus komentar §3a di recipe.py; supersede MATRIKS v2 → v3.
2. `caption.py` (baru) — verbatim+tail/inc, trigger keep + keep_tokens=1.
3. `recipe_engine.py` / isi recipe.py — 3-sumbu + cascade + sanity-guard + `model_plans` loader.
4. `dedup.py` (baru) — dHash+colour, gate+floor.
5. Ganti `subtype_classifier.py` → `category_detection.py` 6-kategori keyword (buang short-circuit).
6. `noise.py` (baru) — multires per-kategori.
7. Extend step budget sublinear (F7) di step_solver / modul step_budget.

**Catatan git:** default branch = `main` (commit `7ac886e`). Refactor akan dikerjakan di branch terpisah (`feat/v3-engine`) — commit hanya saat diminta.
