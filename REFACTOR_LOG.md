# REFACTOR_LOG.md — Langkah 3 (Fase 1 scaffold → v3 engine)

Branch: `feat/v3-engine`. Semua nilai VERIFIED dari kode zayden (path:baris di `MATRIKS_RECIPE_v3.md`) atau INFERRED (ditandai). Belum di-commit (nunggu instruksi).

## Prinsip
Fase 1 berhenti di scaffold (recipe kosong) → refactor v3 mayoritas ADITIF. Fondasi zayden = titik mulai; edge = slot flag default OFF (A/B pas GPU).

## Perubahan

### Buang tesis-lama (Langkah 3.1)
- `runtime/recipe.py`: placeholder + komentar §3a (unconditional, dropout 0.30-0.50) → **deprecation shim**, re-export dari `recipe_engine`. Tesis lama hilang.
- `MATRIKS_RECIPE_FINAL_v2.md`: banner DEPRECATED di atas; disimpan sbg jejak.
- `runtime/subtype_classifier.py`: **disuperseksi** oleh `category_detection.py` (3-bucket + short-circuit trigger `:102` ditinggalkan). File lama dibiarkan (tak diimport lagi) — bisa dihapus nanti.

### Modul BARU (fondasi v3)
| File | Fondasi | Isi | Sumber |
|------|---------|-----|--------|
| `runtime/category_detection.py` | 6-kategori | keyword scoring, threshold=3, style short-circuit, trigger recovery ≥60% | VERIFIED zayden category_detection |
| `runtime/dedup.py` | F4 | dHash+colour ganda, hamming≤6, colour≤14, gate N≥15, floor 0.6N, dry_run | VERIFIED zayden_dedup |
| `runtime/caption.py` | F1/F2 | mode inc/tail/verbatim per kategori, trigger keep, budget 75/220, VLM guarded | VERIFIED auto_caption |
| `runtime/noise.py` | F5 | multires iter6/disc0.3/offset0 utk person+logo/social/design | VERIFIED zayden_recipe:233 |
| `runtime/recipe_engine.py` | Sumbu 1-3, F6/F8/F9 | cascade tabel→guard→generator, capacity preset, family split, Qwen formula | VERIFIED + INFERRED(F8) |
| `runtime/step_budget.py` | F7 | sublinear per-kategori (subject band ketat) | VERIFIED zayden:356 |
| `runtime/data/model_plans.json` | Sumbu 3 | tabel penuh 31×5×2 (x-override preserved) | dari zayden source |
| `runtime/data/model_plans.csv` | ref | tabel human-readable | parse |

### Sanity-guard (Sumbu 3, kunci audit)
`recipe_engine.sanity_ok()`: `optimizer∈{adamw,adamw8bit,lion} & unet_lr>1e-2 → reject → generator`. Menetralkan blue-pencil style/CH (ulr=1.0 @ AdamW) tanpa hardcode.

### Cascade (recipe_engine.resolve_recipe)
```
base di tabel[shape][model][talla]?
  ├─ ada → sanity_ok? → PAKAI baris tabel (source="table")
  │         gagal      → generator (source="generator(guard-reject;fam)")
  └─ tidak → generator (source="generator(fam)")
+ F9 capacity (kecuali x-override pin network_dim)
+ F5 noise per kategori
+ F2/F3 caption reg (keep_tokens=1, shuffle, dropout 0.2/0.1)
```
Output overlay bawa `_source/_talla/_shape` buat trace validasi.

### Entrypoint (scripts/image_trainer.py)
Pipeline lama (classify→skeleton kosong) → v3: **dataset → dedup → detect kategori+trigger → caption → window plan → resolve recipe (sdxl/flux overlay | qwen/z aitoolkit) → write config**. Gate baru: `ALLOW_TRAINING` env (default OFF) → laptop/CPU resolve+tulis config, TAK launch training (validasi $0). Config writer: TOML overlay (sdxl/flux) / JSON (qwen/z).

## Status vs v3 spec
- ✅ F1-F9, Sumbu 1-3, cascade+guard, output per-arch: **ter-code**.
- Slot EDGE (2.7): BELUM (default OFF, A/B pas GPU) — sesuai rencana.
- VLM generation loop (caption tail/inc) & throughput measure: STUB → wiring GPU (Langkah 5).

## ⚠️ BLOCKER eksekusi
Laptop **tidak punya Python** (cuma stub Store; tak ada py/python3/conda). Kode v3 SELESAI ditulis, tapi **Langkah 3b (validasi routing) & Langkah 4 (jalankan harness) butuh interpreter** — tak bisa dieksekusi di laptop apa adanya. Opsi di bagian bawah laporan chat.
