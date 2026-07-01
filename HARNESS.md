# HARNESS.md — Scoring Harness (Langkah 4)

Replikasi metrik skor validator biar checkpoint bisa dinilai **sebelum** A/B bakar GPU.

## Struktur
| File | Isi | CPU-testable? |
|------|-----|---------------|
| `harness/constants.py` | weighting 0.25/0.75, denoise per-arch, seed 42×10, anchor kalibrasi | ✅ |
| `harness/score.py` | `pixel_l2` (mse/rmse/mae), `weighted_score`, `aggregate` — **matematika murni** | ✅ |
| `harness/evaluator.py` | `evaluate_checkpoint` (loop 2-pass × seed), `holdout_split`, `compare_runs`; generasi di-inject via `reconstruct_fn` | ✅ struktur |
| `tests/test_harness_score.py` | 16 unit test matematika (dummy array) | ✅ 16/16 hijau |

## Metrik (VERIFIED dari matriks §0, sitir sumber validator)
```
score = 0.25 · mean(text_L2) + 0.75 · mean(notext_L2)      # lower = better
img2img reconstruction, denoise per-arch: sdxl 0.9 / flux 0.75 / z 0.90 / qwen 0.93
10 seed (master=42), 2 pass (prompt on/off) per held-out image
```

## ⚠️ STATUS KALIBRASI: BELUM (GPU-pending) — TIDAK diklaim dari dummy
Unit test memvalidasi **matematika**-nya benar (L2 identik→0, weighting 0.25/0.75, notext 3× leverage, split deterministik, orkestrasi 2-pass). Test **TIDAK** memvalidasi bahwa skala/reduction cocok dengan validator asli.

**Kalibrasi nyata = Langkah 5 (GPU):** jalankan Flux gs85 vs gs1 di model asli, cek harness mereproduksi urutan+skala anchor (`gs85 0.06490 < gs1 0.07098`). Baru setelah itu `reduction` (mse vs rmse) dikunci & `calibrated=True` boleh di-set. Sampai itu, `evaluate_checkpoint` selalu return `calibrated=False`.

## Pemakaian (GPU phase)
`reconstruct_fn(original, prompt, seed, denoise, model_type) -> image` dibungkus ke pipeline diffusion asli (LoRA loaded, img2img strength=denoise). Di test, `reconstruct_fn` = dummy (return original / offset) → tanpa model, tanpa GPU.
