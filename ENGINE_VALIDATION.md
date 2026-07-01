# ENGINE_VALIDATION.md — Langkah 3b (CPU, $0 GPU)

Engine v3 divalidasi di laptop pakai Python 3.12 + 2 dataset nyata Wen + kasus dummy.
Semua CPU. Bug routing 'David person->style' (stray 'aesthetic') DITEMUKAN & DIPERBAIKI (port style_detection 25%-threshold).

## 1. REAL DATASET routing (post-fix)
| dataset | N | category | shape | dedup | recipe source | rank | opt | ep | dropout |
|---------|---|----------|-------|-------|---------------|------|-----|-----|---------|
| 2ae5b33ce613 | 44 | person | subject | 44->44(rm0) | table | 32 | prodigy | 19 | 0.2 |
| 515e7013c71f | 36 | person | subject | 36->36(rm0) | table | 32 | prodigy | 19 | 0.2 |
| 66d58dd3b23e | 32 | person | subject | 32->32(rm0) | table | 32 | prodigy | 19 | 0.2 |
| 6ae185635ddf | 27 | style | style | 27->27(rm0) | table | 32 | adamw | 30 | 0.1 |
| b709e45e8b8f | 27 | logo | style | 27->27(rm0) | table | 32 | adamw | 30 | 0.1 |
| 39da7a07cf2b | 18 | logo | style | 18->18(rm0) | table | 32 | adamw | 35 | 0.1 |
| 11a4cf76e4aa | 9 | person | subject | 9->9(rm0) | table | 32 | prodigy | 73 | 0.1 |

## 2. CASCADE cases
| category | n | base | source | opt | ulr | rank | ep |
|----------|---|------|--------|-----|-----|------|-----|
| person | 15 | Lykon/dreamshaper-xl-1-0 | table | prodigy | 1.0 | 32 | 33 |
| style | 15 | Lykon/dreamshaper-xl-1-0 | table | adamw | 2.8e-05 | 32 | 35 |
| person | 15 | SG161222/RealVisXL_V4.0 | table | prodigy | 0.9 | 64 | 27 |
| style | 15 | zenless-lab/sdxl-blue-pencil | generator(guard-reject;anime) | adamw | 2.8e-05 | 32 | 32 |
| person | 12 | anon/unknown | generator(other) | prodigy | 1.0 | 32 | 33 |
| style | 40 | anime-illustrious/x | generator(anime) | adamw | 3.6e-05 | 32 | 18 |

## 3. Dummy category routing
| input | detected | shape |
|-------|----------|-------|
| logo | logo | style | OK
| social | social | style | OK
| design | design | style | OK
| product | product | subject | OK
| person | person | subject | OK
| style | style | style | OK

## 4. Qwen/Z adaptive
  qwen-image product n=10: rank=126 steps=158
  qwen-image product n=25: rank=139 steps=344
  qwen-image product n=45: rank=152 steps=567
  qwen-image logo n=10: rank=126 steps=700
  qwen-image logo n=25: rank=139 steps=913
  qwen-image logo n=45: rank=152 steps=1225
  z-image product n=10: rank=32 steps=173
  z-image product n=25: rank=32 steps=377
  z-image product n=45: rank=32 steps=621
  z-image logo n=10: rank=32 steps=800
  z-image logo n=25: rank=32 steps=1004
  z-image logo n=45: rank=32 steps=1347

## 5. TEMUAN & FIX
1. **BUG DITEMUKAN (fixed):** dataset person "David" (2ae5b33c) → salah route ke `style` karena kata "aesthetic" nyasar memicu regex style kasar. **Fix:** port `style_detection.py` asli winner (threshold real-style ≥25% caption). Post-fix → `person/subject/prodigy` benar. Ini persis nilai Langkah 3b: bug ketangkep $0 GPU.
2. **product→person (acceptable):** dataset produk (AuraGlow lamp: 66d58dd/515e7013) → label `person`, TAPI shape `subject/Prodigy` — sama seperti product. Recipe tetap benar (product & person dua-duanya Prodigy subject). Sesuai desain zayden (product mis-detect fallback aman ke person). Kandidat perbaikan label: hybrid image-vote router (Tier 3, A/B).
3. **Sanity-guard TERBUKTI:** blue-pencil style/CH → `generator(guard-reject)`, ep=32 (bukan 180 rusak), ulr aman.
4. **Dedup:** 0 removed di semua dataset turnamen (sudah beragam); gate N<15 benar skip set 9-gambar. Logic jalan, tak ada false-positive.
5. **Qwen/Z adaptif benar:** rank 126→152 skala N; subject-band step (product n=10=158) jauh < logo-band (n=10=700).

## STATUS: Langkah 3b LULUS (1 bug ditemukan & diperbaiki). Engine v3 route benar di data nyata + dummy. Siap Langkah 4 (harness).
