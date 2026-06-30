# VM_GATE_TEST_REPORT.md — Hasil Eksekusi Gerbang di VM

**Eksekutor:** Claude Code (di VM ber-Docker) · **Branch:** `fase1-validator-gate` · **Tanggal:** 2026-06-30 (UTC).
**Verdict singkat:** ✅ **GERBANG VM-LEVEL LOLOS.** Dua Dockerfile build sukses tanpa konflik, keempat arah `--model-type` nyampe recipe-guard + exit 0 + route bener. **Tidak ada training** (recipe kosong by design).

---

## Lingkungan VM

| Item | Nilai |
|---|---|
| Docker | 27.2.1 (storage overlay2) |
| CPU / RAM | 4 vCPU / 3.8 GB |
| Disk | 1 filesystem `/dev/vda1` 97 GB (16 GB kepakai sebelum test) |
| GPU | **tidak ada** (sesuai — gerbang full-CPU) |

**§2 (data-root ke ephemeral): DILEWATI, beralasan.** `/ephemeral` ada di filesystem yang SAMA (`/dev/vda1`) dengan `/` — pindahin data-root nggak nambah ruang. Disk free 82 GB >> kebutuhan ~24 GB, jadi data-root dibiarkan default `/var/lib/docker`. Di VM lo yang punya disk ephemeral terpisah, tetap ikutin §2.

---

## Langkah 2 — Build kedua Dockerfile

| Image | Dockerfile | Hasil | Durasi (launch→image) | Ukuran |
|---|---|---|---|---|
| `stpeppers-image:gate` | `standalone-image-trainer` (SDXL/Flux, kohya sd3, cu124) | ✅ SUKSES | ~3 min (08:36:53→08:39:53) | **9.65 GB** |
| `stpeppers-toolkit:gate` | `standalone-image-toolkit-trainer` (Qwen/Z, ai-toolkit+diffusers source, cu128) | ✅ SUKSES | ~4.5 min (08:47:03→08:51:26) | **14.2 GB** |

> Build cepat karena network VM kenceng. Build = CPU-only, tanpa `--gpus`.

### Verifikasi anti-konflik (inti yang lo minta dibuktiin)

| Cek | Hasil |
|---|---|
| Dependency conflict / `ResolutionImpossible` / `incompatible` / `No matching distribution` | ❌ **NONE — clean** (grep dua log = nol; "liberror-perl" yg muncul itu nama paket Debian, bukan error) |
| torch pin image A | ✅ `2.6.0+cu124` (survived, nggak ke-override sd-scripts reqs) |
| torch pin image B | ✅ `2.9.1+cu128` (survived, nggak ke-override ai-toolkit reqs) |
| diffusers from-source | ✅ `0.38.0.dev0` (`.dev0` = git build, bukan wheel rilis) |
| ai-toolkit clone @ main | ✅ ke-install, reqs resolve tanpa bentrok torch |
| **Bake adapter 324MB** | ✅ `baked adapter -> /app/adapters/zimage_turbo_training_adapter_v2.safetensors` (**325 MB** di image, terverifikasi `ls`) |
| HF offline env aktif setelah bake | ✅ `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` di-set setelah bake (bake jalan online duluan) |

**Fallback SHA (§8): TIDAK PERLU dipakai.** Default pins (kohya `b8d1eb06…`, ai-toolkit `4e505354…`, diffusers `b549ca91…`) build bersih sekali jalan.

---

## Langkah 3 — Run 4 arah (entrypoint di DALAM container)

Mount `/cache` read-only berisi `{task}_tourn.zip` (12 pair dummy) + `models/dummy--model`. **Tanpa `--gpus`.**

| model-type | image | reached guard? | EXIT | route ("Would run") | config ext | window |
|---|---|---|---|---|---|---|
| `sdxl` | image | ✅ | **0** | `accelerate launch … /app/sd-scripts/sdxl_train_network.py` | `.toml` | 0.75h |
| `flux` | image | ✅ | **0** | `accelerate launch … /app/sd-scripts/flux_train_network.py` | `.toml` | 0.75h |
| `qwen-image` | toolkit | ✅ | **0** | `python3 /app/ai-toolkit/run.py` | `.yaml` | **1.25h** (0.75+0.5) |
| `z-image` | toolkit | ✅ | **0** | `python3 /app/ai-toolkit/run.py` | `.yaml` | 0.75h |

Tiap run berakhir di:
```
[image_trainer] RECIPE NOT FILLED — Fase 1 scaffold stops here (no training launched).
EXIT=0
```

**Bukti penting:** torch (cu124/cu128) ADA di dalam image tapi container jalan di host **CPU-only tanpa `--gpus`** dan **nggak ada CUDA error** sebelum guard. Mengonfirmasi audit import: nggak ada import GPU yang ke-eksekusi sebelum recipe-guard.

Catatan kecil: qwen-image dipanggil dengan `--hours-to-complete 0.75` tapi window *prediksi* dari ukuran dataset = 1.25h (12 pair + Qwen +0.5h). `hours_to_complete` (dari validator) dan window prediksi memang dua hal beda — keduanya kebaca bener.

---

## Disk & resource

| Item | Nilai |
|---|---|
| Disk kepakai sebelum test | 16 GB |
| Disk kepakai sesudah (2 image) | 38 GB → **delta ~22 GB** |
| Total ukuran image (dedup base layer) | 23.84 GB |
| Free tersisa | 60 GB |
| RAM | 3.8 GB cukup (build sukses; nggak OOM walau ketat) |

Estimasi §6 (~40 GB) **konservatif** — aktual ~22–24 GB. Aman.

---

## Anomali

- **Tidak ada anomali yang mempengaruhi gerbang.**
- Warning jinak (bukan error): `pip running as root`, dan `HF: sending unauthenticated requests` pas bake adapter (download publik, sukses 4.3s).
- False-positive grep "ERROR": baris apt `liberror-perl` (nama paket), bukan kegagalan.

---

## Kesimpulan & langkah berikut

✅ **Semua hijau:** build A+B sukses tanpa konflik, pins terjaga, adapter ke-bake, 4 run nyampe guard EXIT=0 dengan route benar. Gerbang VM-level **LOLOS**.

- Aman buat **PR `fase1-validator-gate` → `main`** (gw belum bikin PR — nunggu aba-aba lo sesuai aturan).
- Yang BELUM dibuktiin (memang di luar scope gerbang): training beneran (butuh GPU + recipe Fase 2), resolusi adapter offline saat ai-toolkit jalan (Fase 2), upload checkpoint ke `/app/checkpoints`.

Cleanup di VM (opsional): `docker image rm stpeppers-image:gate stpeppers-toolkit:gate && docker builder prune -f`.
