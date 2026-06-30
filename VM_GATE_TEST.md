# VM_GATE_TEST.md — Uji Gerbang di VM (CPU-only)

**Tujuan:** buktiin dua Dockerfile **VALID** (build sukses) + entrypoint **jalan sampai recipe-guard berhenti bersih** (exit 0), TANPA training. Recipe kosong by design → `is_complete()` False → stop sebelum nyentuh torch/training.

**Bisa 100% di CPU.** Hasil audit import: nggak ada lib CUDA-dependent (torch/xformers/bitsandbytes/flash-attn/diffusers/transformers) yang ke-import sebelum recipe-guard. Build CPU-only; run TANPA `--gpus`; **nggak butuh** nvidia-container-toolkit buat gerbang ini. GPU cuma perlu nanti pas Fase 2 (training beneran).

> ⚠️ JANGAN isi recipe. Test ini gerbang doang. Kalau entrypoint sampai "RECIPE NOT FILLED" dan exit 0 = LOLOS.

---

## 0. Audit import (ringkas — referensi)

| Modul | Import | CUDA-dependent? |
|---|---|---|
| `scripts/image_trainer.py` | argparse, os, sys, zipfile, (subprocess **lazy** di baris 156, setelah guard) | ❌ tidak |
| `trainer/constants.py` | — | ❌ |
| `trainer/training_paths.py`, `trainer/runtime.py` | os | ❌ |
| `runtime/recipe.py` | dataclasses | ❌ |
| `runtime/step_solver.py` | math, dataclasses | ❌ |
| `runtime/subtype_classifier.py` | math, os, collections, dataclasses | ❌ |

`"accelerate"`/`"python3"` cuma string di `build_training_command` (bikin list arg), bukan import. **Kesimpulan: gerbang full-CPU.** Tidak ada lazy-import yang perlu diubah.

---

## 1. Prasyarat VM

- Docker terpasang (`docker --version`).
- Disk ephemeral lega (estimasi di §6: **~40 GB**). Root disk JANGAN dipakai buat data-root.
- Akses internet **pas build** (download torch wheels, clone sd-scripts/ai-toolkit/diffusers, bake adapter 324MB). Pas **run** nggak butuh internet.

---

## 2. Pindahin Docker data-root ke ephemeral

Biar layer/image gede (~40GB) nggak penuhin root disk. Ganti `/mnt/ephemeral` ke path ephemeral lo.

```bash
EPH=/mnt/ephemeral            # <-- ganti ke path ephemeral lo
sudo systemctl stop docker
sudo mkdir -p "$EPH/docker"
sudo tee /etc/docker/daemon.json >/dev/null <<EOF
{
  "data-root": "$EPH/docker"
}
EOF
sudo systemctl start docker
docker info | grep -i "docker root dir"   # harus nunjuk ke $EPH/docker
```

Alternatif tanpa ubah daemon (kalau nggak mau restart docker): bind-mount `$EPH/docker` ke `/var/lib/docker` sebelum start docker. Tapi cara daemon.json di atas paling bersih.

---

## 3. Clone repo + checkout branch

```bash
cd "$EPH"
git clone https://github.com/helterskeltersz/St-Peppers-Lonely-Hearts-Club.git
cd St-Peppers-Lonely-Hearts-Club
git checkout fase1-validator-gate
```

---

## 4. Build kedua Dockerfile (CPU-only)

Build context = root repo. **Tanpa `--gpus`** (build emang nggak pernah butuh GPU).

```bash
# A) SDXL + Flux (kohya sd-scripts @ sd3, torch cu124)
docker build -f ops/docker/standalone-image-trainer.dockerfile \
  -t stpeppers-image:gate .

# B) Qwen-Image + Z-Image (ai-toolkit @ main + diffusers source, torch cu128, offline adapter)
docker build -f ops/docker/standalone-image-toolkit-trainer.dockerfile \
  -t stpeppers-toolkit:gate .
```

**Yang harus lolos di build:**
- Image A: clone sd-scripts@`b8d1eb06…`, `pip install` requirements + torch 2.6.0 cu124 + lycoris-lora 3.3.0 → tanpa error.
- Image B: clone ai-toolkit@`4e505354…`, diffusers@`b549ca91…` from source, optimum-quanto, **bake `zimage_turbo_training_adapter_v2.safetensors` (~324MB)** → tanpa error. Bake ini DIUJI pas build (kalau gagal = build gagal, kebaca di sini).

> Catatan: offline env (`HF_HUB_OFFLINE=1`) baru aktif SETELAH adapter ke-bake. Jadi bake (yang butuh internet) jalan duluan, baru di-set offline. Urutan ini sengaja.

---

## 5. Run gerbang — 4 arah model-type

Routing: **image A** handle `sdxl`+`flux`; **toolkit B** handle `qwen-image`+`z-image`.

### 5.1 Siapin fake /cache (read-only mount, niru validator)

```bash
CACHE=$(mktemp -d)
mkdir -p "$CACHE/datasets" "$CACHE/models/dummy--model"

# Bikin dataset zip kecil buat tiap task (pakai python, `zip` sering nggak ada di VM).
python3 - "$CACHE" <<'PY'
import sys, zipfile, os
cache = sys.argv[1]
for tid in ["t_sdxl","t_flux","t_qwen","t_zimg"]:
    p = os.path.join(cache,"datasets",f"{tid}_tourn.zip")
    with zipfile.ZipFile(p,"w") as z:
        for i in range(1,13):                      # 12 pair dummy
            z.writestr(f"img_{i}.png", b"\x89PNG\r\n")   # bukan PNG asli; gerbang cuma hitung ekstensi
            z.writestr(f"img_{i}.txt", "a cat sitting\n")
print("fake datasets siap:", os.listdir(os.path.join(cache,"datasets")))
PY
```

### 5.2 Jalanin tiap arah (TANPA --gpus)

```bash
run_gate () {  # $1=image  $2=model-type  $3=task-id  $4=trigger(optional)
  echo "================= $2 ================="
  docker run --rm \
    -v "$CACHE:/cache:ro" \
    "$1" \
    --task-id "$3" \
    --model "dummy/model" \
    --dataset-zip "${3}_tourn.zip" \
    --model-type "$2" \
    --expected-repo-name "r_$3" \
    --hours-to-complete 0.75 \
    ${4:+--trigger-word "$4"}
  echo "EXIT=$?"
}

run_gate stpeppers-image:gate   sdxl        t_sdxl  zzz
run_gate stpeppers-image:gate   flux        t_flux
run_gate stpeppers-toolkit:gate qwen-image  t_qwen
run_gate stpeppers-toolkit:gate z-image     t_zimg
```

### 5.3 Expected output tiap arah

Harus nyampe baris guard ini + **EXIT=0**:

```
[image_trainer] ---STARTING IMAGE TRAINING (Fase 1 scaffold)---
[image_trainer] model_type=<type> model=dummy/model -> /cache/models/dummy--model
[image_trainer] Extracted dataset -> /dataset/images/<task>
[image_trainer] signature={'image_count': 12, ...}
[image_trainer] bucket=<...> confidence=0.4
[image_trainer] predicted_window_hours=0.75 (...)   # qwen-image: 1.25 (0.75 + 0.5)
[image_trainer] Wrote config skeleton -> /dataset/configs/<task>.(toml|yaml)
[image_trainer] RECIPE NOT FILLED — Fase 1 scaffold stops here (no training launched).
[image_trainer] Would run: <command routing yang BENER>
EXIT=0
```

Cek **"Would run"** ngeroute bener:
- `sdxl` → `accelerate launch ... /app/sd-scripts/sdxl_train_network.py --config_file ....toml`
- `flux` → `... /app/sd-scripts/flux_train_network.py ... .toml`
- `qwen-image` / `z-image` → `python3 /app/ai-toolkit/run.py ....yaml`

Dan ekstensi config: sdxl/flux = `.toml`, qwen/z = `.yaml`.

> Catatan window: qwen-image dapat **+0.5h** (jadi 1.25h utk 12 pair), tiga lainnya 0.75h. Itu benar, bukan bug.

---

## 6. Checklist lolos/gagal + estimasi disk

| Cek | Lolos kalau |
|---|---|
| Build image A | exit 0, image `stpeppers-image:gate` kebentuk |
| Build image B | exit 0, adapter 324MB ke-bake (liat log `baked adapter ->`) |
| Run sdxl | guard message + EXIT=0 + route sd-scripts/sdxl_train_network.py + .toml |
| Run flux | guard message + EXIT=0 + route flux_train_network.py + .toml |
| Run qwen-image | guard message + EXIT=0 + route ai-toolkit/run.py + .yaml + window 1.25h |
| Run z-image | guard message + EXIT=0 + route ai-toolkit/run.py + .yaml |
| Tidak ada crash CUDA | nggak ada `CUDA error`/`no kernel image`/`Found no NVIDIA driver` SEBELUM guard |

**Estimasi disk ephemeral (image + build cache):**
| Item | ~Ukuran |
|---|---|
| base `nvidia/cuda:12.4.1` + torch cu124 + sd-scripts deps (image A) | ~9–13 GB |
| base `nvidia/cuda:12.8.0` + torch cu128 + ai-toolkit + diffusers + quanto + adapter (image B) | ~14–20 GB |
| build cache / layer overhead | ~5–8 GB |
| **Total aman** | **~40 GB** |

Bersihin setelah selesai: `docker image rm stpeppers-image:gate stpeppers-toolkit:gate && docker builder prune -f && rm -rf "$CACHE"`.

---

## 7. Skenario kegagalan umum + cara baca

| Gejala (di log) | Penyebab | Aksi |
|---|---|---|
| `ERROR: Cannot install ...` / `version conflict` pas `pip install -r requirements.txt` | Dependency sd-scripts/ai-toolkit bentrok sama torch pin | Catat paket+versi yang bentrok. Coba fallback SHA lebih lama (§8). Jangan longgarin torch pin tanpa alasan. |
| `Found no NVIDIA driver` / `CUDA error` **pas build** | Ada langkah build yang manggil CUDA (harusnya nggak ada) | Report ke gw — build kita murni pip/clone/bake, harusnya nggak nyentuh GPU. |
| `no kernel image is available` / `CUDA error` **pas run, SEBELUM guard** | Sesuatu meng-import torch sebelum guard (regressi) | Report — audit §0 bilang harusnya nggak ada. Kirim log full. |
| Build B berhenti di langkah bake adapter (`hf_hub_download ... 404 / ConnectionError`) | Repo/nama file adapter berubah, atau no internet pas build | Cek `ostris/zimage_turbo_training_adapter` masih ada + nama file. Pastikan build punya internet. |
| Run B gagal `... offline ... cannot find file` | `HF_HUB_OFFLINE=1` bikin sesuatu nyari file yang belum ke-bake | **Untuk gerbang ini harusnya NGGAK kejadian** — path gerbang nggak import HF sama sekali (berhenti di guard duluan). Kalau muncul = ada import HF prematur, report. (Resolusi adapter offline beneran baru relevan Fase 2.) |
| `toml`/`yaml` skeleton nggak kebikin | dir `/dataset/configs` nggak ada | Dockerfile udah `mkdir`; kalau hilang, cek COPY/mkdir di Dockerfile. |
| Build kelar tapi run `exec format` / `permission denied` di entrypoint | `run_image_trainer.sh` nggak executable | Dockerfile udah `chmod +x`; cek line-ending CRLF (harus LF). |

---

## 8. Fallback SHA (kalau build pecah karena upstream bergerak)

Semua SHA udah jadi `--build-arg`. Override ke commit lebih lama (lebih stabil) tanpa ubah Dockerfile:

```bash
# Image A — kohya sd-scripts (branch sd3)
docker build -f ops/docker/standalone-image-trainer.dockerfile \
  --build-arg SD_SCRIPTS_SHA=<sha_sd3_lebih_lama> \
  -t stpeppers-image:gate .

# Image B — ai-toolkit + diffusers
docker build -f ops/docker/standalone-image-toolkit-trainer.dockerfile \
  --build-arg AI_TOOLKIT_SHA=<sha_ai_toolkit_lebih_lama> \
  --build-arg DIFFUSERS_SHA=<sha_diffusers_kompatibel> \
  -t stpeppers-toolkit:gate .
```

Cara nyari SHA alternatif (tanpa clone penuh):
```bash
git ls-remote https://github.com/kohya-ss/sd-scripts.git sd3
git ls-remote https://github.com/ostris/ai-toolkit.git main
git ls-remote https://github.com/huggingface/diffusers.git main
# atau liat history tanggal: buka commits page repo, ambil SHA ~Des 2025.
```
**Pasangan yang diketahui kompatibel (default sekarang, current HEAD Juni 2026):**
- `SD_SCRIPTS_SHA=b8d1eb067eba32bb105984678b97f05b11452940`
- `AI_TOOLKIT_SHA=4e50535478d59a6e418c4e153c6daa908ad240c5`
- `DIFFUSERS_SHA=b549ca91ac4a31350511ee5625043e5dc922fe3a`

Kalau diffusers terlalu baru bikin ai-toolkit pecah, turunin `DIFFUSERS_SHA` ke commit sezaman sama `AI_TOOLKIT_SHA` (cari diffusers commit di tanggal yang sama / sedikit sebelum SHA ai-toolkit).

---

## 9. Lapor balik ke gw

Setelah eksekusi, kirim:
1. Build A & B: sukses/gagal (+ log error kalau gagal).
2. Tabel 4 run: model-type → reached-guard? → EXIT code → route bener?
3. Disk ephemeral kepakai aktual.
4. Anomali apapun.

Kalau 2 build sukses + 4 run nyampe guard EXIT=0 = **gerbang VM-level LOLOS**, lanjut Fase 2 (isi recipe).
