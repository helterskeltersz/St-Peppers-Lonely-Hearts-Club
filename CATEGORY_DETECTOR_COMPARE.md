# CATEGORY_DETECTOR_COMPARE.md — Poseidon detector vs Classifier baru vs Riset

**Verifikasi, JANGAN ubah kode.** Path:baris dari `/root/poseidon-ref` (f-poseidon, TERUJI di turnamen) & repo baru `fase2a-recipe`.

> **Klarifikasi penting:** poseidon punya **DUA detektor terpisah**, bukan satu:
> 1. `scripts/core/category_detector.py` → **ai-toolkit (Qwen/Z)**: `logo/social/person/art`.
> 2. `scripts/image_trainer.py:detect_image_category` (+`is_person_dataset`) → **SDXL**: `style/logo/design/default`.
>
> Dua task nyata (`254dbded`, `7ea128c8`) **dua-duanya SDXL** → yang relevan = **detektor SDXL**, BUKAN `category_detector.py`. Gw bahas dua-duanya. Dataset asli **nggak ada di VM** → Task C = analisa logika (sesuai aturan).

---

## A — Cara kerja detektor poseidon (path:baris)

### A.1 `category_detector.py` (Qwen/Z) — caption keyword cascade

1. **Taxonomy** (`category_detector.py:61-84`): `logo` | `social` | `person` | `art`.
2. **Sinyal = CAPTION keyword, word-level, fraction ≥ 0.6** (`:49-58`, `FRACTION_THRESHOLD=0.6` `:38`). BUKAN image/CLIP/OCR — murni baca `.txt`.
   - LOGO_KEYWORDS = `("logo",)` (`:32`)
   - SOCIAL_KEYWORDS = `("headline","body","layout","cta")` (`:33`)
   - HUMAN_KEYWORDS = `wearing/smiling/portrait/headshot/his/her/man/woman/suit/shirt/face` (`:34-37`)
   - **Word-level matching, BUKAN substring** (`:54-56`, `:21-24`) — sengaja, biar "his"≠"history", "man"≠"manager". (Ada test-trap.)
3. **Struktur = CASCADE berurutan, berhenti di match pertama** (`:65-84`): logo → social → person → art. Urutan disengaja: "yang paling bahaya-jika-kelewat duluan" (`:65`); logo/social SEBELUM person biar logo/social yg kebetulan punya trigger nggak ke-treat person (`:70-71`).
4. **Trigger handling** (`:81-83`): trigger dipakai **HANYA buat person**, dan **necessary-but-not-sufficient** — person butuh `trigger ADA` **DAN** `human keyword ≥60%`. Logo/social **NGGAK lihat trigger sama sekali** (caption keyword doang). → **trigger BUKAN decisive**, cuma 1 syarat dari 2 buat satu kategori.
5. **Fallback** (`:74-75, :84`): no caption → `art`; nggak match apa-apa → `art`. **`art` = fail-safe sengaja** (recipe art aman: EMA-on, step besar) (`:18-19`).
6. **Validasi empiris** (komentar `:11-13`): LOGO keyword "kebukti 100% di 4 task logo lintas turnamen; non-logo ~0%, product 1 false-pos/32 = 3% < threshold". SOCIAL "11/11 di task social". → **caption-keyword di-validate di task NYATA.**

### A.2 Detektor SDXL (`image_trainer.py`) — relevant buat 2 task nyata

1. **Taxonomy** (`detect_image_category` `:154-164`): `style` | `logo` | `design` | `default`.
2. **Sinyal = trigger (gate) + caption REGEX fraction ≥0.6** (`CATEGORY_FRACTION_THRESHOLD=0.6` `:127`):
   - **trigger ABSENT → `style`** (`:156-157`) — langsung, route DoRA.
   - LOGO_PATTERNS (regex: vector/brand mark/wordmark/monogram/logotype/"X logo") (`:129-131`) ≥0.6 → `logo` (`:160-161`).
   - DESIGN_PATTERNS (UI/UX/wireframe/dashboard/website/landing page/screenshot/SaaS) (`:132-134`) ≥0.6 → `design` (`:162-163`).
   - else → `default` (`:164`).
3. **Person sub-detector** (`is_person_dataset` `:193-199`): person keyword ≥0.6 **AND** product keyword <0.2 → True. **Cuma buat milih `caption_dropout 0.10` di route `default`** (`image_trainer.py:467-469`); **network tetap plain-64**. PRODUCT_EXCLUDE_PATTERNS (`:182-188`) lindungi product dari ke-treat person.
4. **Struktur:** cascade (style-gate → logo → design → default) + sub-check person. Fail-safe = `default` (plain LoRA-64, aman; DoRA dihindari krn "DoRA-default = blunder product -26%" `:126`).
5. **Trigger handling:** trigger ABSENT→style (decisive ke style). trigger PRESENT → caption regex yg nentuin (logo/design/default). **Trigger nggak maksa satu kategori** kalau present.

**Status:** kedua detektor **LENGKAP, nggak tersanitasi** — pure function (`os`+`re`), bisa unit-test CPU.

---

## B — Head-to-head: poseidon vs classifier baru vs riset redesign

| Aspek | `category_detector.py` + SDXL detector (poseidon, **TERUJI R1**) | `subtype_classifier.py` (baru, **CACAT**) | Riset redesign (weighted voting + CLIP) |
|---|---|---|---|
| **taxonomy** | Qwen/Z: logo/social/person/art · SDXL: style/logo/design/default (`category_detector.py:61`, `image_trainer.py:154`) | 3-bucket: aggressive-fit/prior-preserving/logo-specialist | 4-kategori: person/object/style/logo |
| **sinyal** | **CAPTION keyword/regex, word-level, fraction ≥0.6** + trigger gate (`:38`, `:127`) | **IMAGE-based**: image_count/caption_entropy/trigger/subject_variance(CLIP→proxy)/ocr_area_ratio (`subtype_classifier.py:205`) | CLIP zero-shot + face + OCR + DINOv2 variance + caption |
| **struktur** | **cascade berurut + fail-safe**, urutan disengaja (`category_detector.py:65`) | cascade + **short-circuit trigger (CACAT)** (`subtype_classifier.py:231`) | weighted additive voting |
| **trigger** | **NON-decisive**: cuma 1 syarat utk person (butuh +human kw); logo/social/design abaikan trigger | **DECISIVE (bug)**: trigger present → aggressive, override variance (`:231`) | weak evidence (ballot kecil) |
| **logo dikenali via** | **caption kata "logo" ≥60%** (validasi 100%/4 task) | **OCR area ≥0.08** pada pixel (threshold tebakan) | OCR + CLIP zero-shot |
| **fallback** | **art/default = fail-safe SENGAJA** (recipe aman) (`:74-75,84`) | confidence dihitung tapi **nggak dipakai** (`image_trainer.py:124`) | low-conf → prior-preserving (safe) |
| **kalibrasi** | ✅ keyword di-tune dari caption turnamen nyata (`:11-13`) | ❌ threshold 📐 sintetis, CLIP belum aktif | ❌ teori |
| **terbukti?** | ✅ **jalan di R1** (lolos) | ❌ untested | ❌ untested |

**Beda paling fundamental:** poseidon = **caption-driven** (baca apa yang ditulis caption, divalidasi nyata); classifier baru = **pixel-driven** (OCR/CLIP-proxy, belum tervalidasi). Caption-driven lebih murah, deterministik, dan **udah kebukti**; tapi rapuh kalau caption jelek/generik (dan ironisnya strategi baru SENGAJA bikin caption generik + drop trigger → bisa ngerusak detektor caption-based).

---

## C — Analisa 2 task nyata (logika; dataset asli nggak ada)

Dua task = **SDXL photoreal** (dreamshaper-xl / RealVisXL). Detektor yang jalan = **SDXL detector**. Tanpa caption asli, gw analisa berdasar logika rute (JANGAN asumsi subtype dari nama dataset):

| | Poseidon SDXL detector | Classifier baru | Mana lebih masuk akal? |
|---|---|---|---|
| **254dbded** (base dreamshaper, ds `product_luminastream`) | Kalau ada trigger + caption product (bukan logo/design regex, sinyal product tinggi) → **`default`** = **plain LoRA-64 + prodigy size-aware + cd0.05**, person-check gagal (product exclude) → cd tetap 0.05. (`image_trainer.py:464-490`) | trigger present + OCR product-photo biasanya <0.08 → **`aggressive-fit`** = LoRA-32 + AdamW 2e-5 + cd0.30 + conv off. TAPI kalau packaging ada teks/logo → OCR bisa ≥0.08 → salah ke **`logo-specialist`**. | **Poseidon lebih aman**: rute `default` stabil & tervalidasi; classifier baru bisa flip ke logo gara2 teks di kemasan (OCR rapuh). |
| **7ea128c8** (base RealVisXL, ds `social_auraverse`) | SDXL detector **nggak punya kategori "social"** (itu cuma Qwen/Z). Social graphic → kemungkinan caption match DESIGN_PATTERNS sebagian (landing/screen) atau nggak → **`design`** (DoRA+conv4) **atau** **`default`**. Ambigu. | Social graphic sering **teks besar (headline)** → **OCR ≥0.08 → `logo-specialist`** (cd per-arch rendah, conv, flip-off). | **Dua-duanya berisiko**; tapi poseidon fail-safe ke default/design (recipe waras), classifier baru maksa logo-recipe ke task social → mismatch. |

**Catatan jujur:** poseidon **KALAH** dua task ini (tipis: 13.6% & 5.9%). Jadi rute poseidon **bukan jaminan menang** — tapi dia **route ke recipe yang waras & stabil** (default plain-64/prodigy/cd0.05). Kekalahan tipisnya kemungkinan dari **recipe-level** (DoRA/timestep lawan, lihat COMPARE_REPORT), BUKAN dari salah-kategori. Classifier baru, di sisi lain, bisa **salah-kategori dulu** (product/social → logo via OCR) sebelum recipe sempat ngomong.

---

## D — Penilaian & rekomendasi

### D.1 Poseidon lebih robust dari classifier baru?
**YA, untuk pemilihan kategori — di 3 hal:**
1. **Trigger non-decisive** (poseidon) vs **trigger short-circuit** (baru `:231`). Bug paling parah classifier baru NGGAK ada di poseidon.
2. **Fail-safe sengaja** (art/default = recipe aman) vs **confidence nggak dipakai** di baru.
3. **Sinyal tervalidasi** (keyword di-tune dari caption nyata, 100%/4 logo task) vs **threshold tebakan + CLIP belum aktif**.

**Di mana poseidon LEBIH LEMAH:**
1. **Caption-dependent total** — kalau caption generik/kosong → semua jatuh ke art/default (nggak ada sinyal image). Strategi baru (caption generik + drop trigger) **malah ngerusak** detektor caption-based ini.
2. **Logo via kata "logo" doang** — logo tanpa kata "logo" di caption kelewat (walau OCR bisa nangkep). Image-signal (OCR/CLIP) classifier baru **secara prinsip** nangkep ini (kalau threshold bener).
3. **Taxonomy beda Qwen/Z vs SDXL** (4 vs 4 beda) — nggak unified, dua codepath.

### D.2 Taxonomy mana cocok buat 3-bucket recipe kita?
- Poseidon 5-kat (style/logo/design/default/social) & riset 4-kat (person/object/style/logo) → **dua-duanya bisa di-map ke 3-bucket**, riset lebih pas:
  - person/object → **aggressive-fit** · style → **prior-preserving** · logo → **logo-specialist** · design/social → **logo-specialist** atau **prior** (perlu keputusan).
- Poseidon `default` (campur person+product+social) = **terlalu kasar** buat 3-bucket kita (product & person beda kebutuhan). Riset 4-kat lebih bersih buat di-map.

### D.3 Rekomendasi (ranked by confidence) — keputusan di Wen

| # | Opsi | Confidence | Alasan |
|---|---|---|---|
| **R1** | **Hybrid: logika deteksi poseidon (caption-keyword + fail-safe + trigger non-decisive) DI ATAS struktur voting riset** | **TINGGI** | Ambil yg TERUJI (keyword fraction tervalidasi, fail-safe, urutan cascade) + benerin kelemahannya (caption-only) dgn nambah image-signal (OCR/CLIP) sbg **ballot tambahan**, bukan pengganti. Voting additive nutup "caption jelek → buta". |
| **R2** | **Perbaiki classifier baru pakai prinsip riset** (buang short-circuit `:231`, trigger jadi weak ballot, pakai confidence utk fallback ke prior) | **SEDANG-TINGGI** | Murah (1 file), benerin bug utama. TAPI tetap pixel-only + threshold belum kalibrasi → masih perlu validasi nyata. |
| **R3** | **Adopsi detektor poseidon apa adanya + sempurnain** (port `category_detector.py`/SDXL detector ke repo baru, tambah safe-fallback riset) | **SEDANG** | Paling cepat dapat sesuatu yg TERUJI. TAPI caption-based bentrok sama strategi caption-generik repo baru; perlu reconcile dulu. |

**Rekomendasi utama: R1 (hybrid).** Pondasi = **logika poseidon yang terbukti** (caption keyword fraction ≥0.6 word-level, cascade berurut, fail-safe ke recipe aman, trigger non-decisive) — lalu **upgrade ke weighted voting** (riset) dengan nambah **OCR + CLIP zero-shot sebagai ballot** biar nggak buta saat caption jelek, dan **trigger diturunin jadi bukti lemah**. Ini gabungin: TERUJI (poseidon) + nutup-lubang (image signal) + aman (fallback prior).

**⚠️ Konflik yg harus Wen sadari:** strategi recipe baru (caption generik + **drop trigger**) **melemahkan SEMUA detektor caption/trigger-based** (poseidon maupun hybrid). Kalau caption sengaja digenerik-in SEBELUM deteksi, sinyal kategori ilang. → urutan pipeline penting: **deteksi kategori DULU (pakai caption asli), baru caption-strategy di-apply**. Repo baru sekarang: classify (`image_trainer.py:122`) jalan SEBELUM prepare_captions (`:133`) → ✅ urutan udah bener, tapi classifier baru nggak baca caption keyword (cuma entropy), jadi sinyal caption asli itu **belum dimanfaatin**.
