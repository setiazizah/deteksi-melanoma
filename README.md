# Melanoma Detection — Hybrid Feature + HHO + SVM

Kode implementasi thesis S2 Ilmu Komputer UGM:
**"Deteksi Melanoma Menggunakan Fitur Hybrid dan Harris Hawks Optimization"**

---

## Deskripsi

Pipeline klasifikasi melanoma (biner: melanoma vs non-melanoma) yang menggabungkan:
- **Fitur handcrafted**: Shape (area, perimeter, compactness, asymmetry), Tekstur (GLCM, LBP), Warna (histogram HSV/LAB, momen warna)
- **Fitur deep**: Embedding DenseNet121 (pretrained ImageNet)
- **Seleksi fitur**: Harris Hawks Optimization (HHO) — metaheuristik wrapper
- **Klasifikasi**: SVM (terbaik), Random Forest, MLP, XGBoost

Dataset utama: **ISIC 2019** dengan augmentasi melanoma dari ISIC 2016, 2017, 2024, dan HAM10000.

---

## Struktur Notebook

`final-melanoma-detection-v4.ipynb` (76 cells) mencakup:

### Setup & Data Loading (Cell 0–7)
- Import library (OpenCV, scikit-learn, TensorFlow/Keras, XGBoost, skimage)
- Konfigurasi path dataset ISIC 2019, ISIC 2024, ISIC 2017, ISIC 2016, HAM10000
- Load ground truth CSV dan mapping label melanoma

### Augmentasi Data Melanoma (Cell 8–37)
- Ambil data melanoma dari 6 sumber (ISIC 2016/17/24, HAM10000)
- Dedup berdasarkan `isic_id` (bukan path) untuk hindari duplikat lintas dataset
- Downsampling majority class (non-melanoma) agar rasio seimbang di train set

### Fungsi Inti Pipeline (Cell 38–42)
- Preprocessing multi-mode: Resize Only, Gaussian, CLAHE, DullRazor (hair removal)
- Segmentasi: K-means, GrabCut, Otsu thresholding
- Ekstraksi fitur: shape (regionprops), tekstur (GLCM + LBP), warna (histogram + momen), deep (DenseNet121)
- Fitness function HHO: `0.7×Recall + 0.2×Specificity + 0.1×(1 − FeatureRatio)`
- HHO wrapper feature selection

### Eksperimen 1 — Perbandingan Pre-processing (Cell 43–47)
Membandingkan 5 mode preprocessing terhadap performa klasifikasi (val + test):

| Mode | Keterangan |
|------|-----------|
| Resize Only | Baseline (bukan "tanpa preprocessing") |
| Resize + Gaussian | Tambah filter Gaussian blur |
| Resize + Gaussian + CLAHE | Tambah contrast enhancement |
| Resize + Gaussian + DullRazor | Tambah hair removal |
| Resize + Gaussian + CLAHE + DullRazor | Kombinasi lengkap |

### Eksperimen 2 — Perbandingan Segmentasi (Cell 48–50)
Membandingkan metode segmentasi: Otsu Thresholding dan K-means.

### Eksperimen 3 — Ablation Fitur (Cell 51–53)
6 skenario ablasi kontribusi kelompok fitur:

| Skenario | Fitur |
|----------|-------|
| Shape only | Area, perimeter, compactness, asymmetry |
| Texture only | GLCM (contrast, correlation, energy, homogeneity) + LBP |
| Color only | Histogram HSV/LAB + momen warna |
| Handcrafted | Shape + Texture + Color |
| Deep only | DenseNet121 embedding (512 dim) |
| Hybrid | Handcrafted + Deep (512 fitur total setelah HHO) |

### Eksperimen 4a — Hyperparameter Tuning (Cell 57–58)
Tuning 4 model dievaluasi di **validation set** (bukan test set):
- SVM: kernel, C, gamma
- Random Forest: n_estimators, max_depth
- MLP: hidden_layer_sizes, learning_rate
- XGBoost: n_estimators, max_depth, learning_rate

### Eksperimen 4b — Performa Final Model (Cell 61–63)
Evaluasi model terbaik (parameter dari 4a) di **test set**.
Metrik: Accuracy, Sensitivity (Recall), Specificity, Precision, F1, AUC.

### Revisi Penguji (Cell 64–75)
Analisis tambahan untuk menjawab pertanyaan penguji sidang:
- **Feature Importance** — XGBoost surrogate (karena SVM RBF tidak punya `feature_importances_` bawaan)
- **Threshold / Precision-Recall trade-off** — kurva PR dan analisis threshold optimal
- **Visual Error Analysis** — contoh citra TP/FP/FN/TN dari test set
- **Narasi interpretasi** bottleneck model (sensitivity vs precision)

---

## Dataset

| Dataset | Sumber | Keterangan |
|---------|--------|-----------|
| ISIC 2019 | Kaggle `isic-2019` | Dataset utama (train/test split) |
| ISIC 2024 | `setiaazizah/isic-2024` | Augmentasi melanoma train |
| ISIC 2017 | `setiaazizah/isic-2017` | Augmentasi melanoma train |
| ISIC 2016 | `setiaazizah/isic-2016` | Augmentasi melanoma train |
| HAM10000 | `setiaazizah/ham10000` | Augmentasi melanoma train |

Path lokal: `D:\S2\THESIS\DATASET\MELANOMA\`
Path Kaggle: `/kaggle/input/...`

---

## Parameter Utama

```python
# HHO
N_HAWKS  = 3
MAX_ITER = 2

# Fitness function
fitness = 0.7 * Recall + 0.2 * Specificity + 0.1 * (1 - FeatureRatio)

# SVM (model terbaik)
SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')

# Fitur setelah HHO
# Total: 512 fitur hybrid (handcrafted + DenseNet121 embedding)
```

---

## Cara Menjalankan

### Di Kaggle (Recommended)
1. Buka notebook di Kaggle
2. Add datasets: `isic-2019`, `setiaazizah/isic-2024`, `setiaazizah/isic-2017`, `setiaazizah/isic-2016`, `setiaazizah/ham10000`
3. Aktifkan GPU (T4 atau P100)
4. Run All

> **Perhatian:** Full pipeline membutuhkan ~1.5 hari runtime karena ekstraksi fitur DenseNet121 pada seluruh dataset.

### Lokal
```bash
# Pastikan dataset tersedia di D:\S2\THESIS\DATASET\MELANOMA\
# Install dependencies
pip install opencv-python numpy pandas scikit-learn tensorflow xgboost scikit-image scipy tqdm seaborn matplotlib

# Jalankan notebook
jupyter notebook final-melanoma-detection-v4.ipynb
```

---

## File

| File | Keterangan |
|------|-----------|
| `final-melanoma-detection-v4.ipynb` | Notebook utama (27 MB, dengan output gambar) |
| `final-melanoma-detection-v4-compressed.ipynb` | Versi kompresi (2.3 MB, gambar dikompres ke JPEG) |
| `compress_notebook.py` | Script kompresi notebook |
| `README.md` | Dokumentasi ini |

---

## Hasil Kunci

| Metrik | Nilai (ISIC 2019 Test Set) |
|--------|--------------------------|
| Sensitivity | ~59% |
| Fitur setelah HHO | 512 fitur |
| F1 Score | 0.517 |
| Model terbaik | SVM RBF |

> Sensitivity 59% pada ISIC 2019 adalah hasil yang wajar mengingat tingginya class imbalance dataset ISIC 2019 (melanoma <5% dari total).

---

## Konteks Akademik

- **Program:** S2 Ilmu Komputer, Universitas Gadjah Mada
- **Penulis:** Setia Mukti Azizah
- **Pembimbing:** Prof. Agus
- **Status:** Pasca sidang, sedang revisi penguji (Agustus 2026)
- **Kandidat publikasi:** IJ-AI / IJECE (Scopus Q2/Q3, IAES)
