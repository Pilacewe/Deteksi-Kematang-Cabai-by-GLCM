# Deteksi Kematangan Cabai dengan GLCM & KNN

Aplikasi web ini mendeteksi tingkat kematangan cabai berdasarkan citra digital dengan metode Gray Level Co-occurrence Matrix (GLCM) untuk ekstraksi fitur dan K-Nearest Neighbors (KNN) sebagai classifier.

## Struktur Proyek

- **app.py**: Aplikasi utama (Flask) untuk menerima gambar unggahan dan menampilkan hasil prediksi.
- **glcm_feature_extraction.py**: Modul untuk ekstraksi fitur GLCM dari citra.
- **train_model.py**: Script untuk melatih model KNN berdasarkan dataset.
- **dataset/**: Folder yang berisi gambar cabai dengan 3 kelas: 
  - `matang_merah` (cabai matang, merah)
  - `matang_orange` (cabai setengah matang, orange)
  - `tidak_matang` (cabai belum matang, hijau)
- **templates/**: Folder untuk halaman HTML (index dan result).
- **static/uploads/**: Tempat penyimpanan sementara gambar unggahan.
- **model/**: Model KNN yang terlatih (dihasilkan oleh `train_model.py`).

## Cara Menjalankan

1. Kumpulkan gambar dataset sesuai struktur.
2. Install dependency dengan menjalankan:
