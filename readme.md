🌶️ Deteksi Kematangan Cabai Menggunakan GLCM, HSV, dan KNN
Aplikasi web ini dirancang untuk mendeteksi tingkat kematangan cabai berdasarkan citra digital menggunakan metode GLCM (Gray Level Co-occurrence Matrix) untuk fitur tekstur, HSV (Hue, Saturation, Value) untuk fitur warna, serta visualisasi deteksi tepi Sobel. Klasifikasi dilakukan dengan algoritma K-Nearest Neighbors (KNN).

🧠 Teknologi yang Digunakan
Flask – Web framework Python untuk membangun aplikasi web
OpenCV – Untuk pengolahan dan transformasi gambar
scikit-learn – Untuk pelatihan dan prediksi menggunakan KNN
scikit-image – Untuk fitur tekstur (GLCM) dan deteksi tepi Sobel
Bootstrap 4 – Untuk tampilan antarmuka yang responsif dan modern

📁 Struktur Proyek
app.py – Aplikasi utama berbasis Flask
train_model.py – Script pelatihan model KNN menggunakan data cabai
glcm_hsv_feature_extraction.py – Ekstraksi fitur GLCM, HSV, RGB, dan deteksi Sobel
dataset/ – Folder dataset berisi 4 kategori:
  matang_merah – Cabai matang merah
  matang_orange – Cabai setengah matang orange
  matang_hijau – Cabai matang hijau
  tidak_matang – Cabai belum matang
model/knn_model.pkl – Model KNN hasil pelatihan
static/uploads/ – Penyimpanan gambar unggahan user dan deteksi Sobel
templates/ – Folder untuk file HTML (index.html, result.html)

🚀 Cara Menjalankan Aplikasi
1. Clone repositori
git clone https://github.com/username/proyek-deteksi-cabai.git
cd proyek-deteksi-cabai

2. Install dependencies
pip install -r requirements.txt

3. Latih model terlebih dahulu
(Pastikan dataset sudah berada dalam folder dataset/)
python train_model.py

4.Jalankan aplikasi Flask
python app.py

5. Buka di browser
http://127.0.0.1:5000

📷 Cara Kerja Aplikasi
1. Upload gambar cabai melalui halaman awal
2. Sistem akan:
 - Resize gambar menjadi 256x256
 - Ekstraksi fitur GLCM (tekstur)
 - Ekstraksi statistik HSV (warna)
 - Visualisasi deteksi tepi Sobel
3. Prediksi tingkat kematangan cabai menggunakan model KNN
4. Hasil ditampilkan berupa:
 - Nama kategori kematangan
 - Gambar asli + Deteksi tepi Sobel
 - Tabel fitur yang diekstraksi (GLCM & HSV)

📊 Output Klasifikasi
Cabai Matang (Merah)
Cabai Setengah Matang (Orange)
Cabai Matang Hijau
Cabai Tidak Matang

