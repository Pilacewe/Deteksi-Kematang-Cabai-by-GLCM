<h1><b></b>🌶️ Deteksi Kematangan Cabai Menggunakan GLCM, HSV, dan KNN </b></h1><br>
Aplikasi web ini dirancang untuk mendeteksi tingkat kematangan cabai berdasarkan citra digital menggunakan metode GLCM (Gray Level Co-occurrence Matrix) untuk fitur tekstur, HSV (Hue, Saturation, Value) untuk fitur warna, serta visualisasi deteksi tepi Sobel. Klasifikasi dilakukan dengan algoritma K-Nearest Neighbors (KNN). <br>
<br>
🧠 Teknologi yang Digunakan<br>
Flask – Web framework Python untuk membangun aplikasi web<br>
OpenCV – Untuk pengolahan dan transformasi gambar<br>
scikit-learn – Untuk pelatihan dan prediksi menggunakan KNN<br>
scikit-image – Untuk fitur tekstur (GLCM) dan deteksi tepi Sobel<br>
Bootstrap 4 – Untuk tampilan antarmuka yang responsif dan modern<br>
<br>
<br>
📁 Struktur Proyek
app.py – Aplikasi utama berbasis Flask<br>
train_model.py – Script pelatihan model KNN menggunakan data cabai<br>
glcm_hsv_feature_extraction.py – Ekstraksi fitur GLCM, HSV, RGB, dan deteksi Sobel<br>
dataset/ – Folder dataset berisi 4 kategori:<br>
  matang_merah – Cabai matang merah<br>
  matang_orange – Cabai setengah matang orange<br>
  matang_hijau – Cabai matang hijau<br>
  tidak_matang – Cabai belum matang<br>
model/knn_model.pkl – Model KNN hasil pelatihan<br>
static/uploads/ – Penyimpanan gambar unggahan user dan deteksi Sobel<br>
templates/ – Folder untuk file HTML (index.html, result.html)<br>
<br>
<br>
🚀 Cara Menjalankan Aplikasi<br>
1. Clone repositori<br>
git clone https://github.com/username/proyek-deteksi-cabai.git<br>
cd proyek-deteksi-cabai<br>
2. Install dependencies
pip install -r requirements.txt<br>
3. Latih model terlebih dahulu<br>
(Pastikan dataset sudah berada dalam folder dataset/)<br>
python train_model.py<br>
4.Jalankan aplikasi Flask<br>
python app.py<br>
5. Buka di browser<br>
http://127.0.0.1:5000
<br>
<br>
📷 Cara Kerja Aplikasi<br>
1. Upload gambar cabai melalui halaman awal<br>
2. Sistem akan:<br>
 - Resize gambar menjadi 256x256<br>
 - Ekstraksi fitur GLCM (tekstur)<br>
 - Ekstraksi statistik HSV (warna)<br>
 - Visualisasi deteksi tepi Sobel<br>
3. Prediksi tingkat kematangan cabai menggunakan model KNN<br>
4. Hasil ditampilkan berupa:<br>
 - Nama kategori kematangan<br>
 - Gambar asli + Deteksi tepi Sobel<br>
 - Tabel fitur yang diekstraksi (GLCM & HSV)
<br>
<br>
📊 Output Klasifikasi<br>
Cabai Matang (Merah)<br>
Cabai Setengah Matang (Orange)<br>
Cabai Matang Hijau<br>
Cabai Tidak Matang<br>

