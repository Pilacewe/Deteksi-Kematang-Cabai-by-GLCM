# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import joblib
from werkzeug.utils import secure_filename
from glcm_hsv_feature_extraction import extract_combined_features
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Muat model KNN
model_path = 'model/knn_model.pkl'
model = joblib.load(model_path)

# Mapping label ke nama kelas
label_map = {
    0: 'Cabai Matang (Merah)',
    1: 'Cabai Setengah Matang (Orange)',
    2: 'Cabai Matang Hijau',
    3: 'Cabai Tidak Matang'
}

def save_sobel_image(image, save_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    cv2.imwrite(save_path, sobel_combined)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Validasi input
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Simpan gambar
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Baca gambar dan ekstrak fitur GLCM
    image = cv2.imread(filepath)
    image = cv2.resize(image, (256, 256))
    features = extract_combined_features(image)

    # Deteksi Sobel dan simpan
    sobel_filename = 'sobel_' + filename
    sobel_path = os.path.join(app.config['UPLOAD_FOLDER'], sobel_filename)
    save_sobel_image(image, sobel_path)

    # Prediksi dengan KNN
    prediction = model.predict([features])[0]
    result = label_map.get(prediction, "Tidak Diketahui")

    # Persiapan fitur GLCM untuk ditampilkan ke result.html
    # Ambil hanya 8 fitur awal (tanpa r_mean dan g_mean)
    feature_labels = [
    'GLCM Contrast', 'GLCM Homogeneity', 'GLCM Energy',
    'HSV Hue Mean', 'HSV Saturation Mean', 
    'HSV Value Mean'
]
    feature_values = [round(f, 4) for f in features[:6]]  # Ambil hanya 8 fitur


    return render_template(
        'result.html',
        result=result,
        image_path=filepath,
        sobel_path=sobel_path,
        features=zip(feature_labels, feature_values)
    )

if __name__ == '__main__':
    # Aktifkan debug saat pengembangan
    app.run(debug=True)
