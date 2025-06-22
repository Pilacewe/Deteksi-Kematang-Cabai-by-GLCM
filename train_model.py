# train_model.py
import os
import cv2
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from glcm_hsv_feature_extraction import extract_combined_features

# ✅ Update: Mapping kelas 4 kategori
data_dir = 'dataset'
categories = {
    'matang_merah': 0,
    'matang_orange': 1,
    'matang_hijau': 2,
    'tidak_matang': 3
}

X, y = [], []

print("⏳ Memproses dataset dan mengekstrak fitur GLCM + HSV...")
# Proses tiap folder dan ekstrak fitur
for category, label in categories.items():
    folder = os.path.join(data_dir, category)
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(filepath)
            if image is None:
                print(f"⚠️  Gagal membaca gambar: {filepath}")
                continue
            image = cv2.resize(image, (256, 256))
            features = extract_combined_features(image)
            X.append(features)
            y.append(label)
            print(f"✅ Processed: {filepath}")

# Ubah ke array numpy
X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Latih model KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Simpan model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/knn_model.pkl')
print("\n💾 Model berhasil disimpan di 'model/knn_model.pkl'")

# Evaluasi model
print("\n📊 Evaluasi model KNN pada data testing:")
y_pred = model.predict(X_test)

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
target_names = [
    'Cabai Matang (Merah)', 
    'Cabai Setengah Matang (Orange)', 
    'Cabai Matang Hijau', 
    'Cabai Tidak Matang'
]
print(classification_report(y_test, y_pred, target_names=target_names))
