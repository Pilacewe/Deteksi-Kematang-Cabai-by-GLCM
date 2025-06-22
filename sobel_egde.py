import cv2
import numpy as np

def apply_sobel(image_path, save_path='static/sobel_output.jpg'):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi tepi Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # Normalisasi ke rentang 0-255 dan ubah ke uint8
    sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_uint8 = sobel_normalized.astype(np.uint8)

    # Simpan hasil ke file
    cv2.imwrite(save_path, sobel_uint8)
