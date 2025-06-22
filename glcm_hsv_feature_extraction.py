import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_combined_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    # Resize ke ukuran standar
    image = cv2.resize(image, (256, 256))

    # Konversi ke grayscale dan HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ======== GLCM Multi-angle (tekstur) dari grayscale =========
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))

    # ======== Masking cabai (Hue 0–60 untuk merah–orange–hijau muda) =========
    mask_red = cv2.inRange(hsv, (0, 40, 40), (5, 255, 255))      # merah
    mask_orange = cv2.inRange(hsv, (10, 40, 40), (25, 255, 255))  # orange
    mask_green = cv2.inRange(hsv, (25, 30, 30), (60, 255, 255))   # hijau muda
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_orange), mask_green)

    # Ambil pixel HSV dan RGB yang termasuk dalam mask
    masked_hsv = hsv[mask > 0]
    masked_rgb = image[mask > 0]

    if masked_hsv.size == 0 or masked_rgb.size == 0:
        # Fallback fitur dummy jika masking gagal
        hue_mean = hue_std = sat_mean = sat_std = val_mean = 0
        r_mean = g_mean = 0
    else:
        # ======== Statistik HSV =========
        hue_mean = np.mean(masked_hsv[:, 0])
        hue_std  = np.std(masked_hsv[:, 0])
        sat_mean = np.mean(masked_hsv[:, 1])
        sat_std  = np.std(masked_hsv[:, 1])
        val_mean = np.mean(masked_hsv[:, 2])

        # ======== Statistik RGB (R dan G Mean) =========
        r_mean = np.mean(masked_rgb[:, 2])  # OpenCV: BGR, jadi index 2 adalah R
        g_mean = np.mean(masked_rgb[:, 1])  # G

    # ======== Gabungkan Semua Fitur =========
    return [
        contrast,
        homogeneity,
        energy,
        hue_mean,
        hue_std,
        sat_mean,
        sat_std,
        val_mean,
        r_mean,
        g_mean
    ]
