import cv2
import numpy as np

# 1. Buka video input
video_path = 'Microorganism.mp4'
cap = cv2.VideoCapture(video_path)

# 2. Definisikan writer untuk menyimpan hasil video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_combined.mp4', fourcc, 30.0, (int(cap.get(3)) * 2, int(cap.get(4))))

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale

    # Background Subtraction dengan Adaptive Thresholding
    image_bg_subtracted = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2)

    # Konversi kembali ke RGB untuk menggambar bounding box berwarna
    image_rgb = cv2.cvtColor(image_bg_subtracted, cv2.COLOR_GRAY2RGB)

    return image_rgb

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Preprocessing frame
    processed_frame = process_image(frame)

    # 4. Gabungkan frame asli dan frame yang telah diproses secara horizontal
    combined_frame = np.hstack((frame, processed_frame))

    # 5. Simpan frame yang telah digabungkan ke dalam video output
    out.write(combined_frame)

# 6. Tutup semua stream setelah selesai
cap.release()
out.release()