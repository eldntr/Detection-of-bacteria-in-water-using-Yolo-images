# Cara Menjalankan:
# pastikan udah install ultralytics dan model yolo yang udah di train (yolo11n.pt/yolo8n.py)

import cv2
from ultralytics import YOLO

# 1. Muat model YOLOv8 yang telah dilatih
model = YOLO('yolo11n.pt')

# 2. Buka video input
video_path = 'Microorganism.mp4'
cap = cv2.VideoCapture(video_path)

# 3. Definisikan writer untuk menyimpan hasil video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Deteksi objek di setiap frame
    results = model(frame)

    # 5. Gambarkan kotak pembatas dan label pada frame
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Gambarkan kotak pembatas
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Tambahkan label kelas dan confidence score
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Simpan frame yang telah diproses ke dalam video output
    out.write(frame)

# 6. Tutup semua stream setelah selesai
cap.release()
out.release()
