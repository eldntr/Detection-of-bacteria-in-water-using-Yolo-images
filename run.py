from ultralytics import YOLO

def main():
    # Inisialisasi model YOLOv8
    model = YOLO('yolov8s.pt')  # Pre-trained model weights untuk YOLOv8

    # Mulai pelatihan
    model.train(
        data='data.yaml',  # Path to dataset configuration (data.yaml)
        epochs=100,                # Jumlah epoch untuk pelatihan
        imgsz=640,                 # Ukuran gambar untuk input (contohnya 640x640)
        batch=16,                  # Ukuran batch
        name='yolov8_generated_augmented',     # Nama eksperimen pelatihan
        workers=4,                 # Jumlah worker (thread) untuk data loading
        project='runs/train',      # Direktori untuk menyimpan hasil pelatihan
        device=0                   # Tentukan GPU (misal: 0) atau gunakan 'cpu'
    )

    # Evaluasi model setelah pelatihan
    model.val()

    # Export model ke format lain (misal: TorchScript, ONNX, CoreML, TensorRT)
    model.export(format='onnx')  # atau format lain yang diinginkan

if __name__ == '__main__':
    main()
