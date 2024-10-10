# Detection of Bacteria in Water Using YOLO Image

jurnal: 
https://www.mdpi.com/2076-3417/13/22/12406

Dataset:
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9986282/
- https://www.kaggle.com/code/eldintarofarrandi/bacteria-in-water-detection/input


# YOLO-based Deep Learning to Automated Bacterial Colony Counting 

jurnal: 
https://www.researchgate.net/publication/366694140_YOLO-based_Deep_Learning_to_Automated_Bacterial_Colony_Counting

Evaluation Score:
- Mean Average Precision (mAP)

YOLO_data/
    ├── images/
    │   ├── train/       # Folder gambar untuk training
    │   └── val/         # Folder gambar untuk validation
    ├── labels/
    │   ├── train/       # Folder label untuk training (format .txt)
    │   └── val/         # Folder label untuk validation (format .txt)
    └── data.yaml        # File YAML untuk konfigurasi YOLO

Cara untuk training:
1. cd yolov5
2. python train.py --img 640 --batch 16 --epochs 50 --data ../data.yaml --weights yolov5s.pt --device 0