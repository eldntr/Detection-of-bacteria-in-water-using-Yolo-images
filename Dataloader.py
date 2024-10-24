import os
import random
import shutil
import albumentations as A
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path ke folder data
images_dir = "dataset/EMDS7/"
labels_dir = "dataset/EMDS7xml/"
output_dir = "temp/"
sample_output_dir = "sample_output/"
train_val_split = 0.8  # Rasio data train/validation

# Remove output directory if it exists and create fresh output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

if os.path.exists(sample_output_dir):
    shutil.rmtree(sample_output_dir)

# Buat ulang folder output yang fresh
os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

# Buat folder sample_output untuk menyimpan contoh gambar
os.makedirs(sample_output_dir, exist_ok=True)

# Daftar semua gambar dan annotations
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
label_files = [f.replace('.png', '.xml') for f in image_files]

# Shuffle data for randomness
data = list(zip(image_files, label_files))
random.shuffle(data)

# Split data into training and validation sets
split_index = int(train_val_split * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

class_mapping = {
    'G001': 0,  'G002': 1,  'G003': 2,  'G004': 3,  'G005': 4,  'G006': 5,  'G007': 6,  'G008': 7,  'G009': 8,  'G010': 9,
    'G011': 10, 'G012': 11, 'G013': 12, 'G014': 13, 'G015': 14, 'G016': 15, 'G017': 16, 'G018': 17, 'G019': 18, 'G020': 19,
    'G021': 20, 'G022': 21, 'G023': 22, 'G024': 23, 'G025': 24, 'G026': 25, 'G027': 26, 'G028': 27, 'G029': 28, 'G030': 29,
    'G031': 30, 'G032': 31, 'G033': 32, 'G034': 33, 'G035': 34, 'G036': 35, 'G037': 36, 'G038': 37, 'G039': 38, 'G040': 39,
    'G041': 40, 'G042': 41, 'G043': 42, 'G044': 43, 'G045': 44, 'G046': 45, 'G047': 46, 'G048': 47, 'G049': 48, 'G050': 49,
    'G051': 50, 'G052': 51, 'G053': 52
}

# Augmentasi gambar dan bounding box tanpa mirroring, mengisi area kosong dengan putih
augmentor = A.Compose([
    A.Rotate(limit=30, p=0.3),  # Rotasi hingga 360 derajat
    # A.RandomScale(scale_limit=0.2, p=0.8),  # Scaling secara acak
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),  # Shear dan rotasi acak
    A.Resize(512, 512, p=1.0)  # Menjaga ukuran asli dan menyesuaikan resolusi menjadi 512x512
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def fill_with_white(image, desired_size=(512, 512)):
    """Isi area kosong setelah augmentasi dengan warna putih."""
    h, w, c = image.shape
    result = np.full((desired_size[0], desired_size[1], c), 255, dtype=np.uint8)  # Buat background putih
    # Tempelkan gambar ke tengah
    result[:h, :w] = image
    return result

def draw_bboxes(image, bboxes, class_labels):
    """Gambar bounding box pada gambar."""
    for bbox, class_id in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        img_h, img_w = image.shape[:2]
        xmin = int((x_center - width / 2) * img_w)
        ymin = int((y_center - height / 2) * img_h)
        xmax = int((x_center + width / 2) * img_w)
        ymax = int((y_center + height / 2) * img_h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Hijau untuk bounding box
        cv2.putText(image, str(class_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def convert_xml_to_txt(xml_path, txt_output_path, augment=False, img=None):
    """Fungsi ini akan mengonversi format XML ke format YOLO (txt) dan mengabaikan objek dengan nama 'unknown'."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    bboxes = []
    class_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text

        # Abaikan jika class_name adalah 'unknown'
        if class_name == 'unknown':
            continue

        # Class ID mapping (sesuaikan dengan class yang ada)
        class_id = class_mapping.get(class_name, -1)

        if class_id == -1:
            continue  # Jika tidak ada dalam mapping, skip

        # Mendapatkan bounding box koordinat
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Konversi ke format YOLO (x_center, y_center, width, height) (normalized)
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        bboxes.append([x_center, y_center, box_width, box_height])
        class_labels.append(class_id)

    # Jika augmentasi diminta, terapkan augmentasi pada gambar dan bounding boxes
    if augment and img is not None:
        augmented = augmentor(image=img, bboxes=bboxes, class_labels=class_labels)
        img = augmented['image']
        bboxes = augmented['bboxes']
        class_labels = augmented['class_labels']

    # Simpan bounding boxes ke format YOLO
    with open(txt_output_path, 'w') as txt_file:
        for bbox, label in zip(bboxes, class_labels):
            txt_file.write(f"{label} {' '.join(map(str, bbox))}\n")

    return img, bboxes, class_labels  # Return augmented image and bboxes if augment=True

def process_data(img_file, lbl_file, img_dest_folder, lbl_dest_folder, augment=False, sample_output_count=15, sample_counter=0):
    img_src_path = os.path.join(images_dir, img_file)
    lbl_src_path = os.path.join(labels_dir, lbl_file)

    # Load image
    img = cv2.imread(img_src_path)

    # Augmentasi jika diminta
    if augment:
        img_dest_path = os.path.join(img_dest_folder, img_file.replace('.png', '_aug.png'))
        lbl_dest_path = os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '_aug.txt'))

        augmented_img, augmented_bboxes, class_labels = convert_xml_to_txt(lbl_src_path, lbl_dest_path, augment=True, img=img)
        
        # Isi bagian kosong dengan warna putih
        augmented_img = fill_with_white(augmented_img)
        
        cv2.imwrite(img_dest_path, augmented_img)

        # Simpan beberapa contoh gambar augmented ke folder sample_output
        if sample_counter < sample_output_count:
            sample_img = augmented_img.copy()
            draw_bboxes(sample_img, augmented_bboxes, class_labels)
            cv2.imwrite(os.path.join(sample_output_dir, f"{img_file.replace('.png', '_aug_sample.png')}"), sample_img)
            sample_counter += 1

    else:
        img_dest_path = os.path.join(img_dest_folder, img_file)
        lbl_dest_path = os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '.txt'))

        # Simpan tanpa augmentasi
        shutil.copy(img_src_path, img_dest_path)
        convert_xml_to_txt(lbl_src_path, lbl_dest_path)

def copy_and_convert_data(data, img_dest_folder, lbl_dest_folder, augment=False, sample_output_count=30):
    sample_counter = 0  # Counter for saving samples to sample_output

    # Gunakan ThreadPoolExecutor untuk memproses data secara paralel
    with ThreadPoolExecutor() as executor:
        futures = []
        for img_file, lbl_file in data:
            futures.append(executor.submit(process_data, img_file, lbl_file, img_dest_folder, lbl_dest_folder, augment, sample_output_count, sample_counter))

        # Tunggu hingga semua thread selesai
        for future in as_completed(futures):
            future.result()

# Copy dan konversi data untuk train (dengan augmentasi) dan val (tanpa augmentasi)
copy_and_convert_data(train_data, os.path.join(output_dir, 'images/train'), os.path.join(output_dir, 'labels/train'), augment=True)
copy_and_convert_data(val_data, os.path.join(output_dir, 'images/val'), os.path.join(output_dir, 'labels/val'))

print("Data berhasil dipisah, dikonversi, dan diaugmentasi ke format YOLO! Contoh gambar disimpan di folder sample_output.")