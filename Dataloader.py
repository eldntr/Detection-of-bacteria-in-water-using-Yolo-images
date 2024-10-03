import os
import random
import shutil

# Path ke folder data
images_dir = "EMDS7/"
labels_dir = "EMDS7xml/"
output_dir = "YOLO_data/"
train_val_split = 0.8  # Rasio data train/validation

# Hapus folder output jika sudah ada, lalu buat ulang folder output yang fresh
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Buat ulang folder output yang fresh
os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

# Daftar semua gambar dan annotations
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
label_files = [f.replace('.png', '.xml') for f in image_files]

# Shuffle untuk memastikan randomisasi
data = list(zip(image_files, label_files))
random.shuffle(data)

# Split data untuk train dan validation
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

def convert_xml_to_txt(xml_path, txt_output_path):
    """Fungsi ini akan mengonversi format XML ke format YOLO (txt) dan mengabaikan objek dengan nama 'unknown'."""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    with open(txt_output_path, 'w') as txt_file:
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

            # Tulis ke file txt
            txt_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

def copy_and_convert_data(data, img_dest_folder, lbl_dest_folder):
    for img_file, lbl_file in data:
        img_src_path = os.path.join(images_dir, img_file)
        lbl_src_path = os.path.join(labels_dir, lbl_file)

        # Copy images
        img_dest_path = os.path.join(img_dest_folder, img_file)
        shutil.copy(img_src_path, img_dest_path)

        # Convert labels from XML to YOLO txt format
        txt_output_path = os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '.txt'))
        convert_xml_to_txt(lbl_src_path, txt_output_path)

# Copy dan konversi data untuk train dan val
copy_and_convert_data(train_data, os.path.join(output_dir, 'images/train'), os.path.join(output_dir, 'labels/train'))
copy_and_convert_data(val_data, os.path.join(output_dir, 'images/val'), os.path.join(output_dir, 'labels/val'))

print("Data berhasil dipisah dan dikonversi ke format YOLO!")
