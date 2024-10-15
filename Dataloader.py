import os
import random
import shutil
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np

# Define paths
images_dir = "EMDS7/"
labels_dir = "EMDS7xml/"
output_dir = "YOLO_data/"
train_val_split = 0.8  # Ratio for training and validation data split

# Remove output directory if it exists and create fresh output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

# Class mapping for bounding box labels
class_mapping = {
    'G001': 0, 'G002': 1, 'G003': 2, 'G004': 3, 
    'G005': 4, 'G006': 5, 'G007': 6, 'G008': 7, 
    'G009': 8, 'G010': 9, 'G011': 10, 'G012': 11, 
    'G013': 12, 'G014': 13, 'G015': 14, 'G016': 15, 
    'G017': 16, 'G018': 17, 'G019': 18, 'G020': 19, 
    'G021': 20, 'G022': 21, 'G023': 22, 'G024': 23, 
    'G025': 24, 'G026': 25, 'G027': 26, 'G028': 27, 
    'G029': 28, 'G030': 29, 'G031': 30, 'G032': 31, 
    'G033': 32, 'G034': 33, 'G035': 34, 'G036': 35, 
    'G037': 36, 'G038': 37, 'G039': 38, 'G040': 39, 
    'G041': 40, 'G042': 41, 'G043': 42, 
    'G044': 43, 'G045': 44, 'G046': 45, 'G047': 46, 
    'G048': 47, 'G049': 48, 'G050': 49, 
    'G051': 50, 'G052': 51, 'G053': 52
}

# List all image and label files
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
label_files = [f.replace('.png', '.xml') for f in image_files]

# Shuffle data for randomness
data = list(zip(image_files, label_files))
random.shuffle(data)

# Split data into training and validation sets
split_index = int(train_val_split * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# Define the augmentation pipeline with bounding box adjustments
augmentations = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0)),
    # A.ColorJitter(),
    # A.HueSaturationValue(),
    # A.RGBShift(),
    A.RandomBrightnessContrast(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def load_xml_bboxes(xml_path, img_width, img_height):
    """Parse XML and convert bounding boxes to YOLO format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    class_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name == 'unknown':
            continue

        class_id = class_mapping.get(class_name, -1)
        if class_id == -1:
            continue

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height

        bboxes.append([x_center, y_center, box_width, box_height])
        class_labels.append(class_id)

    return bboxes, class_labels

def save_augmentations(img, bboxes, class_labels, img_dest_path, txt_dest_path, suffix=""):
    """Save augmented image and adjusted bounding boxes."""
    img_file = img_dest_path.replace(".png", f"{suffix}.png")
    txt_file = txt_dest_path.replace(".txt", f"{suffix}.txt")

    # Save augmented image
    cv2.imwrite(img_file, img)

    # Save bounding boxes in YOLO format
    with open(txt_file, 'w') as f:
        for bbox, label in zip(bboxes, class_labels):
            f.write(f"{label} {' '.join(map(str, bbox))}\n")

def validate_bboxes(bboxes):
    """Validate and filter out bounding boxes with invalid coordinates."""
    valid_bboxes = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        # Ensure width and height are positive and within bounds
        if width > 0 and height > 0 and x_center - width / 2 >= 0 and y_center - height / 2 >= 0:
            valid_bboxes.append(bbox)
        else:
            print(f"Invalid bbox detected and removed: {bbox}")
    return valid_bboxes

def augment_and_save_data(data, img_dest_folder, lbl_dest_folder):
    for img_file, lbl_file in data:
        img_path = os.path.join(images_dir, img_file)
        xml_path = os.path.join(labels_dir, lbl_file)
        
        # Load image and get dimensions
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        # Load bounding boxes
        bboxes, class_labels = load_xml_bboxes(xml_path, width, height)

        # Validate bounding boxes before augmentation
        bboxes = validate_bboxes(bboxes)
        if not bboxes:
            print(f"Skipping {img_file} due to no valid bounding boxes.")
            continue

        # Apply augmentations
        try:
            augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']
        except ValueError as e:
            print(f"Augmentation error for {img_file}: {e}")
            continue

        # Save the augmented image and labels
        save_augmentations(augmented_image, augmented_bboxes, augmented_class_labels, 
                           os.path.join(img_dest_folder, img_file), 
                           os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '.txt')), suffix='-aug')

        # Additional augmentations with bounding box validation
        for i in range(3):
            try:
                augmented = augmentations(image=augmented_image, bboxes=augmented_bboxes, class_labels=augmented_class_labels)
                if not augmented['bboxes']:
                    print(f"Skipping variation {i+1} for {img_file} due to empty bounding boxes.")
                    continue

                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_class_labels = augmented['class_labels']
                save_augmentations(augmented_image, augmented_bboxes, augmented_class_labels, 
                                   os.path.join(img_dest_folder, img_file), 
                                   os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '.txt')), suffix=f'-aug{i+1}')
            except ValueError as e:
                print(f"Augmentation error in variation {i+1} for {img_file}: {e}")
                continue

# Applying augmentations to train dataset
augment_and_save_data(train_data, os.path.join(output_dir, 'images/train'), os.path.join(output_dir, 'labels/train'))
print("Data train berhasil diproses dengan augmentasi dan bounding box yang disesuaikan!")

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

def copy_and_convert_data(data, img_dest_folder, lbl_dest_folder, apply_processing=False):
    """Copy images and convert XML annotations to YOLO format, with optional image processing."""
    for img_file, lbl_file in data:
        img_src_path = os.path.join(images_dir, img_file)
        lbl_src_path = os.path.join(labels_dir, lbl_file)

        # Load image using OpenCV
        image = cv2.imread(img_src_path)

        # Save image (processed or original)
        img_dest_path = os.path.join(img_dest_folder, img_file)
        cv2.imwrite(img_dest_path, image)

        # Convert labels from XML to YOLO txt format
        txt_output_path = os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '.txt'))
        convert_xml_to_txt(lbl_src_path, txt_output_path)

# Process validation data without augmentations
copy_and_convert_data(train_data, os.path.join(output_dir, 'images/train'), os.path.join(output_dir, 'labels/train'), apply_processing=False)
copy_and_convert_data(val_data, os.path.join(output_dir, 'images/val'), os.path.join(output_dir, 'labels/val'), apply_processing=False)
print("Validation data has been successfully copied and converted without additional processing.")
