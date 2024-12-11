import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import random
import shutil
import albumentations as A
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Path ke folder data
images_dir = "D:\KULIAH\SEMESTER 5\RSBP\Bacteri\Detection-of-bacteria-in-water-using-Yolo-images\Detect\Detection-of-bacteria-in-water-using-Yolo-images\EMDS7"
labels_dir = "D:\KULIAH\SEMESTER 5\RSBP\Bacteri\Detection-of-bacteria-in-water-using-Yolo-images\Detect\Detection-of-bacteria-in-water-using-Yolo-images\EMDS7xml"
output_dir = "D:\KULIAH\SEMESTER 5\RSBP\Bacteri\Detection-of-bacteria-in-water-using-Yolo-images\Detect\Detection-of-bacteria-in-water-using-Yolo-images\\temp"
sample_output_dir = "D:\KULIAH\SEMESTER 5\RSBP\Bacteri\Detection-of-bacteria-in-water-using-Yolo-images\Detect\Detection-of-bacteria-in-water-using-Yolo-images\sample_output"

class ImageProcessor:
    def __init__(self, image_path):
        if not os.path.exists(image_path):
            raise ValueError(f"The image path does not exist: {image_path}")
        
        # Load the image
        self._img_np = cv2.imread(image_path)
        if self._img_np is None:
            raise ValueError(f"Image not found or unable to load: {image_path}")
        
        self._img_np = cv2.cvtColor(self._img_np, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        self._grayscale_np = None
        self._blurred_np = None
        self._gradient_magnitude = None
        self._gradient_direction = None
        self._suppressed_img = None
        self._connected_edges = None
        self._canny_img = None

    def grayscale(self):
        ''' Convert image (RGB) color to grayscale '''
        self._grayscale_np = np.dot(self._img_np[..., :3], [0.2989, 0.5870, 0.1140])
        return self

    def gaussian_kernel(self, size, sigma):
        ''' Generate a Gaussian kernel of a given size and standard deviation '''
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                          np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    def gaussian_blur(self, size, sigma=2.5):
        ''' Apply Gaussian blur to the grayscale image '''
        kernel = self.gaussian_kernel(size, sigma)
        self._blurred_np = convolve2d(self._grayscale_np, kernel, mode='same', boundary='symm', fillvalue=0)
        return self

    def gradient_approximation(self):
        ''' Approximate the magnitude of the gradient '''
        horizontal_filter = np.array([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]])
        vertical_filter = np.transpose(horizontal_filter)
        
        horizontal_gradient = convolve2d(self._blurred_np, horizontal_filter, mode='same', boundary='symm', fillvalue=0)
        vertical_gradient = convolve2d(self._blurred_np, vertical_filter, mode='same', boundary='symm', fillvalue=0)
        
        self._gradient_magnitude = np.sqrt(horizontal_gradient ** 2 + vertical_gradient ** 2)
        self._gradient_direction = np.arctan2(vertical_gradient, horizontal_gradient)
        return self

    def non_max_suppression(self):
        ''' Suppress non-maximum pixels in the direction of the gradient '''
        self._suppressed_img = np.zeros_like(self._gradient_magnitude)
        
        for x in range(self._gradient_magnitude.shape[0]):
            for y in range(self._gradient_magnitude.shape[1]):
                angle = self._gradient_direction[x, y] * (180.0 / np.pi)  # Convert to degrees
                angle = angle + 180 if angle < 0 else angle  # Adjust angle to be in [0, 360]
                
                # Define neighboring pixel indices based on gradient direction
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbor1_i, neighbor1_j = x, y + 1
                    neighbor2_i, neighbor2_j = x, y - 1
                elif (22.5 <= angle < 67.5):
                    neighbor1_i, neighbor1_j = x - 1, y + 1
                    neighbor2_i, neighbor2_j = x + 1, y - 1
                elif (67.5 <= angle < 112.5):
                    neighbor1_i, neighbor1_j = x - 1, y
                    neighbor2_i, neighbor2_j = x + 1, y
                elif (112.5 <= angle < 157.5):
                    neighbor1_i, neighbor1_j = x - 1, y - 1
                    neighbor2_i, neighbor2_j = x + 1, y + 1
                else:
                    neighbor1_i, neighbor1_j = x - 1, y
                    neighbor2_i, neighbor2_j = x + 1, y
                
                # Check if neighbor indices are within bounds
                neighbor1_i = max(0, min(neighbor1_i, self._gradient_magnitude.shape[0] - 1))
                neighbor1_j = max(0, min(neighbor1_j, self._gradient_magnitude.shape[1] - 1))
                neighbor2_i = max(0, min(neighbor2_i, self._gradient_magnitude.shape[0] - 1))
                neighbor2_j = max(0, min(neighbor2_j, self._gradient_magnitude.shape[1] - 1))
                
                # Compare current pixel magnitude with its neighbors along gradient direction
                current_mag = self._gradient_magnitude[x, y]
                neighbor1_mag = self._gradient_magnitude[neighbor1_i, neighbor1_j]
                neighbor2_mag = self._gradient_magnitude[neighbor2_i, neighbor2_j]
                
                # Perform suppression
                if (current_mag >= neighbor1_mag) and (current_mag >= neighbor2_mag):
                    self._suppressed_img[x, y] = current_mag
                else:
                    self._suppressed_img[x, y] = 0
        return self

    def double_thresholding(self, low_threshold_ratio=0.1, high_threshold_ratio=0.3):
        ''' Categorize pixels into strong, weak, and non-edges '''
        h_threshold = np.max(self._gradient_magnitude) * high_threshold_ratio
        l_threshold = h_threshold * low_threshold_ratio
        
        strong_edges = (self._gradient_magnitude >= h_threshold)
        weak_edges = (self._gradient_magnitude >= l_threshold) & (self._gradient_magnitude < h_threshold)
        
        self._connected_edges = np.zeros_like(self._gradient_magnitude)
        self._connected_edges[strong_edges] = 255
        
        for x in range(self._gradient_magnitude.shape[0]):
            for y in range(self._gradient_magnitude.shape[1]):
                if weak_edges[x, y]:
                    if (strong_edges[x - 1:x + 2, y - 1:y + 2].any()):
                        self._connected_edges[x, y] = 255
        return self

    def hysteresis(self, weak_pixel_intensity=50, strong_pixel_intensity=255):
        ''' Connect weak edges to strong edges and reject isolated weak edges '''
        self._canny_img = self._connected_edges.copy()
        
        weak_edges_x, weak_edges_y = np.where(self._canny_img == weak_pixel_intensity)
        for x, y in zip(weak_edges_x, weak_edges_y):
            if np.any(self._connected_edges[x - 1:x + 2, y - 1:y + 2] == strong_pixel_intensity):
                self._canny_img[x, y] = strong_pixel_intensity
            else:
                self._canny_img[x, y] = 0
        return self

    def show_image(self, image, title='Image'):
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        plt.axis('off')
        plt.title(title)
        plt.show()


train_val_split = 0.8  # Rasio data train/validation

# Cek apakah path gambar dan label ada
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Path gambar tidak ditemukan: {images_dir}")
if not os.path.exists(labels_dir):
    raise FileNotFoundError(f"Path label tidak ditemukan: {labels_dir}")

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
    'G001': 0, 'G002': 1, 'G003': 2, 'G004': 3, 'G005': 4, 'G006': 5, 'G007': 6, 'G008': 7, 'G009': 8, 'G010': 9,
    'G011': 10, 'G012': 11, 'G013': 12, 'G014': 13, 'G015': 14, 'G016': 15, 'G017': 16, 'G018': 17, 'G019': 18, 'G020': 19,
    'G021': 20, 'G022': 21, 'G023': 22, 'G024': 23, 'G025': 24, 'G026': 25, 'G027': 26, 'G028': 27, 'G029': 28, 'G030': 29,
    'G031': 30, 'G032': 31, 'G033': 32, 'G034': 33, 'G035': 34, 'G036': 35, 'G037': 36, 'G038': 37, 'G039': 38, 'G040': 39,
    'G041': 40, 'G042': 41, 'G043': 42, 'G044': 43, 'G045': 44, 'G046': 45, 'G047': 46, 'G048': 47, 'G049': 48, 'G050': 49,
    'G051': 50, 'G052': 51, 'G053': 52
}

# Augmentasi gambar dan bounding box tanpa mirroring, mengisi area kosong dengan putih
augmentor = A.Compose([
    A.Resize(1024, 1024, p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

sample_counter_lock = Lock()

def draw_bboxes(image, bboxes, class_labels):
    """Gambar bounding box pada gambar."""
    for bbox, class_id in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        img_h, img_w = image.shape[:2]
        xmin = int((x_center - width / 2) * img_w)
        ymin = int((y_center - height / 2) * img_h)
        xmax = int((x_center + width / 2) * img_w)
        ymax = int((y_center + height / 2) * img_h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, str(class_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def convert_xml_to_txt(xml_path, txt_output_path, augment=False, img=None):
    """Fungsi ini akan mengonversi format XML ke format YOLO (txt)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    bboxes = []
    class_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_mapping.get(class_name, -1)
        if class_id == -1:
            continue
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        bboxes.append([x_center, y_center, box_width, box_height])
        class_labels.append(class_id)

    with open(txt_output_path, 'w') as txt_file:
        for bbox, label in zip(bboxes, class_labels):
            txt_file.write(f"{label} {' '.join(map(str, bbox))}\n")

    return img, bboxes, class_labels

def apply_morphology(img, kernel_size=(10, 10), blur_kernel=(5, 5)):
    """Terapkan adaptive Gaussian thresholding dengan beberapa langkah pra-pemrosesan."""
    # Jika gambar berwarna, konversi ke grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Image preprocessing - Gaussian Blur untuk mengurangi noise awal
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Adaptive Gaussian Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return adaptive_thresh

def process_data(img_file, lbl_file, img_dest_folder, lbl_dest_folder, augment=False, sample_output_count=15):
    img_src_path = os.path.join(images_dir, img_file)
    lbl_src_path = os.path.join(labels_dir, lbl_file)

    # Baca gambar
    img_processor = ImageProcessor(img_src_path)

    # Convert to grayscale, apply Gaussian blur, and perform edge detection steps
    img_processor.grayscale()

    # Get the final Canny edge-detected image
    img = img_processor._grayscale_np.astype(np.uint8)

    if augment:
        img_dest_path = os.path.join(img_dest_folder, img_file.replace('.png', '_aug.png'))
        lbl_dest_path = os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '_aug.txt'))
        augmented_img, augmented_bboxes, class_labels = convert_xml_to_txt(lbl_src_path, lbl_dest_path, augment=True, img=img)

        augmented = augmentor(image=augmented_img, bboxes=augmented_bboxes, class_labels=class_labels)
        augmented_img = augmented['image']
        augmented_bboxes = augmented['bboxes']

        # Draw bounding boxes on augmented image and save to sample output
        with sample_counter_lock:
            if sample_output_count > 0:
                sample_img = augmented_img.copy()
                draw_bboxes(sample_img, augmented_bboxes, class_labels)
                cv2.imwrite(os.path.join(sample_output_dir, f"{img_file.replace('.png', '_aug_sample.png')}") , sample_img)
                sample_output_count -= 1

        # Save the augmented image in the train directory
        cv2.imwrite(img_dest_path, augmented_img)
        print(f"Augmented Image saved to: {img_dest_path}")
    else:
        img_dest_path = os.path.join(img_dest_folder, img_file)
        lbl_dest_path = os.path.join(lbl_dest_folder, lbl_file.replace('.xml', '.txt'))
        cv2.imwrite(img_dest_path, img)
        convert_xml_to_txt(lbl_src_path, lbl_dest_path)
        print(f"Original Image saved to: {img_dest_path}")

def copy_and_convert_data(data, img_dest_folder, lbl_dest_folder, augment=False, sample_output_count=30):
    with ThreadPoolExecutor(max_workers=14) as executor:  # Tentukan jumlah maksimum thread
        futures = []
        for img_file, lbl_file in data:
            futures.append(executor.submit(process_data, img_file, lbl_file, img_dest_folder, lbl_dest_folder, augment, sample_output_count))
        for future in as_completed(futures):
            future.result()

copy_and_convert_data(train_data, os.path.join(output_dir, 'images/train'), os.path.join(output_dir, 'labels/train'), augment=True)
copy_and_convert_data(val_data, os.path.join(output_dir, 'images/val'), os.path.join(output_dir, 'labels/val'))

print("Data berhasil dipisah, dikonversi, dan diaugmentasi ke format YOLO! Contoh gambar disimpan di folder sample_output.")