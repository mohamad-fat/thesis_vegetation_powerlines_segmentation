import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pathlib import Path

def parse_voc_annotation(xml_path):
    """Parse VOC format XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels

def draw_boxes(image, boxes, labels, colors=None):
    """Draw bounding boxes on image"""
    img = image.copy()

    if colors is None:
        colors = {
            'tbar': (0, 255, 0),      # Green
            'insulator': (255, 0, 0),  # Red
            'tower': (0, 0, 255),      # Blue
        }

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        color = colors.get(label, (255, 255, 0))  # Default yellow

        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (xmin, ymin - label_size[1] - 5),
                     (xmin + label_size[0], ymin), color, -1)

        # Draw label text
        cv2.putText(img, label, (xmin, ymin - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def visualize_dataset(data_dir, split='train', num_samples=9, random_samples=True):
    """Visualize labeled samples from the dataset"""

    # Setup paths
    image_dir = os.path.join(data_dir, split)
    annotation_dir = os.path.join(data_dir, 'Annotations', 'VOC', split)

    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    if random_samples:
        image_files = random.sample(image_files, min(num_samples, len(image_files)))
    else:
        image_files = image_files[:num_samples]

    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, img_file in enumerate(image_files):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotation
        xml_file = img_file.replace('.jpg', '.xml')
        xml_path = os.path.join(annotation_dir, xml_file)

        if os.path.exists(xml_path):
            boxes, labels = parse_voc_annotation(xml_path)
            image_with_boxes = draw_boxes(image, boxes, labels)

            # Count objects
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            title = f"{img_file}\n{', '.join([f'{k}: {v}' for k, v in label_counts.items()])}"
        else:
            image_with_boxes = image
            title = f"{img_file}\nNo annotations"

        axes[idx].imshow(image_with_boxes)
        axes[idx].set_title(title, fontsize=8)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_single_image(data_dir, split='train', image_name=None):
    """Visualize a single labeled image"""

    image_dir = os.path.join(data_dir, split)
    annotation_dir = os.path.join(data_dir, 'Annotations', 'VOC', split)

    if image_name is None:
        # Pick a random image
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        image_name = random.choice(image_files)

    # Load image
    img_path = os.path.join(image_dir, image_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotation
    xml_file = image_name.replace('.jpg', '.xml')
    xml_path = os.path.join(annotation_dir, xml_file)

    plt.figure(figsize=(12, 8))

    if os.path.exists(xml_path):
        boxes, labels = parse_voc_annotation(xml_path)
        image_with_boxes = draw_boxes(image, boxes, labels)

        # Print annotation details
        print(f"Image: {image_name}")
        print(f"Size: {image.shape[1]}x{image.shape[0]}")
        print(f"Number of objects: {len(labels)}")
        print("\nObjects:")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            print(f"  {i+1}. {label}: bbox {box}")
    else:
        image_with_boxes = image
        print(f"No annotations found for {image_name}")

    plt.imshow(image_with_boxes)
    plt.title(image_name)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set dataset directory
    data_dir = "./data/prof_data_uav/Aerial_Power_Infrastructure_Detection_Dataset"

    # Visualize multiple samples from training set
    print("Visualizing training samples...")
    visualize_dataset(data_dir, split='train', num_samples=9, random_samples=True)

    # Visualize a single image with details
    print("\nVisualizing single image with details...")
    visualize_single_image(data_dir, split='train')
