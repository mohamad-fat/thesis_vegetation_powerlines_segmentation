import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict


class PowerLineDetectorV4:
    """Simple edge-based power line detection - keeps actual detected segments"""

    def __init__(self, line_width=15):
        """
        Initialize detector

        Args:
            line_width: Width for mask generation
        """
        self.line_width = line_width

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Strong CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def detect_power_line_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect edges and enhance linear features

        Args:
            gray: Grayscale image

        Returns:
            Binary mask of detected power line edges
        """
        # Apply bilateral filter to reduce noise while keeping edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multi-scale edge detection
        # Canny with low thresholds to catch thin lines
        edges1 = cv2.Canny(bilateral, 20, 60)
        edges2 = cv2.Canny(bilateral, 30, 90)

        # Combine
        edges = cv2.bitwise_or(edges1, edges2)

        # Dilate slightly to thicken thin lines
        kernel_dilate = np.ones((2, 2), np.uint8)
        edges_thick = cv2.dilate(edges, kernel_dilate, iterations=1)

        # Use directional morphology to enhance lines
        # Horizontal
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        enhanced_h = cv2.morphologyEx(edges_thick, cv2.MORPH_CLOSE, kernel_h)

        # Diagonal 1 (/)
        kernel_d1 = np.eye(15, dtype=np.uint8)
        enhanced_d1 = cv2.morphologyEx(edges_thick, cv2.MORPH_CLOSE, kernel_d1)

        # Diagonal 2 (\)
        kernel_d2 = np.fliplr(np.eye(15, dtype=np.uint8))
        enhanced_d2 = cv2.morphologyEx(edges_thick, cv2.MORPH_CLOSE, kernel_d2)

        # Vertical (less common but possible)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        enhanced_v = cv2.morphologyEx(edges_thick, cv2.MORPH_CLOSE, kernel_v)

        # Combine all directions
        combined = cv2.bitwise_or(enhanced_h, enhanced_d1)
        combined = cv2.bitwise_or(combined, enhanced_d2)
        combined = cv2.bitwise_or(combined, enhanced_v)

        return combined

    def filter_by_shape(self, mask: np.ndarray,
                       min_length: int = 80,
                       max_width: int = 60,
                       min_aspect_ratio: float = 2.5) -> np.ndarray:
        """
        Filter detected regions by shape criteria

        Args:
            mask: Binary mask
            min_length: Minimum length of power line segment
            max_width: Maximum width
            min_aspect_ratio: Minimum aspect ratio (length/width)

        Returns:
            Filtered mask
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_mask = np.zeros_like(mask)

        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate properties
            length = max(w, h)
            width = min(w, h)
            aspect_ratio = length / (width + 1e-6)
            area = cv2.contourArea(contour)

            # Keep if it's long, thin, and not too small
            if (length >= min_length and
                width <= max_width and
                aspect_ratio >= min_aspect_ratio and
                area > 30):

                # Draw the contour
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        return filtered_mask

    def detect_powerlines(self, image: np.ndarray) -> Dict:
        """
        Detect power lines - returns actual detected segments (not edge-to-edge)

        Args:
            image: Input BGR image

        Returns:
            Dictionary with detection results
        """
        # Preprocess
        gray = self.preprocess_image(image)

        # Detect edges with line enhancement
        edge_mask = self.detect_power_line_edges(gray)

        # Filter by shape
        filtered_mask = self.filter_by_shape(edge_mask,
                                            min_length=80,
                                            max_width=60,
                                            min_aspect_ratio=2.5)

        # Dilate final mask slightly to make lines more visible
        kernel_final = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(filtered_mask, kernel_final, iterations=2)

        return {
            'mask': final_mask,
            'edge_mask': edge_mask,
            'filtered_mask': filtered_mask,
            'preprocessed': gray
        }

    def mask_to_polygons(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Convert mask to polygons"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                # Simplify polygon
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
                if len(polygon) >= 3:
                    polygons.append(polygon)

        return polygons

    def visualize_detection(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Visualize detected power lines"""
        vis = image.copy()

        # Create colored overlay
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 0] = 0  # Remove blue
        mask_colored[:, :, 2] = 0  # Remove red
        # Keep only green

        # Blend
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)

        # Count regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.putText(vis, f"Segments detected: {len(contours)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return vis


def process_test_dataset(input_dir: str, output_dir: str, visualize=True):
    """Process test dataset"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    masks_dir = output_path / "masks"
    vis_dir = output_path / "visualizations"
    debug_dir = output_path / "debug"
    annotations_dir = output_path / "annotations"

    masks_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    # Initialize detector
    detector = PowerLineDetectorV4(line_width=15)

    # Get images
    image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))

    print(f"Processing {len(image_files)} images from {input_dir}")

    # COCO annotations
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "powerline", "supercategory": "infrastructure"}
        ]
    }

    annotation_id = 1

    for idx, image_file in enumerate(image_files):
        if image_file.name.startswith('_'):
            continue

        # Process first 5 for testing
        if idx >= 5:
            break

        print(f"Processing [{idx+1}/{len(image_files)}]: {image_file.name}")

        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Warning: Could not read {image_file.name}")
            continue

        height, width = image.shape[:2]

        # Detect power lines
        result = detector.detect_powerlines(image)
        mask = result['mask']

        # Save outputs
        cv2.imwrite(str(masks_dir / f"{image_file.stem}_mask.png"), mask)
        cv2.imwrite(str(debug_dir / f"{image_file.stem}_edges.png"), result['edge_mask'])
        cv2.imwrite(str(debug_dir / f"{image_file.stem}_filtered.png"), result['filtered_mask'])

        # Visualize
        if visualize:
            vis = detector.visualize_detection(image, mask)
            cv2.imwrite(str(vis_dir / image_file.name), vis)

        # Convert to polygons
        polygons = detector.mask_to_polygons(mask)

        # Add to COCO
        image_id = idx + 1
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        })

        for polygon in polygons:
            if len(polygon) < 3:
                continue

            segmentation = []
            for x, y in polygon:
                segmentation.extend([x, y])

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": bbox_width * bbox_height,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "iscrowd": 0
            })
            annotation_id += 1

        print(f"  Detected {len(polygons)} power line segments")

    # Save COCO annotations
    with open(annotations_dir / "powerlines_coco.json", 'w') as f:
        json.dump(coco_annotations, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Masks: {masks_dir}")
    print(f"Visualizations: {vis_dir}")
    print(f"Debug: {debug_dir}")
    print(f"Annotations: {annotations_dir}")


if __name__ == "__main__":
    input_dir = "Aerial_Power_Infrastructure_Detection_Dataset/test"
    output_dir = "powerline_annotations_test_v4"

    process_test_dataset(input_dir, output_dir, visualize=True)
