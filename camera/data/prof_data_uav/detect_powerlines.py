import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class PowerLineDetector:
    """Detects power lines using multiple computer vision techniques"""

    def __init__(self, line_width=10):
        """
        Initialize detector

        Args:
            line_width: Width of detected power lines in pixels for mask generation
        """
        self.line_width = line_width

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for line detection

        Args:
            image: Input BGR image

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        return denoised

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detector

        Args:
            image: Preprocessed grayscale image

        Returns:
            Binary edge map
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

        # Canny edge detection with adaptive thresholds
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        return edges

    def detect_lines_hough_standard(self, edges: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect lines using standard Hough Transform

        Args:
            edges: Binary edge map

        Returns:
            List of lines in (rho, theta) format
        """
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is None:
            return []

        return [(rho, theta) for rho, theta in lines[:, 0]]

    def detect_lines_hough_probabilistic(self, edges: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect lines using Probabilistic Hough Transform

        Args:
            edges: Binary edge map

        Returns:
            List of line segments in (x1, y1, x2, y2) format
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return []

        return [tuple(line[0]) for line in lines]

    def detect_lines_lsd(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect lines using Line Segment Detector (LSD)

        Args:
            image: Preprocessed grayscale image

        Returns:
            List of line segments in (x1, y1, x2, y2) format
        """
        lsd = cv2.createLineSegmentDetector(0)
        lines, width, prec, nfa = lsd.detect(image)

        if lines is None:
            return []

        return [tuple(map(int, line[0])) for line in lines]

    def filter_lines_by_angle(self, lines: List[Tuple],
                             angle_range: Tuple[float, float] = None,
                             min_length: int = 30) -> List[Tuple]:
        """
        Filter lines by angle and length

        Args:
            lines: List of line segments (x1, y1, x2, y2)
            angle_range: (min_angle, max_angle) in degrees, None to skip angle filtering
            min_length: Minimum line length

        Returns:
            Filtered lines
        """
        filtered = []

        for line in lines:
            if len(line) == 4:  # Line segment format
                x1, y1, x2, y2 = line

                # Calculate length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length < min_length:
                    continue

                # Calculate angle
                if angle_range is not None:
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    min_angle, max_angle = angle_range

                    # Check if angle is within range (considering both directions)
                    if not (min_angle <= angle <= max_angle or
                           min_angle <= (180 - angle) <= max_angle):
                        continue

                filtered.append(line)

        return filtered

    def merge_similar_lines(self, lines: List[Tuple[int, int, int, int]],
                           angle_threshold: float = 10,
                           distance_threshold: float = 20) -> List[Tuple[int, int, int, int]]:
        """
        Merge similar collinear lines

        Args:
            lines: List of line segments
            angle_threshold: Maximum angle difference in degrees
            distance_threshold: Maximum distance between lines

        Returns:
            Merged lines
        """
        if len(lines) == 0:
            return []

        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            x1, y1, x2, y2 = line1
            angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            group = [line1]

            for j, line2 in enumerate(lines[i+1:], start=i+1):
                if j in used:
                    continue

                x3, y3, x4, y4 = line2
                angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi

                # Check angle similarity
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff < angle_threshold:
                    # Check distance between lines
                    mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                    mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
                    dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)

                    if dist < distance_threshold:
                        group.append(line2)
                        used.add(j)

            # Merge lines in group by taking endpoints
            if len(group) > 0:
                all_points = []
                for line in group:
                    all_points.extend([(line[0], line[1]), (line[2], line[3])])

                # Find extreme points
                all_points = np.array(all_points)
                idx1 = np.argmin(all_points[:, 0])
                idx2 = np.argmax(all_points[:, 0])

                merged_line = (
                    int(all_points[idx1, 0]), int(all_points[idx1, 1]),
                    int(all_points[idx2, 0]), int(all_points[idx2, 1])
                )
                merged.append(merged_line)

        return merged

    def detect_powerlines(self, image: np.ndarray,
                         use_angle_filter: bool = True,
                         merge_lines: bool = True) -> Dict:
        """
        Detect power lines using ensemble of methods

        Args:
            image: Input BGR image
            use_angle_filter: Whether to filter lines by angle
            merge_lines: Whether to merge similar lines

        Returns:
            Dictionary containing detected lines and metadata
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)
        edges = self.detect_edges(preprocessed)

        # Detect using multiple methods
        lines_prob_hough = self.detect_lines_hough_probabilistic(edges)
        lines_lsd = self.detect_lines_lsd(preprocessed)

        # Combine all detections
        all_lines = lines_prob_hough + lines_lsd

        # Filter lines
        if use_angle_filter:
            # Power lines are typically horizontal or at slight angles
            # Filter for lines within 0-45 degrees from horizontal
            all_lines = self.filter_lines_by_angle(all_lines, angle_range=(0, 45))
        else:
            all_lines = self.filter_lines_by_angle(all_lines, angle_range=None)

        # Merge similar lines
        if merge_lines and len(all_lines) > 0:
            all_lines = self.merge_similar_lines(all_lines)

        return {
            'lines': all_lines,
            'num_lines': len(all_lines),
            'edges': edges,
            'preprocessed': preprocessed
        }

    def lines_to_mask(self, lines: List[Tuple[int, int, int, int]],
                     image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert detected lines to binary segmentation mask

        Args:
            lines: List of line segments
            image_shape: (height, width) of output mask

        Returns:
            Binary mask with power lines
        """
        mask = np.zeros(image_shape, dtype=np.uint8)

        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(mask, (x1, y1), (x2, y2), 255, self.line_width)

        return mask

    def mask_to_polygons(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Convert binary mask to polygon contours

        Args:
            mask: Binary segmentation mask

        Returns:
            List of polygons, each polygon is a list of (x, y) points
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if len(contour) >= 3:  # Need at least 3 points for a polygon
                polygon = [(int(point[0][0]), int(point[0][1])) for point in contour]
                polygons.append(polygon)

        return polygons

    def visualize_detection(self, image: np.ndarray,
                          detection_result: Dict,
                          show_edges: bool = True) -> np.ndarray:
        """
        Visualize detected power lines

        Args:
            image: Original image
            detection_result: Result from detect_powerlines()
            show_edges: Whether to show edge detection

        Returns:
            Visualization image
        """
        vis = image.copy()

        # Draw detected lines
        for line in detection_result['lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add text
        cv2.putText(vis, f"Lines detected: {detection_result['num_lines']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return vis


def process_test_dataset(input_dir: str, output_dir: str, visualize: bool = True):
    """
    Process test dataset and generate power line annotations

    Args:
        input_dir: Directory containing test images
        output_dir: Directory to save results
        visualize: Whether to save visualization images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    masks_dir = output_path / "masks"
    vis_dir = output_path / "visualizations"
    annotations_dir = output_path / "annotations"

    masks_dir.mkdir(exist_ok=True)
    if visualize:
        vis_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    # Initialize detector
    detector = PowerLineDetector(line_width=10)

    # Get all images
    image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))

    print(f"Processing {len(image_files)} images from {input_dir}")

    # COCO format annotations
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
        # TODO: Remove break to process all images
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
        result = detector.detect_powerlines(image, use_angle_filter=True, merge_lines=True)

        # Generate mask
        mask = detector.lines_to_mask(result['lines'], (height, width))

        # Save mask
        mask_filename = image_file.stem + "_mask.png"
        cv2.imwrite(str(masks_dir / mask_filename), mask)

        # Convert to polygons
        polygons = detector.mask_to_polygons(mask)

        # Add to COCO annotations
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

            # Flatten polygon for COCO format
            segmentation = []
            for x, y in polygon:
                segmentation.extend([x, y])

            # Calculate bounding box
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

        # Visualize if requested
        if visualize:
            vis = detector.visualize_detection(image, result)
            cv2.imwrite(str(vis_dir / image_file.name), vis)

        print(f"  Detected {result['num_lines']} power line segments")

    # Save COCO annotations
    coco_file = annotations_dir / "powerlines_coco.json"
    with open(coco_file, 'w') as f:
        json.dump(coco_annotations, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Masks saved to: {masks_dir}")
    print(f"Annotations saved to: {annotations_dir}")
    if visualize:
        print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    # Test on test dataset
    input_dir = "Aerial_Power_Infrastructure_Detection_Dataset/test"
    output_dir = "powerline_annotations_test"

    process_test_dataset(input_dir, output_dir, visualize=True)
