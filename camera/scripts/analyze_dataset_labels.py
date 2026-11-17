import os
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset_labels(data_dir):
    """Analyze and count all labels in the dataset"""

    splits = ['train', 'valid', 'test']
    overall_stats = {}

    for split in splits:
        annotation_dir = os.path.join(data_dir, 'Annotations', 'VOC', split)

        if not os.path.exists(annotation_dir):
            print(f"Split '{split}' not found, skipping...")
            continue

        # Get all XML files
        xml_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]

        label_counts = Counter()
        images_with_labels = 0
        total_objects = 0

        for xml_file in xml_files:
            xml_path = os.path.join(annotation_dir, xml_file)

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                objects = root.findall('object')
                if objects:
                    images_with_labels += 1

                for obj in objects:
                    label = obj.find('name').text
                    label_counts[label] += 1
                    total_objects += 1

            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")

        overall_stats[split] = {
            'total_images': len(xml_files),
            'images_with_labels': images_with_labels,
            'label_counts': label_counts,
            'total_objects': total_objects
        }

    return overall_stats

def print_statistics(stats):
    """Print dataset statistics"""

    print("="*70)
    print("AERIAL POWER INFRASTRUCTURE DATASET LABEL ANALYSIS")
    print("="*70)

    total_images = 0
    total_objects = 0
    overall_labels = Counter()

    for split, data in stats.items():
        print(f"\n{split.upper()} SET:")
        print("-"*70)
        print(f"  Total images: {data['total_images']}")
        print(f"  Images with labels: {data['images_with_labels']}")
        print(f"  Total objects: {data['total_objects']}")
        print(f"  Average objects per image: {data['total_objects']/max(data['images_with_labels'], 1):.2f}")
        print(f"\n  Label distribution:")

        for label, count in sorted(data['label_counts'].items()):
            percentage = (count / data['total_objects'] * 100) if data['total_objects'] > 0 else 0
            print(f"    - {label}: {count} ({percentage:.1f}%)")
            overall_labels[label] += count

        total_images += data['total_images']
        total_objects += data['total_objects']

    print("\n" + "="*70)
    print("OVERALL DATASET SUMMARY:")
    print("="*70)
    print(f"Total images: {total_images}")
    print(f"Total objects: {total_objects}")
    print(f"\nOverall label distribution:")

    for label, count in sorted(overall_labels.items()):
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  - {label}: {count} ({percentage:.1f}%)")

    return overall_labels

def visualize_label_distribution(overall_labels):
    """Create bar chart of label distribution"""

    if not overall_labels:
        print("No labels to visualize")
        return

    labels = list(overall_labels.keys())
    counts = list(overall_labels.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=['#2ecc71', '#e74c3c', '#3498db'])

    plt.xlabel('Object Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Label Distribution in Aerial Power Infrastructure Dataset', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "data/prof_data_uav/Aerial_Power_Infrastructure_Detection_Dataset"

    print("Analyzing dataset labels...")
    stats = analyze_dataset_labels(data_dir)
    overall_labels = print_statistics(stats)

    print("\nGenerating visualization...")
    visualize_label_distribution(overall_labels)
