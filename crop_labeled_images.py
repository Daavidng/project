import json
import cv2
import numpy as np
import time
from pathlib import Path


def get_bounding_box_with_padding(points, image_shape, padding=50):
    """Calculate bounding box from polygon points with padding."""

    if not points:
        return None
    
    np_points = np.array(points)
    x_coords = np_points[:, 0]
    y_coords = np_points[:, 1]
    
    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(np.min(y_coords))
    y_max = int(np.max(y_coords))
    
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_shape[1], x_max + padding)
    y_max = min(image_shape[0], y_max + padding)
    
    return (x_min, y_min, x_max, y_max)


def crop_image_with_annotations(image_path, json_path, output_dir):
    """Crop image based on polygon annotations and save with label in filename."""
    
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return 0
    
    # Load the JSON annotations
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON {json_path}: {e}")
        return 0
    
    crops_saved = 0
    
    if 'shapes' in annotations:
        for i, shape in enumerate(annotations['shapes']):
            if 'points' not in shape or 'label' not in shape:
                continue
                
            points = shape['points']
            label = shape['label']
            
            if len(points) < 3:  # Need at least 3 points for a polygon
                continue
            
            # Get bounding box with padding
            bbox = get_bounding_box_with_padding(points, image.shape, padding=50)
            if bbox is None:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            # Crop the image
            cropped_image = image[y_min:y_max, x_min:x_max]
            
            # Skip if crop is too small
            if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                continue
            
            # Create filename with label
            base_name = image_path.stem 
            if len(annotations['shapes']) > 1:
                # Multiple shapes, add index
                output_filename = f"{base_name}_{label}_{i+1}.jpg"
            else:
                # Single shape
                output_filename = f"{base_name}_{label}.jpg"
            
            output_path = output_dir / output_filename
            
            # Save the cropped image
            success = cv2.imwrite(str(output_path), cropped_image)
            
            if success:
                print(f"  Cropped: {output_filename} ({cropped_image.shape[1]}x{cropped_image.shape[0]})")
                crops_saved += 1
            else:
                print(f"  Failed to save: {output_filename}")
    
    return crops_saved


def main():
    # Set up paths
    labeled_dir = Path("dataset/Labeled")
    output_dir = Path("dataset/Labeled_Crop")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not labeled_dir.exists():
        print(f"Error: Directory not found: {labeled_dir}")
        return
    
    # Find all JPG files
    jpg_files = list(labeled_dir.glob("*.jpg")) + list(labeled_dir.glob("*.JPG"))
    
    if not jpg_files:
        print(f"No JPG files found in {labeled_dir}")
        return
    
    print(f"Found {len(jpg_files)} JPG files in {labeled_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    processed_count = 0
    total_crops = 0
    skipped_count = 0
    start_time = time.time()
    
    for i, jpg_file in enumerate(sorted(jpg_files), 1):
        json_file = jpg_file.with_suffix('.json')
        
        if not json_file.exists():
            print(f"[{i:3d}/{len(jpg_files)}] No JSON file for {jpg_file.name}")
            skipped_count += 1
            continue
        
        # Show progress
        print(f"[{i:3d}/{len(jpg_files)}] Processing {jpg_file.name}...")
        
        crops_from_image = crop_image_with_annotations(jpg_file, json_file, output_dir)
        if crops_from_image > 0:
            processed_count += 1
            total_crops += crops_from_image
        else:
            skipped_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Total crops generated: {total_crops}")
    print(f"Skipped/Failed: {skipped_count} images")
    print(f"Total time: {duration:.2f} seconds")
    if len(jpg_files) > 0:
        print(f"Average time per image: {duration/len(jpg_files):.2f} seconds")
    print(f"Cropped images saved in: {output_dir}")
    
    # List some sample output files
    output_files = list(output_dir.glob("*.jpg"))
    print(f"\nSample output files ({min(10, len(output_files))} of {len(output_files)}):")
    for output_file in sorted(output_files)[:10]:
        print(f"  - {output_file.name}")
    
    if len(output_files) > 10:
        print(f"  ... and {len(output_files) - 10} more files")
    
    # Show label distribution
    labels = {}
    for output_file in output_files:
        parts = output_file.stem.split('_')
        if len(parts) >= 2:
            if parts[-1].isdigit() and len(parts) >= 3:
                label = parts[-2]
            else:
                label = parts[-1]
            labels[label] = labels.get(label, 0) + 1
    
    if labels:
        print(f"\nLabel distribution:")
        for label, count in sorted(labels.items()):
            print(f"  - {label}: {count} crops")


if __name__ == "__main__":
    main()
