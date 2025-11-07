#!/usr/bin/env python3
"""
Extract and save text regions near city symbols for manual transcription or
offline OCR processing.

This script identifies likely text regions near each city using image processing,
then saves sample regions for review and manual transcription.
"""

import cv2
import numpy as np
import json
from pathlib import Path


def load_filtered_cities(json_path="filtered_cities.json"):
    """Load filtered city detections."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['cities']


def preprocess_for_text_detection(region):
    """Preprocess image region for text detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply bilateral filter to reduce noise while keeping edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return denoised, gray


def detect_text_regions(region):
    """
    Detect potential text regions using MSER (Maximally Stable Extremal Regions).
    Returns list of bounding boxes for detected text regions.
    """
    preprocessed, gray = preprocess_for_text_detection(region)

    # Use MSER to detect text-like regions
    mser = cv2.MSER_create()
    mser.setDelta(5)
    mser.setMinArea(50)
    mser.setMaxArea(2000)

    # Detect regions
    regions, boxes = mser.detectRegions(preprocessed)

    # Filter and merge nearby boxes
    if len(boxes) > 0:
        # Group nearby boxes
        merged_boxes = []
        boxes_list = boxes.tolist()
        used = set()

        for i, (x1, y1, w1, h1) in enumerate(boxes_list):
            if i in used:
                continue

            # Start with this box
            min_x, min_y = x1, y1
            max_x, max_y = x1 + w1, y1 + h1
            used.add(i)

            # Find nearby boxes to merge
            for j, (x2, y2, w2, h2) in enumerate(boxes_list):
                if j in used:
                    continue

                # Check if boxes are close (within 10 pixels)
                if (abs(x1 - x2) < 50 and abs(y1 - y2) < 20):
                    min_x = min(min_x, x2)
                    min_y = min(min_y, y2)
                    max_x = max(max_x, x2 + w2)
                    max_y = max(max_y, y2 + h2)
                    used.add(j)

            # Add merged box
            merged_boxes.append([min_x, min_y, max_x - min_x, max_y - min_y])

        return merged_boxes

    return []


def extract_text_near_city(img, city, search_radius=150):
    """
    Extract likely text regions near city symbol.

    Returns:
        dict with extracted region info
    """
    cx = city['center']['x']
    cy = city['center']['y']

    # Define search region
    x1 = max(0, cx - search_radius)
    y1 = max(0, cy - search_radius)
    x2 = min(img.shape[1], cx + search_radius)
    y2 = min(img.shape[0], cy + search_radius)

    # Extract region
    region = img[y1:y2, x1:x2].copy()

    # Detect text regions
    text_boxes = detect_text_regions(region)

    # Convert to global coordinates and calculate distances
    text_regions = []
    for (bx, by, bw, bh) in text_boxes:
        # Calculate center of text box in global coordinates
        text_cx = x1 + bx + bw // 2
        text_cy = y1 + by + bh // 2

        # Distance from city center
        dist = np.sqrt((text_cx - cx)**2 + (text_cy - cy)**2)

        # Filter out regions that are too small or too far
        if bw > 15 and bh > 10 and dist < search_radius:
            text_regions.append({
                'bbox': {
                    'x': int(x1 + bx),
                    'y': int(y1 + by),
                    'width': int(bw),
                    'height': int(bh)
                },
                'distance': float(dist)
            })

    # Sort by distance
    text_regions.sort(key=lambda x: x['distance'])

    return {
        'text_regions': text_regions[:3],  # Keep top 3 closest
        'search_region': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
        'region_image': region
    }


def process_all_cities(img, cities, search_radius=150):
    """Process all cities and detect text regions."""
    print(f"\nDetecting text regions for {len(cities)} cities...")
    print(f"Search radius: {search_radius}px\n")

    results = []
    regions_with_text = 0

    for i, city in enumerate(cities):
        # Detect text near this city
        detection = extract_text_near_city(img, city, search_radius)

        # Create result entry
        result = {
            'id': city['id'],
            'center': city['center'],
            'confidence': city['confidence'],
            'text_regions': detection['text_regions'],
            'has_detected_text': len(detection['text_regions']) > 0
        }

        results.append(result)

        if result['has_detected_text']:
            regions_with_text += 1

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(cities)} cities ({regions_with_text} with text regions)...")

    print(f"\n✓ Processing complete")
    print(f"  Cities with detected text regions: {regions_with_text}/{len(cities)}")

    return results


def save_sample_regions(img, cities, results, output_dir="text_region_samples", num_samples=100):
    """Save sample text regions for manual review."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving sample text regions to {output_dir}/...")

    # Get cities with detected text
    cities_with_text = [
        (city, result) for city, result in zip(cities, results)
        if result['has_detected_text']
    ]

    # Sample evenly
    if len(cities_with_text) > num_samples:
        step = len(cities_with_text) // num_samples
        samples = cities_with_text[::step][:num_samples]
    else:
        samples = cities_with_text

    for city, result in samples:
        cx = city['center']['x']
        cy = city['center']['y']

        # Extract larger region for context
        radius = 200
        y1 = max(0, cy - radius)
        y2 = min(img.shape[0], cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(img.shape[1], cx + radius)

        region = img[y1:y2, x1:x2].copy()

        # Draw city center
        rel_cx = cx - x1
        rel_cy = cy - y1
        cv2.circle(region, (rel_cx, rel_cy), 8, (0, 255, 0), 2)

        # Draw detected text regions
        for text_region in result['text_regions']:
            bbox = text_region['bbox']
            rx1 = bbox['x'] - x1
            ry1 = bbox['y'] - y1
            rx2 = rx1 + bbox['width']
            ry2 = ry1 + bbox['height']

            cv2.rectangle(region, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

        # Save
        filename = output_path / f"city_{result['id']:04d}_conf{city['confidence']:.3f}.jpg"
        cv2.imwrite(str(filename), region)

    print(f"✓ Saved {len(samples)} sample regions")


def save_results(results, output_path="text_regions_detected.json"):
    """Save detection results."""
    # Calculate statistics
    cities_with_text = sum(1 for r in results if r['has_detected_text'])
    total_regions = sum(len(r['text_regions']) for r in results)

    output_data = {
        'total_cities': len(results),
        'cities_with_detected_text': cities_with_text,
        'total_text_regions_detected': total_regions,
        'cities': results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved results to {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("Extract Text Regions Near Cities")
    print("=" * 70)
    print()
    print("This script detects likely text regions near each city symbol")
    print("using image processing. Results can be used for manual transcription")
    print("or offline OCR processing.")
    print()

    # Paths
    map_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"
    cities_path = "filtered_cities.json"

    # Check files
    for path in [map_path, cities_path]:
        if not Path(path).exists():
            print(f"❌ Error: {path} not found")
            return 1

    try:
        # Load data
        print("Loading map and city detections...")
        img = cv2.imread(map_path)
        cities = load_filtered_cities(cities_path)
        print(f"✓ Loaded {len(cities)} cities\n")

        # Process all cities
        results = process_all_cities(img, cities, search_radius=150)

        # Save results
        save_results(results, "text_regions_detected.json")

        # Save sample regions
        save_sample_regions(img, cities, results, "text_region_samples", num_samples=100)

        print("\n" + "=" * 70)
        print("Text Region Detection Complete!")
        print("=" * 70)
        print("\nOutput files:")
        print("  - text_regions_detected.json (detected text region coordinates)")
        print("  - text_region_samples/ (100 sample images for review)")
        print("\nNext steps:")
        print("  1. Review sample images in text_region_samples/")
        print("  2. For full OCR, use an offline tool like Tesseract on the")
        print("     full map image, or use an online OCR service")
        print("  3. Match extracted text to cities based on proximity")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
