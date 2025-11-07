#!/usr/bin/env python3
"""
Extract text labels for detected city symbols using OCR.
Associates each city coordinate with its corresponding name label.
"""

import cv2
import numpy as np
import json
import easyocr
from pathlib import Path
from collections import defaultdict
import re


def load_filtered_cities(json_path="filtered_cities.json"):
    """Load filtered city detections."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['cities']


def preprocess_for_ocr(region):
    """Preprocess image region for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance text
    # Try to make dark text on light background
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

    # Optionally invert if text is light on dark
    # Check if most pixels are dark (mean < 128)
    if np.mean(denoised) < 128:
        denoised = cv2.bitwise_not(denoised)

    return denoised


def extract_text_near_city(img, city, reader, search_radius=150):
    """
    Extract text from region around city symbol.

    Args:
        img: Full map image
        city: City detection dict with 'center' coordinates
        reader: EasyOCR reader instance
        search_radius: Radius in pixels to search for text

    Returns:
        dict with extracted text and confidence
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

    # Preprocess for OCR
    preprocessed = preprocess_for_ocr(region)

    # Try OCR with EasyOCR
    try:
        # EasyOCR returns list of (bbox, text, confidence)
        results = reader.readtext(
            preprocessed,
            detail=1,  # Return detailed results with bounding boxes
            paragraph=False,  # Detect individual text instances
            min_size=10  # Minimum text size in pixels
        )

        # Filter and process results
        texts = []
        for bbox, text, conf in results:
            text = text.strip()

            # Only keep text with reasonable confidence and length
            if conf > 0.1 and len(text) > 1:
                # Calculate center of bounding box
                bbox_center_x = int(np.mean([pt[0] for pt in bbox]))
                bbox_center_y = int(np.mean([pt[1] for pt in bbox]))

                # Convert to global coordinates
                text_x = x1 + bbox_center_x
                text_y = y1 + bbox_center_y

                # Calculate distance from city center
                dist = np.sqrt((text_x - cx)**2 + (text_y - cy)**2)

                texts.append({
                    'text': text,
                    'confidence': float(conf),
                    'distance': float(dist),
                    'position': {'x': int(text_x), 'y': int(text_y)}
                })

        # Sort by distance and confidence
        texts.sort(key=lambda x: (x['distance'], -x['confidence']))

        return {
            'texts': texts,
            'region_bounds': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        }

    except Exception as e:
        print(f"  OCR error at ({cx}, {cy}): {e}")
        return {
            'texts': [],
            'region_bounds': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        }


def clean_city_name(text):
    """Clean and normalize extracted city name."""
    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove obvious OCR artifacts (single characters, numbers)
    if len(text) <= 1:
        return None

    # Remove if mostly numbers or special characters
    if sum(c.isalnum() for c in text) / len(text) < 0.5:
        return None

    return text


def extract_all_labels(img, cities, reader, search_radius=150):
    """Extract labels for all cities."""
    print(f"\nExtracting text labels for {len(cities)} cities...")
    print(f"Search radius: {search_radius}px\n")

    results = []

    for i, city in enumerate(cities):
        # Extract text near this city
        ocr_result = extract_text_near_city(img, city, reader, search_radius)

        # Find best text candidate
        best_text = None
        best_confidence = 0
        best_position = None

        if ocr_result['texts']:
            # Take the closest text with decent confidence
            for text_data in ocr_result['texts'][:3]:  # Check top 3 candidates
                cleaned = clean_city_name(text_data['text'])
                if cleaned and text_data['confidence'] > best_confidence:
                    best_text = cleaned
                    best_confidence = text_data['confidence']
                    best_position = text_data['position']

        # Create result entry
        result = {
            'id': city['id'],
            'center': city['center'],
            'confidence': city['confidence'],
            'label': best_text if best_text else "",
            'label_confidence': float(best_confidence) if best_confidence else 0.0,
            'label_position': best_position,
            'ocr_candidates': [
                {'text': t['text'], 'conf': float(t['confidence']), 'dist': float(t['distance'])}
                for t in ocr_result['texts'][:5]
            ]
        }

        results.append(result)

        # Progress update
        if (i + 1) % 50 == 0:
            labeled = sum(1 for r in results if r['label'])
            print(f"  Processed {i+1}/{len(cities)} cities ({labeled} with labels)...")

    return results


def save_results(results, output_path="cities_with_labels.json"):
    """Save extraction results."""
    # Calculate statistics
    labeled_count = sum(1 for r in results if r['label'])
    avg_label_conf = np.mean([r['label_confidence'] for r in results if r['label']])

    output_data = {
        'total_cities': len(results),
        'cities_with_labels': labeled_count,
        'cities_without_labels': len(results) - labeled_count,
        'avg_label_confidence': float(avg_label_conf) if labeled_count > 0 else 0,
        'cities': results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved results to {output_path}")
    print(f"\nStatistics:")
    print(f"  Total cities: {len(results)}")
    print(f"  With labels: {labeled_count} ({100*labeled_count/len(results):.1f}%)")
    print(f"  Without labels: {len(results) - labeled_count}")
    if labeled_count > 0:
        print(f"  Avg label confidence: {avg_label_conf:.1f}")


def create_visualization(img, results, output_path="cities_with_labels_viz.jpg", max_display=100):
    """Create visualization showing some cities with their extracted labels."""
    print(f"\nCreating visualization...")

    # Work on a copy
    vis = img.copy()

    # Filter to cities with labels
    labeled_cities = [r for r in results if r['label']]

    # Sample for visualization (show up to max_display)
    if len(labeled_cities) > max_display:
        # Sample evenly across the image
        step = len(labeled_cities) // max_display
        sample = labeled_cities[::step][:max_display]
    else:
        sample = labeled_cities

    print(f"  Displaying {len(sample)} labeled cities...")

    # Draw cities and labels
    for city in sample:
        cx = city['center']['x']
        cy = city['center']['y']

        # Draw circle at city location
        cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)

        # Draw label
        if city['label']:
            label_text = city['label'][:30]  # Truncate long names

            # Position text above city
            text_y = cy - 20
            text_x = cx - 40

            # Draw text background
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(vis,
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (255, 255, 255), -1)

            # Draw text
            cv2.putText(vis, label_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Resize for display if too large
    max_dimension = 4000
    if max(vis.shape[:2]) > max_dimension:
        scale = max_dimension / max(vis.shape[:2])
        new_width = int(vis.shape[1] * scale)
        new_height = int(vis.shape[0] * scale)
        vis = cv2.resize(vis, (new_width, new_height))

    cv2.imwrite(output_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"✓ Saved visualization to {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("Extract City Labels using OCR")
    print("=" * 70)
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
        # Initialize EasyOCR reader
        print("Initializing EasyOCR reader...")
        print("  Languages: Latin, English, Arabic")
        print("  (First run will download models, ~100MB)\n")

        # Initialize reader with Latin, English, and Arabic for historical names
        reader = easyocr.Reader(['la', 'en', 'ar'], gpu=True)
        print("✓ EasyOCR ready\n")

        # Load data
        print("Loading map and city detections...")
        img = cv2.imread(map_path)
        cities = load_filtered_cities(cities_path)
        print(f"✓ Loaded {len(cities)} cities\n")

        # Extract labels
        results = extract_all_labels(img, cities, reader, search_radius=150)

        # Save results
        save_results(results, "cities_with_labels.json")

        # Create visualization
        create_visualization(img, results, "cities_with_labels_viz.jpg", max_display=100)

        print("\n" + "=" * 70)
        print("Label Extraction Complete!")
        print("=" * 70)
        print("\nOutput files:")
        print("  - cities_with_labels.json (coordinates + text labels)")
        print("  - cities_with_labels_viz.jpg (visualization of sample)")
        print("\nNext: Review results and refine as needed")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
