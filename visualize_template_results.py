#!/usr/bin/env python3
"""
Create enhanced visualizations of template-matched cities.
"""

import cv2
import json
import numpy as np
from pathlib import Path


def create_zoomed_samples(image_path, cities_json, num_regions=6):
    """Create zoomed views of different map regions showing detected cities."""

    print("Creating zoomed region samples...")

    # Load map and cities
    img = cv2.imread(image_path)
    with open(cities_json, 'r') as f:
        data = json.load(f)

    cities = data['cities']
    height, width = img.shape[:2]

    # Divide map into grid regions
    regions_per_row = 3
    regions_per_col = 2

    region_width = width // regions_per_row
    region_height = height // regions_per_col

    zoom_files = []

    for row in range(regions_per_col):
        for col in range(regions_per_row):
            region_num = row * regions_per_row + col

            # Define region bounds
            x1 = col * region_width
            x2 = (col + 1) * region_width
            y1 = row * region_height
            y2 = (row + 1) * region_height

            # Find cities in this region
            region_cities = [
                c for c in cities
                if x1 <= c['center']['x'] < x2 and y1 <= c['center']['y'] < y2
            ]

            if len(region_cities) == 0:
                continue

            print(f"  Region {region_num} (row {row}, col {col}): {len(region_cities)} cities")

            # Crop region
            cropped = img[y1:y2, x1:x2].copy()

            # Draw cities
            for city in region_cities:
                x = city['center']['x'] - x1
                y = city['center']['y'] - y1
                w = city['width']
                h = city['height']
                conf = city['confidence']

                # Color by confidence
                if conf > 0.85:
                    color = (0, 255, 0)  # Green
                elif conf > 0.75:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange

                # Draw box and center
                top_left = (city['top_left']['x'] - x1, city['top_left']['y'] - y1)
                cv2.rectangle(cropped, top_left, (top_left[0] + w, top_left[1] + h), color, 2)
                cv2.circle(cropped, (x, y), 4, (255, 0, 255), -1)

                # Add ID
                cv2.putText(cropped, str(city['id']), (x + 8, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Add region info
            cv2.putText(cropped, f"Region {region_num}: {len(region_cities)} cities", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(cropped, f"Region {region_num}: {len(region_cities)} cities", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2, cv2.LINE_AA)

            # Save
            output_file = f"zoom_region_{region_num}.jpg"
            cv2.imwrite(output_file, cropped)
            zoom_files.append(output_file)
            print(f"    ✓ Saved {output_file}")

    return zoom_files


def create_density_visualization(image_path, cities_json, output_path="cities_density_map.jpg"):
    """Create a density heatmap of detected cities."""

    print("\nCreating density heatmap...")

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    with open(cities_json, 'r') as f:
        data = json.load(f)
    cities = data['cities']

    # Create heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    for city in cities:
        x = city['center']['x']
        y = city['center']['y']

        # Gaussian influence
        size = 150
        y1, y2 = max(0, y - size), min(height, y + size)
        x1, x2 = max(0, x - size), min(width, x + size)

        for dy in range(y1, y2):
            for dx in range(x1, x2):
                dist = np.sqrt((dx - x)**2 + (dy - y)**2)
                if dist < size:
                    heatmap[dy, dx] += np.exp(-(dist**2) / (2 * (size/3)**2))

    # Normalize and colorize
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend
    result = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

    # Add legend
    cv2.putText(result, f"City Density: {len(cities)} cities detected", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8, cv2.LINE_AA)

    cv2.imwrite(output_path, result)
    print(f"✓ Saved {output_path}")


def show_statistics(cities_json):
    """Print detailed statistics about detections."""

    with open(cities_json, 'r') as f:
        data = json.load(f)

    cities = data['cities']

    print("\n" + "=" * 70)
    print("DETECTION STATISTICS")
    print("=" * 70)

    print(f"\nTotal cities detected: {len(cities)}")

    # Confidence distribution
    confidences = [c['confidence'] for c in cities]
    print(f"\nConfidence scores:")
    print(f"  Minimum: {min(confidences):.3f}")
    print(f"  Maximum: {max(confidences):.3f}")
    print(f"  Average: {np.mean(confidences):.3f}")
    print(f"  Median:  {np.median(confidences):.3f}")

    # Confidence bins
    high = sum(1 for c in confidences if c > 0.85)
    medium = sum(1 for c in confidences if 0.75 <= c <= 0.85)
    low = sum(1 for c in confidences if c < 0.75)

    print(f"\nConfidence distribution:")
    print(f"  High (>0.85):   {high:4d} ({high/len(cities)*100:.1f}%)")
    print(f"  Medium (0.75-0.85): {medium:4d} ({medium/len(cities)*100:.1f}%)")
    print(f"  Low (<0.75):    {low:4d} ({low/len(cities)*100:.1f}%)")

    # Scale distribution
    scales = [c['scale'] for c in cities]
    print(f"\nScale distribution:")
    for s in sorted(set(scales)):
        count = scales.count(s)
        print(f"  Scale {s:.2f}: {count:4d} ({count/len(cities)*100:.1f}%)")

    # Top detections
    print(f"\nTop 20 highest confidence detections:")
    sorted_cities = sorted(cities, key=lambda c: c['confidence'], reverse=True)
    for i, city in enumerate(sorted_cities[:20]):
        print(f"  {i+1:2d}. City {city['id']:4d}: ({city['center']['x']:5d}, {city['center']['y']:5d}) "
              f"conf={city['confidence']:.3f} scale={city['scale']:.2f}")


def main():
    print("=" * 70)
    print("Template Detection Results Visualization")
    print("=" * 70)
    print()

    map_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"
    cities_json = "detected_cities_template.json"

    if not Path(map_path).exists():
        print(f"❌ Map not found: {map_path}")
        return 1

    if not Path(cities_json).exists():
        print(f"❌ Cities data not found: {cities_json}")
        print("Run detect_cities_template.py first")
        return 1

    try:
        # Show statistics
        show_statistics(cities_json)

        # Create zoomed region views
        print("\n" + "=" * 70)
        zoom_files = create_zoomed_samples(map_path, cities_json)

        # Create density map
        print("\n" + "=" * 70)
        create_density_visualization(map_path, cities_json)

        print("\n" + "=" * 70)
        print("VISUALIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nGenerated files:")
        print(f"  📊 {len(zoom_files)} zoomed region files (zoom_region_*.jpg)")
        print(f"  🌡️  cities_density_map.jpg")
        print(f"\nReview these files to verify the city detections!")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
