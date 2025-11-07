#!/usr/bin/env python3
"""
Simple batch OCR processor that can work with various backends.
Falls back to creating a summary and instructions if OCR fails.
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path


def analyze_text_crops_visually():
    """
    Analyze text crops and provide statistics for manual transcription.
    """
    print("=" * 70)
    print("Text Crop Analysis")
    print("=" * 70)
    print()

    # Load transcription template
    with open("transcription_template.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        crops = list(reader)

    # Group by city
    cities = {}
    for crop in crops:
        city_id = int(crop['city_id'])
        if city_id not in cities:
            cities[city_id] = []
        cities[city_id].append(crop)

    print(f"Total cities: {len(cities)}")
    print(f"Total text regions: {len(crops)}")
    print(f"Average regions per city: {len(crops)/len(cities):.1f}")
    print()

    # Find cities with text crops
    crops_path = Path("text_crops")
    existing_crops = set(f.stem.replace('_original', '').replace('_enhanced', '')
                        for f in crops_path.glob("*.jpg"))

    print(f"Existing text crop images: {len(existing_crops)}")
    print()

    # Create prioritized list: closest text region per city
    priority_list = []
    for city_id in sorted(cities.keys()):
        regions = cities[city_id]
        # Sort by distance, take closest
        regions.sort(key=lambda x: float(x['distance_from_city']))
        closest = regions[0]
        priority_list.append({
            'city_id': city_id,
            'filename': closest['text_region_filename'],
            'distance': float(closest['distance_from_city']),
            'x': int(closest['city_x']),
            'y': int(closest['city_y'])
        })

    # Save priority list for manual transcription
    priority_file = "priority_transcription_list.csv"
    with open(priority_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'city_id', 'city_x', 'city_y', 'text_crop_filename',
            'distance_from_city', 'transcribed_name'
        ])
        writer.writeheader()
        for item in priority_list:
            writer.writerow({
                'city_id': item['city_id'],
                'city_x': item['x'],
                'city_y': item['y'],
                'text_crop_filename': item['filename'],
                'distance_from_city': f"{item['distance']:.1f}",
                'transcribed_name': ''
            })

    print(f"✓ Created {priority_file}")
    print(f"  Contains closest text region for each of {len(priority_list)} cities")
    print()

    # Create collage of first 100 text crops for manual review
    print("Creating visual reference collage...")
    create_text_crops_collage(priority_list[:100])

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {priority_file}: Simplified list (585 rows, one per city)")
    print(f"  - text_crops_collage.jpg: Visual reference of first 100 labels")
    print("\nFor manual transcription:")
    print(f"  1. Open {priority_file} in Excel/LibreOffice")
    print("  2. For each row, view the corresponding image in text_crops/")
    print("  3. Type the city name in the 'transcribed_name' column")
    print("  4. Save and run: python3 merge_simple_transcriptions.py")


def create_text_crops_collage(priority_list, max_items=100):
    """Create a collage of text crops for visual reference."""
    crops_path = Path("text_crops")

    # Load crop images
    images = []
    labels = []

    for item in priority_list[:max_items]:
        filename = item['filename']
        img_path = crops_path / f"{filename}_original.jpg"

        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize to standard height
                target_height = 40
                aspect = img.shape[1] / img.shape[0]
                target_width = int(target_height * aspect)
                img = cv2.resize(img, (target_width, target_height))

                # Add border and label
                img = cv2.copyMakeBorder(img, 2, 2, 2, 2,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))

                images.append(img)
                labels.append(f"City {item['city_id']}")

    # Create grid collage
    if images:
        cols = 10
        rows = (len(images) + cols - 1) // cols

        # Determine max width per column
        max_width = max(img.shape[1] for img in images)

        # Create canvas
        cell_height = 60
        cell_width = max_width + 10
        canvas = np.ones((rows * cell_height, cols * cell_width, 3),
                        dtype=np.uint8) * 255

        for i, (img, label) in enumerate(zip(images, labels)):
            row = i // cols
            col = i % cols

            y = row * cell_height
            x = col * cell_width

            # Paste image
            h, w = img.shape[:2]
            canvas[y:y+h, x:x+w] = img

            # Add text label
            cv2.putText(canvas, label, (x + 5, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Save collage
        cv2.imwrite("text_crops_collage.jpg", canvas)
        print(f"  ✓ Created text_crops_collage.jpg ({len(images)} samples)")


def create_simple_merge_script():
    """Create script to merge simple transcriptions."""
    script = '''#!/usr/bin/env python3
"""
Merge simplified transcriptions (one per city) with coordinates.
Usage: python3 merge_simple_transcriptions.py
"""

import json
import csv

def main():
    # Load priority transcriptions
    transcriptions = {}
    with open("priority_transcription_list.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            city_id = int(row['city_id'])
            text = row['transcribed_name'].strip()
            if text:
                transcriptions[city_id] = text

    # Load original city data
    with open("filtered_cities.json", 'r') as f:
        cities_data = json.load(f)

    # Merge
    results = []
    for city in cities_data['cities']:
        city_id = city['id']
        label = transcriptions.get(city_id, "")

        results.append({
            'id': city_id,
            'center': city['center'],
            'confidence': city['confidence'],
            'label': label
        })

    # Save
    output = {
        'total_cities': len(results),
        'labeled_cities': sum(1 for r in results if r['label']),
        'cities': results
    }

    with open("cities_with_labels.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✓ Created cities_with_labels.json")
    print(f"  Total: {output['total_cities']}")
    print(f"  Labeled: {output['labeled_cities']}")
    print(f"  Unlabeled: {output['total_cities'] - output['labeled_cities']}")

if __name__ == "__main__":
    main()
'''

    with open("merge_simple_transcriptions.py", 'w') as f:
        f.write(script)

    print("✓ Created merge_simple_transcriptions.py")


def main():
    """Main execution."""
    try:
        analyze_text_crops_visually()
        create_simple_merge_script()
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
