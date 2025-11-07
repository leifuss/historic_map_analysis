#!/usr/bin/env python3
"""
Prepare text regions for batch OCR processing.

This script:
1. Extracts individual text region images for each city
2. Creates a template CSV for manual transcription or OCR results
3. Generates instructions for offline OCR processing
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path


def load_data():
    """Load cities and detected text regions."""
    with open("filtered_cities.json", 'r') as f:
        cities_data = json.load(f)

    with open("text_regions_detected.json", 'r') as f:
        regions_data = json.load(f)

    return cities_data, regions_data


def extract_text_region_crops(img, regions_data, output_dir="text_crops"):
    """Extract individual text region images."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nExtracting text region crops to {output_dir}/...")

    crops_info = []
    total_crops = 0

    for city in regions_data['cities']:
        city_id = city['id']
        cx = city['center']['x']
        cy = city['center']['y']

        # Extract each detected text region
        for region_idx, text_region in enumerate(city['text_regions']):
            bbox = text_region['bbox']
            x = bbox['x']
            y = bbox['y']
            w = bbox['width']
            h = bbox['height']

            # Add padding around text
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)

            # Extract crop
            crop = img[y1:y2, x1:x2].copy()

            # Resize if too small (for better OCR)
            if crop.shape[0] < 30 or crop.shape[1] < 50:
                scale = max(30 / crop.shape[0], 50 / crop.shape[1])
                new_w = int(crop.shape[1] * scale)
                new_h = int(crop.shape[0] * scale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Enhance for OCR
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            enhanced = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Save both original and enhanced versions
            crop_name = f"city{city_id:04d}_region{region_idx}"
            cv2.imwrite(str(output_path / f"{crop_name}_original.jpg"), crop)
            cv2.imwrite(str(output_path / f"{crop_name}_enhanced.jpg"), enhanced)

            crops_info.append({
                'city_id': city_id,
                'region_idx': region_idx,
                'filename': crop_name,
                'city_center': {'x': cx, 'y': cy},
                'bbox': bbox,
                'distance': text_region['distance']
            })

            total_crops += 1

        # Progress
        if (city_id + 1) % 100 == 0:
            print(f"  Processed {city_id + 1} cities, extracted {total_crops} text crops...")

    print(f"\n✓ Extracted {total_crops} text region crops")
    print(f"  (2 files per region: original + enhanced for OCR)")

    return crops_info


def create_transcription_template(crops_info, cities_data, output_file="transcription_template.csv"):
    """Create CSV template for manual transcription or OCR results."""
    print(f"\nCreating transcription template: {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'city_id',
            'city_x',
            'city_y',
            'text_region_filename',
            'distance_from_city',
            'transcribed_text',
            'confidence',
            'notes'
        ])

        # Rows for each text region
        for crop in crops_info:
            writer.writerow([
                crop['city_id'],
                crop['city_center']['x'],
                crop['city_center']['y'],
                crop['filename'],
                f"{crop['distance']:.1f}",
                '',  # Empty for manual transcription
                '',  # Empty for confidence
                ''   # Empty for notes
            ])

    print(f"✓ Created template with {len(crops_info)} rows")


def create_instructions(crops_info):
    """Create instructions file for OCR processing."""
    instructions = """
================================================================================
OCR PROCESSING INSTRUCTIONS FOR IDRISI MAP CITY LABELS
================================================================================

OVERVIEW:
This directory contains text region crops extracted from the Idrisi map,
located near each of the 585 detected city symbols. Use these to extract
city name labels.

FILES:
------
- text_crops/          : Directory with {total_crops} text region images
                         (city*_original.jpg and city*_enhanced.jpg)
- transcription_template.csv : Template for recording OCR results
- text_regions_detected.json : Full data with coordinates

IMAGE NAMING:
-------------
  city0123_region0_original.jpg   - Original crop of text near city #123
  city0123_region0_enhanced.jpg   - Enhanced/preprocessed for OCR

OPTION 1: OFFLINE OCR WITH TESSERACT
-------------------------------------
Install Tesseract OCR:
  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-lat
  macOS: brew install tesseract tesseract-lang
  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

Batch process all enhanced images:
  cd text_crops
  for f in *_enhanced.jpg; do
    tesseract "$f" "${{f%.jpg}}" -l lat+ara+eng
  done

This creates .txt files with OCR results for each image.

OPTION 2: MANUAL TRANSCRIPTION
-------------------------------
1. Open transcription_template.csv in Excel/LibreOffice
2. For each row:
   - Open the corresponding image in text_crops/
   - Type the city name in the 'transcribed_text' column
   - Add confidence level (high/medium/low) if desired
   - Add notes for unclear text
3. Save the CSV

OPTION 3: ONLINE OCR SERVICES
------------------------------
Upload images to services like:
  - Google Cloud Vision API (best accuracy, costs apply)
  - Microsoft Azure Computer Vision
  - Amazon Textract
  - Free services: OnlineOCR.net, NewOCR.com

Then fill results into transcription_template.csv

TIPS FOR ACCURACY:
------------------
1. The Idrisi map uses medieval Latin script and Arabic
2. City names may include diacritical marks (ā, ī, ū, etc.)
3. Use *_enhanced.jpg for OCR, *_original.jpg for visual verification
4. Each city may have multiple text regions - use the closest one
5. Some text may be decorative labels or region names, not city names

NEXT STEPS:
-----------
After transcription, create final output with:
  python3 merge_transcriptions.py transcription_template.csv

This will generate cities_with_labels.json combining coordinates + names.

================================================================================
""".format(total_crops=len(crops_info))

    with open("OCR_INSTRUCTIONS.txt", 'w') as f:
        f.write(instructions)

    print("\n✓ Created OCR_INSTRUCTIONS.txt")


def create_merge_script():
    """Create script to merge transcriptions back with city data."""
    script_content = '''#!/usr/bin/env python3
"""
Merge transcribed text back with city coordinates.
Usage: python3 merge_transcriptions.py transcription_template.csv
"""

import json
import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 merge_transcriptions.py <transcription_csv>")
        return 1

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        return 1

    # Load original city data
    with open("filtered_cities.json", 'r') as f:
        cities_data = json.load(f)

    # Create mapping of city_id to city data
    cities_by_id = {city['id']: city for city in cities_data['cities']}

    # Load transcriptions
    transcriptions = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            city_id = int(row['city_id'])
            text = row['transcribed_text'].strip()

            if text:  # Only keep non-empty transcriptions
                if city_id not in transcriptions:
                    transcriptions[city_id] = []

                transcriptions[city_id].append({
                    'text': text,
                    'confidence': row.get('confidence', ''),
                    'distance': float(row['distance_from_city']),
                    'source_file': row['text_region_filename']
                })

    # Merge data
    results = []
    for city_id, city in cities_by_id.items():
        # Get best transcription (closest to city center)
        if city_id in transcriptions:
            trans_list = transcriptions[city_id]
            trans_list.sort(key=lambda x: x['distance'])
            best = trans_list[0]

            label = best['text']
            label_conf = best.get('confidence', '')
            all_candidates = [t['text'] for t in trans_list]
        else:
            label = ""
            label_conf = ""
            all_candidates = []

        results.append({
            'id': city_id,
            'center': city['center'],
            'confidence': city['confidence'],
            'label': label,
            'label_confidence': label_conf,
            'all_detected_labels': all_candidates
        })

    # Save results
    output = {
        'total_cities': len(results),
        'labeled_cities': sum(1 for r in results if r['label']),
        'unlabeled_cities': sum(1 for r in results if not r['label']),
        'cities': results
    }

    with open("cities_with_labels.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✓ Created cities_with_labels.json")
    print(f"  Total cities: {output['total_cities']}")
    print(f"  Labeled: {output['labeled_cities']}")
    print(f"  Unlabeled: {output['unlabeled_cities']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    with open("merge_transcriptions.py", 'w') as f:
        f.write(script_content)

    print("✓ Created merge_transcriptions.py")


def main():
    """Main execution."""
    print("=" * 70)
    print("Prepare Text Regions for OCR Processing")
    print("=" * 70)
    print()

    # Check files
    required_files = [
        "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg",
        "filtered_cities.json",
        "text_regions_detected.json"
    ]

    for path in required_files:
        if not Path(path).exists():
            print(f"❌ Error: {path} not found")
            return 1

    try:
        # Load data
        print("Loading data...")
        cities_data, regions_data = load_data()
        img = cv2.imread("al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg")
        print(f"✓ Loaded {regions_data['total_cities']} cities with {regions_data['total_text_regions_detected']} text regions")

        # Extract crops
        crops_info = extract_text_region_crops(img, regions_data)

        # Create template
        create_transcription_template(crops_info, cities_data)

        # Create instructions
        create_instructions(crops_info)

        # Create merge script
        create_merge_script()

        print("\n" + "=" * 70)
        print("OCR Preparation Complete!")
        print("=" * 70)
        print("\nGenerated files:")
        print(f"  - text_crops/ ({len(crops_info) * 2} images)")
        print("  - transcription_template.csv")
        print("  - OCR_INSTRUCTIONS.txt")
        print("  - merge_transcriptions.py")
        print("\nNext steps:")
        print("  1. Read OCR_INSTRUCTIONS.txt for detailed guidance")
        print("  2. Use offline OCR tool or manual transcription")
        print("  3. Fill in transcription_template.csv")
        print("  4. Run: python3 merge_transcriptions.py transcription_template.csv")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
