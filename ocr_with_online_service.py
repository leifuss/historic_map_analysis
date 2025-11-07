#!/usr/bin/env python3
"""
Use free online OCR service (OCR.space) to extract text from city label crops.
OCR.space offers a free tier with 500 requests/day.
"""

import requests
import json
import time
import csv
from pathlib import Path
from tqdm import tqdm


def ocr_space_file(filename, api_key='helloworld', language='ara'):
    """
    OCR.space API request with local file.

    Args:
        filename: Path to image file
        api_key: OCR.space API key (default uses demo key with rate limits)
        language: Language code (ara=Arabic, eng=English)

    Returns:
        Parsed text or None
    """
    payload = {
        'apikey': api_key,
        'language': language,
        'isOverlayRequired': False,
        'OCREngine': 2,  # Engine 2 supports more languages
    }

    with open(filename, 'rb') as f:
        r = requests.post(
            'https://api.ocr.space/parse/image',
            files={'filename': f},
            data=payload,
            timeout=30
        )

    try:
        result = r.json()
        if result.get('IsErroredOnProcessing'):
            print(f"  Error: {result.get('ErrorMessage', 'Unknown error')}")
            return None

        if result.get('ParsedResults'):
            text = result['ParsedResults'][0].get('ParsedText', '').strip()
            return text if text else None

        return None

    except Exception as e:
        print(f"  Exception: {e}")
        return None


def process_text_crops(crops_dir="text_crops", output_csv="transcription_template.csv",
                      api_key='helloworld', delay=1.5):
    """
    Process all text crop images using OCR.space API.

    Args:
        crops_dir: Directory containing text crop images
        output_csv: Output CSV file to update
        api_key: OCR.space API key (get free key at https://ocr.space/ocrapi)
        delay: Delay between requests in seconds (rate limiting)
    """
    print("=" * 70)
    print("Online OCR Processing with OCR.space")
    print("=" * 70)
    print()

    if api_key == 'helloworld':
        print("⚠ Using demo API key (limited to 10 requests/hour)")
        print("For better results, get a free API key at https://ocr.space/ocrapi")
        print("(Free tier: 500 requests/day)\n")

    # Read existing CSV
    crops_info = []
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            crops_info.append(row)

    print(f"Found {len(crops_info)} text regions to process")
    print(f"Processing enhanced images with {delay}s delay between requests...\n")

    # Process each crop
    processed = 0
    successful = 0
    failed = 0

    crops_path = Path(crops_dir)

    # Group by city to show progress better
    for i, crop_info in enumerate(crops_info):
        filename = crop_info['text_region_filename']
        enhanced_path = crops_path / f"{filename}_enhanced.jpg"

        if not enhanced_path.exists():
            print(f"  Warning: {enhanced_path} not found, skipping...")
            failed += 1
            continue

        # Try OCR with both Arabic and English engines
        text = None

        # Try Arabic first (many city names are Arabic)
        text = ocr_space_file(str(enhanced_path), api_key=api_key, language='ara')

        if not text or len(text) < 2:
            # Try English/Latin
            time.sleep(delay)
            text = ocr_space_file(str(enhanced_path), api_key=api_key, language='eng')

        # Update CSV row
        if text and len(text) > 1:
            crop_info['transcribed_text'] = text
            crop_info['confidence'] = 'auto'
            successful += 1
            status = "✓"
        else:
            crop_info['transcribed_text'] = ''
            failed += 1
            status = "✗"

        processed += 1

        # Show progress
        if processed % 10 == 0 or processed == len(crops_info):
            print(f"  [{processed}/{len(crops_info)}] {status} City {crop_info['city_id']} region: '{text if text else '(none)'}'")

        # Rate limiting
        if processed < len(crops_info):
            time.sleep(delay)

    # Save updated CSV
    print(f"\nSaving results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['city_id', 'city_x', 'city_y', 'text_region_filename',
                     'distance_from_city', 'transcribed_text', 'confidence', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(crops_info)

    print("\n" + "=" * 70)
    print("OCR Processing Complete!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Total regions: {len(crops_info)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {100*successful/len(crops_info):.1f}%")
    print(f"\nUpdated: {output_csv}")
    print(f"\nNext step: python3 merge_transcriptions.py {output_csv}")


def process_sample_only(num_samples=50, crops_dir="text_crops",
                        output_csv="transcription_sample.csv", api_key='helloworld'):
    """
    Process only a sample of text crops for testing.
    """
    print("=" * 70)
    print(f"Processing {num_samples} Sample Text Crops")
    print("=" * 70)
    print()

    # Read CSV and take sample
    with open("transcription_template.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_crops = list(reader)

    # Sample: take first region from first N cities
    seen_cities = set()
    sample_crops = []
    for crop in all_crops:
        city_id = crop['city_id']
        if city_id not in seen_cities:
            sample_crops.append(crop)
            seen_cities.add(city_id)
        if len(sample_crops) >= num_samples:
            break

    print(f"Selected {len(sample_crops)} samples from different cities\n")

    # Process sample
    crops_path = Path(crops_dir)
    results = []

    for i, crop_info in enumerate(sample_crops):
        filename = crop_info['text_region_filename']
        enhanced_path = crops_path / f"{filename}_enhanced.jpg"

        if not enhanced_path.exists():
            continue

        # Try OCR
        text = ocr_space_file(str(enhanced_path), api_key=api_key, language='ara')

        if not text:
            time.sleep(1.5)
            text = ocr_space_file(str(enhanced_path), api_key=api_key, language='eng')

        crop_info['transcribed_text'] = text if text else ''
        crop_info['confidence'] = 'auto' if text else ''
        results.append(crop_info)

        print(f"  [{i+1}/{len(sample_crops)}] City {crop_info['city_id']}: '{text if text else '(none)'}'")

        if i < len(sample_crops) - 1:
            time.sleep(1.5)

    # Save sample results
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['city_id', 'city_x', 'city_y', 'text_region_filename',
                     'distance_from_city', 'transcribed_text', 'confidence', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    successful = sum(1 for r in results if r['transcribed_text'])

    print(f"\n✓ Saved sample results to {output_csv}")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"\nTo process all {len(all_crops)} regions:")
    print(f"  1. Get free API key from https://ocr.space/ocrapi")
    print(f"  2. Run: python3 ocr_with_online_service.py --full --api-key YOUR_KEY")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='OCR text crops using OCR.space API')
    parser.add_argument('--full', action='store_true',
                       help='Process all text crops (default: sample only)')
    parser.add_argument('--api-key', default='helloworld',
                       help='OCR.space API key (default: demo key)')
    parser.add_argument('--delay', type=float, default=1.5,
                       help='Delay between requests in seconds (default: 1.5)')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of samples to process (default: 50)')

    args = parser.parse_args()

    try:
        if args.full:
            # Process all crops
            process_text_crops(
                crops_dir="text_crops",
                output_csv="transcription_template.csv",
                api_key=args.api_key,
                delay=args.delay
            )
        else:
            # Process sample only
            process_sample_only(
                num_samples=args.sample_size,
                crops_dir="text_crops",
                output_csv="transcription_sample.csv",
                api_key=args.api_key
            )

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
