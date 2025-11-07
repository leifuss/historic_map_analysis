#!/usr/bin/env python3
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
