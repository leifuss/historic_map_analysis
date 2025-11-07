#!/usr/bin/env python3
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
