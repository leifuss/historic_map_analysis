#!/usr/bin/env python3
"""
Symbol-to-Label Matching for Idrisi Map
Associates detected city symbols with their corresponding text labels
using spatial proximity and intelligent heuristics.
"""

import json
import math
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    return math.sqrt(dx*dx + dy*dy)


def find_nearest_labels(symbol, labels, max_distance=150, top_k=5):
    """
    Find the k nearest text labels to a symbol.

    Args:
        symbol: Symbol dict with 'center' key
        labels: List of label dicts with 'center' key
        max_distance: Maximum distance to consider (in pixels)
        top_k: Number of nearest labels to return

    Returns:
        list: Nearest labels sorted by distance
    """
    distances = []

    for label in labels:
        dist = calculate_distance(symbol['center'], label['center'])

        if dist <= max_distance:
            distances.append({
                'label': label,
                'distance': dist
            })

    # Sort by distance
    distances.sort(key=lambda x: x['distance'])

    return distances[:top_k]


def match_symbols_to_labels(symbols, labels, strategy='nearest', max_distance=150):
    """
    Match each symbol to its most likely corresponding text label.

    Args:
        symbols: List of detected city symbols
        labels: List of detected text labels
        strategy: Matching strategy ('nearest', 'directional', 'hybrid')
        max_distance: Maximum distance for matching (pixels)

    Returns:
        list: Matched pairs with confidence scores
    """
    print(f"\nMatching {len(symbols)} symbols to {len(labels)} labels...")
    print(f"Strategy: {strategy}, Max distance: {max_distance}px")

    matched_cities = []
    used_labels = set()

    for symbol in symbols:
        nearest = find_nearest_labels(symbol, labels, max_distance=max_distance, top_k=3)

        if not nearest:
            # No labels found within max_distance
            matched_cities.append({
                'symbol_id': symbol['id'],
                'symbol_coords': symbol['center'],
                'city_name': None,
                'label_id': None,
                'distance': None,
                'confidence': 0.0,
                'status': 'unmatched'
            })
            continue

        # Select best match (preferring unused labels)
        best_match = None
        for candidate in nearest:
            if candidate['label']['id'] not in used_labels:
                best_match = candidate
                break

        # If all nearest labels are used, take the closest anyway
        if best_match is None:
            best_match = nearest[0]

        label = best_match['label']
        distance = best_match['distance']

        # Calculate confidence score based on distance
        # Closer = higher confidence (1.0 at distance 0, decreasing linearly)
        confidence = max(0.0, 1.0 - (distance / max_distance))

        # Mark label as used (to avoid multiple symbols claiming same label)
        used_labels.add(label['id'])

        matched_cities.append({
            'symbol_id': symbol['id'],
            'symbol_coords': symbol['center'],
            'city_name': label['text'],
            'label_id': label['id'],
            'label_coords': label['center'],
            'distance': distance,
            'confidence': confidence,
            'status': 'matched',
            'alternative_matches': [
                {
                    'text': n['label']['text'],
                    'distance': n['distance']
                }
                for n in nearest[1:3]
            ]
        })

    # Report matching statistics
    matched_count = sum(1 for c in matched_cities if c['status'] == 'matched')
    high_confidence = sum(1 for c in matched_cities if c.get('confidence', 0) > 0.7)

    print(f"\n✓ Matched {matched_count}/{len(symbols)} symbols to labels")
    print(f"  High confidence matches (>0.7): {high_confidence}")
    print(f"  Unmatched symbols: {len(symbols) - matched_count}")

    return matched_cities


def save_matched_cities(matched_cities, output_path="matched_cities.json"):
    """Save matched city data to JSON file."""

    # Calculate statistics
    matched = [c for c in matched_cities if c['status'] == 'matched']
    avg_distance = sum(c['distance'] for c in matched) / len(matched) if matched else 0
    avg_confidence = sum(c['confidence'] for c in matched) / len(matched) if matched else 0

    output_data = {
        "total_cities": len(matched_cities),
        "matched_cities": len(matched),
        "unmatched_symbols": len([c for c in matched_cities if c['status'] == 'unmatched']),
        "average_distance": avg_distance,
        "average_confidence": avg_confidence,
        "cities": matched_cities,
        "description": "City symbols matched to text labels on Idrisi map"
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(matched_cities)} matched cities to {output_path}")

    # Print sample matches
    print("\nSample matched cities:")
    for city in [c for c in matched_cities if c['status'] == 'matched'][:10]:
        print(f"  '{city['city_name']}' at ({city['symbol_coords']['x']:.0f}, "
              f"{city['symbol_coords']['y']:.0f}) - confidence: {city['confidence']:.2f}")

    return output_data


def create_visualization(image_path, matched_cities, output_path="visualization_matched_cities.jpg"):
    """
    Create a visualization showing matched symbols and labels.

    Args:
        image_path: Path to original map image
        matched_cities: List of matched city data
        output_path: Path to save visualization
    """
    print(f"\nCreating visualization...")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠ Warning: Could not load {image_path} for visualization")
        return

    # Create overlay
    overlay = img.copy()

    for city in matched_cities:
        symbol_x = int(city['symbol_coords']['x'])
        symbol_y = int(city['symbol_coords']['y'])

        if city['status'] == 'matched':
            # Draw green circle around symbol
            cv2.circle(overlay, (symbol_x, symbol_y), 20, (0, 255, 0), 2)
            cv2.circle(overlay, (symbol_x, symbol_y), 3, (0, 255, 0), -1)

            # Draw line to label
            label_x = int(city['label_coords']['x'])
            label_y = int(city['label_coords']['y'])
            cv2.line(overlay, (symbol_x, symbol_y), (label_x, label_y),
                    (0, 255, 0), 1, cv2.LINE_AA)

            # Add city name annotation
            cv2.putText(overlay, city['city_name'], (symbol_x + 10, symbol_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            # Draw red circle for unmatched symbols
            cv2.circle(overlay, (symbol_x, symbol_y), 20, (0, 0, 255), 2)
            cv2.circle(overlay, (symbol_x, symbol_y), 3, (0, 0, 255), -1)

    # Blend overlay with original image
    result = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Save visualization
    cv2.imwrite(output_path, result)
    print(f"✓ Visualization saved to {output_path}")


def export_to_csv(matched_cities, output_path="cities_coordinates.csv"):
    """Export matched cities to CSV format."""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['City Name', 'X Coordinate', 'Y Coordinate', 'Confidence', 'Status'])

        for city in matched_cities:
            writer.writerow([
                city.get('city_name', 'Unknown'),
                f"{city['symbol_coords']['x']:.2f}",
                f"{city['symbol_coords']['y']:.2f}",
                f"{city.get('confidence', 0):.3f}",
                city['status']
            ])

    print(f"✓ Exported to CSV: {output_path}")


def main():
    """Main execution function."""

    print("=" * 70)
    print("Symbol-to-Label Matching - Idrisi Map")
    print("=" * 70)
    print()

    # Check for required input files
    symbols_file = "detected_symbols.json"
    labels_file = "detected_text_labels.json"
    image_file = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"

    if not Path(symbols_file).exists():
        print(f"❌ Error: {symbols_file} not found")
        print("Run detect_city_symbols.py first to detect symbols")
        return 1

    if not Path(labels_file).exists():
        print(f"❌ Error: {labels_file} not found")
        print("Run extract_text_labels.py first to extract text")
        return 1

    try:
        # Load input data
        print("Loading detected symbols and labels...")
        symbols_data = load_json(symbols_file)
        labels_data = load_json(labels_file)

        symbols = symbols_data['symbols']
        labels = labels_data['labels']

        print(f"Loaded {len(symbols)} symbols and {len(labels)} labels")

        # Match symbols to labels
        matched_cities = match_symbols_to_labels(
            symbols,
            labels,
            strategy='nearest',
            max_distance=150  # Adjust based on map scale
        )

        # Save results
        save_matched_cities(matched_cities)

        # Export to CSV
        export_to_csv(matched_cities)

        # Create visualization
        if Path(image_file).exists():
            create_visualization(image_file, matched_cities)

        print("\n" + "=" * 70)
        print("Matching Complete!")
        print("=" * 70)
        print(f"\nOutput files generated:")
        print(f"  - matched_cities.json (detailed matching data)")
        print(f"  - cities_coordinates.csv (simple coordinate list)")
        print(f"  - visualization_matched_cities.jpg (visual verification)")
        print("\nReview the visualization to verify matches and adjust parameters if needed.")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
