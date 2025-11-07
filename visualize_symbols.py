#!/usr/bin/env python3
"""
Create a clean visualization of detected city symbols on the Idrisi map.
"""

import cv2
import json
import numpy as np
from pathlib import Path


def create_symbol_visualization(image_path, symbols_json, output_path="city_symbols_visualization.jpg"):
    """
    Create a clear visualization showing all detected city symbols.

    Args:
        image_path: Path to the original map
        symbols_json: Path to detected_symbols.json
        output_path: Where to save the visualization
    """
    print("Creating city symbols visualization...")

    # Load the map
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Map dimensions: {img.shape[1]}x{img.shape[0]}")

    # Load symbols data
    with open(symbols_json, 'r') as f:
        data = json.load(f)

    symbols = data['symbols']
    print(f"Loaded {len(symbols)} symbols")

    # Create visualization overlay
    overlay = img.copy()

    # Draw each symbol
    for symbol in symbols:
        x = int(symbol['center']['x'])
        y = int(symbol['center']['y'])
        radius = int(symbol['radius'])

        # Draw bright cyan circle around symbol (more visible than green)
        cv2.circle(overlay, (x, y), max(radius + 10, 15), (255, 255, 0), 3)  # Cyan, thicker

        # Draw center point in magenta
        cv2.circle(overlay, (x, y), 5, (255, 0, 255), -1)  # Magenta filled circle

        # Add symbol ID in white text (smaller font, positioned above)
        cv2.putText(overlay, str(symbol['id']), (x - 15, y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Add legend/info text
    info_text = f"Detected City Symbols: {len(symbols)}"
    cv2.putText(overlay, info_text, (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8, cv2.LINE_AA)
    cv2.putText(overlay, info_text, (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4, cv2.LINE_AA)

    # Blend with original for better visibility
    result = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)

    # Save result
    cv2.imwrite(output_path, result)
    print(f"✓ Visualization saved to {output_path}")

    return result


def create_zoomed_regions(image_path, symbols_json):
    """Create zoomed-in views of regions with symbols."""
    print("\nCreating zoomed region samples...")

    # Load the map and symbols
    img = cv2.imread(image_path)
    with open(symbols_json, 'r') as f:
        data = json.load(f)
    symbols = data['symbols']

    # Select a few interesting regions (different parts of the map)
    height, width = img.shape[:2]

    regions = [
        ("northwest", 0, width//4, 0, height//4),
        ("center", width//3, 2*width//3, height//3, 2*height//3),
        ("southeast", 3*width//4, width, 3*height//4, height),
    ]

    for region_name, x1, x2, y1, y2 in regions:
        # Find symbols in this region
        region_symbols = [
            s for s in symbols
            if x1 <= s['center']['x'] < x2 and y1 <= s['center']['y'] < y2
        ]

        if len(region_symbols) == 0:
            continue

        print(f"  Region '{region_name}': {len(region_symbols)} symbols")

        # Crop region
        cropped = img[y1:y2, x1:x2].copy()

        # Draw symbols in this region
        for symbol in region_symbols:
            x = int(symbol['center']['x']) - x1
            y = int(symbol['center']['y']) - y1
            radius = int(symbol['radius'])

            # Draw circle and center
            cv2.circle(cropped, (x, y), max(radius + 10, 15), (255, 255, 0), 2)
            cv2.circle(cropped, (x, y), 4, (255, 0, 255), -1)
            cv2.putText(cropped, str(symbol['id']), (x - 10, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Save zoomed region
        output_path = f"zoom_{region_name}.jpg"
        cv2.imwrite(output_path, cropped)
        print(f"  ✓ Saved {output_path}")


def create_heatmap(image_path, symbols_json, output_path="symbol_density_heatmap.jpg"):
    """Create a heatmap showing density of city symbols."""
    print("\nCreating density heatmap...")

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    with open(symbols_json, 'r') as f:
        data = json.load(f)
    symbols = data['symbols']

    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Add Gaussian blob for each symbol
    for symbol in symbols:
        x = int(symbol['center']['x'])
        y = int(symbol['center']['y'])

        # Create a Gaussian kernel
        size = 200  # Radius of influence
        y1, y2 = max(0, y - size), min(height, y + size)
        x1, x2 = max(0, x - size), min(width, x + size)

        # Add heat
        for dy in range(y1, y2):
            for dx in range(x1, x2):
                dist = np.sqrt((dx - x)**2 + (dy - y)**2)
                if dist < size:
                    heatmap[dy, dx] += np.exp(-(dist**2) / (2 * (size/3)**2))

    # Normalize heatmap
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend with original image
    result = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # Add legend
    cv2.putText(result, "City Symbol Density", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8, cv2.LINE_AA)

    cv2.imwrite(output_path, result)
    print(f"✓ Heatmap saved to {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("City Symbols Visualization Generator")
    print("=" * 70)
    print()

    image_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"
    symbols_json = "detected_symbols.json"

    # Check files exist
    if not Path(image_path).exists():
        print(f"❌ Error: Map image not found: {image_path}")
        return 1

    if not Path(symbols_json).exists():
        print(f"❌ Error: Symbols data not found: {symbols_json}")
        print("Run detect_city_symbols.py first")
        return 1

    try:
        # Create main visualization
        create_symbol_visualization(image_path, symbols_json)

        # Create zoomed regions
        create_zoomed_regions(image_path, symbols_json)

        # Create density heatmap
        create_heatmap(image_path, symbols_json)

        print("\n" + "=" * 70)
        print("Visualization Complete!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  📊 city_symbols_visualization.jpg - All symbols marked on full map")
        print("  🔍 zoom_*.jpg - Zoomed views of different regions")
        print("  🌡️  symbol_density_heatmap.jpg - Heat map showing city concentration")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
