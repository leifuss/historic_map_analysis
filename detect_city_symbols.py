#!/usr/bin/env python3
"""
City Symbol Detection for Idrisi Map
Detects small brown circular symbols (resembling bread loaves/Trivial Pursuit pieces)
that represent cities on the map.
"""

import cv2
import numpy as np
from PIL import Image
import json
import sys
from pathlib import Path


def detect_brown_circles_cv(image_path, output_debug=True):
    """
    Use computer vision to detect brown circular symbols.

    Args:
        image_path: Path to the map image
        output_debug: If True, save debug visualization

    Returns:
        list: Detected circles with coordinates
    """
    print("Loading image for circle detection...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]
    print(f"Image dimensions: {width}x{height}")

    # Convert to different color spaces for brown detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("\nDetecting brown-colored regions...")

    # Define brown color range in HSV
    # Brown is typically: low saturation orange/red with medium value
    # Adjust these ranges based on the specific brown in the map
    lower_brown1 = np.array([0, 30, 50])      # Reddish-brown
    upper_brown1 = np.array([20, 200, 200])

    lower_brown2 = np.array([10, 40, 60])     # Orange-brown
    upper_brown2 = np.array([30, 255, 180])

    # Create masks for brown regions
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    brown_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    print(f"Brown pixels found: {np.sum(brown_mask > 0)}")

    # Find contours in the brown mask
    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} brown regions")

    # Filter contours to find circular symbols
    symbols = []
    debug_img = img.copy() if output_debug else None

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Filter by area (adjust based on symbol size)
        # Small brown circles should be roughly 20-200 pixels in area
        if area < 20 or area > 500:
            continue

        # Get bounding circle
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Calculate circularity (how circle-like the shape is)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Filter by circularity (1.0 = perfect circle, >0.6 is reasonably circular)
        if circularity < 0.5:
            continue

        # Get the center point
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(x), int(y)

        symbol = {
            "id": len(symbols),
            "center": {"x": cx, "y": cy},
            "radius": float(radius),
            "area": float(area),
            "circularity": float(circularity),
            "bounding_box": {
                "x_min": int(x - radius),
                "y_min": int(y - radius),
                "x_max": int(x + radius),
                "y_max": int(y + radius)
            }
        }
        symbols.append(symbol)

        # Draw on debug image
        if output_debug:
            cv2.circle(debug_img, (cx, cy), int(radius), (0, 255, 0), 2)
            cv2.circle(debug_img, (cx, cy), 2, (0, 0, 255), -1)
            cv2.putText(debug_img, str(len(symbols)-1), (cx-10, cy-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    print(f"\n✓ Detected {len(symbols)} circular symbols")

    # Save debug visualization
    if output_debug and len(symbols) > 0:
        debug_path = "debug_symbol_detection.jpg"
        cv2.imwrite(debug_path, debug_img)
        print(f"✓ Debug visualization saved to {debug_path}")

        # Also save the brown mask for inspection
        cv2.imwrite("debug_brown_mask.jpg", brown_mask)
        print(f"✓ Brown mask saved to debug_brown_mask.jpg")

    return symbols, (width, height)


def refine_with_hough_circles(image_path, approximate_locations=None):
    """
    Use Hough Circle Transform as an alternative/complementary method.

    Args:
        image_path: Path to the map image
        approximate_locations: Optional list of approximate locations to focus on

    Returns:
        list: Detected circles
    """
    print("\nTrying Hough Circle Transform method...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    # Parameters may need tuning based on symbol size
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,      # Minimum distance between circle centers
        param1=50,       # Canny edge threshold
        param2=30,       # Accumulator threshold
        minRadius=5,     # Minimum circle radius
        maxRadius=25     # Maximum circle radius
    )

    detected = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Hough detected {len(circles[0])} circles")

        for i, (x, y, r) in enumerate(circles[0, :]):
            detected.append({
                "id": i,
                "center": {"x": int(x), "y": int(y)},
                "radius": int(r),
                "method": "hough"
            })
    else:
        print("Hough method found no circles")

    return detected


def save_symbol_detections(symbols, image_size, output_path="detected_symbols.json"):
    """Save detected symbols to JSON file."""
    output_data = {
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "total_symbols": len(symbols),
        "symbols": symbols,
        "description": "Brown circular city symbols detected on Idrisi map"
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved {len(symbols)} symbol detections to {output_path}")

    # Print sample
    print("\nSample detected symbols:")
    for symbol in symbols[:10]:
        print(f"  Symbol {symbol['id']}: center=({symbol['center']['x']}, {symbol['center']['y']}), "
              f"radius={symbol.get('radius', 'N/A'):.1f}")

    if len(symbols) > 10:
        print(f"  ... and {len(symbols) - 10} more")

    return output_data


def main():
    """Main execution function."""
    image_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"

    print("=" * 70)
    print("City Symbol Detection - Idrisi Map")
    print("=" * 70)
    print("\nTarget: Small brown circular symbols (bread loaf/Trivial Pursuit shape)")
    print()

    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return 1

    try:
        # Method 1: Color-based circle detection
        symbols, image_size = detect_brown_circles_cv(image_path, output_debug=True)

        # Method 2: Hough circles (alternative)
        # hough_symbols = refine_with_hough_circles(image_path)

        # Save results
        if len(symbols) > 0:
            save_symbol_detections(symbols, image_size)

            print("\n" + "=" * 70)
            print("Symbol Detection Complete!")
            print("=" * 70)
            print(f"Total symbols detected: {len(symbols)}")
            print("\nNext steps:")
            print("1. Review debug_symbol_detection.jpg to verify detections")
            print("2. Adjust color/size thresholds if needed")
            print("3. Run text extraction to get city labels")
            print("4. Match symbols to labels using proximity")
        else:
            print("\n⚠ Warning: No symbols detected!")
            print("Suggestions:")
            print("- Review debug_brown_mask.jpg to check color detection")
            print("- Adjust brown color HSV ranges in the code")
            print("- Try the Hough circles method as alternative")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
