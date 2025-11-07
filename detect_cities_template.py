#!/usr/bin/env python3
"""
Template-based city icon detection for Idrisi map.
Uses the provided city_icon.jpg as a template to find all matching symbols.
"""

import cv2
import numpy as np
import json
from pathlib import Path


def load_template(template_path="city_icon.jpg"):
    """Load the city icon template."""
    template = cv2.imread(template_path)
    if template is None:
        raise ValueError(f"Could not load template: {template_path}")

    # Get template dimensions
    h, w = template.shape[:2]
    print(f"Template loaded: {w}x{h} pixels")

    return template, (w, h)


def detect_cities_template_matching(image_path, template, template_size,
                                   threshold=0.7, scales=None):
    """
    Detect city icons using multi-scale template matching.

    Args:
        image_path: Path to the map
        template: Template image
        template_size: (width, height) of template
        threshold: Match threshold (0-1, higher = stricter)
        scales: List of scales to try (default: 0.5 to 1.5)

    Returns:
        list: Detected city locations with confidence scores
    """
    print(f"\nLoading map: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Map dimensions: {img.shape[1]}x{img.shape[0]}")

    # Default scales if not provided
    if scales is None:
        # Try multiple scales from 50% to 150% of template size
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    all_detections = []
    template_w, template_h = template_size

    print(f"\nSearching at {len(scales)} different scales...")
    print(f"Match threshold: {threshold}")

    for scale in scales:
        # Resize template
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)

        if scaled_w < 5 or scaled_h < 5 or scaled_w > img.shape[1] or scaled_h > img.shape[0]:
            continue

        scaled_template = cv2.resize(template, (scaled_w, scaled_h))

        # Perform template matching
        result = cv2.matchTemplate(img, scaled_template, cv2.TM_CCOEFF_NORMED)

        # Find matches above threshold
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):  # Switch x and y
            x, y = pt
            confidence = result[y, x]

            # Calculate center of detection
            center_x = x + scaled_w // 2
            center_y = y + scaled_h // 2

            all_detections.append({
                'center': {'x': int(center_x), 'y': int(center_y)},
                'top_left': {'x': int(x), 'y': int(y)},
                'width': int(scaled_w),
                'height': int(scaled_h),
                'scale': float(scale),
                'confidence': float(confidence)
            })

        print(f"  Scale {scale:.2f} ({scaled_w}x{scaled_h}): {len(locations[0])} matches")

    print(f"\nTotal raw detections: {len(all_detections)}")

    return all_detections


def non_max_suppression(detections, overlap_threshold=0.3):
    """
    Remove overlapping detections, keeping the highest confidence ones.

    Args:
        detections: List of detection dictionaries
        overlap_threshold: Maximum allowed overlap ratio

    Returns:
        list: Filtered detections
    """
    if len(detections) == 0:
        return []

    print(f"\nApplying non-maximum suppression (overlap threshold: {overlap_threshold})...")

    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    kept = []

    for detection in detections:
        # Check if this detection overlaps significantly with any kept detection
        overlap = False

        for kept_det in kept:
            # Calculate distance between centers
            dx = detection['center']['x'] - kept_det['center']['x']
            dy = detection['center']['y'] - kept_det['center']['y']
            distance = np.sqrt(dx*dx + dy*dy)

            # Calculate average radius
            avg_radius = (detection['width'] + detection['height'] +
                         kept_det['width'] + kept_det['height']) / 4

            # If centers are very close, consider it an overlap
            if distance < avg_radius * overlap_threshold:
                overlap = True
                break

        if not overlap:
            kept.append(detection)

    print(f"Kept {len(kept)} detections after NMS")

    return kept


def save_detections(detections, image_size, output_path="detected_cities_template.json"):
    """Save detected cities to JSON."""

    # Add IDs to detections
    for i, det in enumerate(detections):
        det['id'] = i

    output_data = {
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "total_cities": len(detections),
        "detection_method": "template_matching",
        "template_file": "city_icon.jpg",
        "cities": detections,
        "description": "City icons detected using template matching on Idrisi map"
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved {len(detections)} city detections to {output_path}")

    # Print statistics
    if len(detections) > 0:
        confidences = [d['confidence'] for d in detections]
        scales = [d['scale'] for d in detections]

        print(f"\nDetection statistics:")
        print(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Scale range: {min(scales):.2f} - {max(scales):.2f}")
        print(f"  Most common scale: {max(set(scales), key=scales.count):.2f}")

    # Print sample detections
    print("\nSample detected cities:")
    for city in detections[:10]:
        print(f"  City {city['id']}: ({city['center']['x']}, {city['center']['y']}) "
              f"conf={city['confidence']:.3f} scale={city['scale']:.2f}")

    if len(detections) > 10:
        print(f"  ... and {len(detections) - 10} more")

    return output_data


def create_visualization(image_path, detections, output_path="template_detection_visualization.jpg"):
    """Create visualization of template-matched cities."""
    print(f"\nCreating visualization...")

    img = cv2.imread(image_path)
    overlay = img.copy()

    for city in detections:
        x = city['center']['x']
        y = city['center']['y']
        w = city['width']
        h = city['height']
        conf = city['confidence']

        # Color based on confidence (green = high, yellow = medium, red = low)
        if conf > 0.85:
            color = (0, 255, 0)  # Green
        elif conf > 0.75:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange

        # Draw bounding box
        top_left = (city['top_left']['x'], city['top_left']['y'])
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(overlay, top_left, bottom_right, color, 2)

        # Draw center point
        cv2.circle(overlay, (x, y), 5, (255, 0, 255), -1)

        # Add ID label
        cv2.putText(overlay, str(city['id']), (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Add legend
    cv2.putText(overlay, f"Cities Detected: {len(detections)}", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8, cv2.LINE_AA)
    cv2.putText(overlay, f"Cities Detected: {len(detections)}", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4, cv2.LINE_AA)

    # Blend
    result = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    cv2.imwrite(output_path, result)
    print(f"✓ Visualization saved to {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("Template-Based City Icon Detection")
    print("=" * 70)
    print()

    map_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"
    template_path = "city_icon.jpg"

    # Check files exist
    if not Path(map_path).exists():
        print(f"❌ Error: Map not found: {map_path}")
        return 1

    if not Path(template_path).exists():
        print(f"❌ Error: Template not found: {template_path}")
        return 1

    try:
        # Load template
        template, template_size = load_template(template_path)

        # Get map dimensions
        img = cv2.imread(map_path)
        image_size = (img.shape[1], img.shape[0])

        # Detect cities using template matching
        # Adjust threshold if needed (0.65-0.75 is typical, lower = more matches)
        raw_detections = detect_cities_template_matching(
            map_path,
            template,
            template_size,
            threshold=0.65,  # Try 0.65 first, adjust if needed
            scales=np.arange(0.5, 1.6, 0.1)  # More granular scales
        )

        if len(raw_detections) == 0:
            print("\n⚠ Warning: No cities detected!")
            print("Try lowering the threshold (e.g., 0.6 or 0.55)")
            return 1

        # Apply non-maximum suppression to remove duplicates
        final_detections = non_max_suppression(raw_detections, overlap_threshold=0.5)

        # Save results
        save_detections(final_detections, image_size)

        # Create visualization
        create_visualization(map_path, final_detections)

        print("\n" + "=" * 70)
        print("Detection Complete!")
        print("=" * 70)
        print(f"\nTotal cities found: {len(final_detections)}")
        print("\nOutput files:")
        print("  - detected_cities_template.json")
        print("  - template_detection_visualization.jpg")
        print("\nNext step: Review visualization to verify detections")
        print("If needed, adjust threshold in script and rerun")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
