#!/usr/bin/env python3
"""
Phase 1 (Lightweight): Extract detection patches and compute visual features.
Uses hand-crafted features instead of deep learning (no model download needed).
"""

import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from skimage.feature import hog, local_binary_pattern
from skimage.metrics import structural_similarity as ssim


def load_detections(json_path="detected_cities_template.json"):
    """Load detection data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['cities'], data['image_size']


def extract_patch(img, center_x, center_y, patch_size=80):
    """Extract a square patch centered on the detection."""
    half_size = patch_size // 2

    # Calculate bounds
    y1 = max(0, center_y - half_size)
    y2 = min(img.shape[0], center_y + half_size)
    x1 = max(0, center_x - half_size)
    x2 = min(img.shape[1], center_x + half_size)

    # Extract patch
    patch = img[y1:y2, x1:x2].copy()

    # Pad if necessary
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        py = (patch_size - patch.shape[0]) // 2
        px = (patch_size - patch.shape[1]) // 2
        padded[py:py+patch.shape[0], px:px+patch.shape[1]] = patch
        patch = padded

    return patch


def compute_visual_features(patch, template):
    """
    Compute comprehensive visual features for clustering.

    Returns feature vector combining:
    - Template correlation & SSIM
    - Color histograms (HSV)
    - HOG features (shape/edges)
    - LBP texture features
    - Hu moments (shape invariant)
    """
    features = []

    # Resize patch to template size for comparison
    patch_resized = cv2.resize(patch, (template.shape[1], template.shape[0]))

    # 1. Template Similarity (2 features)
    result = cv2.matchTemplate(patch_resized, template, cv2.TM_CCOEFF_NORMED)
    template_corr = result[0, 0]
    features.append(template_corr)

    patch_gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(template_gray, patch_gray)
    features.append(ssim_score)

    # 2. Color Features (60 features)
    # HSV histogram
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([patch_hsv], [0], None, [20], [0, 180])
    hist_s = cv2.calcHist([patch_hsv], [1], None, [20], [0, 256])
    hist_v = cv2.calcHist([patch_hsv], [2], None, [20], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    features.extend(hist_h)
    features.extend(hist_s)
    features.extend(hist_v)

    # 3. HOG Features (81 features)
    # Histogram of Oriented Gradients
    patch_gray_full = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hog_features = hog(patch_gray_full, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=False)
    features.extend(hog_features)

    # 4. LBP Texture Features (26 features)
    # Local Binary Patterns
    lbp = local_binary_pattern(patch_gray_full, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    features.extend(lbp_hist)

    # 5. Hu Moments (7 features)
    # Shape invariant moments
    moments = cv2.moments(patch_gray_full)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log transform for better scaling
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    features.extend(hu_moments)

    # 6. Edge Statistics (4 features)
    edges = cv2.Canny(patch_gray_full, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)

    # Horizontal and vertical edge projections
    h_proj = np.sum(edges, axis=0).astype(float)
    v_proj = np.sum(edges, axis=1).astype(float)
    features.append(np.std(h_proj) / (np.mean(h_proj) + 1))
    features.append(np.std(v_proj) / (np.mean(v_proj) + 1))

    # Edge orientation histogram
    sobelx = cv2.Sobel(patch_gray_full, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(patch_gray_full, cv2.CV_64F, 0, 1)
    orientation = np.arctan2(sobely, sobelx)
    orientation = (orientation + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    features.append(np.std(orientation))

    return np.array(features, dtype=np.float32)


def main():
    """Main execution."""
    print("=" * 70)
    print("Phase 1: Extract Patches and Compute Visual Features (Lightweight)")
    print("=" * 70)
    print()

    # Paths
    map_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"
    detections_path = "detected_cities_template.json"
    template_path = "city_icon.jpg"

    # Check files
    for path in [map_path, detections_path, template_path]:
        if not Path(path).exists():
            print(f"❌ Error: {path} not found")
            return 1

    try:
        # Load data
        print("Loading map and detections...")
        img = cv2.imread(map_path)
        template = cv2.imread(template_path)
        detections, img_size = load_detections(detections_path)

        print(f"Map: {img_size['width']}x{img_size['height']}")
        print(f"Detections: {len(detections)}")
        print(f"Template: {template.shape[1]}x{template.shape[0]}")

        # Extract patches and features
        print("\nExtracting patches and computing visual features...")
        print("Feature vector composition:")
        print("  - Template similarity: 2 features")
        print("  - Color histograms (HSV): 60 features")
        print("  - HOG (edges/shape): 81 features")
        print("  - LBP (texture): 26 features")
        print("  - Hu moments (shape): 7 features")
        print("  - Edge statistics: 4 features")
        print("  Total: 180 features per patch\n")

        all_features = []
        patches = []

        for i, det in enumerate(detections):
            cx = det['center']['x']
            cy = det['center']['y']

            # Extract patch
            patch = extract_patch(img, cx, cy, patch_size=80)
            patches.append(patch)

            # Compute features
            features = compute_visual_features(patch, template)
            all_features.append(features)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(detections)} patches...")

        all_features = np.array(all_features)

        print(f"\n✓ Feature extraction complete!")
        print(f"  Feature matrix shape: {all_features.shape}")

        # Prepare feature dataset
        print("\nPreparing feature dataset...")
        feature_data = {
            'features': all_features,  # Nx180
            'detections': detections,
            'patch_size': 80,
            'template_file': template_path,
            'feature_description': {
                'total_dims': 180,
                'template_similarity': (0, 2),
                'color_hsv': (2, 62),
                'hog': (62, 143),
                'lbp': (143, 169),
                'hu_moments': (169, 176),
                'edge_stats': (176, 180)
            }
        }

        # Save features
        output_path = "detection_features.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(feature_data, f)

        print(f"✓ Saved features to {output_path}")

        # Save sample patches
        print("\nSaving sample patches...")
        sample_dir = Path("sample_patches")
        sample_dir.mkdir(exist_ok=True)

        for i in range(min(100, len(patches))):
            patch_path = sample_dir / f"patch_{i:04d}_conf{detections[i]['confidence']:.3f}.jpg"
            cv2.imwrite(str(patch_path), patches[i])

        print(f"✓ Saved 100 sample patches to {sample_dir}/")

        # Print feature statistics
        print("\n" + "=" * 70)
        print("Feature Extraction Complete!")
        print("=" * 70)
        print(f"\nFeature matrix: {all_features.shape}")
        print(f"  Mean: {np.mean(all_features):.3f}")
        print(f"  Std:  {np.std(all_features):.3f}")
        print(f"  Min:  {np.min(all_features):.3f}")
        print(f"  Max:  {np.max(all_features):.3f}")

        # Template similarity stats
        template_sims = all_features[:, 0]
        print(f"\nTemplate correlation distribution:")
        print(f"  Mean: {np.mean(template_sims):.3f}")
        print(f"  Std:  {np.std(template_sims):.3f}")
        print(f"  Min:  {np.min(template_sims):.3f}")
        print(f"  Max:  {np.max(template_sims):.3f}")

        print(f"\nOutput files:")
        print(f"  - detection_features.pkl (feature dataset)")
        print(f"  - sample_patches/ (100 example patches)")
        print(f"\nNext step: Run phase 2 clustering (cluster_detections.py)")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
