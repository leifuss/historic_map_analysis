#!/usr/bin/env python3
"""
Phase 3: Filter detections based on clustering results.
Keeps only detections from the "true city icon" cluster.
"""

import numpy as np
import pickle
import json
import cv2
from pathlib import Path
from scipy.spatial import distance_matrix


def load_clustering_results(clustering_path="clustering_results.pkl",
                            features_path="detection_features.pkl"):
    """Load clustering results and feature data."""
    with open(clustering_path, 'rb') as f:
        clustering_data = pickle.load(f)

    with open(features_path, 'rb') as f:
        feature_data = pickle.load(f)

    return clustering_data, feature_data


def filter_by_cluster(detections, labels, selected_clusters):
    """
    Filter detections to keep only those in selected clusters.

    Args:
        detections: List of detection dicts
        labels: Cluster labels array
        selected_clusters: List of cluster IDs to keep

    Returns:
        filtered_detections: List of filtered detection dicts
        filtered_indices: Original indices of kept detections
    """
    filtered_detections = []
    filtered_indices = []

    for i, (det, label) in enumerate(zip(detections, labels)):
        if label in selected_clusters:
            filtered_detections.append(det)
            filtered_indices.append(i)

    return filtered_detections, filtered_indices


def apply_spatial_nms(detections, min_distance=75):
    """
    Apply spatial non-maximum suppression.
    Remove detections that are too close together, keeping higher confidence ones.

    Args:
        detections: List of detection dicts
        min_distance: Minimum distance between detections (pixels)

    Returns:
        filtered_detections: List after NMS
    """
    if len(detections) == 0:
        return []

    print(f"\nApplying spatial NMS (min distance: {min_distance}px)...")

    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    # Extract coordinates
    coords = np.array([[d['center']['x'], d['center']['y']] for d in sorted_dets])

    # Compute pairwise distances
    dist_matrix = distance_matrix(coords, coords)

    # Keep track of which detections to keep
    keep = np.ones(len(sorted_dets), dtype=bool)

    for i in range(len(sorted_dets)):
        if not keep[i]:
            continue

        # Mark all detections within min_distance as suppressed
        # (except the current one)
        close_indices = np.where(dist_matrix[i] < min_distance)[0]
        for j in close_indices:
            if j > i:  # Only suppress lower confidence ones
                keep[j] = False

    filtered_dets = [det for det, k in zip(sorted_dets, keep) if k]

    print(f"  Before NMS: {len(sorted_dets)}")
    print(f"  After NMS:  {len(filtered_dets)}")
    print(f"  Removed:    {len(sorted_dets) - len(filtered_dets)}")

    return filtered_dets


def apply_confidence_filter(detections, min_confidence=0.75):
    """Filter detections by confidence threshold."""
    filtered = [d for d in detections if d['confidence'] >= min_confidence]

    print(f"\nConfidence filter (>={min_confidence}):")
    print(f"  Before: {len(detections)}")
    print(f"  After:  {len(filtered)}")
    print(f"  Removed: {len(detections) - len(filtered)}")

    return filtered


def apply_scale_filter(detections, min_scale=0.7, max_scale=1.3):
    """Filter detections by scale range."""
    filtered = [d for d in detections
                if min_scale <= d['scale'] <= max_scale]

    print(f"\nScale filter ({min_scale}-{max_scale}):")
    print(f"  Before: {len(detections)}")
    print(f"  After:  {len(filtered)}")
    print(f"  Removed: {len(detections) - len(filtered)}")

    return filtered


def save_filtered_detections(detections, output_path="filtered_cities.json"):
    """Save filtered detections."""

    # Calculate statistics
    if len(detections) > 0:
        confidences = [d['confidence'] for d in detections]
        scales = [d['scale'] for d in detections]

        stats = {
            'total_cities': len(detections),
            'confidence_stats': {
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences))
            },
            'scale_stats': {
                'min': float(np.min(scales)),
                'max': float(np.max(scales)),
                'mean': float(np.mean(scales)),
                'median': float(np.median(scales))
            }
        }
    else:
        stats = {
            'total_cities': 0
        }

    output_data = {
        **stats,
        'detection_method': 'template_matching + clustering + filtering',
        'cities': detections,
        'description': 'Filtered city detections from Idrisi map (clustering-based refinement)'
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved {len(detections)} filtered cities to {output_path}")

    return output_data


def create_filtered_visualization(detections, output_path="filtered_cities_visualization.jpg"):
    """Create visualization of filtered detections."""
    print("\nCreating visualization...")

    img = cv2.imread("al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg")
    overlay = img.copy()

    for city in detections:
        x = city['center']['x']
        y = city['center']['y']
        w = city['width']
        h = city['height']
        conf = city['confidence']

        # Color by confidence
        if conf > 0.85:
            color = (0, 255, 0)  # Green
        elif conf > 0.75:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange

        # Draw bounding box
        top_left = (city['top_left']['x'], city['top_left']['y'])
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(overlay, top_left, bottom_right, color, 3)

        # Draw center
        cv2.circle(overlay, (x, y), 6, (255, 0, 255), -1)

        # Add ID
        cv2.putText(overlay, str(city['id']), (x + 12, y - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Add title
    cv2.putText(overlay, f"Filtered Cities: {len(detections)}", (50, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA)
    cv2.putText(overlay, f"Filtered Cities: {len(detections)}", (50, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)

    # Blend
    result = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)

    cv2.imwrite(output_path, result)
    print(f"✓ Visualization saved to {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("Phase 3: Filter Detections by Clustering")
    print("=" * 70)
    print()

    clustering_path = "clustering_results.pkl"
    features_path = "detection_features.pkl"

    # Check files
    for path in [clustering_path, features_path]:
        if not Path(path).exists():
            print(f"❌ Error: {path} not found")
            return 1

    try:
        # Load data
        print("Loading clustering results...")
        clustering_data, feature_data = load_clustering_results(
            clustering_path, features_path
        )

        detections = feature_data['detections']
        print(f"Original detections: {len(detections)}")

        # Get clustering results (prefer k-means)
        clustering_results = clustering_data['clustering_results']

        if 'kmeans' in clustering_results:
            method = 'kmeans'
            result = clustering_results['kmeans']
        elif 'dbscan' in clustering_results:
            method = 'dbscan'
            result = clustering_results['dbscan']
        else:
            print("❌ Error: No clustering results found")
            return 1

        labels = result['labels']
        best_cluster = result.get('best_cluster')

        if best_cluster is None:
            print("❌ Error: No best cluster identified")
            return 1

        print(f"\nUsing {method.upper()} clustering")
        print(f"Best cluster (true city icons): {best_cluster}")

        # Show cluster scores
        if 'cluster_scores' in result:
            print("\nCluster scores:")
            for cluster_id, score, size, corr, ssim, color in result['cluster_scores'][:5]:
                print(f"  Cluster {cluster_id}: score={score:.3f}, size={size}, "
                      f"corr={corr:.3f}, ssim={ssim:.3f}, color={color:.3f}")

        # Step 1: Filter by cluster membership
        print(f"\n{'='*70}")
        print("Step 1: Filter by Cluster")
        print(f"{'='*70}")

        # You can select multiple clusters if needed
        # For now, just use the best cluster
        selected_clusters = [best_cluster]

        # Optionally, also include clusters with high scores
        if 'cluster_scores' in result:
            # Add any other clusters with score > 0.6
            for cluster_id, score, size, _, _, _ in result['cluster_scores']:
                if score > 0.6 and cluster_id != best_cluster:
                    selected_clusters.append(cluster_id)
                    print(f"  Also including cluster {cluster_id} (score={score:.3f})")

        filtered_dets, filtered_indices = filter_by_cluster(
            detections, labels, selected_clusters
        )

        print(f"\nCluster filtering:")
        print(f"  Selected clusters: {selected_clusters}")
        print(f"  Before: {len(detections)}")
        print(f"  After:  {len(filtered_dets)}")
        print(f"  Kept:   {len(filtered_dets)/len(detections)*100:.1f}%")

        # Step 2: Confidence filter
        print(f"\n{'='*70}")
        print("Step 2: Additional Filters")
        print(f"{'='*70}")

        filtered_dets = apply_confidence_filter(filtered_dets, min_confidence=0.70)

        # Step 3: Scale filter
        filtered_dets = apply_scale_filter(filtered_dets, min_scale=0.7, max_scale=1.3)

        # Step 4: Spatial NMS
        filtered_dets = apply_spatial_nms(filtered_dets, min_distance=75)

        # Re-assign IDs
        for i, det in enumerate(filtered_dets):
            det['id'] = i

        # Save results
        print(f"\n{'='*70}")
        print("Saving Results")
        print(f"{'='*70}")

        output_data = save_filtered_detections(filtered_dets)

        # Create visualization
        create_filtered_visualization(filtered_dets)

        # Print final statistics
        print("\n" + "=" * 70)
        print("Filtering Complete!")
        print("=" * 70)
        print(f"\nFinal results:")
        print(f"  Original detections: {len(detections)}")
        print(f"  Filtered detections: {len(filtered_dets)}")
        print(f"  Reduction: {(1 - len(filtered_dets)/len(detections))*100:.1f}%")

        if len(filtered_dets) > 0:
            print(f"\n  Confidence range: {output_data['confidence_stats']['min']:.3f} - "
                  f"{output_data['confidence_stats']['max']:.3f}")
            print(f"  Average confidence: {output_data['confidence_stats']['mean']:.3f}")

        print(f"\nOutput files:")
        print(f"  - filtered_cities.json")
        print(f"  - filtered_cities_visualization.jpg")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
