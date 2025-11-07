#!/usr/bin/env python3
"""
Phase 2: Cluster detections and identify true city icon cluster.
Uses extracted features to group similar detections.
"""

import numpy as np
import pickle
import json
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path


def load_features(features_path="detection_features.pkl"):
    """Load extracted features."""
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    return data


def find_optimal_k(features, k_range=(3, 15)):
    """Find optimal number of clusters using elbow method and silhouette score."""
    print(f"\nFinding optimal k (range {k_range[0]}-{k_range[1]})...")

    inertias = []
    silhouettes = []
    k_values = range(k_range[0], k_range[1] + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        inertias.append(kmeans.inertia_)

        if k < len(features):  # Silhouette needs at least k < n_samples
            sil_score = silhouette_score(features, labels)
            silhouettes.append(sil_score)
        else:
            silhouettes.append(0)

        print(f"  k={k:2d}: inertia={kmeans.inertia_:10.2f}, silhouette={silhouettes[-1]:.3f}")

    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)

    ax2.plot(k_values, silhouettes, 'ro-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('cluster_optimization.png', dpi=150)
    print(f"✓ Saved cluster_optimization.png")

    # Recommend k (highest silhouette score)
    best_k = k_values[np.argmax(silhouettes)]
    print(f"\nRecommended k={best_k} (highest silhouette score)")

    return best_k


def run_clustering(features, method='both', n_clusters=8):
    """
    Run clustering algorithms.

    Args:
        features: Feature array (Nx2048)
        method: 'kmeans', 'dbscan', or 'both'
        n_clusters: Number of clusters for k-means

    Returns:
        dict with clustering results
    """
    print(f"\nRunning clustering...")
    print(f"Features shape: {features.shape}")

    results = {}

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means
    if method in ['kmeans', 'both']:
        print(f"\nK-means (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        kmeans_labels = kmeans.fit_predict(features_scaled)

        unique, counts = np.unique(kmeans_labels, return_counts=True)
        print(f"  Cluster sizes: {dict(zip(unique, counts))}")

        silhouette = silhouette_score(features_scaled, kmeans_labels)
        print(f"  Silhouette score: {silhouette:.3f}")

        results['kmeans'] = {
            'labels': kmeans_labels,
            'centers': kmeans.cluster_centers_,
            'silhouette': silhouette,
            'model': kmeans
        }

    # DBSCAN
    if method in ['dbscan', 'both']:
        print(f"\nDBSCAN...")
        # Try different eps values
        best_eps = None
        best_labels = None
        best_n_clusters = 0

        for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            dbscan = DBSCAN(eps=eps, min_samples=10)
            labels = dbscan.fit_predict(features_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            print(f"  eps={eps:.1f}: {n_clusters} clusters, {n_noise} noise points")

            # Choose eps with 5-15 clusters
            if 5 <= n_clusters <= 15:
                if best_eps is None or n_clusters > best_n_clusters:
                    best_eps = eps
                    best_labels = labels
                    best_n_clusters = n_clusters

        if best_labels is not None:
            unique, counts = np.unique(best_labels, return_counts=True)
            print(f"  Selected eps={best_eps:.1f}")
            print(f"  Cluster sizes: {dict(zip(unique, counts))}")

            results['dbscan'] = {
                'labels': best_labels,
                'eps': best_eps,
                'n_clusters': best_n_clusters
            }
        else:
            print("  Warning: No suitable DBSCAN clustering found")

    return results, features_scaled


def visualize_clusters(features_scaled, clustering_results, detections, save_prefix="clusters"):
    """Create visualizations of clustering results."""
    print("\nCreating cluster visualizations...")

    # PCA for 2D projection
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    for method_name, result in clustering_results.items():
        labels = result['labels']

        # 2D scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points (DBSCAN)
                color = 'gray'
                marker = 'x'
                label_text = 'Noise'
            else:
                marker = 'o'
                label_text = f'Cluster {label}'

            mask = labels == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[color], marker=marker, s=50, alpha=0.6,
                      edgecolors='black', linewidths=0.5,
                      label=f'{label_text} ({mask.sum()})')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Cluster Visualization: {method_name.upper()}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{method_name}_scatter.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved {save_prefix}_{method_name}_scatter.png")
        plt.close()


def create_cluster_sample_grids(feature_data, clustering_results, samples_per_cluster=50):
    """Create grid images showing sample patches from each cluster."""
    print("\nCreating cluster sample grids...")

    map_img = cv2.imread("al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg")
    detections = feature_data['detections']

    for method_name, result in clustering_results.items():
        labels = result['labels']
        unique_labels = [l for l in set(labels) if l != -1]  # Skip noise

        print(f"\n{method_name.upper()}:")

        for cluster_id in sorted(unique_labels):
            # Get indices of this cluster
            indices = np.where(labels == cluster_id)[0]

            if len(indices) == 0:
                continue

            print(f"  Cluster {cluster_id}: {len(indices)} samples")

            # Sample up to N patches
            sample_indices = np.random.choice(indices,
                                            size=min(samples_per_cluster, len(indices)),
                                            replace=False)

            # Extract patches
            patches = []
            for idx in sample_indices:
                det = detections[idx]
                cx = det['center']['x']
                cy = det['center']['y']

                # Extract patch
                y1 = max(0, cy - 40)
                y2 = min(map_img.shape[0], cy + 40)
                x1 = max(0, cx - 40)
                x2 = min(map_img.shape[1], cx + 40)

                patch = map_img[y1:y2, x1:x2]

                # Resize to standard size
                patch = cv2.resize(patch, (80, 80))
                patches.append(patch)

            # Create grid
            grid_rows = int(np.ceil(np.sqrt(len(patches))))
            grid_cols = grid_rows

            grid_img = np.zeros((grid_rows * 80, grid_cols * 80, 3), dtype=np.uint8)

            for i, patch in enumerate(patches):
                row = i // grid_cols
                col = i % grid_cols
                grid_img[row*80:(row+1)*80, col*80:(col+1)*80] = patch

            # Save grid
            output_path = f"cluster_{method_name}_c{cluster_id}_samples.jpg"
            cv2.imwrite(output_path, grid_img)
            print(f"    ✓ Saved {output_path}")


def analyze_cluster_similarity_to_template(feature_data, clustering_results):
    """Find which cluster best matches the original template."""
    print("\n" + "=" * 70)
    print("Cluster Similarity to Original Template")
    print("=" * 70)

    # Handle both feature types
    if 'visual_features' in feature_data:
        # ResNet+visual features
        visual_features = feature_data['visual_features']
        use_dict_features = True
    elif 'features' in feature_data:
        # Hand-crafted features (first 2 dims are template_corr and ssim)
        all_features = feature_data['features']
        use_dict_features = False
    else:
        raise ValueError("No compatible features found")

    for method_name, result in clustering_results.items():
        labels = result['labels']
        unique_labels = [l for l in set(labels) if l != -1]

        print(f"\n{method_name.upper()}:")
        print(f"{'Cluster':<10} {'Size':<8} {'Avg Corr':<12} {'Avg SSIM':<12} {'Score':<10}")
        print("-" * 70)

        cluster_scores = []

        for cluster_id in sorted(unique_labels):
            indices = np.where(labels == cluster_id)[0]

            # Calculate average similarity metrics
            if use_dict_features:
                corrs = [visual_features[i]['template_correlation'] for i in indices]
                ssims = [visual_features[i]['ssim'] for i in indices]
                colors = [visual_features[i]['color_similarity'] for i in indices]
                avg_color = np.mean(colors)
            else:
                # Extract from feature vector (dim 0=corr, dim 1=ssim)
                corrs = [all_features[i, 0] for i in indices]
                ssims = [all_features[i, 1] for i in indices]
                avg_color = 0.0  # Not available separately

            avg_corr = np.mean(corrs)
            avg_ssim = np.mean(ssims)

            # Combined score (weighted average)
            if use_dict_features:
                score = 0.5 * avg_corr + 0.3 * avg_ssim + 0.2 * avg_color
            else:
                score = 0.6 * avg_corr + 0.4 * avg_ssim  # Simplified for hand-crafted

            cluster_scores.append((cluster_id, score, len(indices), avg_corr, avg_ssim, avg_color))

            if use_dict_features:
                print(f"{cluster_id:<10} {len(indices):<8} {avg_corr:<12.3f} {avg_ssim:<12.3f} {avg_color:<12.3f} {score:<10.3f}")
            else:
                print(f"{cluster_id:<10} {len(indices):<8} {avg_corr:<12.3f} {avg_ssim:<12.3f} {score:<10.3f}")

        # Identify best cluster
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        best_cluster_id = cluster_scores[0][0]
        best_score = cluster_scores[0][1]

        print(f"\n⭐ Best match: Cluster {best_cluster_id} (score={best_score:.3f})")
        print(f"   This cluster likely contains true city icons")

        # Save recommendation
        result['best_cluster'] = best_cluster_id
        result['cluster_scores'] = cluster_scores

    return clustering_results


def save_clustering_results(feature_data, clustering_results, output_path="clustering_results.pkl"):
    """Save clustering results."""
    output_data = {
        'clustering_results': clustering_results,
        'feature_data_path': "detection_features.pkl"
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Saved clustering results to {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("Phase 2: Cluster Detections")
    print("=" * 70)
    print()

    features_path = "detection_features.pkl"

    if not Path(features_path).exists():
        print(f"❌ Error: {features_path} not found")
        print("Run extract_features.py first")
        return 1

    try:
        # Load features
        print("Loading features...")
        feature_data = load_features(features_path)

        # Support both ResNet features and visual features
        if 'deep_features' in feature_data:
            features = feature_data['deep_features']
        elif 'features' in feature_data:
            features = feature_data['features']
        else:
            raise ValueError("No features found in feature data")

        print(f"Loaded {features.shape[0]} samples with {features.shape[1]}D features")

        # Find optimal k
        optimal_k = find_optimal_k(features, k_range=(3, 12))

        # Run clustering
        clustering_results, features_scaled = run_clustering(
            features,
            method='both',
            n_clusters=optimal_k
        )

        # Visualize clusters
        visualize_clusters(features_scaled, clustering_results,
                          feature_data['detections'])

        # Create sample grids
        create_cluster_sample_grids(feature_data, clustering_results,
                                   samples_per_cluster=50)

        # Analyze similarity to template
        clustering_results = analyze_cluster_similarity_to_template(
            feature_data, clustering_results
        )

        # Save results
        save_clustering_results(feature_data, clustering_results)

        print("\n" + "=" * 70)
        print("Clustering Complete!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - cluster_optimization.png")
        print("  - clusters_kmeans_scatter.png")
        print("  - clusters_dbscan_scatter.png")
        print("  - cluster_*_c*_samples.jpg (sample grids for each cluster)")
        print("  - clustering_results.pkl")
        print("\nNext step: Review cluster sample grids and run phase 3 filtering")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
