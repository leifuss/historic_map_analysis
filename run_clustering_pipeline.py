#!/usr/bin/env python3
"""
Master script to run the complete clustering-based refinement pipeline.

Phases:
1. Extract patches and compute features
2. Cluster detections and identify true city icons
3. Filter detections based on clustering results
"""

import subprocess
import sys
import time
from pathlib import Path


def run_phase(script_name, description):
    """Run a pipeline phase."""
    print("\n" + "=" * 80)
    print(f"PHASE: {description}")
    print("=" * 80)
    print(f"Running: {script_name}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Phase completed successfully in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Phase failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    required_packages = [
        'torch',
        'torchvision',
        'sklearn',
        'scipy',
        'matplotlib',
        'cv2',
        'PIL',
        'numpy'
    ]

    missing = []

    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("✓ All dependencies installed\n")
    return True


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("CLUSTERING-BASED REFINEMENT PIPELINE")
    print("=" * 80)
    print("\nThis pipeline will:")
    print("1. Extract patches and compute ResNet features (Phase 1)")
    print("2. Cluster detections and identify true city icons (Phase 2)")
    print("3. Filter detections based on clustering (Phase 3)")
    print()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check input files
    required_files = [
        "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg",
        "detected_cities_template.json",
        "city_icon.jpg"
    ]

    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Error: Required file not found: {file}")
            return 1

    print("✓ All required files present\n")

    total_start = time.time()

    # Phase 1: Feature extraction
    success = run_phase(
        "extract_features_simple.py",
        "Phase 1 - Extract Patches and Compute Visual Features"
    )
    if not success:
        print("\n❌ Pipeline failed at Phase 1")
        return 1

    # Phase 2: Clustering
    success = run_phase(
        "cluster_detections.py",
        "Phase 2 - Cluster Detections"
    )
    if not success:
        print("\n❌ Pipeline failed at Phase 2")
        return 1

    # Phase 3: Filtering
    success = run_phase(
        "filter_by_clustering.py",
        "Phase 3 - Filter by Clustering"
    )
    if not success:
        print("\n❌ Pipeline failed at Phase 3")
        return 1

    # Success!
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nTotal execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print("\nGenerated files:")
    print("\n📊 Phase 1 Output:")
    print("  - detection_features.pkl (feature dataset)")
    print("  - sample_patches/ (100 example patches)")
    print("\n📊 Phase 2 Output:")
    print("  - cluster_optimization.png (elbow method + silhouette)")
    print("  - clusters_kmeans_scatter.png (PCA visualization)")
    print("  - clusters_dbscan_scatter.png (PCA visualization)")
    print("  - cluster_*_c*_samples.jpg (sample grids for each cluster)")
    print("  - clustering_results.pkl")
    print("\n📊 Phase 3 Output:")
    print("  - filtered_cities.json (refined city detections)")
    print("  - filtered_cities_visualization.jpg (final results)")
    print("\n🎯 Next steps:")
    print("  1. Review cluster sample grids to verify true city icon cluster")
    print("  2. Check filtered_cities_visualization.jpg for quality")
    print("  3. If needed, adjust filters and re-run Phase 3")
    print("  4. Proceed to text label extraction (Florence-2)")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
