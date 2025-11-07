#!/usr/bin/env python3
"""
Complete Pipeline Runner for Idrisi Map City Extraction

This script runs the complete 3-stage pipeline:
1. Detect brown circular city symbols
2. Extract text labels using Florence-2 OCR
3. Match symbols to labels using proximity
"""

import sys
import subprocess
from pathlib import Path
import time


def run_script(script_name, description):
    """Run a Python script and report results."""
    print("\n" + "=" * 70)
    print(f"STAGE: {description}")
    print("=" * 70)
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
        print(f"\n✓ Stage completed successfully in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Stage failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False


def check_prerequisites():
    """Check if all required files and dependencies exist."""
    print("Checking prerequisites...")

    # Check for map image
    if not Path("al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg").exists():
        print("❌ Error: Map image not found")
        print("   Expected: al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg")
        return False

    # Check for required scripts
    required_scripts = [
        "detect_city_symbols.py",
        "extract_text_labels.py",
        "match_symbols_to_labels.py"
    ]

    for script in required_scripts:
        if not Path(script).exists():
            print(f"❌ Error: Required script not found: {script}")
            return False

    # Check if dependencies are installed
    try:
        import cv2
        import torch
        import transformers
        from PIL import Image
    except ImportError as e:
        print(f"❌ Error: Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

    print("✓ All prerequisites met\n")
    return True


def main():
    """Main pipeline execution."""
    print("=" * 70)
    print("IDRISI MAP CITY EXTRACTION PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("1. Detect brown circular city symbols using computer vision")
    print("2. Extract text labels using Florence-2 VLM")
    print("3. Match symbols to labels using spatial proximity")
    print()

    # Check prerequisites
    if not check_prerequisites():
        return 1

    total_start = time.time()

    # Stage 1: Detect city symbols
    success = run_script(
        "detect_city_symbols.py",
        "Stage 1 - Detect Brown Circular City Symbols"
    )
    if not success:
        print("\n❌ Pipeline failed at Stage 1")
        return 1

    # Stage 2: Extract text labels
    success = run_script(
        "extract_text_labels.py",
        "Stage 2 - Extract Text Labels with Florence-2"
    )
    if not success:
        print("\n❌ Pipeline failed at Stage 2")
        return 1

    # Stage 3: Match symbols to labels
    success = run_script(
        "match_symbols_to_labels.py",
        "Stage 3 - Match Symbols to Labels"
    )
    if not success:
        print("\n❌ Pipeline failed at Stage 3")
        return 1

    # Success!
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nTotal execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print("\nOutput files generated:")
    print("  📄 detected_symbols.json         - Detected city symbols with coordinates")
    print("  📄 detected_text_labels.json     - Extracted text labels with coordinates")
    print("  📄 matched_cities.json           - Final matched cities with all data")
    print("  📄 cities_coordinates.csv        - Simple CSV export")
    print("  🖼️  debug_symbol_detection.jpg   - Symbol detection visualization")
    print("  🖼️  debug_brown_mask.jpg         - Brown color detection mask")
    print("  🖼️  visualization_matched_cities.jpg - Final matched results")
    print("\nNext steps:")
    print("  1. Review visualization_matched_cities.jpg to verify results")
    print("  2. Check matched_cities.json for detailed matching data")
    print("  3. Use cities_coordinates.csv for further analysis")
    print("  4. Adjust parameters in individual scripts if needed and re-run")

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
