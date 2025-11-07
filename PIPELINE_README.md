# Idrisi Map City Coordinate Extraction

Automated pipeline to extract city coordinates from the Tabula Rogeriana (al-Idrisi's 12th-century world map, Konrad Miller 1929 reconstruction).

## Overview

This project identifies **city symbols** (small brown circles resembling bread loaves or Trivial Pursuit game pieces) on the historic map and associates them with their **text labels** to extract precise image coordinates for each city.

### Why Symbol Detection?

Unlike simple text extraction, this approach:
- Provides coordinates of the **actual city markers** on the map (not just label positions)
- Maintains geographic accuracy to the map's visual representation
- Associates symbols with their corresponding city names using spatial proximity

## Pipeline Architecture

### Stage 1: Symbol Detection (`detect_city_symbols.py`)
**Method**: Computer Vision (OpenCV)

- Detects brown-colored regions using HSV color space filtering
- Identifies circular shapes using contour analysis and circularity metrics
- Filters by size and shape to find city symbols
- Outputs: `detected_symbols.json`, `debug_symbol_detection.jpg`

**Key Parameters**:
```python
# Brown color HSV ranges
lower_brown = [0-30, 30-40, 50-60]
upper_brown = [20-30, 200-255, 180-200]

# Symbol size constraints
min_area = 20 pixels
max_area = 500 pixels
min_circularity = 0.5
```

### Stage 2: Label Extraction (`extract_text_labels.py`)
**Method**: Vision-Language Model (Florence-2)

- Uses Microsoft's Florence-2-large model for OCR with region detection
- Extracts all text labels with quad bounding boxes (handles rotated text)
- Calculates center points and bounding rectangles for each label
- Outputs: `detected_text_labels.json`

**Model**: `microsoft/Florence-2-large`
- Task: `<OCR_WITH_REGION>`
- Returns: Text labels with 8-point quad boxes
- Handles: Medieval Latin script, multiple orientations

### Stage 3: Symbol-Label Matching (`match_symbols_to_labels.py`)
**Method**: Spatial Proximity Analysis

- For each detected symbol, finds nearest text labels within radius
- Matches symbols to labels using distance-based confidence scoring
- Handles edge cases (unmatched symbols, multiple candidates)
- Generates visualization with matched pairs
- Outputs: `matched_cities.json`, `cities_coordinates.csv`, `visualization_matched_cities.jpg`

**Matching Algorithm**:
```python
max_distance = 150 pixels  # Adjustable
confidence = 1.0 - (distance / max_distance)
# Prefers unused labels to avoid double-matching
```

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU recommended (CPU fallback supported)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, torch, transformers; print('✓ All dependencies installed')"
```

### Dependencies
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.45.0` - Hugging Face models
- `opencv-python>=4.8.0` - Computer vision
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations

## Usage

### Quick Start (Full Pipeline)
```bash
# Run complete pipeline
python run_pipeline.py
```

This executes all three stages automatically and generates all output files.

### Individual Stages

Run stages separately for debugging or parameter tuning:

```bash
# Stage 1: Detect symbols
python detect_city_symbols.py

# Stage 2: Extract labels
python extract_text_labels.py

# Stage 3: Match symbols to labels
python match_symbols_to_labels.py
```

## Output Files

### JSON Data Files

**`detected_symbols.json`**
```json
{
  "total_symbols": 150,
  "symbols": [
    {
      "id": 0,
      "center": {"x": 1234.5, "y": 567.8},
      "radius": 12.3,
      "area": 475.2,
      "circularity": 0.87
    }
  ]
}
```

**`detected_text_labels.json`**
```json
{
  "total_labels": 450,
  "labels": [
    {
      "id": 0,
      "text": "Roma",
      "center": {"x": 1250.0, "y": 580.0},
      "quad_box": [x1, y1, x2, y2, x3, y3, x4, y4],
      "bounding_box": {"x_min": 1240, "y_min": 575, "x_max": 1260, "y_max": 585}
    }
  ]
}
```

**`matched_cities.json`**
```json
{
  "total_cities": 150,
  "matched_cities": 145,
  "unmatched_symbols": 5,
  "cities": [
    {
      "symbol_id": 0,
      "symbol_coords": {"x": 1234.5, "y": 567.8},
      "city_name": "Roma",
      "label_id": 0,
      "label_coords": {"x": 1250.0, "y": 580.0},
      "distance": 18.2,
      "confidence": 0.88,
      "status": "matched"
    }
  ]
}
```

**`cities_coordinates.csv`**
```csv
City Name,X Coordinate,Y Coordinate,Confidence,Status
Roma,1234.50,567.80,0.878,matched
Alexandria,2345.67,890.12,0.912,matched
```

### Visualization Files

- **`debug_symbol_detection.jpg`**: Shows all detected brown circular symbols with IDs
- **`debug_brown_mask.jpg`**: Binary mask of brown color detection (for tuning)
- **`visualization_matched_cities.jpg`**: Final results with symbols, labels, and matching lines

## Parameter Tuning

### Symbol Detection Sensitivity

Edit `detect_city_symbols.py`:

```python
# Make detection more sensitive (finds more symbols, more false positives)
lower_brown = np.array([0, 20, 40])  # Wider range
min_area = 15  # Smaller symbols
min_circularity = 0.4  # Less strict shape

# Make detection more strict (fewer symbols, fewer false positives)
lower_brown = np.array([5, 40, 60])  # Narrower range
min_area = 30  # Larger symbols only
min_circularity = 0.6  # Stricter shape
```

### Matching Distance

Edit `match_symbols_to_labels.py`:

```python
# Increase if labels are far from symbols
max_distance = 200  # pixels

# Decrease for tighter matching
max_distance = 100  # pixels
```

## Methodology

### Design Decisions

**Q: Why computer vision for symbols instead of VLM?**
A: Small brown circles are well-suited for traditional CV methods:
- Color thresholding is fast and accurate
- Circle detection is mathematically precise
- VLMs may struggle with very small objects (<20px)

**Q: Why Florence-2 for text extraction?**
A: Florence-2 excels at:
- Multi-orientation text (medieval maps have rotated labels)
- Providing precise bounding boxes (quad boxes for rotated text)
- Efficiency (0.7B parameters, fast inference)

**Q: Why not use a single VLM for everything?**
A: Hybrid approach provides:
- Better accuracy (CV excels at geometric shapes, VLM excels at text)
- Debugging capability (can tune each stage independently)
- Flexibility (can swap out components)

### Known Limitations

1. **Brown color variation**: Map aging/scanning may alter brown hue
   - *Solution*: Adjust HSV ranges in `detect_city_symbols.py`

2. **Symbol size variation**: Some cities may use larger/smaller markers
   - *Solution*: Adjust `min_area`/`max_area` parameters

3. **Overlapping text**: Densely labeled regions may confuse matching
   - *Solution*: Reduce `max_distance` or manually review matches

4. **Non-circular symbols**: Some cities might use different icon shapes
   - *Solution*: Modify circularity threshold or add alternative detection method

## Troubleshooting

### No symbols detected
```bash
# Check brown color mask
# Open debug_brown_mask.jpg - should show white regions where brown is detected
# If mostly black: adjust HSV ranges to be more permissive
```

### Too many false positive symbols
```bash
# Increase minimum area: min_area = 30
# Increase circularity threshold: min_circularity = 0.6
# Narrow brown color range
```

### Poor symbol-label matching
```bash
# Check distances in matched_cities.json
# If avg_distance > 100: increase max_distance parameter
# Review visualization_matched_cities.jpg for mismatches
```

### Model download failures
```bash
# Florence-2 is ~1GB, requires good internet
# Models cache to ~/.cache/huggingface/
# Can manually download and point to local path
```

## Advanced Usage

### Analyzing Specific Map Regions

Crop the map before processing:

```python
from PIL import Image

img = Image.open("al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg")
region = img.crop((x1, y1, x2, y2))  # Define region of interest
region.save("map_region.jpeg")

# Then process map_region.jpeg instead
```

### Custom Symbol Detection

For non-circular symbols, modify `detect_city_symbols.py`:

```python
# Example: Detect rectangular symbols
x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = w / h

if 0.8 < aspect_ratio < 1.2:  # Square-ish
    # This is a potential city symbol
```

### Batch Processing Multiple Maps

```bash
# Process multiple map files
for map in map1.jpg map2.jpg map3.jpg; do
    cp "$map" "current_map.jpeg"
    python run_pipeline.py
    mv matched_cities.json "results_${map%.jpg}.json"
done
```

## Research Applications

This pipeline enables:
- **Historical cartography analysis**: Study city distribution patterns
- **Map accuracy assessment**: Compare detected positions to known coordinates
- **Medieval geography research**: Analyze which cities were known/important
- **Cartographic evolution**: Compare different historical map editions

## Citation

If you use this pipeline in research, please cite:

```
Idrisi Map City Coordinate Extraction Pipeline (2025)
https://github.com/[repository]
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **al-Idrisi**: Original 12th-century cartographer
- **Konrad Miller**: 1929 reconstruction
- **Microsoft Florence-2**: Vision-language model for OCR
- **OpenCV**: Computer vision library
