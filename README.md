# Historic Map Analysis: Idrisi Map City Extraction

Automated extraction of city coordinates from the Tabula Rogeriana (al-Idrisi's 12th-century world map, Konrad Miller 1929 reconstruction).

## Overview

This project uses a hybrid computer vision and VLM approach to:
1. **Detect** brown circular city symbols on the map
2. **Extract** text labels using Florence-2 VLM
3. **Match** symbols to labels using spatial proximity

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py
```

## Output

- `matched_cities.json` - City names with symbol coordinates
- `cities_coordinates.csv` - Simple CSV export
- `visualization_matched_cities.jpg` - Visual verification

## Documentation

See **[PIPELINE_README.md](PIPELINE_README.md)** for:
- Detailed methodology
- Parameter tuning guide
- Troubleshooting
- Advanced usage

## Key Features

- **Symbol-centric**: Extracts coordinates of actual city markers, not just labels
- **Hybrid approach**: Combines CV (circle detection) + VLM (OCR)
- **Robust matching**: Spatial proximity algorithm with confidence scoring
- **Fully automated**: Single command to process entire map

## Requirements

- Python 3.8+
- PyTorch + Transformers (Florence-2 model)
- OpenCV
- CUDA GPU recommended (CPU supported)
