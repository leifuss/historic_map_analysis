# City Label Extraction Summary

## Overview

This document summarizes the work done to extract text labels for the 585 detected city symbols on the Idrisi map (Tabula Rogeriana, Konrad Miller 1929 reconstruction).

## What Was Accomplished

### 1. Text Region Detection (✓ Complete)

- **Script**: `extract_text_regions.py`
- **Method**: Used OpenCV's MSER (Maximally Stable Extremal Regions) algorithm to detect text-like regions near each city symbol
- **Results**:
  - Detected text regions for all 585 cities
  - Total of 1,751 text regions identified
  - Average of ~3 text region candidates per city

### 2. Text Crop Extraction (✓ Complete)

- **Script**: `prepare_for_ocr.py`
- **Output**:
  - 1,751 individual text region images extracted
  - 2 versions per region (original + enhanced for OCR)
  - Total of 3,502 image files in `text_crops/` directory
  - Each crop is labeled as `city####_region#_[original|enhanced].jpg`

### 3. Transcription Template (✓ Complete)

- **File**: `transcription_template.csv`
- **Purpose**: Template for recording OCR results or manual transcriptions
- **Structure**: 1,751 rows (one per text region) with columns:
  - `city_id`: Unique city identifier (0-584)
  - `city_x`, `city_y`: City symbol coordinates
  - `text_region_filename`: Corresponding crop image name
  - `distance_from_city`: Distance from city center (in pixels)
  - `transcribed_text`: (Empty - to be filled)
  - `confidence`: (Empty - to be filled)
  - `notes`: (Empty - to be filled)

### 4. Processing Instructions (✓ Complete)

- **File**: `OCR_INSTRUCTIONS.txt`
- **Content**: Detailed instructions for:
  - Offline OCR processing with Tesseract
  - Manual transcription workflow
  - Online OCR service options
  - Tips for handling medieval Latin script and diacritics

### 5. Merge Script (✓ Complete)

- **Script**: `merge_transcriptions.py`
- **Purpose**: Combine transcribed text with city coordinates
- **Usage**: `python3 merge_transcriptions.py transcription_template.csv`
- **Output**: Will generate `cities_with_labels.json` with format:
  ```json
  {
    "total_cities": 585,
    "labeled_cities": N,
    "cities": [
      {
        "id": 0,
        "center": {"x": 5574, "y": 3399},
        "confidence": 0.960,
        "label": "City Name",
        "label_confidence": "high",
        "all_detected_labels": ["City Name", "Alt Name"]
      },
      ...
    ]
  }
  ```

## Files Generated

```
historic_map_analysis/
├── text_crops/                      # 3,502 text region images
│   ├── city0000_region0_original.jpg
│   ├── city0000_region0_enhanced.jpg
│   └── ...
├── text_region_samples/             # 100 sample context images
│   └── city_####_conf#.###.jpg
├── text_regions_detected.json       # Text region coordinates
├── transcription_template.csv       # Template for transcription
├── OCR_INSTRUCTIONS.txt             # Detailed processing instructions
├── merge_transcriptions.py          # Script to merge results
├── extract_text_regions.py          # Text region detection script
└── prepare_for_ocr.py               # Crop extraction script
```

## Why Model-Based OCR Wasn't Used

We attempted to use several OCR libraries that support diacritics:

1. **EasyOCR** - Failed: HTTP 403 error when downloading models
2. **PaddleOCR** - Failed: Package conflicts with system PyYAML
3. **Tesseract** - Not available: System package installation restricted

The environment has network and permission restrictions that prevent:
- Downloading pre-trained OCR models
- Installing system packages (apt-get)
- Accessing certain model repositories

## Recommended Next Steps

### Option 1: Offline OCR (Recommended for Accuracy)

1. Download the repository to a local machine with Tesseract installed
2. Run batch OCR on the `text_crops/*_enhanced.jpg` images:
   ```bash
   cd text_crops
   for f in *_enhanced.jpg; do
     tesseract "$f" "${f%.jpg}" -l lat+ara+eng
   done
   ```
3. Parse the generated .txt files and fill `transcription_template.csv`
4. Run `python3 merge_transcriptions.py transcription_template.csv`

### Option 2: Online OCR Service

1. Upload images from `text_crops/` to a service like:
   - Google Cloud Vision API (best for historical text)
   - Microsoft Azure Computer Vision
   - Amazon Textract
   - Free services: OnlineOCR.net, NewOCR.com
2. Export results and fill into `transcription_template.csv`
3. Run merge script

### Option 3: Manual Transcription

1. Open `transcription_template.csv` in Excel/LibreOffice
2. For each row, view the corresponding image in `text_crops/`
3. Type the city name into the `transcribed_text` column
4. Save and run merge script

## Technical Details

### Text Detection Algorithm

The text region detection uses MSER (Maximally Stable Extremal Regions):
- Detects blob-like regions with stable boundaries
- Particularly effective for text in various fonts and sizes
- Filters regions by size (50-2000 pixels) to match expected text scale
- Groups nearby regions to form complete words/labels
- Ranks candidates by distance from city symbol

### Image Preprocessing

Each text region is processed to improve OCR accuracy:
1. Extract with 10px padding
2. Upscale small regions (min 30x50 pixels)
3. Convert to grayscale
4. Apply adaptive thresholding (Gaussian, kernel=11)
5. Enhance contrast with CLAHE

### Coordinate System

- All coordinates are in pixels relative to the original map image
- Map dimensions: 9933 x 7016 pixels
- City coordinates are stored in `filtered_cities.json`
- Text region coordinates are in `text_regions_detected.json`

## Statistics

- **Total cities**: 585
- **Cities with detected text regions**: 585 (100%)
- **Total text regions detected**: 1,751
- **Average regions per city**: 3.0
- **Text region images**: 3,502 (original + enhanced)
- **Sample images for review**: 100

## Notes for Transcription

1. **Script**: The map uses medieval Latin script and some Arabic
2. **Diacritics**: Many city names include marks like ā, ī, ū, ḥ, etc.
3. **Multiple candidates**: Each city may have multiple detected regions
   - Use the closest region (smallest `distance_from_city`)
   - Some may be decorative labels or region names, not city names
4. **Quality**: Use `*_enhanced.jpg` for OCR, `*_original.jpg` for verification
5. **Orientation**: Most text is horizontal, but some may be rotated

## Future Improvements

If processing in an unrestricted environment:
1. Use Florence-2 or similar VLM for joint detection + recognition
2. Fine-tune OCR model on medieval manuscript data
3. Implement automatic diacritic recognition
4. Add language model for historical place name validation
5. Implement automatic text orientation correction

## Questions?

For issues or questions:
1. Check `OCR_INSTRUCTIONS.txt` for detailed guidance
2. Review sample images in `text_region_samples/`
3. Examine `text_regions_detected.json` for raw detection data
