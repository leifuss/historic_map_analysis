# Historic Map Analysis - Large PDF Processing System

A Python system for efficiently processing large PDF files (like historic maps) with memory-efficient lazy loading and image extraction capabilities.

## Features

- **Memory-Efficient Processing**: Lazy page loading with LRU caching prevents memory overflow
- **Large File Support**: Process PDFs of any size with constant memory usage
- **High-Quality Image Extraction**: Configurable DPI (up to 600+) with multiple output formats
- **Thumbnail Generation**: Automatic creation of preview images
- **Batch Processing**: Process specific page ranges or entire documents
- **Progress Tracking**: Real-time progress indicators with time estimates
- **Flexible Configuration**: YAML-based settings for easy customization

## Project Structure

```
historic_map_analysis/
├── src/
│   ├── pdf_processor.py      # Core PDF handler with lazy loading
│   ├── image_extractor.py    # Image extraction and conversion
│   └── utils.py              # Configuration and utilities
├── examples/
│   └── simple_example.py     # API usage examples
├── config/
│   └── processing_config.yaml # Processing configuration
├── process_pdf.py            # Main CLI tool
├── requirements.txt          # Python dependencies
└── USAGE.md                  # Detailed usage guide
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Analyze a PDF (no extraction):

```bash
python process_pdf.py analyze descriptiondela00goejgoog.pdf
```

Extract all pages as images:

```bash
python process_pdf.py extract descriptiondela00goejgoog.pdf
```

Extract first 10 pages:

```bash
python process_pdf.py extract descriptiondela00goejgoog.pdf --max-pages 10
```

## Example Output

The system was tested with a 676-page, 21MB historic map PDF:

```
============================================================
PDF METADATA
============================================================
File Name:       descriptiondela00goejgoog.pdf
File Size:       21.21 MB
Pages:           676
Title:           Description de l'Afrique et de l'Espagne
Author:          Idrīsī, Abou 'Abd Allāh Moḥammad...
============================================================

MEMORY ESTIMATE
============================================================
DPI:             300
Image Dimensions: 2550x3300
Per Page:        24.08 MB
Cache (3 pages): 72.23 MB
All Pages:       16275.04 MB  (processed page-by-page!)
============================================================
```

## Key Capabilities

### Memory Management

The system uses an LRU (Least Recently Used) cache that keeps only a configurable number of pages in memory:

- **Small cache** (3 pages default) = ~72 MB memory
- **Process 676 pages** without loading all into memory
- **Automatic garbage collection** when pages are evicted

### Image Extraction

- DPI range: 150-600+ (configurable)
- Formats: PNG, JPEG, TIFF
- Automatic thumbnail generation
- Embedded image extraction
- Progress tracking with ETA

### Batch Processing

Process PDFs in chunks:

```bash
# First 100 pages
python process_pdf.py extract large.pdf --start 0 --end 100

# Next 100 pages
python process_pdf.py extract large.pdf --start 100 --end 200
```

## Performance

Tested with 676-page historic map PDF:

- **Analysis**: ~3.5 seconds for all pages
- **Extraction**: ~3-4 seconds per page at 300 DPI
- **Memory**: Constant (72 MB) regardless of PDF size
- **Output**: ~1 MB per page (PNG, 300 DPI)

## Python API

```python
from src.pdf_processor import LargePDFProcessor
from src.image_extractor import ImageExtractor

# Process PDF programmatically
with LargePDFProcessor("document.pdf") as processor:
    extractor = ImageExtractor()

    for page_num, page in processor.iter_pages():
        extractor.save_page_as_image(
            page,
            f"output/page_{page_num}.png",
            dpi=300
        )
```

See `examples/simple_example.py` for more examples.

## Configuration

Edit `config/processing_config.yaml`:

```yaml
processing:
  default_dpi: 300          # Image resolution
  page_cache_size: 3        # Pages in memory
  max_dpi: 600             # Maximum allowed DPI

output:
  image_format: "png"       # png, jpeg, tiff
  create_thumbnails: true
  thumbnail_size: [200, 200]
```

## Documentation

- **USAGE.md**: Comprehensive usage guide with examples
- **examples/simple_example.py**: Programmatic API examples
- **Source code**: Well-documented with docstrings

## Use Cases

- Historic map digitization
- Large document archival
- Scanned book processing
- High-resolution image extraction
- Batch PDF conversion

## Technical Details

### Architecture

- **LargePDFProcessor**: Main class with lazy page loading
- **PageCache**: LRU cache for memory management
- **ImageExtractor**: High-quality image rendering
- **Configuration System**: YAML-based settings

### Dependencies

- PyMuPDF (fitz): Fast PDF processing
- Pillow: Image manipulation
- PyYAML: Configuration management
- tqdm: Progress tracking

## Tested With

- **PDF**: 676-page historic map document (21 MB)
- **Python**: 3.x
- **Platform**: Linux

## License

Open source - feel free to use and modify.
