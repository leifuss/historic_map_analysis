# Large PDF Processing - Usage Guide

This system efficiently processes large PDF files (like historic maps) with memory management and lazy loading.

## Features

- **Memory-efficient processing**: Pages loaded lazily, not all at once
- **Page-by-page extraction**: Process specific pages or ranges
- **High-quality image output**: Configurable DPI and formats
- **Thumbnail generation**: Automatic creation of preview images
- **Progress tracking**: Visual feedback during processing
- **Configurable settings**: YAML-based configuration

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Analyze a PDF

Get metadata and page information without extracting images:

```bash
python process_pdf.py analyze descriptiondela00goejgoog.pdf
```

This will show:
- File size and page count
- PDF metadata (title, author, dates)
- Memory usage estimates
- Page dimensions
- Embedded image count

### 2. Extract Images

Extract all pages as images:

```bash
python process_pdf.py extract descriptiondela00goejgoog.pdf
```

Extract first 5 pages only:

```bash
python process_pdf.py extract descriptiondela00goejgoog.pdf --max-pages 5
```

Extract specific page range (pages 10-20):

```bash
python process_pdf.py extract descriptiondela00goejgoog.pdf --start 10 --end 20
```

Extract at higher resolution:

```bash
python process_pdf.py extract descriptiondela00goejgoog.pdf --dpi 600
```

## Configuration

Edit `config/processing_config.yaml` to customize:

```yaml
processing:
  default_dpi: 300          # Image resolution
  page_cache_size: 3        # Pages kept in memory
  parallel_workers: 4       # For future parallel processing

output:
  image_format: "png"       # Output format (png, jpeg, tiff)
  compression_quality: 95   # JPEG quality (1-100)
  create_thumbnails: true   # Generate preview thumbnails
  thumbnail_size: [200, 200]
```

## Output Structure

```
output/
├── images/              # Full-resolution extracted images
│   ├── filename_0000.png
│   ├── filename_0001.png
│   └── ...
├── thumbnails/          # Preview thumbnails
│   ├── filename_0000_thumb.png
│   └── ...
└── processing.log       # Processing log file
```

## Python API Usage

### Basic Usage

```python
from src.pdf_processor import LargePDFProcessor

# Open PDF
with LargePDFProcessor("my_document.pdf") as processor:
    # Get metadata
    print(processor.metadata)

    # Iterate through pages
    for page_num, page in processor.iter_pages():
        print(f"Processing page {page_num}")
        # Process page...
```

### Extract Images Programmatically

```python
from src.pdf_processor import LargePDFProcessor
from src.image_extractor import ImageExtractor

processor = LargePDFProcessor("my_document.pdf")
processor.open()

extractor = ImageExtractor(output_dir="output/images")

# Process specific page
page = processor.get_page(0)
extractor.save_page_as_image(
    page,
    output_path="output/page_0.png",
    dpi=300
)

processor.close()
```

### Memory Usage Estimation

```python
from src.pdf_processor import LargePDFProcessor

with LargePDFProcessor("my_document.pdf") as processor:
    # Estimate memory for 300 DPI processing
    estimate = processor.estimate_memory_usage(dpi=300)
    print(f"Memory per page: {estimate['per_page_mb']} MB")
    print(f"Total memory: {estimate['total_mb']} MB")
```

## Performance Tips

1. **For large PDFs (>100MB)**:
   - Use `--max-pages` to process in batches
   - Lower DPI (150-200) for previews
   - Consider JPEG format to save disk space

2. **For high-quality extraction**:
   - Use 600 DPI for archival quality
   - PNG format for lossless compression
   - Process in smaller batches to manage memory

3. **Batch processing**:
   ```bash
   # Process in chunks of 10 pages
   python process_pdf.py extract document.pdf --start 0 --end 10
   python process_pdf.py extract document.pdf --start 10 --end 20
   # etc...
   ```

## Architecture

### Core Components

- **LargePDFProcessor**: Main PDF handler with lazy loading and caching
- **ImageExtractor**: Converts pages to images with various options
- **PageCache**: LRU cache for managing memory usage
- **Configuration**: YAML-based settings management

### Memory Management

The system uses an LRU (Least Recently Used) cache to keep only a few pages in memory at once:

1. Pages are loaded on-demand
2. Oldest pages are evicted when cache is full
3. Garbage collection runs after eviction
4. Configurable cache size (default: 3 pages)

This allows processing PDFs of any size with constant memory usage.

## Troubleshooting

### Out of Memory Errors

- Reduce `page_cache_size` in config
- Lower DPI setting
- Process fewer pages at once

### Slow Processing

- Increase `page_cache_size` (if memory available)
- Use JPEG instead of PNG
- Lower DPI for non-archival purposes

### Image Quality Issues

- Increase DPI (300-600)
- Use PNG format
- Check source PDF quality

## Examples

See the PDF file included in the repository (`descriptiondela00goejgoog.pdf`) for testing.

```bash
# Analyze the included PDF
python process_pdf.py analyze descriptiondela00goejgoog.pdf

# Extract first 3 pages as a test
python process_pdf.py extract descriptiondela00goejgoog.pdf --max-pages 3
```
