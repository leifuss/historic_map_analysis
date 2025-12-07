#!/usr/bin/env python3
"""
Main script for processing large PDF files.

This script demonstrates how to use the LargePDFProcessor and ImageExtractor
to efficiently process large historic map PDFs.
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pdf_processor import LargePDFProcessor
from image_extractor import ImageExtractor
from utils import (
    load_config,
    setup_logging,
    format_bytes,
    ProgressTracker
)


def print_metadata(processor: LargePDFProcessor):
    """Print PDF metadata in a formatted way."""
    metadata = processor.metadata

    print("\n" + "=" * 60)
    print("PDF METADATA")
    print("=" * 60)
    print(f"File Name:       {metadata['file_name']}")
    print(f"File Size:       {metadata['file_size_mb']:.2f} MB")
    print(f"Pages:           {metadata['page_count']}")
    print(f"PDF Version:     {metadata['pdf_version']}")
    if metadata['title']:
        print(f"Title:           {metadata['title']}")
    if metadata['author']:
        print(f"Author:          {metadata['author']}")
    if metadata['creation_date']:
        print(f"Created:         {metadata['creation_date']}")
    print("=" * 60 + "\n")


def print_memory_estimate(processor: LargePDFProcessor, dpi: int):
    """Print memory usage estimates."""
    estimate = processor.estimate_memory_usage(dpi=dpi)

    print("\n" + "=" * 60)
    print("MEMORY ESTIMATE")
    print("=" * 60)
    print(f"DPI:             {estimate['dpi']}")
    print(f"Image Dimensions: {estimate['dimensions']}")
    print(f"Per Page:        {estimate['per_page_mb']:.2f} MB")
    print(f"Cache ({processor.cache_size} pages): {estimate['cached_mb']:.2f} MB")
    print(f"All Pages:       {estimate['total_mb']:.2f} MB")
    print("=" * 60 + "\n")


def analyze_pdf(pdf_path: str, config: dict):
    """
    Analyze PDF without extracting images.

    Args:
        pdf_path: Path to PDF file
        config: Configuration dictionary
    """
    print(f"\nAnalyzing PDF: {pdf_path}\n")

    with LargePDFProcessor(
        pdf_path,
        cache_size=config['processing']['page_cache_size']
    ) as processor:
        # Print metadata
        print_metadata(processor)

        # Print memory estimates
        dpi = config['processing']['default_dpi']
        print_memory_estimate(processor, dpi)

        # Analyze pages
        print("Analyzing pages...")
        page_info_list = []

        for page_num, page in tqdm(
            processor.iter_pages(),
            total=processor.page_count,
            desc="Scanning pages"
        ):
            info = processor.get_page_info(page_num)
            page_info_list.append(info)

        # Print summary
        print("\n" + "=" * 60)
        print("PAGE ANALYSIS")
        print("=" * 60)

        if page_info_list:
            first_page = page_info_list[0]
            print(f"Typical page size: {first_page['width']:.0f} x {first_page['height']:.0f} points")
            print(f"                   ({first_page['width_inches']:.1f} x {first_page['height_inches']:.1f} inches)")

            total_images = sum(p['image_count'] for p in page_info_list)
            total_text_len = sum(p['text_length'] for p in page_info_list)

            print(f"Total embedded images: {total_images}")
            print(f"Total text length: {total_text_len} characters")
            print(f"Average text per page: {total_text_len / len(page_info_list):.0f} characters")

        print("=" * 60 + "\n")


def extract_images(pdf_path: str, config: dict, start_page: int = 0, end_page: int = None, max_pages: int = None):
    """
    Extract images from PDF pages.

    Args:
        pdf_path: Path to PDF file
        config: Configuration dictionary
        start_page: Starting page number (0-indexed)
        end_page: Ending page number (exclusive)
        max_pages: Maximum number of pages to process
    """
    print(f"\nExtracting images from PDF: {pdf_path}\n")

    # Get configuration
    proc_config = config['processing']
    out_config = config['output']

    dpi = proc_config['default_dpi']
    img_format = out_config['image_format']
    quality = out_config['compression_quality']
    create_thumbs = out_config['create_thumbnails']
    thumb_size = tuple(out_config['thumbnail_size'])

    # Initialize processors
    processor = LargePDFProcessor(
        pdf_path,
        cache_size=proc_config['page_cache_size']
    )
    processor.open()

    extractor = ImageExtractor(
        output_dir=out_config['image_dir'],
        thumbnail_dir=out_config['thumbnail_dir']
    )

    try:
        # Determine page range
        total_pages = processor.page_count
        if end_page is None:
            end_page = total_pages
        if max_pages:
            end_page = min(start_page + max_pages, end_page)

        num_pages = end_page - start_page

        print(f"Processing pages {start_page} to {end_page-1} ({num_pages} pages)")
        print(f"Output directory: {extractor.output_dir}")
        print(f"DPI: {dpi}, Format: {img_format}")
        print()

        # Print memory estimate
        estimate = processor.estimate_memory_usage(dpi=dpi)
        print(f"Estimated memory per page: {estimate['per_page_mb']:.2f} MB")
        print(f"Estimated total for selected pages: {estimate['per_page_mb'] * num_pages:.2f} MB\n")

        # Get file prefix from PDF name
        prefix = Path(pdf_path).stem

        # Process pages
        results = []
        for page_num, page in tqdm(
            processor.iter_pages(start=start_page, end=end_page),
            total=num_pages,
            desc="Extracting images"
        ):
            if create_thumbs:
                result = extractor.process_page_with_thumbnail(
                    page=page,
                    page_num=page_num,
                    prefix=prefix,
                    dpi=dpi,
                    format=img_format,
                    quality=quality,
                    thumbnail_size=thumb_size
                )
            else:
                # Generate filename
                image_filename = f"{prefix}_{page_num:04d}.{img_format}"
                image_path = extractor.output_dir / image_filename

                result = extractor.save_page_as_image(
                    page=page,
                    output_path=image_path,
                    dpi=dpi,
                    format=img_format,
                    quality=quality
                )
                result['page_number'] = page_num

            results.append(result)

        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Pages processed: {len(results)}")
        total_size = sum(r['file_size_mb'] for r in results)
        print(f"Total size: {total_size:.2f} MB")
        print(f"Average per page: {total_size / len(results):.2f} MB")
        print(f"Images saved to: {extractor.output_dir}")
        if create_thumbs:
            print(f"Thumbnails saved to: {extractor.thumbnail_dir}")
        print("=" * 60 + "\n")

    finally:
        processor.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process large PDF files efficiently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a PDF without extracting images
  python process_pdf.py analyze my_document.pdf

  # Extract all pages as images
  python process_pdf.py extract my_document.pdf

  # Extract first 10 pages only
  python process_pdf.py extract my_document.pdf --max-pages 10

  # Extract specific page range
  python process_pdf.py extract my_document.pdf --start 5 --end 15

  # Use custom DPI
  python process_pdf.py extract my_document.pdf --dpi 600
        """
    )

    parser.add_argument(
        'command',
        choices=['analyze', 'extract'],
        help="Command to execute"
    )

    parser.add_argument(
        'pdf_file',
        help="Path to PDF file"
    )

    parser.add_argument(
        '--config',
        default='config/processing_config.yaml',
        help="Path to configuration file"
    )

    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help="Starting page number (0-indexed)"
    )

    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help="Ending page number (exclusive)"
    )

    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help="Maximum number of pages to process"
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=None,
        help="Override DPI setting"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Override DPI if specified
    if args.dpi:
        config['processing']['default_dpi'] = args.dpi

    # Set up logging
    setup_logging(config)

    # Check if PDF exists
    if not Path(args.pdf_file).exists():
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)

    # Execute command
    if args.command == 'analyze':
        analyze_pdf(args.pdf_file, config)
    elif args.command == 'extract':
        extract_images(
            args.pdf_file,
            config,
            start_page=args.start,
            end_page=args.end,
            max_pages=args.max_pages
        )


if __name__ == '__main__':
    main()
