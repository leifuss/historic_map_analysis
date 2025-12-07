#!/usr/bin/env python3
"""
Simple example demonstrating programmatic PDF processing.

This example shows how to use the PDF processor and image extractor
programmatically in your own Python code.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pdf_processor import LargePDFProcessor
from image_extractor import ImageExtractor


def example_1_basic_info():
    """Example 1: Get basic PDF information."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic PDF Information")
    print("=" * 60 + "\n")

    pdf_path = "descriptiondela00goejgoog.pdf"

    with LargePDFProcessor(pdf_path, cache_size=3) as processor:
        # Get metadata
        metadata = processor.metadata
        print(f"File: {metadata['file_name']}")
        print(f"Size: {metadata['file_size_mb']:.2f} MB")
        print(f"Pages: {metadata['page_count']}")
        print(f"Title: {metadata['title']}")
        print(f"Author: {metadata['author']}")

        # Get memory estimate
        estimate = processor.estimate_memory_usage(dpi=300)
        print(f"\nMemory per page at 300 DPI: {estimate['per_page_mb']:.2f} MB")


def example_2_iterate_pages():
    """Example 2: Iterate through pages."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Iterating Through Pages")
    print("=" * 60 + "\n")

    pdf_path = "descriptiondela00goejgoog.pdf"

    with LargePDFProcessor(pdf_path) as processor:
        # Process first 5 pages
        print("Processing first 5 pages:\n")
        for page_num, page in processor.iter_pages(start=0, end=5):
            info = processor.get_page_info(page_num)
            print(f"Page {page_num}:")
            print(f"  Size: {info['width']:.0f} x {info['height']:.0f} points")
            print(f"  Images: {info['image_count']}")
            print(f"  Text length: {info['text_length']} characters")
            print()


def example_3_extract_single_page():
    """Example 3: Extract a single page as image."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Extract Single Page")
    print("=" * 60 + "\n")

    pdf_path = "descriptiondela00goejgoog.pdf"
    output_dir = "output/examples"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    processor = LargePDFProcessor(pdf_path)
    processor.open()

    extractor = ImageExtractor(output_dir=output_dir)

    try:
        # Extract page 0
        page = processor.get_page(0)

        result = extractor.save_page_as_image(
            page,
            output_path=f"{output_dir}/single_page.png",
            dpi=300,
            format="png"
        )

        print(f"Extracted page to: {result['path']}")
        print(f"Dimensions: {result['width']} x {result['height']}")
        print(f"File size: {result['file_size_mb']:.2f} MB")

    finally:
        processor.close()


def example_4_batch_extract():
    """Example 4: Batch extract multiple pages."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Extract Pages")
    print("=" * 60 + "\n")

    pdf_path = "descriptiondela00goejgoog.pdf"
    output_dir = "output/examples/batch"
    thumbnail_dir = "output/examples/batch_thumbs"

    processor = LargePDFProcessor(pdf_path, cache_size=3)
    processor.open()

    extractor = ImageExtractor(
        output_dir=output_dir,
        thumbnail_dir=thumbnail_dir
    )

    try:
        # Extract pages 0-2
        pages_to_extract = range(0, 3)
        results = []

        print(f"Extracting pages {min(pages_to_extract)} to {max(pages_to_extract)}...\n")

        for page_num in pages_to_extract:
            page = processor.get_page(page_num)

            result = extractor.process_page_with_thumbnail(
                page=page,
                page_num=page_num,
                prefix="batch",
                dpi=300,
                format="png",
                thumbnail_size=(150, 150)
            )

            results.append(result)
            print(f"Processed page {page_num}: {result['image_path']}")

        print(f"\nExtracted {len(results)} pages")
        total_size = sum(r['file_size_mb'] for r in results)
        print(f"Total size: {total_size:.2f} MB")

    finally:
        processor.close()


def example_5_memory_efficient():
    """Example 5: Memory-efficient processing of many pages."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Memory-Efficient Processing")
    print("=" * 60 + "\n")

    pdf_path = "descriptiondela00goejgoog.pdf"

    # Use small cache for memory efficiency
    with LargePDFProcessor(pdf_path, cache_size=2) as processor:
        print(f"Processing {processor.page_count} pages with minimal memory...")
        print(f"Cache size: 2 pages\n")

        # Get info for first 10 pages without storing in memory
        for page_num, page in processor.iter_pages(start=0, end=10):
            info = processor.get_page_info(page_num)
            print(f"Page {page_num}: {info['image_count']} images, "
                  f"{info['text_length']} chars text")

        print("\nNote: Only 2 pages kept in memory at a time!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PDF PROCESSING EXAMPLES")
    print("=" * 60)

    try:
        example_1_basic_info()
        example_2_iterate_pages()
        example_3_extract_single_page()
        example_4_batch_extract()
        example_5_memory_efficient()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")

    except FileNotFoundError:
        print("\nError: PDF file not found!")
        print("Please run this script from the repository root:")
        print("  python examples/simple_example.py\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
