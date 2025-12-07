#!/usr/bin/env python3
"""
OCR pages 197-266 specifically for the Fourth Climate Iberian section.
This should contain the French translation with itinerary information.
"""

import sys
from pathlib import Path
import json
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pdf_processor import LargePDFProcessor
import pytesseract
from PIL import Image
import fitz
from tqdm import tqdm


def ocr_page(page, dpi: int = 200) -> str:
    """Extract text from page using OCR."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang='ara+fra+eng')


def ocr_page_range(pdf_path: str, start_page: int, end_page: int, dpi: int = 200) -> List[Dict]:
    """
    OCR a specific range of pages.

    Args:
        pdf_path: Path to PDF
        start_page: Starting page (0-indexed)
        end_page: Ending page (exclusive)
        dpi: OCR resolution

    Returns:
        List of page data dictionaries
    """
    print(f"\n{'='*70}")
    print(f"OCR PROCESSING: Pages {start_page} to {end_page-1}")
    print(f"{'='*70}\n")
    print(f"Target: Fourth Climate - Iberian Peninsula Section")
    print(f"DPI: {dpi}")
    print(f"Pages to process: {end_page - start_page}\n")

    pages_data = []

    with LargePDFProcessor(pdf_path) as processor:
        print(f"Total pages in PDF: {processor.page_count}\n")

        for page_num in tqdm(range(start_page, end_page), desc="OCR Progress"):
            try:
                page = processor.get_page(page_num)
                text = ocr_page(page, dpi=dpi)

                pages_data.append({
                    'page_num': page_num,
                    'text': text,
                    'text_length': len(text)
                })

            except Exception as e:
                print(f"\n✗ Error on page {page_num}: {e}")
                pages_data.append({
                    'page_num': page_num,
                    'text': '',
                    'text_length': 0,
                    'error': str(e)
                })

    return pages_data


def save_ocr_results(pages_data: List[Dict], output_file: str):
    """Save OCR results to JSON and text files."""

    # Save JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved JSON: {output_file}")

    # Save as combined text file
    text_file = output_file.replace('.json', '.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        for page in pages_data:
            f.write(f"\n{'='*70}\n")
            f.write(f"PAGE {page['page_num']}\n")
            f.write(f"{'='*70}\n\n")
            f.write(page['text'])
            f.write('\n\n')

    print(f"✓ Saved text: {text_file}")

    # Statistics
    total_chars = sum(p['text_length'] for p in pages_data)
    successful = sum(1 for p in pages_data if p['text_length'] > 0)

    print(f"\n{'='*70}")
    print("OCR STATISTICS")
    print(f"{'='*70}")
    print(f"Total pages processed: {len(pages_data)}")
    print(f"Successful extractions: {successful}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average per page: {total_chars // len(pages_data):,}")
    print(f"{'='*70}\n")


def main():
    pdf_path = "descriptiondela00goejgoog.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    # Process pages 197-266 (0-indexed: 196-265)
    start_page = 196
    end_page = 266
    dpi = 200

    # OCR the pages
    pages_data = ocr_page_range(pdf_path, start_page, end_page, dpi=dpi)

    # Save results
    output_file = "fourth_climate_pages_197-266.json"
    save_ocr_results(pages_data, output_file)

    print(f"✓ Complete! OCR data saved.")
    print(f"\nNext step: Run extract_iberian_itineraries.py to parse routes\n")


if __name__ == '__main__':
    main()
