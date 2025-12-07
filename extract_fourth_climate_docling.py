#!/usr/bin/env python3
"""
Extract pages 197-266 using Docling for better OCR and structure preservation.
This section contains the Fourth Climate - Iberian Peninsula itineraries.
"""

import sys
from pathlib import Path
import json
from typing import List, Dict
import re

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling not available. Run: pip install docling")


def extract_pages_with_docling(pdf_path: str, start_page: int, end_page: int):
    """
    Extract specific pages using docling.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed for docling)
        end_page: Ending page number (inclusive)

    Returns:
        Extracted content
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("docling not installed. Run: pip install docling")

    print(f"\n{'='*70}")
    print(f"DOCLING EXTRACTION: Pages {start_page} to {end_page}")
    print(f"{'='*70}\n")
    print("Processing with Docling for high-quality extraction...")
    print("This may take several minutes...\n")

    # Configure docling with Tesseract OCR for French text
    # RapidOCR (default) only supports Chinese/English, so we use Tesseract
    ocr_options = TesseractCliOcrOptions(
        lang=["fra", "eng"],  # French and English
        force_full_page_ocr=True  # Required for scanned PDFs
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.ocr_options = ocr_options

    print(f"Using Tesseract OCR with languages: {ocr_options.lang}")

    # Initialize converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document
    print(f"Converting {pdf_path}...")
    result = converter.convert(pdf_path)

    print("\n✓ Conversion complete!")

    return result


def extract_text_by_pages(doc_result, start_page: int, end_page: int) -> List[Dict]:
    """
    Extract text organized by page numbers.

    Args:
        doc_result: Docling document result
        start_page: Starting page (1-indexed)
        end_page: Ending page (inclusive)

    Returns:
        List of page data
    """
    print("\nExtracting text by pages...")

    # Export to markdown
    full_text = doc_result.document.export_to_markdown()

    # Try to split by page if docling provides page markers
    # Otherwise we'll get the full text
    pages_data = []

    # Get page count from document
    if hasattr(doc_result.document, 'pages'):
        print(f"Document has {len(doc_result.document.pages)} pages")

        # Extract text from each page in our range
        for page_num in range(start_page - 1, min(end_page, len(doc_result.document.pages))):
            try:
                page = doc_result.document.pages[page_num]
                # Get text from this page
                page_text = ""

                # Docling organizes content by elements
                for item in page.children:
                    if hasattr(item, 'text'):
                        page_text += item.text + "\n"

                pages_data.append({
                    'page_num': page_num + 1,  # 1-indexed
                    'text': page_text,
                    'text_length': len(page_text)
                })

                print(f"  ✓ Page {page_num + 1}: {len(page_text)} characters")

            except Exception as e:
                print(f"  ✗ Page {page_num + 1}: Error - {e}")
                pages_data.append({
                    'page_num': page_num + 1,
                    'text': '',
                    'text_length': 0,
                    'error': str(e)
                })
    else:
        # Fallback: use full markdown text
        print("Using full markdown export (page separation not available)")
        pages_data.append({
            'page_num': f"{start_page}-{end_page}",
            'text': full_text,
            'text_length': len(full_text)
        })

    return pages_data


def search_iberian_keywords(pages_data: List[Dict]) -> List[Dict]:
    """
    Search for pages containing Iberian peninsula content.

    Args:
        pages_data: List of page dictionaries

    Returns:
        Filtered list with only Iberian-related pages
    """
    iberian_keywords = [
        'espagne', 'spain', 'andalus', 'portugal',
        'cordoue', 'cordoba', 'grenade', 'granada',
        'séville', 'seville', 'toledo', 'tolède',
        'valence', 'valencia', 'barcelone', 'barcelona',
        'lisbonne', 'lisbon', 'malaga', 'málaga',
        'saragosse', 'zaragoza', 'murcie', 'murcia'
    ]

    distance_keywords = ['milles', 'mille', 'journée', 'journées', 'lieue', 'lieues']

    relevant_pages = []

    for page in pages_data:
        text_lower = page['text'].lower()

        # Check for Iberian keywords
        iberian_matches = [kw for kw in iberian_keywords if kw in text_lower]

        # Check for distance keywords
        distance_matches = [kw for kw in distance_keywords if kw in text_lower]

        if iberian_matches or distance_matches:
            page['iberian_keywords'] = iberian_matches
            page['distance_keywords'] = distance_matches
            page['has_itinerary'] = len(distance_matches) > 0
            relevant_pages.append(page)

    return relevant_pages


def save_results(pages_data: List[Dict], output_file: str):
    """Save extraction results."""

    # Save JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved JSON: {output_file}")

    # Save as text
    text_file = output_file.replace('.json', '.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        for page in pages_data:
            f.write(f"\n{'='*70}\n")
            f.write(f"PAGE {page['page_num']}\n")

            if 'iberian_keywords' in page and page['iberian_keywords']:
                f.write(f"Iberian Keywords: {', '.join(page['iberian_keywords'][:5])}\n")
            if 'distance_keywords' in page and page['distance_keywords']:
                f.write(f"Distance Keywords: {', '.join(page['distance_keywords'][:3])}\n")

            f.write(f"{'='*70}\n\n")
            f.write(page['text'])
            f.write('\n\n')

    print(f"✓ Saved text: {text_file}")

    # Statistics
    total_chars = sum(p['text_length'] for p in pages_data)
    with_itineraries = sum(1 for p in pages_data if p.get('has_itinerary', False))

    print(f"\n{'='*70}")
    print("EXTRACTION STATISTICS")
    print(f"{'='*70}")
    print(f"Total pages: {len(pages_data)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Pages with itineraries: {with_itineraries}")
    if pages_data:
        print(f"Average chars/page: {total_chars // len(pages_data):,}")
    print(f"{'='*70}\n")


def main():
    pdf_path = "descriptiondela00goejgoog.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    # Pages 197-266 (1-indexed for docling)
    start_page = 197
    end_page = 266

    try:
        # Extract with docling
        doc_result = extract_pages_with_docling(pdf_path, start_page, end_page)

        # Extract text by pages
        pages_data = extract_text_by_pages(doc_result, start_page, end_page)

        # Filter for Iberian content
        print("\nSearching for Iberian peninsula content...")
        relevant_pages = search_iberian_keywords(pages_data)

        print(f"\nFound {len(relevant_pages)} pages with Iberian content")

        if relevant_pages:
            # Save results
            output_file = "fourth_climate_iberian_docling.json"
            save_results(relevant_pages, output_file)

            print(f"\n✓ Complete! Found {len(relevant_pages)} relevant pages")
            print(f"\nNext: Run parse_itineraries_from_docling.py to extract routes\n")

        else:
            print("\nNo Iberian content found. Saving all extracted pages...")
            output_file = "fourth_climate_all_docling.json"
            save_results(pages_data, output_file)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
