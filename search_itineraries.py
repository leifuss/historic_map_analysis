#!/usr/bin/env python3
"""
Search for itinerary sections with distance information in al-Idrisi's document.
Focus on finding route descriptions with distances between locations.
"""

import sys
from pathlib import Path
import re
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pdf_processor import LargePDFProcessor
import pytesseract
from PIL import Image
import fitz


def ocr_page(page, dpi: int = 200) -> str:
    """Extract text from page using OCR."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang='ara+fra+eng')


def has_distance_keywords(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains distance/itinerary keywords.

    Returns:
        Tuple of (has_match, list_of_matches)
    """
    # French distance terms
    distance_keywords = [
        'milles', 'mille', 'lieues', 'lieue',  # miles, leagues
        'journée', 'journées',  # days of travel
        'distance', 'itinéraire', 'route',
        'entre', 'de.*à',  # between, from...to
        'parasanges',  # Persian measure
        'marches', 'étapes',  # stages
    ]

    text_lower = text.lower()
    matches = []

    for keyword in distance_keywords:
        if re.search(keyword, text_lower):
            matches.append(keyword)

    return len(matches) > 0, matches


def search_for_itineraries(pdf_path: str, start_page: int = 40, max_pages: int = 150, dpi: int = 200):
    """
    Search PDF for pages containing itinerary/distance information.

    Args:
        pdf_path: Path to PDF
        start_page: Starting page (skip intro pages)
        max_pages: Number of pages to scan
        dpi: OCR resolution
    """
    print(f"\nSearching for itinerary sections with distances...")
    print(f"Scanning pages {start_page} to {start_page + max_pages}")
    print(f"OCR DPI: {dpi}\n")

    itinerary_pages = []

    with LargePDFProcessor(pdf_path) as processor:
        end_page = min(start_page + max_pages, processor.page_count)

        for page_num in range(start_page, end_page):
            if page_num % 20 == 0:
                print(f"Processing page {page_num}/{end_page}...")

            try:
                page = processor.get_page(page_num)
                text = ocr_page(page, dpi=dpi)

                # Check for distance keywords
                has_distance, matches = has_distance_keywords(text)

                # Check for Iberian locations
                iberian_terms = ['espagne', 'spain', 'andalus', 'portugal',
                                'cordoue', 'cordoba', 'grenade', 'granada',
                                'séville', 'toledo', 'valence', 'lisbonne']

                has_iberian = any(term in text.lower() for term in iberian_terms)

                if has_distance and has_iberian:
                    itinerary_pages.append({
                        'page_num': page_num,
                        'text': text,
                        'distance_keywords': matches,
                        'text_length': len(text)
                    })

                    iberian_found = [t for t in iberian_terms if t in text.lower()]
                    print(f"  ✓ Page {page_num}: Found itinerary content")
                    print(f"    Distance terms: {', '.join(matches[:3])}")
                    print(f"    Locations: {', '.join(iberian_found[:3])}")

            except Exception as e:
                print(f"  ✗ Page {page_num}: Error - {e}")
                continue

    return itinerary_pages


def main():
    pdf_path = "descriptiondela00goejgoog.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("SEARCHING FOR ITINERARY SECTIONS")
    print("="*70)

    # Search pages 40-200 (skip intro material)
    results = search_for_itineraries(
        pdf_path,
        start_page=40,
        max_pages=160,
        dpi=200
    )

    print("\n" + "="*70)
    print(f"FOUND {len(results)} PAGES WITH ITINERARY CONTENT")
    print("="*70)

    if results:
        # Save results
        import json
        output_file = "itinerary_pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nSaved to: {output_file}")
        print(f"\nPage numbers: {[r['page_num'] for r in results[:20]]}")

        # Show sample
        print("\n" + "="*70)
        print("SAMPLE FROM FIRST PAGE:")
        print("="*70)
        print(results[0]['text'][:500])
    else:
        print("\nNo itinerary content found. Try different page range.")


if __name__ == '__main__':
    main()
