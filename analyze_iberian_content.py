#!/usr/bin/env python3
"""
Analyze PDF text content to understand structure and identify
Iberian peninsula related sections.
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pdf_processor import LargePDFProcessor
from tqdm import tqdm


def analyze_text_content(pdf_path: str, sample_pages: int = 20):
    """
    Analyze first N pages to understand text structure.

    Args:
        pdf_path: Path to PDF file
        sample_pages: Number of pages to sample
    """
    print(f"Analyzing text content structure...\n")

    with LargePDFProcessor(pdf_path) as processor:
        print(f"Total pages: {processor.page_count}\n")

        for page_num, page in processor.iter_pages(start=0, end=min(sample_pages, processor.page_count)):
            text = page.get_text()

            # Get page info
            info = processor.get_page_info(page_num)

            print(f"\n{'='*70}")
            print(f"PAGE {page_num}")
            print(f"{'='*70}")
            print(f"Text length: {len(text)} characters")
            print(f"Embedded images: {info['image_count']}")

            if len(text) > 0:
                # Show first 500 characters
                print(f"\nText sample (first 500 chars):")
                print("-" * 70)
                print(text[:500])
                print("-" * 70)

                # Check for keywords
                keywords = [
                    'Espagne', 'Spain', 'Iberia', 'Ibérie',
                    'al-Andalus', 'Andalusia', 'Andalousie',
                    'Portugal', 'Catalogne', 'Valencia',
                    'Arabic', 'arabe', 'traduction', 'translation'
                ]

                found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                if found_keywords:
                    print(f"\n*** KEYWORDS FOUND: {', '.join(found_keywords)} ***")
            else:
                print("No extractable text (likely scanned image)")


def search_iberian_content(pdf_path: str):
    """
    Search entire PDF for pages containing Iberian peninsula references.

    Args:
        pdf_path: Path to PDF file
    """
    print("\nSearching for Iberian peninsula content...\n")

    # Keywords related to Iberian peninsula
    search_terms = [
        'espagne', 'spain', 'iberia', 'ibérie', 'iberian',
        'al-andalus', 'andalus', 'andalusia', 'andalousie',
        'portugal', 'lusitanie', 'lusitania',
        'catalogne', 'catalonia', 'barcelone', 'barcelona',
        'valence', 'valencia',
        'cordoue', 'cordoba', 'córdoba',
        'séville', 'sevilla', 'seville',
        'grenade', 'granada',
        'toledo', 'tolède',
        'madrid', 'lisbonne', 'lisbon'
    ]

    relevant_pages = []

    with LargePDFProcessor(pdf_path) as processor:
        for page_num, page in tqdm(processor.iter_pages(), total=processor.page_count, desc="Scanning pages"):
            text = page.get_text().lower()

            # Check for search terms
            matches = [term for term in search_terms if term in text]

            if matches:
                relevant_pages.append({
                    'page_num': page_num,
                    'text_length': len(text),
                    'matches': matches,
                    'text_preview': text[:300]
                })

    print(f"\n{'='*70}")
    print(f"FOUND {len(relevant_pages)} PAGES WITH IBERIAN CONTENT")
    print(f"{'='*70}\n")

    for item in relevant_pages:
        print(f"Page {item['page_num']}:")
        print(f"  Matches: {', '.join(item['matches'])}")
        print(f"  Text length: {item['text_length']} chars")
        print(f"  Preview: {item['text_preview'][:150]}...")
        print()

    return relevant_pages


def main():
    pdf_path = "descriptiondela00goejgoog.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("PDF TEXT ANALYSIS FOR IBERIAN PENINSULA CONTENT")
    print("="*70 + "\n")

    # First, analyze structure of first pages
    print("STEP 1: Analyzing text structure")
    analyze_text_content(pdf_path, sample_pages=10)

    # Then search for relevant content
    print("\n\nSTEP 2: Searching for Iberian content")
    relevant_pages = search_iberian_content(pdf_path)

    if relevant_pages:
        print(f"\nRelevant page numbers: {[p['page_num'] for p in relevant_pages]}")
    else:
        print("\nNote: No text found. PDF may contain scanned images requiring OCR.")


if __name__ == '__main__':
    main()
