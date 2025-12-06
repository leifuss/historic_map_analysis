#!/usr/bin/env python3
"""
Extract Arabic text and French translations related to the Iberian peninsula
using OCR (Tesseract).
"""

import sys
from pathlib import Path
import json
import re
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pdf_processor import LargePDFProcessor
from image_extractor import ImageExtractor

try:
    import pytesseract
    from PIL import Image
    import fitz
    TESSERACT_AVAILABLE = True
except ImportError:
    print("Error: pytesseract required. Run: pip install pytesseract")
    sys.exit(1)


def ocr_page(page, dpi: int = 300, lang: str = 'ara+fra+eng') -> str:
    """
    Extract text from a PDF page using OCR.

    Args:
        page: PyMuPDF page object
        dpi: Resolution for OCR
        lang: Languages to use (ara=Arabic, fra=French, eng=English)

    Returns:
        Extracted text
    """
    # Render page to image
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Perform OCR
    text = pytesseract.image_to_string(img, lang=lang)

    return text


def contains_iberian_keywords(text: str) -> tuple:
    """
    Check if text contains Iberian peninsula keywords.

    Args:
        text: Text to search

    Returns:
        Tuple of (has_match, list_of_matches)
    """
    keywords = [
        'espagne', 'spain', 'iberia', 'ibérie',
        'al-andalus', 'andalus', 'andalousie',
        'portugal', 'lusitanie',
        'cordoue', 'cordoba', 'córdoba',
        'séville', 'sevilla', 'seville',
        'grenade', 'granada',
        'toledo', 'tolède',
        'valence', 'valencia',
        'barcelone', 'barcelona',
        'lisbonne', 'lisbon',
        'catalogne', 'aragon', 'castille',
    ]

    text_lower = text.lower()
    matches = [kw for kw in keywords if kw in text_lower]

    return len(matches) > 0, matches


def separate_arabic_french(text: str) -> Dict:
    """
    Separate Arabic and French text.

    Args:
        text: Mixed text content

    Returns:
        Dictionary with separated Arabic and French text
    """
    # Arabic Unicode ranges
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+'

    lines = text.split('\n')
    arabic_lines = []
    french_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if line contains Arabic
        if re.search(arabic_pattern, line):
            arabic_lines.append(line)
        else:
            # Likely French/Latin text
            french_lines.append(line)

    return {
        'arabic_text': '\n'.join(arabic_lines),
        'french_text': '\n'.join(french_lines),
        'has_arabic': len(arabic_lines) > 0
    }


def process_pdf_for_iberian_content(pdf_path: str, max_pages: int = None, dpi: int = 200):
    """
    Process PDF to find and extract Iberian peninsula content.

    Args:
        pdf_path: Path to PDF
        max_pages: Maximum pages to process (None = all)
        dpi: OCR resolution (lower = faster, higher = more accurate)
    """
    print(f"\nProcessing: {pdf_path}")
    print(f"OCR DPI: {dpi}")
    print("This will take some time...\n")

    relevant_pages = []

    with LargePDFProcessor(pdf_path) as processor:
        total_pages = min(max_pages, processor.page_count) if max_pages else processor.page_count

        print(f"Scanning {total_pages} pages for Iberian content...\n")

        for page_num in range(total_pages):
            if page_num % 10 == 0:
                print(f"Processing page {page_num}/{total_pages}...")

            page = processor.get_page(page_num)

            # Perform OCR
            try:
                text = ocr_page(page, dpi=dpi)

                # Check for Iberian keywords
                has_match, matches = contains_iberian_keywords(text)

                if has_match:
                    # Separate Arabic and French
                    lang_split = separate_arabic_french(text)

                    relevant_pages.append({
                        'page_num': page_num,
                        'matches': matches,
                        'full_text': text,
                        'arabic_text': lang_split['arabic_text'],
                        'french_text': lang_split['french_text'],
                        'has_arabic': lang_split['has_arabic']
                    })

                    print(f"  ✓ Page {page_num}: Found Iberian content ({', '.join(matches[:3])})")

            except Exception as e:
                print(f"  ✗ Page {page_num}: OCR failed - {e}")
                continue

    return relevant_pages


def save_results(results: List[Dict], output_dir: str = "output"):
    """
    Save extraction results to files.

    Args:
        results: List of extracted page data
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_file = output_path / "iberian_content.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved JSON to: {json_file}")

    # Save Markdown
    md_file = output_path / "iberian_content.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Iberian Peninsula Content from al-Idrisi's Description\n\n")
        f.write(f"Extracted from {len(results)} pages\n\n")
        f.write("---\n\n")

        for result in results:
            f.write(f"## Page {result['page_num']}\n\n")
            f.write(f"**Keywords found:** {', '.join(result['matches'])}\n\n")

            if result['has_arabic']:
                f.write("### Arabic Text\n\n")
                f.write("```\n")
                f.write(result['arabic_text'])
                f.write("\n```\n\n")

            f.write("### French Translation\n\n")
            f.write(result['french_text'])
            f.write("\n\n---\n\n")

    print(f"Saved Markdown to: {md_file}")

    # Save separate Arabic and French files
    arabic_file = output_path / "iberian_arabic.txt"
    french_file = output_path / "iberian_french.txt"

    with open(arabic_file, 'w', encoding='utf-8') as f:
        for result in results:
            if result['has_arabic']:
                f.write(f"=== Page {result['page_num']} ===\n")
                f.write(result['arabic_text'])
                f.write("\n\n")

    with open(french_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"=== Page {result['page_num']} ===\n")
            f.write(result['french_text'])
            f.write("\n\n")

    print(f"Saved Arabic text to: {arabic_file}")
    print(f"Saved French text to: {french_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Iberian peninsula content from al-Idrisi PDF"
    )
    parser.add_argument(
        'pdf_file',
        nargs='?',
        default='descriptiondela00goejgoog.pdf',
        help="PDF file to process"
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=50,
        help="Maximum pages to process (default: 50, use 0 for all)"
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help="OCR resolution (default: 200, higher=slower but better)"
    )

    args = parser.parse_args()

    pdf_path = args.pdf_file
    max_pages = None if args.max_pages == 0 else args.max_pages

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("IBERIAN CONTENT EXTRACTION WITH OCR")
    print("="*70)

    # Process PDF
    results = process_pdf_for_iberian_content(
        pdf_path,
        max_pages=max_pages,
        dpi=args.dpi
    )

    print("\n" + "="*70)
    print(f"EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total pages with Iberian content: {len(results)}")
    print(f"Pages with Arabic text: {sum(1 for r in results if r['has_arabic'])}")

    if results:
        # Save results
        save_results(results)

        # Display sample
        print("\n" + "="*70)
        print("SAMPLE FROM FIRST MATCH:")
        print("="*70)
        first = results[0]
        print(f"\nPage {first['page_num']}")
        print(f"Keywords: {', '.join(first['matches'][:5])}\n")

        if first['has_arabic']:
            print("Arabic text (first 200 chars):")
            print(first['arabic_text'][:200])
            print("\n")

        print("French text (first 300 chars):")
        print(first['french_text'][:300])
    else:
        print("\nNo Iberian content found in the scanned pages.")

    print("\n")


if __name__ == '__main__':
    main()
