#!/usr/bin/env python3
"""
Extract Arabic text and French translations related to the Iberian peninsula
from the historic al-Idrisi document using docling.
"""

import sys
from pathlib import Path
import json
from typing import Dict, List
import re

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling not available. Installing...")


def extract_document_with_docling(pdf_path: str) -> Dict:
    """
    Extract text from PDF using docling.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary with extracted content
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("docling not installed. Run: pip install docling")

    print(f"Processing PDF with docling: {pdf_path}\n")
    print("This may take several minutes for a large PDF...\n")

    # Initialize converter
    converter = DocumentConverter()

    # Convert PDF
    result = converter.convert(pdf_path)

    return result


def find_iberian_sections(doc_result) -> List[Dict]:
    """
    Find sections related to the Iberian peninsula.

    Args:
        doc_result: Docling document result

    Returns:
        List of relevant sections
    """
    # Keywords to search for
    iberian_keywords = [
        # Spanish/Portuguese terms
        'espagne', 'spain', 'iberia', 'ibérie', 'iberian',
        'al-andalus', 'andalus', 'andalusia', 'andalousie',
        'portugal', 'lusitanie', 'lusitania',

        # Cities
        'cordoue', 'cordoba', 'córdoba',
        'séville', 'sevilla', 'seville',
        'grenade', 'granada',
        'toledo', 'tolède',
        'valence', 'valencia',
        'barcelone', 'barcelona',
        'saragosse', 'zaragoza',
        'lisbonne', 'lisbon',
        'madrid',

        # Regions
        'catalogne', 'catalonia',
        'aragon', 'aragón',
        'castille', 'castile',
        'galice', 'galicia',
        'léon', 'leon',

        # Arabic geographical terms
        'الأندلس',  # al-Andalus
        'إسبانيا',  # Spain
        'البرتغال',  # Portugal
    ]

    relevant_sections = []

    # Extract text from document
    doc_text = doc_result.document.export_to_markdown()

    # Split into sections (paragraphs or pages)
    sections = doc_text.split('\n\n')

    for idx, section in enumerate(sections):
        section_lower = section.lower()

        # Check if section contains any keywords
        matches = [kw for kw in iberian_keywords if kw.lower() in section_lower]

        if matches:
            relevant_sections.append({
                'section_id': idx,
                'text': section,
                'matches': matches,
                'length': len(section)
            })

    return relevant_sections


def identify_arabic_and_translation(text: str) -> Dict:
    """
    Identify Arabic text blocks and their French translations.

    Args:
        text: Text content

    Returns:
        Dictionary with Arabic and French text separated
    """
    # Arabic Unicode range: \u0600-\u06FF, \u0750-\u077F, \uFB50-\uFDFF, \uFE70-\uFEFF
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+'

    # Find all Arabic text segments
    arabic_segments = re.findall(arabic_pattern, text)

    # Split text into lines
    lines = text.split('\n')

    arabic_lines = []
    french_lines = []

    for line in lines:
        if re.search(arabic_pattern, line):
            arabic_lines.append(line)
        elif line.strip():  # Non-empty, non-Arabic line (likely French)
            french_lines.append(line)

    return {
        'arabic_text': '\n'.join(arabic_lines),
        'french_text': '\n'.join(french_lines),
        'arabic_segments': arabic_segments,
        'has_arabic': len(arabic_segments) > 0
    }


def main():
    pdf_path = "descriptiondela00goejgoog.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("EXTRACTING IBERIAN CONTENT FROM AL-IDRISI'S DOCUMENT")
    print("="*70 + "\n")

    try:
        # Extract document with docling
        print("Step 1: Processing PDF with docling...")
        doc_result = extract_document_with_docling(pdf_path)

        print("\nDocument processing complete!\n")

        # Find Iberian sections
        print("Step 2: Searching for Iberian peninsula content...")
        iberian_sections = find_iberian_sections(doc_result)

        print(f"\nFound {len(iberian_sections)} sections related to Iberian peninsula\n")

        if not iberian_sections:
            print("No Iberian content found.")
            return

        # Process and save results
        output_file = "iberian_content_extracted.json"
        results = []

        for section in iberian_sections:
            # Analyze Arabic/French content
            lang_analysis = identify_arabic_and_translation(section['text'])

            result = {
                'section_id': section['section_id'],
                'matches': section['matches'],
                'full_text': section['text'],
                'arabic_text': lang_analysis['arabic_text'],
                'french_text': lang_analysis['french_text'],
                'has_arabic': lang_analysis['has_arabic']
            }

            results.append(result)

            # Display section
            print("="*70)
            print(f"SECTION {section['section_id']}")
            print(f"Keywords found: {', '.join(section['matches'][:5])}")
            print("="*70)

            if lang_analysis['has_arabic']:
                print("\n--- ARABIC TEXT ---")
                print(lang_analysis['arabic_text'][:500])
                if len(lang_analysis['arabic_text']) > 500:
                    print("... (truncated)")

            print("\n--- FRENCH TEXT ---")
            print(lang_analysis['french_text'][:500])
            if len(lang_analysis['french_text']) > 500:
                print("... (truncated)")

            print("\n")

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {output_file}")
        print(f"Total sections extracted: {len(results)}")
        print(f"Sections with Arabic text: {sum(1 for r in results if r['has_arabic'])}")
        print("="*70 + "\n")

        # Also save as markdown for readability
        md_file = "iberian_content_extracted.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Iberian Peninsula Content from al-Idrisi's Description\n\n")
            f.write(f"Extracted {len(results)} sections\n\n")
            f.write("---\n\n")

            for result in results:
                f.write(f"## Section {result['section_id']}\n\n")
                f.write(f"**Keywords:** {', '.join(result['matches'][:10])}\n\n")

                if result['has_arabic']:
                    f.write("### Arabic Text\n\n")
                    f.write(result['arabic_text'])
                    f.write("\n\n")

                f.write("### French Translation\n\n")
                f.write(result['french_text'])
                f.write("\n\n---\n\n")

        print(f"Markdown version saved to: {md_file}\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
