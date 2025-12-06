# Extracting Iberian Peninsula Content from al-Idrisi's Document

## Overview

This guide explains how to extract Arabic text and French translations related to the Iberian peninsula from the historic document "Description de l'Afrique et de l'Espagne" by al-Idrisi.

## The Document

- **Author**: Abu Abdullah Muhammad ibn Muhammad ibn Idris al-Idrisi (الإدريسي)
- **Title**: Description de l'Afrique et de l'Espagne (وصف أفريقيا وإسبانيا)
- **Date**: Original 12th century, this edition published 1866
- **Editors**: R. Dozy and M. J. de Goeje
- **Language**: Arabic text with French translation
- **Format**: Scanned PDF (676 pages, 21 MB)

## Extraction Method

Since the PDF contains scanned images rather than extractable text, we use OCR (Optical Character Recognition) with Tesseract to extract the content.

### Languages Supported

The extraction system uses Tesseract with three languages:
- **Arabic** (`ara`) - For the original al-Idrisi text
- **French** (`fra`) - For the scholarly translation
- **English** (`eng`) - For auxiliary text

## Usage

### Basic Extraction

Extract Iberian content from the first 50 pages:

```bash
python extract_iberian_with_ocr.py --max-pages 50
```

### Full Document Extraction

Process the entire document (this will take several hours):

```bash
python extract_iberian_with_ocr.py --max-pages 0
```

### High-Quality Extraction

Use higher DPI for better OCR accuracy (slower):

```bash
python extract_iberian_with_ocr.py --max-pages 100 --dpi 300
```

### Command Options

- `--max-pages N`: Process first N pages (0 = all pages, default: 50)
- `--dpi N`: OCR resolution (default: 200, range: 150-600)
  - 150 DPI: Fast, lower quality
  - 200 DPI: Good balance
  - 300 DPI: High quality, slower

## Search Keywords

The system searches for these Iberian peninsula keywords:

**General Terms:**
- espagne, spain, iberia, ibérie
- al-andalus, andalus, andalusia, andalousie
- portugal, lusitanie, lusitania

**Major Cities:**
- Cordoue (Córdoba/Cordoba)
- Séville (Seville/Sevilla)
- Grenade (Granada)
- Tolède (Toledo)
- Valence (Valencia)
- Barcelone (Barcelona)
- Lisbonne (Lisbon)
- Saragosse (Zaragoza)
- Madrid

**Regions:**
- Catalogne (Catalonia)
- Aragon, Aragón
- Castille (Castile)
- Galice (Galicia)
- Léon (Leon)

## Output Files

The extraction creates several output files in the `output/` directory:

### 1. JSON Format (`iberian_content.json`)

Structured data with all extracted content:

```json
[
  {
    "page_num": 14,
    "matches": ["espagne", "cordoue", "grenade"],
    "full_text": "...",
    "arabic_text": "...",
    "french_text": "...",
    "has_arabic": true
  }
]
```

### 2. Markdown Format (`iberian_content.md`)

Human-readable format with Arabic and French text separated:

```markdown
## Page 14

**Keywords found:** espagne, cordoue, grenade

### Arabic Text
[Arabic text here]

### French Translation
[French translation here]
```

### 3. Separate Text Files

- `iberian_arabic.txt`: Only Arabic text from all pages
- `iberian_french.txt`: Only French text from all pages

## Example Output

From the initial extraction (pages 0-20), we found:

**Page 14** - About al-Idrisi's biography:

*Arabic:*
```
آنا
بن محمد بن عبد الله بن ادريس
```

*French:*
```
Son bisaïeul, Edris II al-'Aali bi-amri-'l-lāh, de la famille des
Hammoudites, qui se distinguait par une grande bonté de cœur aussi
bien que par une extrême faiblesse de caractère, avait régné sur la prin-
cipauté de Malaga... Edris II mourut en 1056 ; deux années après,
Malaga fut annexée au royaume de Grenade...

Il ajoute qu'Edrisi fit ses études à Cordoue, car, comme l'a observé
Quatremère, "si l'on considère le soin que notre géographe a pris d'en
donner une description complète..."
```

This page discusses:
- al-Idrisi's ancestor who ruled Málaga
- The annexation of Málaga to the Kingdom of Granada (1056-1058)
- al-Idrisi's studies in Córdoba
- His descriptions of Spanish cities

## Processing Performance

- **OCR Speed**: ~2-5 seconds per page (at 150 DPI)
- **100 pages**: ~5-10 minutes
- **Full document (676 pages)**: ~45-90 minutes

## Technical Details

### OCR Preprocessing

The system:
1. Renders each PDF page at specified DPI
2. Converts to PIL image
3. Applies Tesseract OCR with multi-language support
4. Separates Arabic and French text using Unicode ranges

### Arabic Text Detection

Arabic characters are identified by Unicode ranges:
- Main Arabic: `\u0600-\u06FF`
- Arabic Supplement: `\u0750-\u077F`
- Arabic Presentation Forms: `\uFB50-\uFDFF`, `\uFE70-\uFEFF`

## Tips for Better Results

1. **For faster preview**: Use `--max-pages 50 --dpi 150`
2. **For archival extraction**: Use `--dpi 300` with all pages
3. **Monitor progress**: The script shows progress every 10 pages
4. **Check output files**: Review the markdown file for readability

## Historical Context

Al-Idrisi (1100-1165) was a Muslim geographer who worked at the court of King Roger II of Sicily. His "Nuzhat al-Mushtaq" (commonly known as the "Book of Roger") includes detailed descriptions of the Iberian peninsula during the Islamic period (al-Andalus).

This 1866 edition by Dozy and de Goeje presents the Arabic original text alongside French translation, making it a valuable source for:
- Medieval Islamic geography
- History of al-Andalus
- Arabic-French translation studies
- Historical toponymy of Iberia

## Next Steps

After extraction:
1. Review the markdown file for accuracy
2. Use the JSON file for programmatic analysis
3. Compare Arabic text with French translation
4. Extract specific city or region descriptions
5. Create thematic collections (cities, routes, regions)

## Troubleshooting

**No content found:**
- Try different page ranges
- Increase DPI for better OCR
- Check that keywords match the document language

**OCR errors:**
- Historical fonts may cause recognition issues
- Arabic text in scanned images can be challenging
- Consider manual verification of critical passages

**Long processing time:**
- Reduce page count or DPI
- Process in batches (e.g., pages 0-100, then 100-200)
