"""
OCR processing for extracting text from scanned PDF pages.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OCRProcessor:
    """
    Extract text from scanned images using OCR.

    Supports multiple languages including Arabic, French, and English.
    """

    def __init__(self, languages: List[str] = None):
        """
        Initialize OCR processor.

        Args:
            languages: List of language codes (e.g., ['ara', 'fra', 'eng'])
                      Default: ['ara', 'fra', 'eng'] for Arabic, French, English
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract not available. Install with: pip install pytesseract"
            )

        self.languages = languages or ['ara', 'fra', 'eng']
        self.logger = logging.getLogger(__name__)

        # Test if tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                f"Tesseract OCR not found. Please install tesseract-ocr: {e}"
            )

    def extract_text_from_image(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Extract text from an image using OCR.

        Args:
            image: PIL Image object
            languages: Optional language override

        Returns:
            Extracted text
        """
        langs = languages or self.languages
        lang_string = '+'.join(langs)

        try:
            text = pytesseract.image_to_string(image, lang=lang_string)
            return text
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return ""

    def extract_text_from_page(
        self,
        page,
        dpi: int = 300,
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Extract text from a PDF page using OCR.

        Args:
            page: PyMuPDF page object
            dpi: Resolution for rendering (higher = better quality, slower)
            languages: Optional language override

        Returns:
            Extracted text
        """
        import fitz

        # Render page to image
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Extract text
        text = self.extract_text_from_image(img, languages=languages)

        self.logger.debug(f"Extracted {len(text)} characters via OCR")

        return text

    def extract_with_layout(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None
    ) -> Dict:
        """
        Extract text with layout information (bounding boxes, confidence).

        Args:
            image: PIL Image object
            languages: Optional language override

        Returns:
            Dictionary with detailed OCR data
        """
        langs = languages or self.languages
        lang_string = '+'.join(langs)

        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang_string,
                output_type=pytesseract.Output.DICT
            )
            return data
        except Exception as e:
            self.logger.error(f"OCR with layout failed: {e}")
            return {}

    def detect_language(self, image: Image.Image) -> str:
        """
        Detect the primary language in an image.

        Args:
            image: PIL Image object

        Returns:
            Detected language code
        """
        try:
            osd = pytesseract.image_to_osd(image)
            # Parse language from OSD output
            for line in osd.split('\n'):
                if 'Script:' in line:
                    return line.split(':')[1].strip()
            return 'unknown'
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return 'unknown'

    def extract_regions(
        self,
        image: Image.Image,
        regions: List[Tuple[int, int, int, int]],
        languages: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract text from specific regions of an image.

        Args:
            image: PIL Image object
            regions: List of (x, y, width, height) tuples
            languages: Optional language override

        Returns:
            List of extracted text strings for each region
        """
        results = []

        for x, y, w, h in regions:
            # Crop region
            region_img = image.crop((x, y, x + w, y + h))

            # Extract text
            text = self.extract_text_from_image(region_img, languages=languages)
            results.append(text)

        return results

    def preprocess_for_ocr(
        self,
        image: Image.Image,
        enhance: bool = True,
        threshold: bool = False
    ) -> Image.Image:
        """
        Preprocess image for better OCR results.

        Args:
            image: PIL Image object
            enhance: Apply contrast enhancement
            threshold: Apply binary thresholding

        Returns:
            Preprocessed image
        """
        from PIL import ImageEnhance, ImageFilter

        processed = image.copy()

        # Convert to grayscale
        if processed.mode != 'L':
            processed = processed.convert('L')

        # Enhance contrast
        if enhance:
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(2.0)

        # Apply threshold
        if threshold:
            processed = processed.point(lambda x: 0 if x < 128 else 255, '1')

        # Denoise
        processed = processed.filter(ImageFilter.MedianFilter(size=3))

        return processed
