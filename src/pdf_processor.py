"""
Core PDF Processor for handling large PDF files efficiently.
Uses lazy loading and memory management to process PDFs page by page.
"""

import fitz  # PyMuPDF
import gc
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import logging
from collections import OrderedDict


class PageCache:
    """LRU cache for PDF pages to manage memory usage."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: int):
        """Get a page from cache, moving it to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: int, value):
        """Add a page to cache, evicting oldest if necessary."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                # Remove oldest item
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                gc.collect()

    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        gc.collect()


class LargePDFProcessor:
    """
    Efficiently process large PDF files with memory management.

    Features:
    - Lazy page loading
    - Memory-efficient page iteration
    - Metadata extraction
    - Page caching with LRU eviction
    - Progress tracking
    """

    def __init__(self, pdf_path: str, cache_size: int = 3):
        """
        Initialize the PDF processor.

        Args:
            pdf_path: Path to the PDF file
            cache_size: Number of pages to keep in memory cache
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.cache_size = cache_size
        self.page_cache = PageCache(max_size=cache_size)
        self._doc = None
        self._metadata = None

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self):
        """Open the PDF document."""
        if self._doc is None:
            self._doc = fitz.open(self.pdf_path)
            self.logger.info(f"Opened PDF: {self.pdf_path.name} ({self.page_count} pages)")

    def close(self):
        """Close the PDF document and clear cache."""
        if self._doc is not None:
            self.page_cache.clear()
            self._doc.close()
            self._doc = None
            gc.collect()
            self.logger.info(f"Closed PDF: {self.pdf_path.name}")

    @property
    def page_count(self) -> int:
        """Get total number of pages in the PDF."""
        if self._doc is None:
            self.open()
        return len(self._doc)

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Extract PDF metadata.

        Returns:
            Dictionary containing PDF metadata
        """
        if self._metadata is None:
            if self._doc is None:
                self.open()

            self._metadata = {
                'file_name': self.pdf_path.name,
                'file_size_mb': self.pdf_path.stat().st_size / (1024 * 1024),
                'page_count': self.page_count,
                'pdf_version': self._doc.metadata.get('format', 'Unknown'),
                'title': self._doc.metadata.get('title', ''),
                'author': self._doc.metadata.get('author', ''),
                'subject': self._doc.metadata.get('subject', ''),
                'creator': self._doc.metadata.get('creator', ''),
                'producer': self._doc.metadata.get('producer', ''),
                'creation_date': self._doc.metadata.get('creationDate', ''),
                'modification_date': self._doc.metadata.get('modDate', ''),
            }

        return self._metadata

    def get_page(self, page_num: int):
        """
        Get a specific page with caching.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            PyMuPDF page object
        """
        if self._doc is None:
            self.open()

        if page_num < 0 or page_num >= self.page_count:
            raise ValueError(f"Page number {page_num} out of range (0-{self.page_count-1})")

        # Check cache first
        cached_page = self.page_cache.get(page_num)
        if cached_page is not None:
            return cached_page

        # Load page and add to cache
        page = self._doc[page_num]
        self.page_cache.put(page_num, page)

        return page

    def iter_pages(self, start: int = 0, end: Optional[int] = None) -> Iterator:
        """
        Iterate through pages lazily.

        Args:
            start: Starting page number (0-indexed)
            end: Ending page number (exclusive), None for all pages

        Yields:
            Tuple of (page_number, page_object)
        """
        if self._doc is None:
            self.open()

        if end is None:
            end = self.page_count

        for page_num in range(start, min(end, self.page_count)):
            page = self.get_page(page_num)
            yield page_num, page

    def get_page_info(self, page_num: int) -> Dict[str, Any]:
        """
        Get information about a specific page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with page information
        """
        page = self.get_page(page_num)
        rect = page.rect

        return {
            'page_number': page_num,
            'width': rect.width,
            'height': rect.height,
            'width_inches': rect.width / 72,  # Convert from points to inches
            'height_inches': rect.height / 72,
            'rotation': page.rotation,
            'image_count': len(page.get_images()),
            'text_length': len(page.get_text()),
        }

    def get_all_pages_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all pages.

        Returns:
            List of page information dictionaries
        """
        pages_info = []
        for page_num, _ in self.iter_pages():
            pages_info.append(self.get_page_info(page_num))
        return pages_info

    def estimate_memory_usage(self, dpi: int = 300) -> Dict[str, float]:
        """
        Estimate memory usage for processing at given DPI.

        Args:
            dpi: Target DPI for rendering

        Returns:
            Dictionary with memory estimates in MB
        """
        if not self.page_count:
            return {'total_mb': 0, 'per_page_mb': 0}

        # Get first page to estimate
        page = self.get_page(0)
        rect = page.rect

        # Calculate pixel dimensions at target DPI
        width_px = int(rect.width / 72 * dpi)
        height_px = int(rect.height / 72 * dpi)

        # Estimate bytes (RGB = 3 bytes per pixel)
        bytes_per_page = width_px * height_px * 3
        mb_per_page = bytes_per_page / (1024 * 1024)
        total_mb = mb_per_page * self.page_count

        return {
            'per_page_mb': round(mb_per_page, 2),
            'total_mb': round(total_mb, 2),
            'cached_mb': round(mb_per_page * self.cache_size, 2),
            'dpi': dpi,
            'dimensions': f"{width_px}x{height_px}"
        }
