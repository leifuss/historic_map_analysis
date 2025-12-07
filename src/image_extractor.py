"""
Image extraction utilities for PDF processing.
Handles extracting images from PDF pages with memory-efficient operations.
"""

import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import io
import logging


class ImageExtractor:
    """
    Extract and process images from PDF pages.

    Features:
    - Render PDF pages as images at specified DPI
    - Extract embedded images from PDF
    - Create thumbnails
    - Memory-efficient processing
    """

    def __init__(self, output_dir: str = "output/images", thumbnail_dir: str = "output/thumbnails"):
        """
        Initialize image extractor.

        Args:
            output_dir: Directory for full-size images
            thumbnail_dir: Directory for thumbnails
        """
        self.output_dir = Path(output_dir)
        self.thumbnail_dir = Path(thumbnail_dir)

        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def render_page_to_image(
        self,
        page,
        dpi: int = 300,
        format: str = "png"
    ) -> Image.Image:
        """
        Render a PDF page to a PIL Image.

        Args:
            page: PyMuPDF page object
            dpi: Resolution in DPI
            format: Output format (png, jpeg, etc.)

        Returns:
            PIL Image object
        """
        # Calculate zoom factor (72 DPI is default)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        self.logger.debug(f"Rendered page to {pix.width}x{pix.height} image at {dpi} DPI")

        return img

    def save_page_as_image(
        self,
        page,
        output_path: str,
        dpi: int = 300,
        format: str = "png",
        quality: int = 95
    ) -> Dict[str, Any]:
        """
        Save a PDF page as an image file.

        Args:
            page: PyMuPDF page object
            output_path: Path to save the image
            dpi: Resolution in DPI
            format: Output format (png, jpeg, etc.)
            quality: Quality for JPEG compression (1-100)

        Returns:
            Dictionary with image information
        """
        output_path = Path(output_path)

        # Render page
        img = self.render_page_to_image(page, dpi=dpi, format=format)

        # Save with appropriate settings
        save_kwargs = {}
        if format.lower() in ['jpeg', 'jpg']:
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        elif format.lower() == 'png':
            save_kwargs['optimize'] = True

        img.save(output_path, format=format, **save_kwargs)

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB

        self.logger.info(f"Saved page image: {output_path.name} ({file_size:.2f} MB)")

        return {
            'path': str(output_path),
            'width': img.width,
            'height': img.height,
            'dpi': dpi,
            'format': format,
            'file_size_mb': round(file_size, 2)
        }

    def create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (200, 200),
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Create a thumbnail from an image.

        Args:
            image: PIL Image object
            size: Thumbnail size (width, height)
            output_path: Optional path to save thumbnail

        Returns:
            PIL Image thumbnail
        """
        # Create thumbnail (maintains aspect ratio)
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)

        if output_path:
            output_path = Path(output_path)
            thumbnail.save(output_path, optimize=True)
            self.logger.debug(f"Saved thumbnail: {output_path.name}")

        return thumbnail

    def extract_embedded_images(self, page, output_dir: Optional[str] = None) -> list:
        """
        Extract embedded images from a PDF page.

        Args:
            page: PyMuPDF page object
            output_dir: Optional directory to save extracted images

        Returns:
            List of extracted image information
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        extracted_images = []
        image_list = page.get_images()

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            # Extract image
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Convert to PIL Image
            img = Image.open(io.BytesIO(image_bytes))

            img_data = {
                'index': img_index,
                'width': img.width,
                'height': img.height,
                'format': image_ext,
                'size_kb': len(image_bytes) / 1024
            }

            # Optionally save
            if output_dir:
                filename = f"embedded_img_{img_index}.{image_ext}"
                img_path = output_dir / filename
                img.save(img_path)
                img_data['path'] = str(img_path)

            extracted_images.append(img_data)

        self.logger.info(f"Extracted {len(extracted_images)} embedded images from page")

        return extracted_images

    def process_page_with_thumbnail(
        self,
        page,
        page_num: int,
        prefix: str = "page",
        dpi: int = 300,
        format: str = "png",
        quality: int = 95,
        thumbnail_size: Tuple[int, int] = (200, 200)
    ) -> Dict[str, Any]:
        """
        Process a page: save as image and create thumbnail.

        Args:
            page: PyMuPDF page object
            page_num: Page number for naming
            prefix: Filename prefix
            dpi: Resolution in DPI
            format: Output format
            quality: JPEG quality
            thumbnail_size: Thumbnail dimensions

        Returns:
            Dictionary with paths and metadata
        """
        # Generate filenames
        image_filename = f"{prefix}_{page_num:04d}.{format}"
        thumb_filename = f"{prefix}_{page_num:04d}_thumb.{format}"

        image_path = self.output_dir / image_filename
        thumb_path = self.thumbnail_dir / thumb_filename

        # Save full image
        img_info = self.save_page_as_image(
            page,
            output_path=image_path,
            dpi=dpi,
            format=format,
            quality=quality
        )

        # Render and create thumbnail
        img = self.render_page_to_image(page, dpi=150)  # Lower DPI for thumbnail source
        self.create_thumbnail(img, size=thumbnail_size, output_path=thumb_path)

        return {
            'page_number': page_num,
            'image_path': str(image_path),
            'thumbnail_path': str(thumb_path),
            'dimensions': f"{img_info['width']}x{img_info['height']}",
            'file_size_mb': img_info['file_size_mb'],
            'dpi': dpi
        }
