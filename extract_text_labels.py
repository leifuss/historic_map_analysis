#!/usr/bin/env python3
"""
Text Label Extraction for Idrisi Map
Uses Florence-2 to extract all text labels from the map with their coordinates.
"""

import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import sys
from pathlib import Path


def load_florence_model(model_id="microsoft/Florence-2-large"):
    """Load Florence-2 model and processor."""
    print(f"Loading Florence-2 model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print(f"Model loaded on device: {device}")
    return model, processor, device


def extract_text_with_ocr(image_path, model, processor, device):
    """
    Extract all text labels from the map using Florence-2 OCR.

    Returns:
        dict: OCR results with text and bounding boxes
    """
    print(f"Processing image: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Use OCR_WITH_REGION task to get text with coordinates
    task_prompt = "<OCR_WITH_REGION>"

    # Process inputs
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)

    # Generate OCR results
    print("Running OCR to extract text labels...")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False
    )

    # Decode results
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Parse results
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer, image.size


def process_ocr_results(ocr_results, image_size):
    """
    Process OCR results into structured format.

    Args:
        ocr_results: Raw OCR output from Florence-2
        image_size: Tuple of (width, height)

    Returns:
        list: Structured list of text labels with coordinates
    """
    # Extract OCR_WITH_REGION results
    ocr_data = ocr_results.get("<OCR_WITH_REGION>", {})
    quad_boxes = ocr_data.get("quad_boxes", [])
    labels = ocr_data.get("labels", [])

    print(f"\nProcessing {len(labels)} detected text labels...")

    text_labels = []
    for i, (box, text) in enumerate(zip(quad_boxes, labels)):
        # Calculate center point from quad box
        # quad_box format: [x1, y1, x2, y2, x3, y3, x4, y4]
        center_x = sum(box[::2]) / 4  # average of x coordinates
        center_y = sum(box[1::2]) / 4  # average of y coordinates

        # Calculate bounding rectangle
        x_coords = box[::2]
        y_coords = box[1::2]

        label_data = {
            "id": i,
            "text": text.strip(),
            "center": {"x": center_x, "y": center_y},
            "quad_box": box,
            "bounding_box": {
                "x_min": min(x_coords),
                "y_min": min(y_coords),
                "x_max": max(x_coords),
                "y_max": max(y_coords)
            },
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords)
        }
        text_labels.append(label_data)

    return text_labels


def save_text_labels(labels, image_size, output_path="detected_text_labels.json"):
    """Save extracted text labels to JSON file."""
    output_data = {
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "total_labels": len(labels),
        "labels": labels,
        "description": "Text labels extracted from Idrisi map using Florence-2 OCR"
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(labels)} text labels to {output_path}")

    # Print statistics
    if len(labels) > 0:
        avg_width = sum(l['width'] for l in labels) / len(labels)
        avg_height = sum(l['height'] for l in labels) / len(labels)
        print(f"\nText label statistics:")
        print(f"  Average width: {avg_width:.1f}px")
        print(f"  Average height: {avg_height:.1f}px")

    # Print sample labels
    print("\nSample extracted labels:")
    for label in labels[:15]:
        print(f"  '{label['text']}' at ({label['center']['x']:.1f}, {label['center']['y']:.1f})")

    if len(labels) > 15:
        print(f"  ... and {len(labels) - 15} more")

    return output_data


def main():
    """Main execution function."""
    image_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"

    print("=" * 70)
    print("Text Label Extraction - Idrisi Map")
    print("=" * 70)
    print()

    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return 1

    try:
        # Load Florence-2 model
        model, processor, device = load_florence_model()

        # Extract text labels
        ocr_results, image_size = extract_text_with_ocr(image_path, model, processor, device)

        # Process results
        text_labels = process_ocr_results(ocr_results, image_size)

        # Save results
        if len(text_labels) > 0:
            save_text_labels(text_labels, image_size)

            print("\n" + "=" * 70)
            print("Text Extraction Complete!")
            print("=" * 70)
            print(f"Total text labels extracted: {len(text_labels)}")
            print("\nNext step: Run match_symbols_to_labels.py to associate symbols with labels")
        else:
            print("\n⚠ Warning: No text labels extracted!")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
