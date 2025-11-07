#!/usr/bin/env python3
"""
Stage 1: Text Detection using Florence-2
Detects all text regions on the Idrisi map with bounding box coordinates.
"""

import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import sys

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

def detect_text_regions(image_path, model, processor, device):
    """
    Detect all text regions in the image using Florence-2's OCR_WITH_REGION task.

    Returns:
        dict: Detection results with quad_boxes and labels
    """
    print(f"Processing image: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Prepare task prompt
    task_prompt = "<OCR_WITH_REGION>"

    # Process inputs
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)

    # Generate detections
    print("Running text detection...")
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

def save_results(results, image_size, output_path="stage1_detections.json"):
    """Save detection results to JSON file."""

    # Extract OCR_WITH_REGION results
    ocr_results = results.get("<OCR_WITH_REGION>", {})
    quad_boxes = ocr_results.get("quad_boxes", [])
    labels = ocr_results.get("labels", [])

    # Create structured output
    detections = []
    for i, (box, label) in enumerate(zip(quad_boxes, labels)):
        # Calculate center point from quad box
        # quad_box format: [x1, y1, x2, y2, x3, y3, x4, y4]
        center_x = sum(box[::2]) / 4  # average of x coordinates
        center_y = sum(box[1::2]) / 4  # average of y coordinates

        # Calculate bounding rectangle (min/max of quad points)
        x_coords = box[::2]
        y_coords = box[1::2]
        bbox = {
            "x_min": min(x_coords),
            "y_min": min(y_coords),
            "x_max": max(x_coords),
            "y_max": max(y_coords)
        }

        detection = {
            "id": i,
            "text": label,
            "quad_box": box,
            "center": {"x": center_x, "y": center_y},
            "bounding_box": bbox
        }
        detections.append(detection)

    output_data = {
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "total_detections": len(detections),
        "detections": detections
    }

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(detections)} text detections to {output_path}")

    # Print summary
    print("\nSample detections:")
    for detection in detections[:10]:
        print(f"  - '{detection['text']}' at ({detection['center']['x']:.1f}, {detection['center']['y']:.1f})")

    if len(detections) > 10:
        print(f"  ... and {len(detections) - 10} more")

    return output_data

def main():
    """Main execution function."""
    image_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"

    print("=" * 60)
    print("Stage 1: Florence-2 Text Detection")
    print("=" * 60)

    try:
        # Load model
        model, processor, device = load_florence_model()

        # Detect text regions
        results, image_size = detect_text_regions(image_path, model, processor, device)

        # Save results
        output_data = save_results(results, image_size)

        print("\n" + "=" * 60)
        print("Stage 1 Complete!")
        print("=" * 60)
        print(f"Total text regions detected: {output_data['total_detections']}")
        print("Next step: Run stage2_qwen_classification.py to classify city names")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
