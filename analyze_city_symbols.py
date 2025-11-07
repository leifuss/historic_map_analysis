#!/usr/bin/env python3
"""
Interactive analysis to understand what city symbols look like on the Idrisi map.
Uses VLM to identify and describe the visual characteristics of city markers.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import sys

def analyze_map_symbols():
    """Use Florence-2 to understand the map's visual elements."""

    print("Analyzing Idrisi map to identify city symbols...")

    # Load Florence-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "microsoft/Florence-2-large"
    print(f"Loading {model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Load the map
    image_path = "al-Idrisi - Tabula Rogeriana (Konrad Miller 1929).jpeg"
    image = Image.open(image_path).convert("RGB")

    print(f"Image loaded: {image.size}")
    print("\n" + "="*60)

    # Strategy 1: Get detailed caption to understand map elements
    print("\n1. Getting detailed caption of the map...")
    task = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(caption, task=task, image_size=image.size)
    print(f"Caption: {parsed}")

    # Strategy 2: Dense region detection
    print("\n2. Detecting dense regions on the map...")
    task = "<DENSE_REGION_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        num_beams=3
    )
    regions = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(regions, task=task, image_size=image.size)

    print(f"\nFound {len(parsed.get('<DENSE_REGION_CAPTION>', {}).get('labels', []))} regions")
    labels = parsed.get('<DENSE_REGION_CAPTION>', {}).get('labels', [])
    bboxes = parsed.get('<DENSE_REGION_CAPTION>', {}).get('bboxes', [])

    # Show first 20 regions
    print("\nFirst 20 detected regions:")
    for i, (label, bbox) in enumerate(zip(labels[:20], bboxes[:20])):
        print(f"  {i+1}. '{label}' at {bbox}")

    print("\n" + "="*60)
    print("\nAnalysis complete!")
    print("Look for patterns like 'circle', 'dot', 'symbol' in the region labels")
    print("These might indicate city markers on the map")

    return parsed

if __name__ == "__main__":
    try:
        analyze_map_symbols()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
