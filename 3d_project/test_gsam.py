import os
import cv2
import torch
import numpy as np
from PIL import Image
import sys

from segment_anything import sam_model_registry, SamPredictor

# GroundingDINO paths
sys.path.append(os.path.expanduser("~/Documents/3d_project/Grounded-Segment-Anything"))
sys.path.insert(0, os.path.expanduser("~/Documents/3d_project/Grounded-Segment-Anything/GroundingDINO"))

import groundingdino.util.inference as gd
import groundingdino.datasets.transforms as GDT

# Load models
print("Loading GroundingDINO...")
dino_model = gd.load_model(
    os.path.expanduser("~/Documents/3d_project/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    os.path.expanduser("~/Documents/3d_project/Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
)

print("Loading SAM...")
sam = sam_model_registry["vit_h"](
    checkpoint=os.path.expanduser("~/Documents/3d_project/Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
sam.to(device)
predictor = SamPredictor(sam)

# Test on first image 
test_dir = os.path.expanduser("~/Documents/3d_project/test_images")
test_image = os.path.join(test_dir, os.listdir(test_dir)[0])
print(f"Testing on: {test_image}")

image_bgr = cv2.imread(test_image)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

TEXT_PROMPT = "candle"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Transform image for GroundingDINO
transform = GDT.Compose([
    GDT.RandomResize([800], max_size=1333),
    GDT.ToTensor(),
    GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image_pil = Image.fromarray(image_rgb)
image_transformed, _ = transform(image_pil, None)

boxes, logits, phrases = gd.predict(
    model=dino_model,
    image=image_transformed,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device="cpu"
)

print(f"Detected {len(boxes)} instance(s) of '{TEXT_PROMPT}'")

if len(boxes) > 0:
    predictor.set_image(image_rgb)
    print(f"Boxes: {boxes}")
    print(f"Box shape: {boxes.shape}")
    h, w = image_rgb.shape[:2]
    print(f"Image size: {w}x{h}")
    box = boxes[0] * torch.tensor([w, h, w, h])
    print(f"Scaled box: {box}")

    h, w = image_rgb.shape[:2]

    # Convert from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2]
    box = boxes[0] * torch.tensor([w, h, w, h])
    cx, cy, bw, bh = box
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    box_xyxy = np.array([x1.item(), y1.item(), x2.item(), y2.item()])
    print(f"Converted box: {box_xyxy}")

    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(box=box_xyxy, multimask_output=False)
    print(f"Mask shape: {masks.shape}")
    print(f"Mask dtype: {masks.dtype}")
    print(f"Mask min: {masks.min()}, max: {masks.max()}")
    print(f"Mask unique values: {np.unique(masks)}")

    mask_overlay = image_bgr.copy()
    mask_overlay[masks[0]] = [0, 255, 0]
    result_path = os.path.expanduser("~/Documents/3d_project/test_result.jpg")
    cv2.imwrite(result_path, cv2.addWeighted(image_bgr, 0.5, mask_overlay, 0.5, 0))
    print(f"Result saved to {result_path} — open it to see the mask overlay")

    # save the raw mask for clarity
    raw_mask_path = os.path.expanduser("~/Documents/3d_project/test_mask.jpg")
    cv2.imwrite(raw_mask_path, (masks[0] * 255).astype(np.uint8))
    print(f"Raw mask saved to {raw_mask_path}")
else:
    print("Nothing detected — try changing TEXT_PROMPT to something in your photo")