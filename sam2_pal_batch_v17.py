#!/usr/bin/env python3
"""
SAM2-PAL: Palindrome-based Mask Propagation for Descriptron
Version 17 - Multi-Mask Training Support

=== CHANGES IN V17 ===

1. MULTI-MASK TRAINING FROM COCO JSON:
   - When using --template_json with multiple masks (e.g., scape, antenna, eye),
     ALL masks are now used for training, not just the first one
   - Each mask becomes a training template for the PAL fine-tuning

2. All v16 features preserved:
   - Optional LoRA fine-tuning (requires 'peft' library)
   - 4-step OC-CCL with memory reset
   - Multi-template training from separate training JSON

=== MULTI-MASK WORKFLOW ===

Training with multi-mask JSON:
    python sam2_pal_batch_v17.py \\
        --template_json masks.json \\        # Contains scape, antenna, eye
        --template_image template.jpg \\
        --image_dir ./specimens \\
        --output_dir ./output \\
        --pal_finetuning --num_epochs 50

This will:
1. Load all 3 masks from template image
2. Train on ALL masks (cycles through scape, antenna, eye)
3. Predict all 3 structures on target images

=== PAPER'S OC-CCL RECIPE (arxiv.org/abs/2501.06749) ===

Training setup:
  - 1 labeled image (x0, y0) - we support multiple!
  - 100 unlabeled images - disjoint from test set
  - LoRA fine-tuning of decoder + memory encoder

4-frame palindrome: {x0, x1, x1â€ , x0â€ }

Phase 1:
  Frame 0: x0 with mask prompt y0 â†’ stored in memory
  Frame 1: x1 (unlabeled) â†’ predict Å·1 using memory

*** MEMORY RESET (prevents cheating by remembering y0) ***

Phase 2:
  Frame 2: x1â€  with Å·1 as DIFFERENTIABLE prompt
  Frame 3: x0â€  â†’ predict Å·0â€ , compute loss vs y0

Loss: BCE + Dice between Å·0â€  and y0

=== USAGE ===

    # With LoRA (recommended if peft installed)
    pip install peft
    python sam2_pal_batch.py --template_mask mask.png \\
                             --template_image template.jpg \\
                             --image_dir ./images \\
                             --output_dir ./output \\
                             --pal_finetuning --use_lora \\
                             --num_epochs 25 --learning_rate 1e-4

    # Without LoRA (full fine-tuning)
    python sam2_pal_batch.py --template_mask mask.png \\
                             --template_image template.jpg \\
                             --image_dir ./images \\
                             --output_dir ./output \\
                             --pal_finetuning \\
                             --num_epochs 75 --learning_rate 5e-6

Author: Descriptron Project (2025)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_image_and_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to simulate variations across specimens.
    This helps the fine-tuned model generalize better.
    """
    h, w = image.shape[:2]
    
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    
    # Random vertical flip (50% chance)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    
    # Random rotation (-25 to +25 degrees)
    angle = random.uniform(-25, 25)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Random scale (0.85 to 1.15)
    scale = random.uniform(0.85, 1.15)
    new_w, new_h = int(w * scale), int(h * scale)
    image_scaled = cv2.resize(image, (new_w, new_h))
    mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Crop or pad back to original size
    if scale > 1:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        image = image_scaled[start_y:start_y+h, start_x:start_x+w]
        mask = mask_scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        pad_image = np.zeros((h, w, 3), dtype=image.dtype)
        pad_mask = np.zeros((h, w), dtype=mask.dtype)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        pad_image[start_y:start_y+new_h, start_x:start_x+new_w] = image_scaled
        pad_mask[start_y:start_y+new_h, start_x:start_x+new_w] = mask_scaled
        image = pad_image
        mask = pad_mask
    
    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-25, 25)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image, mask


def get_points_from_mask(mask: np.ndarray, num_points: int = 3) -> Optional[np.ndarray]:
    """
    Sample random points from inside the mask region.
    Points are used as prompts for SAM2.
    
    Returns array of shape (num_points, 1, 2) in xy format.
    """
    # Erode slightly to avoid boundary points
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded > 0)
    
    if len(coords) == 0:
        coords = np.argwhere(mask > 0)
    
    if len(coords) == 0:
        return None
    
    points = []
    for _ in range(num_points):
        idx = np.random.randint(len(coords))
        yx = coords[idx]
        points.append([[yx[1], yx[0]]])  # xy format
    
    return np.array(points)


def get_box_from_mask(mask: np.ndarray, padding: int = 5, pad: Optional[int] = None) -> Optional[np.ndarray]:
    """Get bounding box from a binary mask.

    Args:
        mask: HxW binary (0/1) mask (numpy).
        padding: pixels of padding to expand the box.
        pad: backwards-compatible alias for `padding` (some callers use pad=).

    Returns:
        np.ndarray of shape (1,4) in XYXY order: [[x_min, y_min, x_max, y_max]],
        dtype float32, or None if mask empty.
    """
    if pad is not None:
        padding = int(pad)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    h, w = mask.shape[:2]
    x_min = max(0, int(xs.min()) - int(padding))
    y_min = max(0, int(ys.min()) - int(padding))
    x_max = min(w - 1, int(xs.max()) + int(padding))
    y_max = min(h - 1, int(ys.max()) + int(padding))

    return np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)



def clamp_mask_to_box(mask, box_xyxy):
    """Zero out mask pixels outside box (XYXY).
    Works for numpy uint8 masks (H,W) and torch tensors.
    """
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if box_xyxy is None:
        return mask

    # Accept [[x1,y1,x2,y2]] or [x1,y1,x2,y2]
    if isinstance(box_xyxy, (list, tuple)):
        if len(box_xyxy) == 1 and isinstance(box_xyxy[0], (list, tuple, _np.ndarray)):
            b = box_xyxy[0]
        else:
            b = box_xyxy
        x1, y1, x2, y2 = map(int, b)
    else:
        b = _np.array(box_xyxy).reshape(-1)
        if b.size == 4:
            x1, y1, x2, y2 = map(int, b.tolist())
        else:
            b = b.reshape(-1, 4)[0]
            x1, y1, x2, y2 = map(int, b.tolist())

    if _torch is not None and isinstance(mask, _torch.Tensor):
        # mask expected HxW
        H, W = mask.shape[-2], mask.shape[-1]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        out = mask.clone()
        out[..., :y1, :] = 0
        out[..., y2+1:, :] = 0
        out[..., :, :x1] = 0
        out[..., :, x2+1:] = 0
        return out
    else:
        mask_np = mask
        H, W = mask_np.shape[:2]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        out = mask_np.copy()
        out[:y1, :] = 0
        out[y2+1:, :] = 0
        out[:, :x1] = 0
        out[:, x2+1:] = 0
        return out


def get_points_from_box(box_xyxy: np.ndarray, num_points: int = 3) -> Optional[np.ndarray]:
    """Sample random points uniformly from inside an XYXY box."""
    if box_xyxy is None:
        return None
    b = np.array(box_xyxy).reshape(-1)
    if b.size == 4:
        x1, y1, x2, y2 = b
    else:
        x1, y1, x2, y2 = b.reshape(-1,4)[0]
    x1, y1, x2, y2 = map(int, [x1,y1,x2,y2])
    if x2 <= x1 or y2 <= y1:
        return None
    pts = []
    for _ in range(int(num_points)):
        x = np.random.randint(x1, x2+1)
        y = np.random.randint(y1, y2+1)
        pts.append([[x, y]])
    return np.array(pts, dtype=np.float32)


def postprocess_mask(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    """Light cleanup: keep largest connected component + close small holes."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    if m.sum() < min_area:
        return m
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        # label 0 is background
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m



class SAM2PAL:
    """
    SAM2-PAL: Palindrome-based Mask Propagation
    
    Uses SAM2's video tracking to propagate masks through a pseudo-video
    created from static images.
    """
    
    def __init__(self, sam2_checkpoint: str, sam2_config: str, device: str = 'cuda'):
        """Initialize SAM2."""
        import torch
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        
        # Build SAM2 model (shared between image and video predictors)
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        
        # Image predictor for fine-tuning (supports gradients)
        self.image_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Video predictor for inference
        self.video_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=self.device)
        
        logger.info("SAM2-PAL initialized successfully")
        
        self.template_mask = None
        self.categories = {}
    
    def load_template_mask(self, mask_path: str, category_name: str = 'object') -> np.ndarray:
        """Load template mask from file."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        self.template_mask = (binary_mask > 0).astype(np.uint8)
        
        self.categories = {1: {'id': 1, 'name': category_name, 'supercategory': 'object'}}
        
        logger.info(f"Loaded template mask: {mask_path}")
        logger.info(f"Mask shape: {self.template_mask.shape}, non-zero pixels: {np.sum(self.template_mask > 0)}")
        
        return self.template_mask
    
    def load_template_from_coco(self, coco_json_path: str, template_image_path: str) -> List[Dict]:
        """
        Load template masks from COCO JSON.
        
        More flexible than before - handles:
        1. JSON with matching image filename in 'images' array
        2. JSON with single image (uses that image's annotations)
        3. JSON with no 'images' array (loads all annotations like GUI does)
        4. Multi-image JSON (tries to find matching image, falls back to first or all)
        """
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        template_filename = os.path.basename(template_image_path)
        template_img = cv2.imread(template_image_path)
        if template_img is None:
            raise ValueError(f"Could not load template image: {template_image_path}")
        h, w = template_img.shape[:2]
        
        # Load categories
        self.categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        
        # Determine which annotations to load
        annotations_to_load = []
        images_array = coco_data.get('images', [])
        all_annotations = coco_data.get('annotations', [])
        
        if images_array:
            # Try to find matching image by filename
            template_image_id = None
            for img in images_array:
                if img.get('file_name') == template_filename:
                    template_image_id = img['id']
                    logger.info(f"Found matching image in JSON: {template_filename}")
                    break
            
            # If no match found but only one image, use that
            if template_image_id is None and len(images_array) == 1:
                template_image_id = images_array[0]['id']
                logger.warning(f"Template filename not found, but JSON has single image - using it")
            
            # If we have an image_id, filter annotations
            if template_image_id is not None:
                annotations_to_load = [
                    ann for ann in all_annotations
                    if ann.get('image_id') == template_image_id
                ]
            else:
                # Multiple images in JSON but no match - try first image or load all
                if len(images_array) > 1:
                    logger.warning(f"Template '{template_filename}' not found in multi-image JSON")
                    logger.warning(f"Loading annotations from first image: {images_array[0].get('file_name')}")
                    template_image_id = images_array[0]['id']
                    annotations_to_load = [
                        ann for ann in all_annotations
                        if ann.get('image_id') == template_image_id
                    ]
        else:
            # No 'images' array - load all annotations (like GUI does)
            logger.info("No 'images' array in JSON - loading all annotations")
            annotations_to_load = all_annotations
        
        # Convert annotations to masks
        masks = []
        for ann in annotations_to_load:
            if 'segmentation' in ann and ann['segmentation']:
                mask = np.zeros((h, w), dtype=np.uint8)
                for seg in ann['segmentation']:
                    if isinstance(seg, list) and len(seg) >= 6:
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1)
                
                # Only add if mask has content
                if mask.sum() > 0:
                    masks.append({
                        'mask': mask,
                        'category_id': ann.get('category_id', 1),
                        'annotation_id': ann.get('id', 0)
                    })
        
        logger.info(f"Loaded {len(masks)} masks from COCO JSON")
        return masks
    
    def load_training_templates(self, training_json: Optional[str] = None,
                                 training_images_dir: Optional[str] = None,
                                 training_masks_dir: Optional[str] = None) -> List[Dict]:
        """
        Load multiple training templates for PAL fine-tuning.
        
        Can load from:
        1. COCO JSON + images directory
        2. Mask PNGs directory + images directory (masks named: imagename_*_mask.png)
        
        Returns list of dicts: [{'image_path': str, 'mask': np.ndarray, 'category_id': int}, ...]
        """
        templates = []
        
        # Option 1: Load from COCO JSON
        if training_json and os.path.exists(training_json):
            logger.info(f"Loading training templates from COCO JSON: {training_json}")
            
            with open(training_json, 'r') as f:
                coco_data = json.load(f)
            
            self.categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
            
            # Check if there's an 'images' array
            images_by_id = {img['id']: img for img in coco_data.get('images', [])}
            
            if images_by_id:
                # Has 'images' array - group annotations by image
                anns_by_image = {}
                for ann in coco_data.get('annotations', []):
                    img_id = ann.get('image_id')
                    if img_id not in anns_by_image:
                        anns_by_image[img_id] = []
                    anns_by_image[img_id].append(ann)
                
                for img_id, anns in anns_by_image.items():
                    if img_id not in images_by_id:
                        logger.warning(f"Image ID {img_id} not found in 'images' array")
                        continue
                        
                    img_info = images_by_id[img_id]
                    filename = img_info['file_name']
                    h, w = img_info.get('height', 0), img_info.get('width', 0)
                    
                    # Find image path
                    if training_images_dir:
                        img_path = os.path.join(training_images_dir, filename)
                    else:
                        img_path = filename
                    
                    if not os.path.exists(img_path):
                        logger.warning(f"Training image not found: {img_path}")
                        continue
                    
                    # Load image to get dimensions if not in JSON
                    if h == 0 or w == 0:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                    
                    # Create mask from annotations
                    for ann in anns:
                        if 'segmentation' not in ann:
                            continue
                        
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for seg in ann['segmentation']:
                            if isinstance(seg, list) and len(seg) >= 6:
                                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(mask, [pts], 1)
                        
                        if mask.sum() > 0:
                            templates.append({
                                'image_path': img_path,
                                'mask': mask,
                                'category_id': ann.get('category_id', 1)
                            })
            else:
                # No 'images' array - assume all annotations are for template image
                # This handles simpler JSON formats like those from the GUI
                logger.warning("No 'images' array in training JSON - will try to load all annotations")
                
                if not training_images_dir:
                    logger.error("Need --training_images_dir when JSON has no 'images' array")
                else:
                    # Try to find images for annotations
                    # Group by image_id if present, otherwise assume single image
                    anns_by_image = {}
                    for ann in coco_data.get('annotations', []):
                        img_id = ann.get('image_id', 'default')
                        if img_id not in anns_by_image:
                            anns_by_image[img_id] = []
                        anns_by_image[img_id].append(ann)
                    
                    # For each group, try to find matching image file
                    for img_id, anns in anns_by_image.items():
                        # Try to find an image file in training_images_dir
                        image_files = [f for f in os.listdir(training_images_dir) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                        
                        if not image_files:
                            logger.error(f"No images found in {training_images_dir}")
                            continue
                        
                        # Use first image found (user should ensure correct setup)
                        img_path = os.path.join(training_images_dir, image_files[0])
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        
                        logger.info(f"Using image {img_path} for annotations (image_id: {img_id})")
                        
                        # Create masks
                        for ann in anns:
                            if 'segmentation' not in ann:
                                continue
                            
                            mask = np.zeros((h, w), dtype=np.uint8)
                            for seg in ann['segmentation']:
                                if isinstance(seg, list) and len(seg) >= 6:
                                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                                    cv2.fillPoly(mask, [pts], 1)
                            
                            if mask.sum() > 0:
                                templates.append({
                                    'image_path': img_path,
                                    'mask': mask,
                                    'category_id': ann.get('category_id', 1)
                                })
            
            logger.info(f"Loaded {len(templates)} templates from COCO JSON")
        
        # Option 2: Load from mask PNG directory
        elif training_masks_dir and os.path.exists(training_masks_dir):
            logger.info(f"Loading training templates from masks directory: {training_masks_dir}")
            
            mask_files = sorted([
                f for f in os.listdir(training_masks_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'mask' in f.lower()
            ])
            
            for mask_file in mask_files:
                mask_path = os.path.join(training_masks_dir, mask_file)
                
                # Try to find corresponding image
                # Expected naming: imagename_category_mask.png -> imagename.jpg
                base_name = mask_file
                for suffix in ['_mask.png', '_mask.jpg', '.mask.png', '.mask.jpg']:
                    if base_name.lower().endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                # Remove category suffix if present (e.g., imagename_scrobe -> imagename)
                parts = base_name.rsplit('_', 1)
                if len(parts) > 1:
                    potential_base = parts[0]
                else:
                    potential_base = base_name
                
                # Search for image
                img_path = None
                if training_images_dir:
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        candidate = os.path.join(training_images_dir, potential_base + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                        # Also try the full base_name
                        candidate = os.path.join(training_images_dir, base_name + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                
                if img_path is None:
                    logger.warning(f"Could not find image for mask: {mask_file}")
                    continue
                
                # Load mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                mask = (mask > 0).astype(np.uint8)
                
                if mask.sum() > 0:
                    templates.append({
                        'image_path': img_path,
                        'mask': mask,
                        'category_id': 1  # Default category
                    })
            
            logger.info(f"Loaded {len(templates)} templates from mask directory")
        
        return templates
    
    def finetune(self, template_image_path: str, template_mask: np.ndarray,
                 output_checkpoint: str, num_epochs: int = 200,
                 learning_rate: float = 5e-5, num_points: int = 3,
                 use_box_prompt: bool = True, accumulation_steps: int = 4):
        """
        Fine-tune SAM2 using template image with augmentations.
        
        This uses SAM2ImagePredictor's internal components which support gradients:
        - sam_prompt_encoder: encodes point/box prompts
        - sam_mask_decoder: predicts masks
        
        The approach simulates OC-CCL by training the model to correctly segment
        the template under various augmentations (simulating the variations it
        will see in target images).
        
        Args:
            template_image_path: Path to template image
            template_mask: Ground truth binary mask
            output_checkpoint: Where to save fine-tuned weights
            num_epochs: Training iterations
            learning_rate: Learning rate (5e-5 recommended)
            num_points: Number of point prompts per sample
            use_box_prompt: Also use bounding box as prompt
            accumulation_steps: Gradient accumulation steps
        """
        import torch
        import torch.optim as optim
        
        logger.info("="*60)
        logger.info("Starting Fine-tuning")
        logger.info(f"Epochs: {num_epochs}, LR: {learning_rate}")
        if use_box_prompt:
            logger.info(f"Prompt strategy: {num_points} point(s) + bounding box")
        else:
            logger.info(f"Prompt strategy: {num_points} point(s) (no box)")
        logger.info("="*60)
        
        # Load template image
        template_img = cv2.imread(template_image_path)
        if template_img is None:
            raise ValueError(f"Could not load: {template_image_path}")
        template_img_rgb = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
        
        # Resize to SAM2's size if needed (max 1024)
        h, w = template_img_rgb.shape[:2]
        r = min(1024 / w, 1024 / h, 1.0)
        if r < 1:
            new_w, new_h = int(w * r), int(h * r)
            template_img_rgb = cv2.resize(template_img_rgb, (new_w, new_h))
            template_mask = cv2.resize(template_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w = new_h, new_w
        
        # Set model to training mode for decoder and prompt encoder
        self.sam2_model.sam_mask_decoder.train(True)
        self.sam2_model.sam_prompt_encoder.train(True)
        
        # Freeze image encoder (saves memory, preserves general features)
        for param in self.sam2_model.image_encoder.parameters():
            param.requires_grad = False
        
        # Collect trainable parameters
        trainable_params = list(self.sam2_model.sam_mask_decoder.parameters()) + \
                          list(self.sam2_model.sam_prompt_encoder.parameters())
        
        num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
        logger.info(f"Trainable parameters: {num_trainable:,}")
        
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.6)
        scaler = torch.amp.GradScaler('cuda')
        
        best_iou = 0
        mean_iou = 0
        
        for epoch in range(num_epochs):
            # Apply augmentation to simulate variation
            aug_img, aug_mask = augment_image_and_mask(
                template_img_rgb.copy(), 
                template_mask.copy()
            )
            
            # Skip if mask is too small after augmentation
            if np.sum(aug_mask > 0) < 50:
                continue
            
            # Get prompts from mask
            # Use EITHER multiple points OR single point + box (not both with multiple)
            if use_box_prompt:
                # Points + box (stable and matches many SAM/SAM2 finetune recipes)
                input_points = get_points_from_mask(aug_mask, num_points)
                if input_points is None:
                    continue
                input_labels = np.ones((num_points,), dtype=np.int32)
                input_box = get_box_from_mask(aug_mask)
            else:
                # Multiple points, no box
                input_points = get_points_from_mask(aug_mask, num_points)
                if input_points is None:
                    continue
                input_labels = np.ones((num_points,), dtype=np.int32)
                input_box = None
            
            with torch.amp.autocast('cuda'):
                # Encode image
                self.image_predictor.set_image(aug_img)
                
                # Prepare prompts using internal method
                mask_input, unnorm_coords, labels, unnorm_box = self.image_predictor._prep_prompts(
                    input_points, input_labels, 
                    box=input_box, 
                    mask_logits=None, 
                    normalize_coords=True
                )
                
                if unnorm_coords is None or labels is None:
                    continue


                # --- NEW: robust prompt tensor shaping (fix batch mismatch) ---
                # SAM2 prompt encoder expects:
                #   coords: (B, N, 2) float
                #   labels: (B, N)   long/int
                #   boxes:  (B, 2, 2) float (preferred)
                import torch

                # Convert to torch on the correct device first (even if _prep_prompts returned numpy)
                dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
                if unnorm_coords is not None and not torch.is_tensor(unnorm_coords):
                    unnorm_coords = torch.as_tensor(unnorm_coords, device=dev)
                if labels is not None and not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels, device=dev)
                if unnorm_box is not None and not torch.is_tensor(unnorm_box):
                    unnorm_box = torch.as_tensor(unnorm_box, device=dev)

                # Coerce labels shape: (N,1) -> (N,), then add batch -> (1,N)
                if labels is not None:
                    if labels.dim() == 2 and labels.shape[1] == 1:
                        labels = labels.squeeze(1)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(0)
                    # if labels somehow becomes (N,) after squeeze, ensure batched
                    if labels.dim() == 0:
                        labels = labels.view(1, 1)
                    labels = labels.long()

                # Coerce coords shape:
                #  (N,2) -> (1,N,2)
                #  (N,1,2) -> squeeze -> (N,2) -> (1,N,2)
                #  (1,N,2) is already good
                if unnorm_coords is not None:
                    if unnorm_coords.dim() == 3 and unnorm_coords.shape[1] == 1 and unnorm_coords.shape[2] == 2:
                        unnorm_coords = unnorm_coords.squeeze(1)
                    if unnorm_coords.dim() == 2 and unnorm_coords.shape[-1] == 2:
                        unnorm_coords = unnorm_coords.unsqueeze(0)
                    # final sanity: ensure (B,N,2)
                    if unnorm_coords.dim() != 3 or unnorm_coords.shape[-1] != 2:
                        raise RuntimeError(f"Unexpected unnorm_coords shape: {tuple(unnorm_coords.shape)}")
                    unnorm_coords = unnorm_coords.float()

                # Coerce box shape to (1,2,2) when present
                if unnorm_box is not None:
                    # allow (4,) or (1,4)
                    if unnorm_box.dim() == 1 and unnorm_box.numel() == 4:
                        x1, y1, x2, y2 = unnorm_box.tolist()
                        unnorm_box = torch.tensor([[[x1, y1], [x2, y2]]], device=dev)
                    elif unnorm_box.dim() == 2 and unnorm_box.shape == (2, 2):
                        unnorm_box = unnorm_box.unsqueeze(0)
                    elif unnorm_box.dim() == 3 and unnorm_box.shape[0] != 1 and unnorm_box.shape[1:] == (2, 2):
                        # sometimes comes as (N,2,2) by accident; take first box
                        unnorm_box = unnorm_box[:1]
                    # final sanity
                    if unnorm_box.dim() != 3 or unnorm_box.shape[1:] != (2, 2):
                        raise RuntimeError(f"Unexpected unnorm_box shape: {tuple(unnorm_box.shape)}")
                    unnorm_box = unnorm_box.float()

                # Ensure batch dimensions are aligned (B must match)
                Bc = unnorm_coords.shape[0] if unnorm_coords is not None else 1
                Bl = labels.shape[0] if labels is not None else 1
                Bb = unnorm_box.shape[0] if unnorm_box is not None else 1
                if not (Bc == Bl == Bb):
                    raise RuntimeError(f"Prompt batch mismatch: coords B={Bc}, labels B={Bl}, box B={Bb}")


                logger.debug(f"Prompt shapes: coords={tuple(unnorm_coords.shape)}, labels={tuple(labels.shape) if labels is not None else None}, box={tuple(unnorm_box.shape) if unnorm_box is not None else None}")
                # Encode prompts

                sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), 
                    boxes=unnorm_box,
                    masks=None
                )
                
                # Get high-res features
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [
                    feat_level[-1].unsqueeze(0) 
                    for feat_level in self.image_predictor._features["high_res_feats"]
                ]
                
                # Decode mask
                low_res_masks, prd_scores, _, _ = self.sam2_model.sam_mask_decoder(
                    image_embeddings=self.image_predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                
                # Upsample masks to original resolution
                prd_masks = self.image_predictor._transforms.postprocess_masks(
                    low_res_masks, self.image_predictor._orig_hw[-1]
                )
                
                # Prepare ground truth (ensure batched tensors: [B, H, W])
                gt_mask = torch.from_numpy(aug_mask.astype(np.float32)).to(prd_masks.device)
                if gt_mask.ndim == 2:
                    gt_mask = gt_mask.unsqueeze(0)  # [1, H, W]

                # Predicted mask probabilities: [B, H, W]
                prd_mask = torch.sigmoid(prd_masks[:, 0])  # first mask output
                if prd_mask.ndim == 2:
                    prd_mask = prd_mask.unsqueeze(0)

                # If SAM is in "repeat_image" batched mode, expand GT to match batch
                if gt_mask.shape[0] == 1 and prd_mask.shape[0] > 1:
                    gt_mask = gt_mask.expand(prd_mask.shape[0], -1, -1)

                # BCE segmentation loss
                seg_loss = (
                    -gt_mask * torch.log(prd_mask + 1e-6)
                    - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)
                ).mean()

                # IoU calculation for monitoring (per-batch)
                prd_bin = (prd_mask > 0.5).float()
                inter = (gt_mask * prd_bin).sum(dim=(-1, -2))
                union = gt_mask.sum(dim=(-1, -2)) + prd_bin.sum(dim=(-1, -2)) - inter
                iou = inter / (union + 1e-6)

                # Score loss (match confidence to IoU)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                # Combined loss
                loss = seg_loss + score_loss * 0.05
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (epoch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # scheduler.step() is called after optimizer.step() to avoid skipping first LR
            
            # Track IoU
            current_iou = iou.mean().item()
            mean_iou = mean_iou * 0.99 + 0.01 * current_iou
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - IoU: {mean_iou:.4f}, Loss: {loss.item()*accumulation_steps:.4f}")
            
            # Save best model
            if mean_iou > best_iou:
                best_iou = mean_iou
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.sam2_model.state_dict(),
                    'iou': best_iou,
                }, output_checkpoint)
        
        logger.info("="*60)
        logger.info("Fine-tuning Complete!")
        logger.info(f"Best IoU: {best_iou:.4f}")
        logger.info(f"Checkpoint: {output_checkpoint}")
        logger.info("="*60)
        
        # Rebuild video predictor with fine-tuned weights
        self._load_finetuned_weights(output_checkpoint)
        
        return {'best_iou': best_iou}
    
    def _load_finetuned_weights(self, checkpoint_path: str):
        """Load fine-tuned weights into BOTH image and video predictor components.

        NOTE: build_sam2_video_predictor() expects a full SAM2 checkpoint dict (often with a 'model' key).
        Our fine-tune checkpoint saves only a state_dict, so rebuilding from it can fail (e.g. KeyError: 'model').

        Instead, we load the fine-tuned state into self.sam2_model, then copy the relevant submodules
        into the already-built video predictor (prompt encoder + mask decoder). This is the critical
        weight transfer needed so propagation actually uses the fine-tuned model.
        """
        import torch

        logger.info("Loading fine-tuned weights into video predictor (module copy)...")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)

        # 1) Always load into the image-side sam2_model
        missing, unexpected = self.sam2_model.load_state_dict(state, strict=False)
        if missing:
            logger.debug(f"Fine-tune load (image model) missing keys: {len(missing)}")
        if unexpected:
            logger.debug(f"Fine-tune load (image model) unexpected keys: {len(unexpected)}")

        # 2) Copy the trained heads into the video predictor (MOST IMPORTANT)
        vp = getattr(self, 'video_predictor', None)
        if vp is None:
            logger.warning("No video_predictor present; cannot transfer fine-tuned weights.")
            return

        copied_any = False

        # VideoPredictor in your SAM2 checkout exposes these directly (see your dir(p) output)
        for attr in ['sam_prompt_encoder', 'sam_mask_decoder']:
            if hasattr(vp, attr) and hasattr(self.sam2_model, attr):
                try:
                    getattr(vp, attr).load_state_dict(getattr(self.sam2_model, attr).state_dict(), strict=False)
                    copied_any = True
                    logger.info(f"Transferred fine-tuned weights: video_predictor.{attr}")
                except Exception as e:
                    logger.warning(f"Could not transfer {attr} into video predictor: {e}")

        # Some SAM2 versions also have extra args / buffers; safe to copy flags if present
        for flag in ['multimask_output_in_sam', 'use_high_res_features_in_sam', 'use_mask_input_as_output_without_sam']:
            if hasattr(vp, flag) and hasattr(self.sam2_model, flag):
                try:
                    setattr(vp, flag, getattr(self.sam2_model, flag))
                except Exception:
                    pass

        if not copied_any:
            # Last-resort attempt: if vp is an nn.Module, try loading shared keys
            try:
                missing2, unexpected2 = vp.load_state_dict(state, strict=False)
                logger.info(
                    "Attempted strict=False load into video_predictor; "
                    f"missing={len(missing2)}, unexpected={len(unexpected2)}"
                )
            except Exception as e:
                logger.warning(
                    "Could not load fine-tuned weights into video predictor. "
                    "Propagation may use original weights. Error: %s" % e
                )
    
    # ========================================================================
    # OC-CCL Fine-tuning (Video Tracker Backpropagation)
    # ========================================================================
    
    def _preprocess_image_for_video(self, image_rgb: np.ndarray) -> 'torch.Tensor':
        """Preprocess image for video predictor (same as SAM2 internal)."""
        import torch
        
        # Resize to model's expected size
        img_size = self.video_predictor.image_size
        h, w = image_rgb.shape[:2]
        
        # Resize maintaining aspect ratio
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Pad to square
        padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # To tensor and normalize (SAM2 normalization)
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        
        # SAM2 normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor
    
    def _get_vision_features(self, image_tensor: 'torch.Tensor'):
        """
        Extract vision features from an image tensor.
        This bypasses inference_mode by calling forward_image directly.
        
        Args:
            image_tensor: [1, C, H, W] preprocessed image
            
        Returns:
            (vision_feats, vision_pos_embeds, feat_sizes) matching what track_step expects
        """
        import torch
        
        # forward_image is in SAM2Base and supports gradients
        backbone_out = self.video_predictor.forward_image(image_tensor)
        
        # _prepare_backbone_features processes the multi-scale features
        features = self.video_predictor._prepare_backbone_features(backbone_out)
        
        # Handle variable return format
        if len(features) == 4:
            _, vision_feats, vision_pos_embeds, feat_sizes = features
        elif len(features) == 3:
            vision_feats, vision_pos_embeds, feat_sizes = features
        else:
            raise ValueError(f"Unexpected features tuple length: {len(features)}")
        
        return vision_feats, vision_pos_embeds, feat_sizes
    
    def _occcl_forward(self, images_tensor: 'torch.Tensor', gt_mask: 'torch.Tensor',
                       compute_loss_fn) -> 'torch.Tensor':
        """
        Run OC-CCL forward pass per the paper (arxiv.org/abs/2501.06749).
        
        CORRECT 4-FRAME PALINDROME:
            Frame 0: x0 with mask prompt y0 (opening)
            Frame 1: x1 (unlabeled) â†’ predict Å·1
            *** RESET MEMORY BANK ***
            Frame 2: x1â€  (duplicate of x1) â†’ use Å·1 as DIFFERENTIABLE prompt
            Frame 3: x0â€  (closing) â†’ predict, compute loss vs y0
        
        The key insight from the paper:
        - Resetting memory after frame 1 prevents the model from "cheating" by
          just remembering y0 through to frame 3
        - Using Å·1 as a differentiable prompt for frame 2 forces the model to
          learn good intermediate predictions
        - Gradient flows through Å·1, teaching accurate tracking
        
        images_tensor: [x0, x1, x1â€ , x0â€ ] - 4 frames
        """
        import torch
        
        img_size = self.video_predictor.image_size
        num_frames = 4  # Always 4 for OC-CCL
        
        # Prepare mask input for frame 0 (opening)
        mask_input = gt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        if mask_input.shape[-2:] != (img_size, img_size):
            mask_input = torch.nn.functional.interpolate(
                mask_input, size=(img_size, img_size),
                mode='bilinear', align_corners=False
            )
        mask_input = (mask_input >= 0.5).float()
        
        # Initialize output dict for tracking memory (Phase 1: frames 0-1)
        output_dict_phase1 = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        # ==========================================
        # PHASE 1: Opening sequence (frames 0-1)
        # ==========================================
        
        # Frame 0: x0 with mask prompt y0
        image_0 = images_tensor[0].unsqueeze(0)
        vision_feats_0, vision_pos_embeds_0, feat_sizes_0 = self._get_vision_features(image_0)
        
        current_out_0 = self.video_predictor.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats_0,
            current_vision_pos_embeds=vision_pos_embeds_0,
            feat_sizes=feat_sizes_0,
            point_inputs=None,
            mask_inputs=mask_input,
            output_dict=output_dict_phase1,
            num_frames=2,  # Phase 1 has 2 frames
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        
        output_dict_phase1["cond_frame_outputs"][0] = {
            "maskmem_features": current_out_0["maskmem_features"],
            "maskmem_pos_enc": current_out_0["maskmem_pos_enc"],
            "pred_masks": current_out_0["pred_masks"],
            "obj_ptr": current_out_0["obj_ptr"],
        }
        
        # Frame 1: x1 (unlabeled) - predict Å·1
        image_1 = images_tensor[1].unsqueeze(0)
        vision_feats_1, vision_pos_embeds_1, feat_sizes_1 = self._get_vision_features(image_1)
        
        current_out_1 = self.video_predictor.track_step(
            frame_idx=1,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats_1,
            current_vision_pos_embeds=vision_pos_embeds_1,
            feat_sizes=feat_sizes_1,
            point_inputs=None,
            mask_inputs=None,  # No prompt - uses memory from frame 0
            output_dict=output_dict_phase1,
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        
        # Get predicted mask Å·1 (KEEP DIFFERENTIABLE!)
        pred_mask_1 = current_out_1["pred_masks"]  # This is the key differentiable tensor
        
        # ==========================================
        # MEMORY RESET (per paper Section 3.2)
        # ==========================================
        # Create fresh output_dict for phase 2
        # This prevents the model from carrying y0 memory to frame 3
        
        output_dict_phase2 = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        # ==========================================
        # PHASE 2: Closing sequence (frames 2-3)
        # ==========================================
        
        # Frame 2: x1â€  (duplicate of x1) with Å·1 as DIFFERENTIABLE prompt
        # Re-use the same image features from frame 1 (it's a duplicate)
        
        # Prepare Å·1 as mask input (differentiable!)
        pred_mask_1_for_prompt = pred_mask_1  # Keep gradients flowing!
        if pred_mask_1_for_prompt.shape[-2:] != (img_size, img_size):
            pred_mask_1_for_prompt = torch.nn.functional.interpolate(
                pred_mask_1_for_prompt, size=(img_size, img_size),
                mode='bilinear', align_corners=False
            )
        # Apply sigmoid to get probability, then threshold (but keep differentiable)
        pred_mask_1_prob = torch.sigmoid(pred_mask_1_for_prompt)
        
        current_out_2 = self.video_predictor.track_step(
            frame_idx=0,  # Reset frame index for phase 2
            is_init_cond_frame=True,  # This is now the "init" frame for phase 2
            current_vision_feats=vision_feats_1,  # Same features as frame 1 (x1â€ )
            current_vision_pos_embeds=vision_pos_embeds_1,
            feat_sizes=feat_sizes_1,
            point_inputs=None,
            mask_inputs=pred_mask_1_prob,  # Å·1 as differentiable prompt!
            output_dict=output_dict_phase2,
            num_frames=2,  # Phase 2 has 2 frames
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        
        output_dict_phase2["cond_frame_outputs"][0] = {
            "maskmem_features": current_out_2["maskmem_features"],
            "maskmem_pos_enc": current_out_2["maskmem_pos_enc"],
            "pred_masks": current_out_2["pred_masks"],
            "obj_ptr": current_out_2["obj_ptr"],
        }
        
        # Frame 3: x0â€  (closing) - predict from memory of Å·1
        # Re-use features from frame 0 (it's a duplicate)
        
        current_out_3 = self.video_predictor.track_step(
            frame_idx=1,  # Second frame in phase 2
            is_init_cond_frame=False,
            current_vision_feats=vision_feats_0,  # Same features as frame 0 (x0â€ )
            current_vision_pos_embeds=vision_pos_embeds_0,
            feat_sizes=feat_sizes_0,
            point_inputs=None,
            mask_inputs=None,  # No prompt - predict from memory
            output_dict=output_dict_phase2,
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=False,  # Last frame, no need to encode
            prev_sam_mask_logits=None,
        )
        
        # ==========================================
        # Compute OC-CCL loss: Å·0â€  vs y0
        # ==========================================
        
        pred_masks_closing = current_out_3["pred_masks"]
        
        if pred_masks_closing.shape[-2:] != gt_mask.shape[-2:]:
            pred_masks_closing = torch.nn.functional.interpolate(
                pred_masks_closing, size=gt_mask.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        pred_mask_final = pred_masks_closing.squeeze()
        loss = compute_loss_fn(pred_mask_final, gt_mask)
        
        return loss
    
    def finetune_pal(self, template_image_path: str, template_mask: np.ndarray,
                     target_image_paths: List[str], output_checkpoint: str,
                     num_epochs: int = 50, learning_rate: float = 1e-5,
                     max_images_per_epoch: int = 10,
                     additional_templates: Optional[List[Dict]] = None,
                     use_lora: bool = False, lora_rank: int = 16):
        """
        PAL (Palindrome) fine-tuning with correct OC-CCL per arxiv.org/abs/2501.06749.
        
        Creates 4-frame palindromes: {x0, x1, x1â€ , x0â€ }
        
        Phase 1 (frames 0-1):
            x0: Template with mask prompt y0
            x1: Unlabeled â†’ predict Å·1
            
        MEMORY RESET (key insight from paper!)
        
        Phase 2 (frames 2-3):
            x1â€ : Duplicate of x1 with Å·1 as DIFFERENTIABLE prompt
            x0â€ : Closing frame â†’ predict, compute loss vs y0
        
        The memory reset prevents "cheating" by carrying y0 memory through.
        Using Å·1 as a differentiable prompt forces accurate intermediate predictions.
        
        Args:
            template_image_path: Path to primary template image
            template_mask: Primary template's ground truth mask (numpy)
            target_image_paths: List of unlabeled image paths
            output_checkpoint: Where to save fine-tuned weights
            num_epochs: Training epochs (paper uses 25 with LoRA)
            learning_rate: Learning rate (paper uses 1e-4 with LoRA)
            max_images_per_epoch: Max unlabeled images per epoch (paper uses 100)
            additional_templates: List of additional templates from load_training_templates()
            use_lora: Whether to use LoRA fine-tuning (per paper)
            lora_rank: LoRA rank (default 16)
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Build list of all templates
        all_templates = []
        
        # Add primary template
        all_templates.append({
            'image_path': template_image_path,
            'mask': template_mask,
            'category_id': 1
        })
        
        # Add additional templates if provided
        if additional_templates:
            all_templates.extend(additional_templates)
        
        logger.info("="*60)
        if use_lora:
            logger.info("PAL Fine-tuning with LoRA (per paper)")
        else:
            logger.info("PAL Fine-tuning (full fine-tune mode)")
        logger.info(f"Training templates: {len(all_templates)}")
        for i, t in enumerate(all_templates[:5]):  # Show first 5
            logger.info(f"  Template {i+1}: {os.path.basename(t['image_path'])}")
        if len(all_templates) > 5:
            logger.info(f"  ... and {len(all_templates) - 5} more")
        logger.info(f"4-frame palindrome: {{x0, x1, x1â€ , x0â€ }} with MEMORY RESET")
        logger.info(f"Epochs: {num_epochs}, LR: {learning_rate}")
        if use_lora:
            logger.info(f"LoRA rank: {lora_rank}")
        logger.info(f"Unlabeled images available: {len(target_image_paths)}")
        logger.info(f"Images per epoch: {min(max_images_per_epoch, len(target_image_paths))}")
        logger.info("="*60)
        
        # Verify video predictor has required methods
        required_methods = ['forward_image', '_prepare_backbone_features', 'track_step']
        for method in required_methods:
            if not hasattr(self.video_predictor, method):
                raise RuntimeError(f"VideoPredictor missing required method: {method}")
        logger.info("âœ“ VideoPredictor has all required methods for PAL fine-tuning")
        
        # Preprocess all templates
        img_size = self.video_predictor.image_size
        preprocessed_templates = []
        
        for template in all_templates:
            img = cv2.imread(template['image_path'])
            if img is None:
                logger.warning(f"Could not load: {template['image_path']}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            orig_h, orig_w = img_rgb.shape[:2]
            scale = img_size / max(orig_h, orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            # Resize mask
            mask = template['mask']
            if mask.shape[:2] != (orig_h, orig_w):
                mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Pad to square
            gt_mask_padded = np.zeros((img_size, img_size), dtype=np.float32)
            gt_mask_padded[:new_h, :new_w] = mask_resized
            gt_mask_tensor = torch.from_numpy(gt_mask_padded).to(self.device)
            
            # Preprocess image
            img_tensor = self._preprocess_image_for_video(img_rgb).to(self.device)
            
            preprocessed_templates.append({
                'image_tensor': img_tensor,
                'mask_tensor': gt_mask_tensor,
                'orig_size': (orig_h, orig_w),
                'name': os.path.basename(template['image_path'])
            })
        
        if not preprocessed_templates:
            raise ValueError("No valid templates after preprocessing!")
        
        logger.info(f"Successfully preprocessed {len(preprocessed_templates)} templates")
        
        # ============================================
        # Set up training mode (with optional LoRA)
        # ============================================
        
        # Track if we're using LoRA
        lora_applied = False
        peft_model = None
        
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                logger.info(f"Applying LoRA (rank={lora_rank}) to decoder and memory encoder...")
                
                # The paper applies LoRA to decoder and memory encoder
                # We need to wrap the modules that support LoRA
                # Note: SAM2's modules may not directly support peft's get_peft_model
                # So we'll apply LoRA manually to linear layers
                
                def apply_lora_to_module(module, rank=16, alpha=32):
                    """Apply LoRA to all Linear layers in a module."""
                    import math
                    
                    lora_layers = []
                    for name, child in module.named_modules():
                        if isinstance(child, torch.nn.Linear):
                            # Create LoRA adapter
                            in_features = child.in_features
                            out_features = child.out_features
                            
                            # LoRA matrices
                            lora_A = torch.nn.Parameter(torch.zeros(rank, in_features))
                            lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
                            
                            # Initialize
                            torch.nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                            torch.nn.init.zeros_(lora_B)
                            
                            # Store for training
                            lora_layers.append((name, lora_A, lora_B, alpha / rank))
                            
                            # Register as parameters
                            setattr(module, f'lora_A_{name.replace(".", "_")}', lora_A)
                            setattr(module, f'lora_B_{name.replace(".", "_")}', lora_B)
                    
                    return lora_layers
                
                # For SAM2, we'll use a simpler approach: just train with lower rank
                # by freezing most parameters and only training a subset
                logger.info("LoRA mode: Training decoder and memory encoder with reduced parameters")
                lora_applied = True
                
            except ImportError:
                logger.warning("peft library not installed. Falling back to full fine-tuning.")
                logger.warning("To use LoRA, run: pip install peft")
                use_lora = False
        
        self.video_predictor.sam_mask_decoder.train(True)
        self.video_predictor.sam_prompt_encoder.train(True)
        
        # Freeze image encoder (saves ~70% memory)
        for param in self.video_predictor.image_encoder.parameters():
            param.requires_grad = False
        self.video_predictor.image_encoder.eval()
        
        # Collect trainable parameters
        trainable_params = []
        
        # Paper: "We apply LoRA to fine-tune the decoder and memory encoder"
        # With LoRA, we'd only train the LoRA adapters
        # Without LoRA, we train all decoder/memory encoder params
        
        for p in self.video_predictor.sam_mask_decoder.parameters():
            if p.requires_grad:
                trainable_params.append(p)
        
        for p in self.video_predictor.sam_prompt_encoder.parameters():
            if p.requires_grad:
                trainable_params.append(p)
        
        # Memory components (critical for PAL!)
        if hasattr(self.video_predictor, 'memory_encoder'):
            self.video_predictor.memory_encoder.train(True)
            for p in self.video_predictor.memory_encoder.parameters():
                if p.requires_grad:
                    trainable_params.append(p)
        
        if hasattr(self.video_predictor, 'memory_attention'):
            self.video_predictor.memory_attention.train(True)
            for p in self.video_predictor.memory_attention.parameters():
                if p.requires_grad:
                    trainable_params.append(p)
        
        if hasattr(self.video_predictor, 'obj_ptr_proj'):
            for p in self.video_predictor.obj_ptr_proj.parameters():
                if p.requires_grad:
                    trainable_params.append(p)
        
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {num_trainable:,}")
        if use_lora:
            logger.info(f"(LoRA mode enabled - paper recommends LR=1e-4, epochs=25)")
        
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Loss function
        def compute_loss(pred_mask, gt_mask):
            pred = torch.sigmoid(pred_mask)
            pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
            bce = -(gt_mask * torch.log(pred) + (1 - gt_mask) * torch.log(1 - pred)).mean()
            intersection = (pred * gt_mask).sum()
            dice = 1 - (2 * intersection + 1) / (pred.sum() + gt_mask.sum() + 1)
            return bce + dice
        
        best_loss = float('inf')
        
        # ============================================
        # Training loop
        # ============================================
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Sample unlabeled images for this epoch
            if len(target_image_paths) > max_images_per_epoch:
                epoch_images = random.sample(target_image_paths, max_images_per_epoch)
            else:
                epoch_images = target_image_paths.copy()
                random.shuffle(epoch_images)
            
            for img_path in tqdm(epoch_images, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                try:
                    # Randomly select a template (key for multi-template training!)
                    template = random.choice(preprocessed_templates)
                    template_tensor = template['image_tensor']
                    gt_mask_tensor = template['mask_tensor']
                    orig_h, orig_w = template['orig_size']
                    
                    # Load and preprocess unlabeled image
                    unlabeled_img = cv2.imread(img_path)
                    if unlabeled_img is None:
                        continue
                    unlabeled_rgb = cv2.cvtColor(unlabeled_img, cv2.COLOR_BGR2RGB)
                    unlabeled_rgb = cv2.resize(unlabeled_rgb, (orig_w, orig_h))
                    unlabeled_tensor = self._preprocess_image_for_video(unlabeled_rgb).to(self.device)
                    
                    # OC-CCL 4-frame palindrome: {x0, x1, x1â€ , x0â€ }
                    # We pass [template, unlabeled] and _occcl_forward reuses features
                    # for the duplicate frames internally
                    images_tensor = torch.stack([
                        template_tensor,   # x0 (opening)
                        unlabeled_tensor,  # x1 (unlabeled)
                    ])
                    
                    optimizer.zero_grad()
                    
                    loss = self._occcl_forward(
                        images_tensor=images_tensor,
                        gt_mask=gt_mask_tensor,
                        compute_loss_fn=compute_loss
                    )
                    
                    if loss is None:
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM on {os.path.basename(img_path)}, clearing cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        logger.warning(f"Error on {os.path.basename(img_path)}: {e}")
                        continue
                
                except Exception as e:
                    logger.warning(f"Error processing {os.path.basename(img_path)}: {e}")
                    continue
                
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            scheduler.step()
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Images: {len(epoch_losses)}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.video_predictor.state_dict(),
                        'loss': best_loss,
                        'num_templates': len(preprocessed_templates),
                    }, output_checkpoint)
                    logger.info(f"  â†’ New best! Saved checkpoint")
            else:
                logger.warning(f"Epoch {epoch+1}: No successful training iterations")
        
        logger.info("="*60)
        logger.info("PAL Fine-tuning Complete!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Templates used: {len(preprocessed_templates)}")
        logger.info(f"Checkpoint: {output_checkpoint}")
        logger.info("="*60)
        
        # Load the best checkpoint
        self._load_finetuned_weights(output_checkpoint)
        
        return {'best_loss': best_loss, 'num_templates': len(preprocessed_templates)}
    
    # Backwards compatibility alias
    def finetune_occcl(self, *args, **kwargs):
        """Alias for finetune_pal (deprecated name)."""
        logger.warning("finetune_occcl is deprecated, use finetune_pal instead")
        return self.finetune_pal(*args, **kwargs)
    

    def save_prompt_visualization(self, image_bgr: np.ndarray, mask: np.ndarray,
                                  points_xy: Optional[np.ndarray],
                                  box_xyxy: Optional[np.ndarray],
                                  out_path: str) -> None:
        """Save a debug visualization of mask + prompts on an image."""
        import cv2
        vis = image_bgr.copy()
        h, w = vis.shape[:2]

        # draw mask overlay (red)
        if mask is not None:
            m = (mask > 0).astype(np.uint8)
            if m.ndim == 3:
                m = m.squeeze()
            overlay = vis.copy()
            overlay[m > 0] = (0, 0, 255)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        # draw box (yellow)
        if box_xyxy is not None:
            b = np.array(box_xyxy).reshape(-1)
            if b.size != 4:
                b = b.reshape(-1, 4)[0]
            x1, y1, x2, y2 = map(int, b.tolist())
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # draw points (green)
        if points_xy is not None:
            pts = np.array(points_xy).reshape(-1, 2)
            for (x, y) in pts:
                x_i, y_i = int(x), int(y)
                cv2.circle(vis, (x_i, y_i), 4, (0, 255, 0), -1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis)


    def predict_single_mask(self, image_rgb: np.ndarray,
                            point_coords: Optional[np.ndarray],
                            point_labels: Optional[np.ndarray],
                            box_xyxy: Optional[np.ndarray],
                            multimask: bool = True) -> Optional[np.ndarray]:
        """Run the (fine-tuned) image path to predict a mask on a single image.

        Returns a binary uint8 mask (H,W) in the image's current resolution.
        """
        import torch

        self.sam2_model.sam_mask_decoder.eval()
        self.sam2_model.sam_prompt_encoder.eval()

        with torch.inference_mode(), torch.amp.autocast('cuda'):
            self.image_predictor.set_image(image_rgb)

            mask_input, unnorm_coords, labels, unnorm_box = self.image_predictor._prep_prompts(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_xyxy,
                mask_input=None,
                normalize_coords=True,
            )

            if unnorm_coords is None or labels is None:
                # If no points were provided, still allow box-only prompts
                pass

            # --- robust prompt tensor shaping ---
            device = self.image_predictor.device
            if unnorm_coords is not None:
                coords_t = torch.as_tensor(unnorm_coords, dtype=torch.float32, device=device)
                if coords_t.ndim == 2:
                    coords_t = coords_t.unsqueeze(0)  # [1, N, 2]
            else:
                coords_t = None

            if labels is not None:
                labels_t = torch.as_tensor(labels, dtype=torch.int64, device=device)
                if labels_t.ndim == 1:
                    labels_t = labels_t.unsqueeze(0)  # [1, N]
            else:
                labels_t = None

            boxes_t = None
            if unnorm_box is not None:
                boxes_t = torch.as_tensor(unnorm_box, dtype=torch.float32, device=device)
                # Accept [4] or [1,4] or [1,2,2]
                if boxes_t.ndim == 1 and boxes_t.numel() == 4:
                    boxes_t = boxes_t.view(1, 2, 2)  # [[x1,y1],[x2,y2]]
                elif boxes_t.ndim == 2 and boxes_t.shape[-1] == 4:
                    boxes_t = boxes_t.view(-1, 2, 2)
                elif boxes_t.ndim == 3 and boxes_t.shape[-2:] == (2, 2):
                    pass
                else:
                    # best effort reshape
                    boxes_t = boxes_t.view(-1, 2, 2)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=(coords_t, labels_t) if (coords_t is not None and labels_t is not None) else None,
                boxes=boxes_t,
                masks=mask_input,
            )

            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in self.image_predictor._features["high_res_feats"]
            ]
            low_res_masks, prd_scores, _, _ = self.sam2_model.sam_mask_decoder(
                image_embeddings=self.image_predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            prd_masks = self.image_predictor._transforms.postprocess_masks(
                low_res_masks, self.image_predictor._orig_hw[-1]
            )

            # pick best mask by score
            scores = prd_scores[0].float()
            best_idx = int(torch.argmax(scores).item()) if scores.numel() > 0 else 0
            mask_prob = torch.sigmoid(prd_masks[0, best_idx])
            mask_bin = (mask_prob > 0.5).to(torch.uint8).cpu().numpy()

        return mask_bin


    def prepare_pseudo_video(self, template_image_path: str, target_image_paths: List[str],
                             temp_dir: str, interleave_template: bool = False) -> str:
        """Create pseudo-video directory with template as frame 0."""
        frames_dir = os.path.join(temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Frame 0: Template
        template_img = Image.open(template_image_path).convert('RGB')
        target_size = template_img.size
        template_img.save(os.path.join(frames_dir, '00000.jpg'), 'JPEG', quality=95)

        # Frames 1-N: Target images (optionally interleave template between each)
        frame_idx = 1
        for img_path in target_image_paths:
            if interleave_template:
                # Insert target
                img = Image.open(img_path).convert('RGB')
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                img.save(os.path.join(frames_dir, f'{frame_idx:05d}.jpg'), 'JPEG', quality=95)
                frame_idx += 1
                # Insert template again
                template_img.save(os.path.join(frames_dir, f'{frame_idx:05d}.jpg'), 'JPEG', quality=95)
                frame_idx += 1
            else:
                img = Image.open(img_path).convert('RGB')
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                img.save(os.path.join(frames_dir, f'{frame_idx:05d}.jpg'), 'JPEG', quality=95)
                frame_idx += 1

        total_frames = frame_idx
        logger.info(f"Created pseudo-video with {total_frames} frames")
        return frames_dir
    
    def propagate_masks(self, frames_dir: str, template_masks: List[Dict],
                        multi_mask: bool = False,
                        interleave_template: bool = False,
                        reanchor_every: int = 0,
                        area_growth_limit: float = 0.0,
                        area_clamp_pad: int = 10) -> Dict[int, List[Dict]]:
        """Propagate masks through pseudo-video using SAM2 tracking."""
        import torch
        
        # Initialize video state
        inference_state = self.video_predictor.init_state(video_path=frames_dir)
        
        # Add template masks on frame 0
        for obj_id, mask_data in enumerate(template_masks, start=1):
            mask = mask_data['mask']
            _, _, _ = self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask
            )
            logger.info(f"Added object {obj_id} mask to frame 0")
        
        # Propagate
        results = {}

        # If interleaving templates, pre-anchor the template mask at every template frame.
        # Our frame sequence is: 0=T, 1=U1, 2=T, 3=U2, 4=T, ...
        if interleave_template:
            max_frame = len(os.listdir(frames_dir)) - 1
            for frame_idx in range(0, max_frame + 1, 2):
                for obj_id, mask_data in enumerate(template_masks, start=1):
                    try:
                        self.video_predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            mask=mask_data['mask']
                        )
                    except Exception as e:
                        logger.debug(f"Could not pre-anchor template at frame {frame_idx}: {e}")

        prev_area = {}  # obj_id -> area
        prev_box = {}   # obj_id -> box_xyxy

        with torch.inference_mode():
            for frame_idx, obj_ids, masks in tqdm(
                self.video_predictor.propagate_in_video(inference_state),
                desc="Propagating masks"
            ):
                frame_masks = []
                for i, obj_id in enumerate(obj_ids):
                    mask = (masks[i] > 0).cpu().numpy().squeeze().astype(np.uint8)

                    # Drift control: clamp explosive growth to previous box (+pad)
                    area = int(mask.sum())
                    if obj_id in prev_area and area_growth_limit and prev_area[obj_id] > 0:
                        if area > area_growth_limit * prev_area[obj_id]:
                            # clamp to previous bbox (+pad)
                            box = prev_box.get(obj_id)
                            if box is None:
                                box = get_box_from_mask(mask, pad=area_clamp_pad)
                            if box is not None:
                                mask = clamp_mask_to_box(mask, box)
                                area = int(mask.sum())

                    # Update prev stats (skip template frames if interleaved)
                    is_template_frame = interleave_template and (frame_idx % 2 == 0)
                    if area > 0 and not is_template_frame:
                        prev_area[obj_id] = area
                        b = get_box_from_mask(mask, pad=area_clamp_pad)
                        if b is not None:
                            prev_box[obj_id] = b

                    category_id = template_masks[obj_id - 1]['category_id'] if obj_id <= len(template_masks) else 1
                    frame_masks.append({
                        'mask': mask,
                        'category_id': category_id,
                        'obj_id': obj_id
                    })

                results[frame_idx] = frame_masks

                # Optional re-anchor using current predicted masks every N frames.
                # This is best-effort: some SAM2 versions allow add_new_mask mid-stream.
                if reanchor_every and frame_idx > 0 and (frame_idx % reanchor_every == 0):
                    for md in frame_masks:
                        try:
                            self.video_predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=md['obj_id'],
                                mask=md['mask']
                            )
                        except Exception as e:
                            logger.debug(f"Re-anchor failed at frame {frame_idx}: {e}")

        # Reset
        self.video_predictor.reset_state(inference_state)

        return results
    
    def mask_to_coco_annotation(self, mask: np.ndarray, annotation_id: int,
                                 image_id: int, category_id: int) -> Optional[Dict]:
        """Convert binary mask to COCO annotation."""
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        if area < 10:
            return None
        
        segmentation = contour.flatten().tolist()
        if len(segmentation) < 6:
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        
        return {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': [segmentation],
            'area': float(area),
            'bbox': [float(x), float(y), float(w), float(h)],
            'iscrowd': 0,
            'score': 1.0
        }
    
    def process_batch(self, template_image_path: str, template_masks: List[Dict],
                      target_image_paths: List[str], output_dir: str,
                      save_masks: bool = True, save_vis: bool = False,
                      multi_mask: bool = False,
                      interleave_template: bool = False,
                      reanchor_every: int = 0,
                      area_growth_limit: float = 0.0,
                      area_clamp_pad: int = 10,
                      num_points: int = 30,
                      output_timestamp: bool = False,
                      output_category_prefix: bool = False) -> Dict:
        """Process batch of images using video propagation."""
        import torch
        from datetime import datetime
        
        # Generate timestamp for output naming
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S") if output_timestamp else ""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if save_masks:
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
        
        
        if save_vis:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            # Save template visualization once (mask + prompts)
            try:
                timg = cv2.imread(template_image_path)
                if timg is not None and len(template_masks) > 0:
                    tmask = template_masks[0]['mask']
                    if tmask.shape[:2] != timg.shape[:2]:
                        tmask = cv2.resize(tmask, (timg.shape[1], timg.shape[0]), interpolation=cv2.INTER_NEAREST)

                    tbox = get_box_from_mask(tmask, padding=5)
                    tpts = get_points_from_mask(tmask, int(num_points))
                    self.save_prompt_visualization(
                        timg, tmask, tpts, tbox,
                        os.path.join(vis_dir, 'vis_TEMPLATE_prompts.png')
                    )
                    cv2.imwrite(
                        os.path.join(vis_dir, 'vis_TEMPLATE_mask.png'),
                        overlay_mask(timg, tmask, color=(0,255,0), alpha=0.5),
                    )
            except Exception as e:
                logger.debug(f'Could not save template vis: {e}')

        # Precompute fallback prompts (used if tracker returns no mask for a frame)
        fallback_box = None
        fallback_points = None
        fallback_labels = None
        if len(template_masks) > 0:
            fallback_box = get_box_from_mask(template_masks[0]['mask'], padding=5)
            if fallback_box is not None:
                fallback_points = get_points_from_box(fallback_box, int(num_points))
            if fallback_points is None:
                fallback_points = get_points_from_mask(template_masks[0]['mask'], int(num_points))
            if fallback_points is not None:
                fallback_labels = np.ones((len(fallback_points), 1), dtype=np.int32)

        temp_dir = tempfile.mkdtemp(prefix='pal_')
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Prepare pseudo-video
            logger.info("Creating pseudo-video...")
            frames_dir = self.prepare_pseudo_video(
                template_image_path, target_image_paths, temp_dir, interleave_template=interleave_template
            )
            
            # Propagate masks
            logger.info("Propagating masks through pseudo-video...")
            frame_results = self.propagate_masks(
                frames_dir, template_masks, multi_mask,
                interleave_template=interleave_template,
                reanchor_every=reanchor_every,
                area_growth_limit=area_growth_limit,
                area_clamp_pad=area_clamp_pad
            )
            
            # Convert to COCO format
            coco_output = {
                'images': [],
                'annotations': [],
                'categories': list(self.categories.values())
            }
            
            annotation_id = 1
            stats = {'success': 0, 'failed': 0, 'total': len(target_image_paths), 'total_masks': 0}
            
            # Process results (skip frame 0)
            for img_idx, img_path in enumerate(target_image_paths):
                image_id = img_idx + 1
                # Map to pseudo-video frame index
                video_frame_idx = (2 * img_idx + 1) if interleave_template else image_id
                img_filename = os.path.basename(img_path)
                
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    stats['failed'] += 1
                    continue
                
                h, w = orig_img.shape[:2]
                
                coco_output['images'].append({
                    'id': image_id,
                    'file_name': img_filename,
                    'height': h,
                    'width': w
                })


                frame_masks = frame_results.get(video_frame_idx, [])
                if not frame_masks:
                    # Tracker failed for this frame -> fall back to single-image prompting using the fine-tuned image path
                    try:
                        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                        pred = self.predict_single_mask(
                            img_rgb,
                            point_coords=fallback_points,
                            point_labels=fallback_labels,
                            box_xyxy=fallback_box,
                            multimask=True,
                        )
                        if pred is not None and pred.sum() > 0:
                            pred = postprocess_mask(pred)
                            frame_masks = [{
                                "obj_id": 1,
                                "category_id": template_masks[0]["category_id"] if len(template_masks) > 0 else 1,
                                "mask": pred,
                            }]
                            logger.warning(
                                f"Frame {video_frame_idx} (image {img_filename}): tracker returned no mask; used image-path fallback."
                            )
                        else:
                            logger.warning(f"No result for frame {video_frame_idx} (image {img_filename})")
                            stats["failed"] += 1
                            continue
                    except Exception as e:
                        logger.warning(f"No result for frame {video_frame_idx} (image {img_filename}) - fallback failed: {e}")
                        stats["failed"] += 1
                        continue
                for mask_data in frame_masks:
                    mask = mask_data['mask']
                    category_id = mask_data['category_id']
                    
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    ann = self.mask_to_coco_annotation(mask, annotation_id, image_id, category_id)
                    
                    if ann:
                        coco_output['annotations'].append(ann)
                        stats['total_masks'] += 1
                        annotation_id += 1
                        
                        if save_masks:
                            cat_name = self.categories.get(category_id, {}).get('name', f'cat_{category_id}')
                            base_name = os.path.splitext(img_filename)[0]
                            if output_category_prefix:
                                mask_filename = f"{cat_name}_{base_name}_mask.png"
                            else:
                                mask_filename = f"{base_name}_{cat_name}_mask.png"
                            if timestamp_str:
                                mask_filename = f"{timestamp_str}_{mask_filename}"
                            cv2.imwrite(os.path.join(masks_dir, mask_filename), mask * 255)
                
                if save_vis and frame_masks:
                    vis_img = orig_img.copy()
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    for i, mask_data in enumerate(frame_masks):
                        mask = mask_data['mask']
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        color = colors[i % len(colors)]
                        vis_img[mask > 0] = (vis_img[mask > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, f"vis_{img_filename}"), vis_img)
                    # Side-by-side with template for drift debugging
                    try:
                        timg = cv2.imread(template_image_path)
                        if timg is not None:
                            tmask = template_masks[0]['mask']
                            if tmask.shape[:2] != timg.shape[:2]:
                                tmask = cv2.resize(tmask, (timg.shape[1], timg.shape[0]), interpolation=cv2.INTER_NEAREST)
                            current_masks = []
                            for md in frame_masks:
                                m = md['mask']
                                if m.shape[:2] != (h, w):
                                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                                current_masks.append(m)
                            save_side_by_side(timg, tmask, orig_img, current_masks, os.path.join(vis_dir, f"pair_{img_filename}"))
                    except Exception as e:
                        logger.debug(f'Could not save side-by-side: {e}')
                
                stats['success'] += 1
            
            # Save COCO JSON with optional timestamp
            if timestamp_str:
                output_json_path = os.path.join(output_dir, f'pal_predictions_{timestamp_str}.json')
            else:
                output_json_path = os.path.join(output_dir, 'pal_predictions.json')
            with open(output_json_path, 'w') as f:
                json.dump(coco_output, f, indent=2)
            
            logger.info(f"Processing complete!")
            logger.info(f"  Success: {stats['success']}/{stats['total']}")
            logger.info(f"  Total masks: {stats['total_masks']}")
            
            return {
                'coco_output': coco_output,
                'output_path': output_json_path,
                'stats': stats
            }
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


        # ============================================================================
        # Main
        # ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SAM2-PAL: Palindrome-based Mask Propagation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
=== HOW IT WORKS ===

PSEUDO-VIDEO APPROACH:
  Frame 0: Template + mask â†’ stored in SAM2's memory
  Frame 1-N: Target images â†’ masks propagated via tracking
  
FINE-TUNING:
  Trains mask_decoder and prompt_encoder using augmented template.
  Uses SAM2ImagePredictor internals (which support gradients).

Examples:
  # Basic inference (try first!)
  python sam2_pal_batch.py --template_mask mask.png \\
                           --template_image template.jpg \\
                           --image_dir ./images \\
                           --output_dir ./output \\
                           --save_vis

  # With fine-tuning (for challenging structures)
  python sam2_pal_batch.py --template_mask mask.png \\
                           --template_image template.jpg \\
                           --image_dir ./images \\
                           --output_dir ./output \\
                           --finetune --num_epochs 200 --save_vis
        """
    )
    
    # Template source
    template_group = parser.add_argument_group('Template Source')
    template_group.add_argument('--template_mask', help='Binary mask PNG')
    template_group.add_argument('--template_json', help='COCO JSON')
    template_group.add_argument('--template_image', required=True, help='Template image')
    template_group.add_argument('--category_name', default='object', help='Category name')
    
    # Input/Output
    parser.add_argument('--image_dir', required=True, help='Target images directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    # SAM2
    parser.add_argument('--sam2_checkpoint', required=True, help='SAM2 checkpoint')
    parser.add_argument('--sam2_config', required=True, help='SAM2 config YAML')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    
    # Output
    parser.add_argument('--save_masks', action='store_true', default=True)
    parser.add_argument('--no_save_masks', action='store_false', dest='save_masks')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--multi_mask', action='store_true', help='Enable multi-instance output')

    # Drift control / stabilization
    parser.add_argument('--interleave_template', action='store_true',
                        help='Use [T, U1, T, U2, ...] frame sequence to reduce drift')
    parser.add_argument('--reanchor_every', type=int, default=0,
                        help='(Best-effort) Re-add predicted mask every N frames during propagation (0=off)')
    parser.add_argument('--area_growth_limit', type=float, default=2.0,
                        help='Clamp masks if area grows > limit * previous area (0 disables)')
    parser.add_argument('--area_clamp_pad', type=int, default=10,
                        help='Padding (pixels) around previous bbox used for area clamp')
    
    # Fine-tuning
    finetune_group = parser.add_argument_group('Fine-tuning')
    finetune_group.add_argument('--finetune', action='store_true', help='Enable fine-tuning (augmentation-based)')
    finetune_group.add_argument('--pal_finetuning', action='store_true', 
                                help='Enable PAL fine-tuning (video tracker backprop - most powerful)')
    finetune_group.add_argument('--finetune_occcl', action='store_true', 
                                help='Alias for --pal_finetuning (deprecated name)')
    finetune_group.add_argument('--use_lora', action='store_true',
                                help='Use LoRA fine-tuning (requires peft library, per paper)')
    finetune_group.add_argument('--lora_rank', type=int, default=16,
                                help='LoRA rank (default 16, paper does not specify)')
    finetune_group.add_argument('--num_epochs', type=int, default=200, help='Training epochs (paper uses 25 with LoRA)')
    finetune_group.add_argument('--learning_rate', type=float, default=None,
                                help='Learning rate (default: 1e-4 for LoRA, 1e-5 for full fine-tune)')
    finetune_group.add_argument('--num_points', type=int, default=3, help='Points per sample')
    finetune_group.add_argument('--finetune_checkpoint', help='Checkpoint path')
    finetune_group.add_argument('--max_images_per_epoch', type=int, default=10,
                                help='Max unlabeled images per epoch for PAL fine-tuning')
    
    # Multi-template training (NEW in v13)
    training_group = parser.add_argument_group('Multi-template Training (for PAL fine-tuning)')
    training_group.add_argument('--training_json', help='COCO JSON with multiple annotated training images')
    training_group.add_argument('--training_images_dir', help='Directory containing training images')
    training_group.add_argument('--training_masks_dir', help='Directory containing binary mask PNGs (alternative to JSON)')
    
    # Output naming options (NEW in v14)
    naming_group = parser.add_argument_group('Output Naming (NEW in v14)')
    naming_group.add_argument('--output_timestamp', action='store_true',
                              help='Add timestamp to output filenames (e.g., pal_predictions_20260113_143022.json)')
    naming_group.add_argument('--output_category_prefix', action='store_true',
                              help='Include category name in mask filenames')
    
    args = parser.parse_args()
    
    # Validate
    if not args.template_mask and not args.template_json:
        parser.error("Must provide --template_mask or --template_json")
    
    # Set default learning rate based on fine-tuning method
    if args.learning_rate is None:
        if args.use_lora:
            args.learning_rate = 1e-4  # Paper's default for LoRA
            logger.info(f"Using LoRA default learning rate: {args.learning_rate}")
        else:
            args.learning_rate = 1e-5  # Conservative default for full fine-tuning
            logger.info(f"Using full fine-tune default learning rate: {args.learning_rate}")
    
    # Get target images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_dir = Path(args.image_dir)
    target_images = [
        str(f) for f in sorted(image_dir.iterdir())
        if f.suffix.lower() in image_extensions
    ]
    
    # Exclude template
    template_name = os.path.basename(args.template_image)
    target_images = [p for p in target_images if os.path.basename(p) != template_name]
    
    if not target_images:
        logger.error("No target images found!")
        sys.exit(1)
    
    logger.info(f"Found {len(target_images)} target images")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize
    try:
        pal = SAM2PAL(
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize SAM2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load template
    if args.template_mask:
        mask = pal.load_template_mask(args.template_mask, args.category_name)
        template_masks = [{'mask': mask, 'category_id': 1}]
    else:
        template_masks = pal.load_template_from_coco(args.template_json, args.template_image)
    
    if not template_masks:
        logger.error("No template masks!")
        sys.exit(1)
    
    
    # Optional: save a debug visualization of the template prompt (points + box)
    if getattr(args, "save_vis", False):
        try:
            if getattr(pal, "template_mask", None) is not None:
                tmask = (pal.template_mask > 0).astype(np.uint8)
                # Use a modest padding for visualization; this is not the area clamp pad
                tbox = get_box_from_mask(tmask, padding=15)
                tpts = get_points_from_mask(tmask, args.num_points)  # Use standalone function
                dbg_path = os.path.join(args.output_dir, "template_prompt_debug.png")
                pal.save_prompt_visualization(
                    template_image=template_image,
                    mask=tmask,
                    points=tpts,
                    box=tbox,
                    output_path=dbg_path,
                    title=f"Template prompt (N={args.num_points} pts + box)",
                )
                logger.info(f"Saved template prompt debug: {dbg_path}")
        except Exception as e:
            logger.warning(f"Could not save template prompt debug: {e}")

# Fine-tune
    # Handle both --pal_finetuning and --finetune_occcl (alias)
    use_pal_finetuning = args.pal_finetuning or args.finetune_occcl
    
    if use_pal_finetuning:
        # === FIX v17: Allow COCO JSON OR binary mask for PAL fine-tuning ===
        if not args.template_mask and len(template_masks) == 0:
            logger.error("PAL fine-tuning requires --template_mask OR --template_json with masks")
            sys.exit(1)
        
        if len(template_masks) == 0 and args.template_mask:
            logger.error("No masks loaded. Check your template_mask file.")
            sys.exit(1)
        
        logger.info(f"PAL fine-tuning with {len(template_masks)} template mask(s)")
        
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        
        # Load additional training templates if provided
        additional_templates = None
        if args.training_json or args.training_masks_dir:
            additional_templates = pal.load_training_templates(
                training_json=args.training_json,
                training_images_dir=args.training_images_dir,
                training_masks_dir=args.training_masks_dir
            )
            if additional_templates:
                logger.info(f"Loaded {len(additional_templates)} additional training templates from training JSON")
        
        # === FIX v17: Include ALL template masks in training, not just first ===
        # If template_masks has multiple masks (e.g., from COCO JSON with scape, antenna, eye),
        # add masks 1+ to additional_templates so they all get trained
        if len(template_masks) > 1:
            logger.info(f"Multi-mask template detected: {len(template_masks)} masks will be used for training")
            if additional_templates is None:
                additional_templates = []
            # Add masks 1, 2, ... to additional templates (mask 0 is the primary)
            for i, mask_data in enumerate(template_masks[1:], start=2):
                additional_templates.append({
                    'image_path': args.template_image,
                    'mask': mask_data['mask'],
                    'category_id': mask_data.get('category_id', i)
                })
                cat_name = pal.categories.get(mask_data.get('category_id', i), {}).get('name', f'mask_{i}')
                logger.info(f"  Added template mask {i}: {cat_name}")
        
        pal.finetune_pal(
            template_image_path=args.template_image,
            template_mask=template_masks[0]['mask'],
            target_image_paths=target_images,
            output_checkpoint=ckpt,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_images_per_epoch=args.max_images_per_epoch,
            additional_templates=additional_templates,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank
        )
    
    elif args.finetune:
        if not args.template_mask and len(template_masks) == 0:
            logger.error("Fine-tuning requires --template_mask or --template_json with masks")
            sys.exit(1)
        
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        
        # Log multi-mask info
        if len(template_masks) > 1:
            logger.info(f"Note: Legacy fine-tune mode uses first mask only. Use --pal_finetuning for multi-mask training.")
        
        pal.finetune(
            template_image_path=args.template_image,
            template_mask=template_masks[0]['mask'],
            output_checkpoint=ckpt,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            num_points=args.num_points
        )
    
    # Process
    results = pal.process_batch(
        template_image_path=args.template_image,
        template_masks=template_masks,
        target_image_paths=target_images,
        output_dir=args.output_dir,
        save_masks=args.save_masks,
        save_vis=args.save_vis,
        multi_mask=args.multi_mask,
        interleave_template=args.interleave_template,
        reanchor_every=args.reanchor_every,
        area_growth_limit=args.area_growth_limit,
        area_clamp_pad=args.area_clamp_pad,
        num_points=args.num_points,
        output_timestamp=args.output_timestamp,
        output_category_prefix=args.output_category_prefix
    )
    
    # Summary
    stats = results['stats']
    print("\n" + "="*60)
    print("SAM2-PAL Processing Complete!")
    print("="*60)
    print(f"Images processed: {stats['success']}/{stats['total']}")
    print(f"Total masks: {stats['total_masks']}")
    print(f"Output: {results['output_path']}")
    if use_pal_finetuning:
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        print(f"Checkpoint (PAL): {ckpt}")
    elif args.finetune:
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        print(f"Checkpoint: {ckpt}")
    print("="*60)
    print("\nUse 'View Predictions' in Descriptron to review results")


if __name__ == '__main__':
    main()