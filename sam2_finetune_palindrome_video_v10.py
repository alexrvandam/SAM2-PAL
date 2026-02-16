#!/usr/bin/env python3
"""
SAM2 Palindrome Fine-Tuning for Video - Version 10.0 (Multi-object Fix)
CRITICAL BUG FIX: Instance-specific user points now properly associated with each mask!

=== KEY V10 FIX ===
V9 and earlier had a critical bug where user points were loaded per instance
but NEVER USED during prediction, causing multiple objects in the same frame
to produce identical or overlapping masks.

V10 fixes this by:
1. Using instance.user_points and instance.point_labels during prediction
2. Adding these points AFTER the mask to provide instance-specific guidance
3. Ensuring negative points properly constrain each mask independently

This fix is essential for tracking multiple objects (e.g., different anatomical
structures) in the same frame that need distinct boundaries.

Previous fixes:
- V9: Proper LoRA weight handling (don't merge during training)
- V8: Fixed LoRA-wrapped state_dict compatibility

Author: Descriptron Project (2025)
"""

import argparse
import json
import logging
import os
import sys
import random
import gc
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# GPU Memory Management
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# Video Extraction
# ============================================================================

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None
) -> Tuple[int, float]:
    """Extract frames from video file."""
    from tqdm import tqdm
    
    logger.info(f"Extracting frames from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if target_fps and target_fps < native_fps:
        frame_skip = int(native_fps / target_fps)
    else:
        frame_skip = 1
        target_fps = native_fps
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                output_path = os.path.join(output_dir, f"{saved_count:06d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    logger.info(f"Extracted {saved_count} frames")
    
    return saved_count, target_fps


def is_video_file(path: str) -> bool:
    """Check if path is a video file."""
    extensions = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.m4v', '.flv', '.wmv'}
    return Path(path).suffix.lower() in extensions


# ============================================================================
# COCO JSON Utilities
# ============================================================================

def load_coco_annotations(coco_path: str) -> Dict:
    """Load COCO JSON annotations."""
    logger.info(f"Loading COCO JSON from: {coco_path}")
    with open(coco_path, 'r') as f:
        data = json.load(f)
    
    if 'categories' in data:
        logger.info(f"Found {len(data['categories'])} categories:")
        for cat in data['categories']:
            logger.info(f"  ID {cat['id']}: {cat['name']}")
    
    if 'annotations' in data:
        logger.info(f"Found {len(data['annotations'])} annotations")
    
    return data


def get_keyframe_annotations(coco_data: Dict) -> Dict[int, List[Dict]]:
    """Extract keyframe annotations from COCO JSON."""
    image_id_to_frame = {}
    for img in coco_data.get('images', []):
        filename = img['file_name']
        try:
            stem = Path(filename).stem
            frame_idx = int(stem)
            image_id_to_frame[img['id']] = frame_idx
        except ValueError:
            image_id_to_frame[img['id']] = img['id']
    
    frame_annotations = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        frame_idx = image_id_to_frame.get(img_id, img_id)
        
        if frame_idx not in frame_annotations:
            frame_annotations[frame_idx] = []
        frame_annotations[frame_idx].append(ann)
    
    return frame_annotations


def polygon_to_mask(segmentation: List, height: int, width: int) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for seg in segmentation:
        if isinstance(seg, list) and len(seg) >= 6:
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    
    return mask


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """Convert binary mask to polygon segmentation."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    segmentations = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) >= 3:
            seg = contour.flatten().tolist()
            if len(seg) >= 6:
                segmentations.append(seg)
    
    return segmentations


def get_box_from_mask(mask: np.ndarray, pad: int = 5) -> Optional[np.ndarray]:
    """Get bounding box from mask."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    h, w = mask.shape
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)
    
    return np.array([x_min, y_min, x_max, y_max])


# ============================================================================
# Augmentation
# ============================================================================

def augment_image_and_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentations."""
    aug_img = image.copy()
    aug_mask = mask.copy()
    
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
        aug_mask = cv2.flip(aug_mask, 1)
    
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        h, w = aug_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h))
        aug_mask = cv2.warpAffine(aug_mask, M, (w, h))
    
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        aug_img = np.clip(aug_img * factor, 0, 255).astype(np.uint8)
    
    return aug_img, aug_mask


def get_points_from_mask(mask: np.ndarray, num_points: int = 3) -> Optional[np.ndarray]:
    """Sample points from mask."""
    coords = np.argwhere(mask > 0)
    if len(coords) < num_points:
        if len(coords) == 0:
            return None
        num_points = len(coords)
    
    indices = np.random.choice(len(coords), num_points, replace=False)
    points = coords[indices][:, [1, 0]]
    return points.reshape(-1, 1, 2)


# ============================================================================
# Instance Data Structure
# ============================================================================

class MaskInstance:
    """Single mask instance."""
    
    def __init__(self, instance_id: int, mask: np.ndarray, category_id: int,
                 frame_idx: int, user_points=None, point_labels=None):
        self.instance_id = instance_id
        self.mask = mask
        self.category_id = category_id
        self.frame_idx = frame_idx
        self.user_points = user_points
        self.point_labels = point_labels


# ============================================================================
# Streaming JSON Writer
# ============================================================================

class StreamingCOCOWriter:
    """Writes COCO JSON incrementally."""
    
    def __init__(self, output_path: str, categories: List[Dict], images: List[Dict]):
        self.output_path = output_path
        self.categories = categories
        self.images = images
        self.temp_path = output_path + '.tmp'
        self.ann_count = 0
        self.file = open(self.temp_path, 'w')
        self.first_annotation = True
        self.file.write('[\n')
    
    def write_annotation(self, ann: Dict):
        """Write a single annotation."""
        if not self.first_annotation:
            self.file.write(',\n')
        self.first_annotation = False
        json.dump(ann, self.file)
        self.ann_count += 1
        if self.ann_count % 100 == 0:
            self.file.flush()
    
    def finalize(self):
        """Finalize and create the complete COCO JSON."""
        self.file.write('\n]')
        self.file.close()
        
        with open(self.temp_path, 'r') as f:
            annotations = json.load(f)
        
        coco = {
            'info': {
                'description': 'SAM2 Palindrome V10 Predictions',
                'version': '9.0',
                'date_created': datetime.now().isoformat()
            },
            'images': self.images,
            'annotations': annotations,
            'categories': self.categories
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(coco, f)
        
        os.remove(self.temp_path)
        return self.ann_count


# ============================================================================
# SAM2 Palindrome V10 - Multi-object Fix + LoRA Handling
# ============================================================================

class SAM2VideoPalindromeV10:
    """SAM2 with proper LoRA weight handling."""
    
    def __init__(self, sam2_checkpoint: str, model_cfg: str, device: str = 'cuda'):
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(self.sam2_model)
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        
        logger.info("SAM2 initialized")
        
        self.categories = []
        self.category_id_to_name = {}
        self.mask_instances = []
        self.instance_to_category = {}
        self.use_lora = False
        self.peft_model = None
    
    def setup_lora(self, rank: int = 8):
        """Setup LoRA with proper tracking."""
        try:
            from peft import LoraConfig, get_peft_model
            import torch.nn as nn
            
            target_modules = []
            for name, module in self.sam2_model.named_modules():
                if isinstance(module, nn.Linear):
                    if 'mask_decoder' in name.lower() or 'prompt_encoder' in name.lower():
                        target_modules.append(name)
            
            if len(target_modules) > 20:
                target_modules = target_modules[:20]
            
            if target_modules:
                lora_config = LoraConfig(
                    r=rank, lora_alpha=rank * 2,
                    target_modules=target_modules,
                    lora_dropout=0.05, bias="none"
                )
                self.sam2_model = get_peft_model(self.sam2_model, lora_config)
                self.peft_model = self.sam2_model
                self.use_lora = True
                logger.info(f"LoRA enabled with rank={rank}, targeting {len(target_modules)} modules")
            
        except ImportError:
            logger.warning("peft not installed, using full fine-tuning")
        except Exception as e:
            logger.warning(f"LoRA setup failed: {e}")
    
    def load_keyframe_data(self, frame_dir: str, coco_data: Dict) -> List[Dict]:
        """Load keyframe data."""
        logger.info("Loading keyframe data...")
        
        keyframes = []
        frame_annotations = get_keyframe_annotations(coco_data)
        
        self.categories = coco_data.get('categories', [{'id': 1, 'name': 'object', 'supercategory': 'object'}])
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        
        logger.info(f"Categories: {self.category_id_to_name}")
        
        next_instance_id = 1
        self.mask_instances = []
        self.instance_to_category = {}
        
        for frame_idx, annotations in sorted(frame_annotations.items()):
            frame_path = os.path.join(frame_dir, f"{frame_idx:06d}.jpg")
            if not os.path.exists(frame_path):
                continue
            
            image = cv2.imread(frame_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            frame_instances = []
            
            for ann in annotations:
                if 'segmentation' not in ann:
                    continue
                
                mask = polygon_to_mask(ann['segmentation'], h, w)
                if mask.sum() == 0:
                    continue
                
                category_id = ann.get('category_id', 1)
                
                user_points = None
                point_labels = None
                if 'points' in ann and ann['points']:
                    user_points = np.array(ann['points'])
                    point_labels = np.array(ann.get('point_labels', [1] * len(ann['points'])), dtype=np.int32)
                
                instance = MaskInstance(
                    instance_id=next_instance_id,
                    mask=mask,
                    category_id=category_id,
                    frame_idx=frame_idx,
                    user_points=user_points,
                    point_labels=point_labels
                )
                
                frame_instances.append(instance)
                self.mask_instances.append(instance)
                self.instance_to_category[next_instance_id] = category_id
                
                cat_name = self.category_id_to_name.get(category_id, f"cat_{category_id}")
                logger.info(f"  Instance {next_instance_id}: frame={frame_idx}, cat={category_id} ({cat_name})")
                
                next_instance_id += 1
            
            if frame_instances:
                keyframes.append({
                    'frame_idx': frame_idx,
                    'image': image_rgb,
                    'instances': frame_instances
                })
        
        logger.info(f"Loaded {len(keyframes)} keyframes, {len(self.mask_instances)} instances")
        return keyframes
    
    def _save_training_checkpoint(self, output_path: str, iou: float):
        """
        Save checkpoint DURING training - does NOT merge LoRA!
        This preserves the gradient computation graph for continued training.
        """
        # Just save raw state dict - don't touch LoRA!
        checkpoint = {
            'model_state_dict': self.sam2_model.state_dict(),
            'iou': iou,
            'categories': self.categories,
            'category_id_to_name': self.category_id_to_name,
            'lora_merged': False,
            'version': 'v10_training'
        }
        torch.save(checkpoint, output_path)
        logger.info(f"  💾 Saved training checkpoint (IoU: {iou:.4f})")
    
    def _save_final_checkpoint(self, output_path: str, iou: float):
        """
        Save FINAL checkpoint AFTER training - merges LoRA weights.
        Called only ONCE after training loop completes.
        """
        logger.info("\n" + "="*60)
        logger.info("📦 Preparing FINAL checkpoint...")
        logger.info("="*60)
        
        if self.use_lora and self.peft_model is not None:
            try:
                logger.info("Merging LoRA weights into base model...")
                
                # merge_and_unload merges LoRA weights and returns clean model
                merged_model = self.peft_model.merge_and_unload()
                
                # Extract weights
                prompt_encoder_state = {}
                mask_decoder_state = {}
                full_state_dict = {}
                
                for name, param in merged_model.state_dict().items():
                    # Clean up key names (remove any remaining prefixes)
                    clean_name = name.replace('base_model.model.', '').replace('base_model.', '')
                    full_state_dict[clean_name] = param.cpu()
                    
                    if 'sam_prompt_encoder' in clean_name or 'prompt_encoder' in clean_name:
                        prompt_encoder_state[clean_name] = param.cpu()
                    elif 'sam_mask_decoder' in clean_name or 'mask_decoder' in clean_name:
                        mask_decoder_state[clean_name] = param.cpu()
                
                checkpoint = {
                    'model_state_dict': full_state_dict,
                    'prompt_encoder_state': prompt_encoder_state,
                    'mask_decoder_state': mask_decoder_state,
                    'iou': iou,
                    'categories': self.categories,
                    'category_id_to_name': self.category_id_to_name,
                    'lora_merged': True,
                    'version': 'v10'
                }
                
                logger.info(f"  ✅ prompt_encoder: {len(prompt_encoder_state)} params")
                logger.info(f"  ✅ mask_decoder: {len(mask_decoder_state)} params")
                logger.info(f"  ✅ total: {len(full_state_dict)} params")
                
                # Update reference (training is done, safe to do this now)
                self.sam2_model = merged_model
                self.use_lora = False
                
            except Exception as e:
                logger.warning(f"LoRA merge failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback - save raw state with cleaned keys
                raw_state = {}
                for name, param in self.sam2_model.state_dict().items():
                    clean_name = name.replace('base_model.model.', '').replace('base_model.', '')
                    raw_state[clean_name] = param.cpu()
                
                checkpoint = {
                    'model_state_dict': raw_state,
                    'iou': iou,
                    'categories': self.categories,
                    'category_id_to_name': self.category_id_to_name,
                    'lora_merged': False,
                    'version': 'v9_fallback'
                }
        else:
            # No LoRA - save normally
            state_dict = {}
            prompt_encoder_state = {}
            mask_decoder_state = {}
            
            for name, param in self.sam2_model.state_dict().items():
                state_dict[name] = param.cpu()
                if 'sam_prompt_encoder' in name or 'prompt_encoder' in name:
                    prompt_encoder_state[name] = param.cpu()
                elif 'sam_mask_decoder' in name or 'mask_decoder' in name:
                    mask_decoder_state[name] = param.cpu()
            
            checkpoint = {
                'model_state_dict': state_dict,
                'prompt_encoder_state': prompt_encoder_state,
                'mask_decoder_state': mask_decoder_state,
                'iou': iou,
                'categories': self.categories,
                'category_id_to_name': self.category_id_to_name,
                'lora_merged': True,
                'version': 'v10'
            }
        
        torch.save(checkpoint, output_path)
        logger.info(f"✅ Saved FINAL checkpoint to: {output_path}")
    
    def finetune_palindrome(self, frame_dir: str, keyframes: List[Dict],
                            output_checkpoint: str, num_epochs: int = 25,
                            learning_rate: float = 1e-4, use_lora: bool = True,
                            lora_rank: int = 8) -> Dict:
        """Fine-tuning loop with proper checkpoint saving."""
        if use_lora:
            self.setup_lora(lora_rank)
        
        if not self.use_lora:
            for param in self.sam2_model.parameters():
                param.requires_grad = False
            for name, param in self.sam2_model.named_parameters():
                if 'mask_decoder' in name.lower() or 'prompt_encoder' in name.lower():
                    param.requires_grad = True
        
        trainable_params = [p for p in self.sam2_model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = torch.amp.GradScaler('cuda')
        
        best_iou = 0
        mean_iou = 0
        
        logger.info(f"Starting fine-tuning: {num_epochs} epochs, {len(self.mask_instances)} instances")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_iou = 0
            num_samples = 0
            
            for kf_data in keyframes:
                for instance in kf_data['instances']:
                    aug_img, aug_mask = augment_image_and_mask(kf_data['image'].copy(), instance.mask.copy())
                    
                    if np.sum(aug_mask > 0) < 50:
                        continue
                    
                    input_points = get_points_from_mask(aug_mask, num_points=3)
                    if input_points is None:
                        continue
                    input_labels = np.ones((3,), dtype=np.int32)
                    input_box = get_box_from_mask(aug_mask)
                    
                    with torch.amp.autocast('cuda'):
                        self.image_predictor.set_image(aug_img)
                        
                        mask_input, unnorm_coords, labels, unnorm_box = \
                            self.image_predictor._prep_prompts(
                                input_points, input_labels, box=input_box,
                                mask_logits=None, normalize_coords=True
                            )
                        
                        if unnorm_coords is None:
                            continue
                        
                        dev = self.device
                        unnorm_coords = torch.as_tensor(unnorm_coords, device=dev).float()
                        labels = torch.as_tensor(labels, device=dev).long()
                        if unnorm_box is not None:
                            unnorm_box = torch.as_tensor(unnorm_box, device=dev).float()
                        
                        if labels.dim() == 2:
                            labels = labels.squeeze(1)
                        if labels.dim() == 1:
                            labels = labels.unsqueeze(0)
                        if unnorm_coords.dim() == 3 and unnorm_coords.shape[1] == 1:
                            unnorm_coords = unnorm_coords.squeeze(1)
                        if unnorm_coords.dim() == 2:
                            unnorm_coords = unnorm_coords.unsqueeze(0)
                        
                        if unnorm_box is not None:
                            if unnorm_box.dim() == 1:
                                x1, y1, x2, y2 = unnorm_box.tolist()
                                unnorm_box = torch.tensor([[[x1, y1], [x2, y2]]], device=dev).float()
                        
                        sparse_emb, dense_emb = self.sam2_model.sam_prompt_encoder(
                            points=(unnorm_coords, labels), boxes=unnorm_box, masks=None
                        )
                        
                        high_res_features = [
                            feat_level[-1].unsqueeze(0)
                            for feat_level in self.image_predictor._features["high_res_feats"]
                        ]
                        
                        low_res_masks, prd_scores, _, _ = self.sam2_model.sam_mask_decoder(
                            image_embeddings=self.image_predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=True,
                            repeat_image=False,
                            high_res_features=high_res_features,
                        )
                        
                        prd_masks = self.image_predictor._transforms.postprocess_masks(
                            low_res_masks, self.image_predictor._orig_hw[-1]
                        )
                        
                        gt_mask = torch.from_numpy(aug_mask.astype(np.float32)).to(prd_masks.device)
                        if gt_mask.ndim == 2:
                            gt_mask = gt_mask.unsqueeze(0)
                        
                        prd_mask = torch.sigmoid(prd_masks[:, 0])
                        if prd_mask.ndim == 2:
                            prd_mask = prd_mask.unsqueeze(0)
                        
                        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)).mean()
                        
                        prd_bin = (prd_mask > 0.5).float()
                        inter = (gt_mask * prd_bin).sum(dim=(-1, -2))
                        union = gt_mask.sum(dim=(-1, -2)) + prd_bin.sum(dim=(-1, -2)) - inter
                        iou = inter / (union + 1e-6)
                        
                        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                        loss = seg_loss + score_loss * 0.05
                    
                    scaler.scale(loss).backward()
                    epoch_loss += loss.item()
                    epoch_iou += iou.mean().item()
                    num_samples += 1
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            if num_samples > 0:
                mean_iou = mean_iou * 0.9 + (epoch_iou / num_samples) * 0.1
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs} - IoU: {mean_iou:.4f}")
                    print(f"PROGRESS: {(epoch + 1) / num_epochs * 100:.1f}%")
                    clear_gpu_memory()
                
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    # Save training checkpoint (NO MERGE - preserves gradients!)
                    self._save_training_checkpoint(output_checkpoint, best_iou)
        
        logger.info(f"\n✅ Fine-tuning Complete! Best IoU: {best_iou:.4f}")
        
        # NOW merge and save final checkpoint (training is done)
        self._save_final_checkpoint(output_checkpoint, best_iou)
        
        # Transfer weights to video predictor
        self._transfer_to_video_predictor()
        
        return {'best_iou': best_iou}
    
    def _transfer_to_video_predictor(self):
        """Transfer trained weights to video predictor."""
        logger.info("Transferring weights to video predictor...")
        
        for attr in ['sam_prompt_encoder', 'sam_mask_decoder']:
            if hasattr(self.video_predictor, attr) and hasattr(self.sam2_model, attr):
                try:
                    src_state = getattr(self.sam2_model, attr).state_dict()
                    getattr(self.video_predictor, attr).load_state_dict(src_state, strict=False)
                    logger.info(f"  ✅ Transferred {attr}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not transfer {attr}: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint with v10 merged weights support."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        if 'prompt_encoder_state' in ckpt and 'mask_decoder_state' in ckpt:
            logger.info("Loading v10 format checkpoint...")
            
            if hasattr(self.video_predictor, 'sam_prompt_encoder'):
                try:
                    self.video_predictor.sam_prompt_encoder.load_state_dict(
                        ckpt['prompt_encoder_state'], strict=False
                    )
                    logger.info("  ✅ Loaded prompt_encoder")
                except Exception as e:
                    logger.warning(f"  ⚠️ prompt_encoder: {e}")
            
            if hasattr(self.video_predictor, 'sam_mask_decoder'):
                try:
                    self.video_predictor.sam_mask_decoder.load_state_dict(
                        ckpt['mask_decoder_state'], strict=False
                    )
                    logger.info("  ✅ Loaded mask_decoder")
                except Exception as e:
                    logger.warning(f"  ⚠️ mask_decoder: {e}")
        else:
            logger.info("Loading full state dict...")
            state_dict = ckpt.get('model_state_dict', ckpt)
            self.video_predictor.load_state_dict(state_dict, strict=False)
        
        if 'categories' in ckpt:
            self.categories = ckpt['categories']
        if 'category_id_to_name' in ckpt:
            self.category_id_to_name = ckpt['category_id_to_name']
    
    def predict_video_streaming(self, frame_dir: str, keyframes: List[Dict], output_path: str) -> int:
        """Run prediction with streaming output."""
        logger.info("\n" + "="*60)
        logger.info("🎬 VIDEO PREDICTION (V10 - Streaming)")
        logger.info("="*60)
        
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        total_frames = len(frame_files)
        
        first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
        img_h, img_w = first_frame.shape[:2]
        del first_frame
        
        images = []
        for frame_idx, frame_file in enumerate(frame_files):
            images.append({
                'id': frame_idx + 1,
                'file_name': frame_file,
                'width': img_w,
                'height': img_h
            })
        
        writer = StreamingCOCOWriter(output_path, self.categories, images)
        
        logger.info("Initializing inference state...")
        try:
            inference_state = self.video_predictor.init_state(
                video_path=frame_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False
            )
        except TypeError:
            inference_state = self.video_predictor.init_state(video_path=frame_dir)
        
        # Use CATEGORY as obj_id
        cat_to_obj = {}
        next_obj_id = 1
        for inst in self.mask_instances:
            if inst.category_id not in cat_to_obj:
                cat_to_obj[inst.category_id] = next_obj_id
                next_obj_id += 1
        
        obj_to_cat = {v: k for k, v in cat_to_obj.items()}
        
        logger.info(f"Category → Object mapping: {cat_to_obj}")
        
        has_add_new_mask = hasattr(self.video_predictor, 'add_new_mask')
        
        logger.info("\nAdding mask prompts...")
        for kf_data in keyframes:
            frame_idx = kf_data['frame_idx']
            
            for instance in kf_data['instances']:
                obj_id = cat_to_obj[instance.category_id]
                cat_name = self.category_id_to_name.get(instance.category_id, f"cat_{instance.category_id}")
                
                mask = instance.mask.astype(np.float32)
                if mask.max() > 1.0:
                    mask = mask / 255.0
                
                if has_add_new_mask:
                    try:
                        self.video_predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            mask=mask
                        )
                        
                        mask_area = int(instance.mask.sum())
                        logger.info(f"  Frame {frame_idx}: obj_id={obj_id}, cat={instance.category_id} ({cat_name}), MASK prompt (area={mask_area} px)")
                        
                        # V10 FIX: Add instance-specific user points if they exist
                        if instance.user_points is not None and len(instance.user_points) > 0:
                            n_pos = np.sum(instance.point_labels == 1) if instance.point_labels is not None else len(instance.user_points)
                            n_neg = np.sum(instance.point_labels == 0) if instance.point_labels is not None else 0
                            logger.info(f"    + {len(instance.user_points)} user points ({n_pos} pos, {n_neg} neg)")
                            
                            pts_labels = instance.point_labels if instance.point_labels is not None else np.ones(len(instance.user_points), dtype=np.int32)
                            self.video_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                points=instance.user_points,
                                labels=pts_labels
                            )
                        
                        continue
                    except Exception as e:
                        logger.warning(f"add_new_mask failed: {e}")
                
                # Fallback: use point-based prompts
                all_points = []
                all_labels = []
                
                # Sample points from mask
                coords = np.argwhere(instance.mask > 0)
                if len(coords) > 0:
                    n_pts = min(10, len(coords))
                    indices = np.random.choice(len(coords), n_pts, replace=False)
                    pts = coords[indices][:, [1, 0]]
                    all_points.append(pts)
                    all_labels.append(np.ones(n_pts, dtype=np.int32))
                    logger.info(f"  Frame {frame_idx}: obj_id={obj_id}, cat={instance.category_id} ({cat_name}), {n_pts} mask points")
                
                # V10 FIX: Add instance-specific user points if they exist
                if instance.user_points is not None and len(instance.user_points) > 0:
                    all_points.append(instance.user_points)
                    pts_labels = instance.point_labels if instance.point_labels is not None else np.ones(len(instance.user_points), dtype=np.int32)
                    all_labels.append(pts_labels)
                    
                    n_pos = np.sum(pts_labels == 1)
                    n_neg = np.sum(pts_labels == 0)
                    logger.info(f"    + {len(instance.user_points)} user points ({n_pos} pos, {n_neg} neg)")
                
                # Combine and send to SAM2
                if all_points:
                    combined_pts = np.vstack(all_points)
                    combined_labels = np.concatenate(all_labels)
                    
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=combined_pts,
                        labels=combined_labels,
                        box=get_box_from_mask(instance.mask)
                    )
        
        logger.info(f"\nPropagating {total_frames} frames...")
        
        ann_id = 1
        frame_count = 0
        
        clear_gpu_memory()
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            
            if isinstance(out_mask_logits, dict):
                for obj_id, mask_logits in out_mask_logits.items():
                    mask = (mask_logits > 0).squeeze().cpu().numpy().astype(np.uint8)
                    del mask_logits
                    
                    if mask.sum() > 0:
                        segmentation = mask_to_polygon(mask)
                        
                        if segmentation:
                            coords = np.argwhere(mask > 0)
                            y_min, x_min = coords.min(axis=0)
                            y_max, x_max = coords.max(axis=0)
                            area = float(mask.sum())
                            
                            del mask
                            del coords
                            
                            cat_id = obj_to_cat.get(obj_id, 1)
                            
                            writer.write_annotation({
                                'id': ann_id,
                                'image_id': out_frame_idx + 1,
                                'category_id': cat_id,
                                'segmentation': segmentation,
                                'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                                'area': area,
                                'iscrowd': 0
                            })
                            ann_id += 1
                        else:
                            del mask
            else:
                for obj_idx, obj_id in enumerate(out_obj_ids):
                    if obj_idx >= len(out_mask_logits):
                        continue
                    
                    mask_logits = out_mask_logits[obj_idx]
                    mask = (mask_logits > 0).squeeze().cpu().numpy().astype(np.uint8)
                    del mask_logits
                    
                    if mask.sum() > 0:
                        segmentation = mask_to_polygon(mask)
                        
                        if segmentation:
                            coords = np.argwhere(mask > 0)
                            y_min, x_min = coords.min(axis=0)
                            y_max, x_max = coords.max(axis=0)
                            area = float(mask.sum())
                            
                            del mask
                            del coords
                            
                            cat_id = obj_to_cat.get(obj_id, 1)
                            
                            writer.write_annotation({
                                'id': ann_id,
                                'image_id': out_frame_idx + 1,
                                'category_id': cat_id,
                                'segmentation': segmentation,
                                'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                                'area': area,
                                'iscrowd': 0
                            })
                            ann_id += 1
                        else:
                            del mask
            
            frame_count += 1
            
            if frame_count % 20 == 0:
                clear_gpu_memory()
                pct = frame_count / total_frames * 100
                print(f"  Frame {frame_count}/{total_frames} ({pct:.0f}%) - {ann_id-1} annotations")
        
        total_anns = writer.finalize()
        logger.info(f"\n✅ Saved {total_anns} annotations to {output_path}")
        
        return total_anns


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SAM2 Palindrome V10 (Multi-object Fix)')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='Input video file')
    input_group.add_argument('--video_dir', help='Directory with frames')
    
    parser.add_argument('--annotations', '-a', required=True, help='COCO JSON')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    parser.add_argument('--export_predictions', help='Output COCO JSON')
    parser.add_argument('--checkpoint', help='Existing checkpoint')
    parser.add_argument('--skip_training', action='store_true')
    
    parser.add_argument('--sam2_checkpoint', default='./checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_cfg', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.input:
        if not is_video_file(args.input):
            sys.exit(1)
        frame_dir = os.path.join(args.output_dir, 'frames')
        extract_frames_from_video(args.input, frame_dir)
    else:
        frame_dir = args.video_dir
    
    coco_data = load_coco_annotations(args.annotations)
    
    sam2 = SAM2VideoPalindromeV10(
        sam2_checkpoint=args.sam2_checkpoint,
        model_cfg=args.model_cfg,
        device=args.device
    )
    
    keyframes = sam2.load_keyframe_data(frame_dir, coco_data)
    if not keyframes:
        logger.error("No keyframes found!")
        sys.exit(1)
    
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, 'sam2_palindrome_v10.pt')
    
    try:
        ckpt_txt = Path(output_dir) / "checkpoint_path.txt"
        ckpt_txt.write_text(str(checkpoint_path) + "\n", encoding="utf-8")
    except:
        pass
    
    if not args.skip_training:
        sam2.finetune_palindrome(
            frame_dir=frame_dir,
            keyframes=keyframes,
            output_checkpoint=checkpoint_path,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank
        )
        clear_gpu_memory()
    elif args.checkpoint:
        sam2._load_checkpoint(args.checkpoint)
    
    if args.export_predictions:
        output_json = args.export_predictions
        if not os.path.isabs(output_json):
            output_json = os.path.join(args.output_dir, output_json)
        
        sam2.predict_video_streaming(frame_dir, keyframes, output_json)
    
    print("\n" + "="*60)
    print("✅ SAM2 Palindrome V10 Complete")
    print("="*60)
    print(f"Categories: {list(sam2.category_id_to_name.values())}")
    print("\nV10 KEY FIXES:")
    print("  ✓ Instance-specific user points now properly used (multi-object fix!)")
    print("  ✓ Negative points correctly constrain each mask independently")
    print("  ✓ LoRA weights saved RAW during training (preserves gradients)")
    print("  ✓ LoRA weights MERGED only at end of training")
    print("  ✓ Checkpoint compatible with vanilla SAM2 predictors")
    print("="*60)


if __name__ == '__main__':
    main()
