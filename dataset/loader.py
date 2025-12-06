# loader.py - pytorch dataset and dataloader for training
# handles pairing source and target faces for the swap task

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FaceSwapDataset(Dataset):
    """
    dataset for face swap training
    
    each sample contains:
    - source face (identity to transfer)
    - target face (pose/expression to preserve)
    - same_identity flag (for reconstruction loss)
    
    the tricky part is balancing same-identity pairs with different-identity
    pairs during training
    """
    
    def __init__(self, data_dir: str, image_size: int = 256, 
                 same_identity_ratio: float = 0.3, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.same_identity_ratio = same_identity_ratio
        self.augment = augment
        
        # load all images grouped by identity
        self.identity_images = self._load_identity_map()
        self.identities = list(self.identity_images.keys())
        
        # flatten for indexing
        self.all_images = []
        for identity, paths in self.identity_images.items():
            for path in paths:
                self.all_images.append((identity, path))
        
        # image transforms
        self.transform = self._get_transforms()
        
        logger.info(f"loaded {len(self.all_images)} images from {len(self.identities)} identities")
    
    def _load_identity_map(self) -> Dict[str, List[Path]]:
        """
        scan data directory and group images by identity
        
        expects folder structure: data_dir/identity_name/image.jpg
        or filename format: identity_xxx.jpg
        """
        identity_map = {}
        
        aligned_dir = self.data_dir / "aligned"
        if not aligned_dir.exists():
            # fallback to root data dir
            aligned_dir = self.data_dir
        
        # check for subdirectory structure first
        for item in aligned_dir.iterdir():
            if item.is_dir():
                # folder name is identity
                identity = item.name
                images = list(item.glob("*.jpg")) + list(item.glob("*.png"))
                if images:
                    identity_map[identity] = images
            elif item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # extract identity from filename
                identity = item.stem.split('_')[0]
                if identity not in identity_map:
                    identity_map[identity] = []
                identity_map[identity].append(item)
        
        # filter out identities with too few images
        # need at least 2 for same-identity pairs
        identity_map = {k: v for k, v in identity_map.items() if len(v) >= 2}
        
        return identity_map
    
    def _get_transforms(self) -> transforms.Compose:
        """build transform pipeline"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        
        if self.augment:
            # add augmentations for training
            # being careful not to distort faces too much
            transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(2, transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05  # small hue shift only
            ))
        
        # normalize to [-1, 1] range - works better for gans
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.all_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        get a training sample
        
        randomly decides whether to use same or different identity
        for source and target
        """
        target_identity, target_path = self.all_images[idx]
        
        # decide if same identity pair
        use_same = random.random() < self.same_identity_ratio
        
        if use_same and len(self.identity_images[target_identity]) > 1:
            # pick different image of same person
            source_path = random.choice([
                p for p in self.identity_images[target_identity] 
                if p != target_path
            ])
            source_identity = target_identity
            same_identity = True
        else:
            # pick random different identity
            other_identities = [i for i in self.identities if i != target_identity]
            source_identity = random.choice(other_identities)
            source_path = random.choice(self.identity_images[source_identity])
            same_identity = False
        
        # load and transform images
        source_img = self._load_image(source_path)
        target_img = self._load_image(target_path)
        
        return {
            'source': source_img,
            'target': target_img,
            'same_identity': torch.tensor(same_identity, dtype=torch.float32),
            'source_identity': source_identity,
            'target_identity': target_identity
        }
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """load and transform single image"""
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            logger.error(f"failed to load {path}: {e}")
            # return black image as fallback - not ideal but prevents crash
            return torch.zeros(3, self.image_size, self.image_size)


class InferenceDataset(Dataset):
    """
    simpler dataset for inference - just loads images without pairing
    """
    
    def __init__(self, image_paths: List[str], image_size: int = 256):
        self.image_paths = image_paths
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


def get_dataloader(data_dir: str, batch_size: int = 8, 
                   image_size: int = 256, num_workers: int = 4,
                   shuffle: bool = True, augment: bool = True) -> DataLoader:
    """
    convenience function to create dataloader with good defaults
    
    num_workers=4 seems to be sweet spot on most machines
    more workers doesnt help much and uses more memory
    """
    dataset = FaceSwapDataset(
        data_dir=data_dir,
        image_size=image_size,
        augment=augment
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # faster gpu transfer
        drop_last=True    # avoid weird batch sizes at end
    )
    
    return loader


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    convert from [-1,1] back to [0,1] for visualization
    
    use this before saving images or displaying
    """
    return (tensor + 1) / 2


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """convert tensor to numpy image for opencv/matplotlib"""
    # handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # denormalize and convert
    img = denormalize(tensor)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    
    return img
