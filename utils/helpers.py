# helpers.py - misc utility functions used throughout the project
# nothing fancy here, just common operations i got tired of rewriting

import os
import random
import logging
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
import cv2


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    configure logging for the project
    
    outputs to both console and file if log_file is provided
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # quiet down some noisy libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def set_seed(seed: int = 42):
    """
    set random seeds for reproducibility
    
    not perfect but helps get consistent results across runs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # these slow things down but make results more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    get the best available device
    
    prefers cuda, falls back to mps (apple silicon), then cpu
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # log gpu info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"using gpu: {gpu_name} ({gpu_mem:.1f} GB)")
            return device
        
        # check for apple silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logging.info("using apple silicon gpu (mps)")
            return torch.device('mps')
    
    logging.info("using cpu")
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_image(path: Union[str, Path], size: Optional[Tuple[int, int]] = None,
               as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    load image from file
    
    handles various formats and optionally resizes
    returns rgb numpy array or tensor
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    
    # load with pil for consistent handling
    img = Image.open(path).convert('RGB')
    
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    
    img_array = np.array(img)
    
    if as_tensor:
        # convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        tensor = (tensor / 127.5) - 1.0
        return tensor
    
    return img_array


def save_image(image: Union[np.ndarray, torch.Tensor], path: Union[str, Path],
               quality: int = 95):
    """
    save image to file
    
    handles both numpy arrays and tensors
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(image, torch.Tensor):
        # convert tensor to numpy
        if image.dim() == 4:
            image = image[0]  # remove batch dim
        
        # denormalize from [-1, 1] to [0, 255]
        image = ((image + 1) * 127.5).clamp(0, 255)
        image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    # ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # save with pil
    pil_img = Image.fromarray(image)
    
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        pil_img.save(path, quality=quality)
    else:
        pil_img.save(path)


def resize_image(image: np.ndarray, size: Tuple[int, int],
                 keep_aspect: bool = True) -> np.ndarray:
    """
    resize image with optional aspect ratio preservation
    
    if keep_aspect is true, pads with black to reach target size
    """
    h, w = image.shape[:2]
    target_w, target_h = size
    
    if not keep_aspect:
        return cv2.resize(image, (target_w, target_h))
    
    # compute scale to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # create padded output
    output = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    
    # center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return output


def create_grid(images: list, nrow: int = 4, padding: int = 2) -> np.ndarray:
    """
    arrange multiple images into a grid
    
    useful for visualizing batches of results
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # ensure all images same size
    h, w = images[0].shape[:2]
    
    ncol = (len(images) + nrow - 1) // nrow
    
    grid_h = ncol * (h + padding) - padding
    grid_w = nrow * (w + padding) - padding
    
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        
        y = row * (h + padding)
        x = col * (w + padding)
        
        # resize if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        grid[y:y + h, x:x + w] = img
    
    return grid


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """convert pytorch tensor to numpy image"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # denormalize
    img = (tensor + 1) / 2
    img = img.clamp(0, 1)
    
    # to numpy
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    
    return img


def numpy_to_tensor(image: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """convert numpy image to pytorch tensor"""
    # normalize to [-1, 1]
    tensor = torch.from_numpy(image).float()
    tensor = (tensor / 127.5) - 1.0
    
    # add batch and channel dims if needed
    if tensor.dim() == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor.to(device)


class AverageMeter:
    """
    tracks running average of a value
    
    useful for tracking loss during training
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """simple timer for profiling"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        import time
        self.start_time = time.time()
    
    def stop(self) -> float:
        import time
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
