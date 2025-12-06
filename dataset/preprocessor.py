# preprocessor.py - face detection, alignment and normalization
# tried a few different detectors before settling on this approach

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# standard face landmark positions for alignment
# these are based on the arcface paper - took me a while to get these right
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041]    # right mouth corner
], dtype=np.float32)


class FacePreprocessor:
    """
    handles all the face preprocessing steps:
    1. detection - find faces in image
    2. landmark extraction - get key points
    3. alignment - rotate and scale to standard position
    4. cropping - extract just the face region
    """
    
    def __init__(self, target_size: int = 256, detector_type: str = "mtcnn"):
        self.target_size = target_size
        self.detector_type = detector_type
        self.detector = None
        
        # lazy load detector - saves memory if not used
        self._init_detector()
    
    def _init_detector(self):
        """initialize face detector based on type"""
        if self.detector_type == "mtcnn":
            # mtcnn is slower but more accurate for varied poses
            try:
                from facenet_pytorch import MTCNN
                import torch
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.detector = MTCNN(
                    image_size=160,
                    margin=0,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],  # tuned for fewer false positives
                    factor=0.709,
                    post_process=False,
                    device=device,
                    keep_all=False  # only return largest face
                )
                logger.info("initialized mtcnn detector")
            except ImportError:
                logger.warning("facenet_pytorch not installed, falling back to opencv")
                self.detector_type = "opencv"
        
        if self.detector_type == "opencv":
            # opencv cascade is fast but less accurate
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            logger.info("initialized opencv cascade detector")
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        detect face and return bounding box + landmarks
        
        returns None if no face found - caller should handle this
        """
        if self.detector_type == "mtcnn":
            return self._detect_mtcnn(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """mtcnn detection - returns box and 5 landmarks"""
        try:
            # mtcnn expects rgb
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # convert to pil for mtcnn
            pil_image = Image.fromarray(image)
            
            boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                return None
            
            # take the detection with highest confidence
            best_idx = np.argmax(probs)
            box = boxes[best_idx].astype(np.int32)
            lms = landmarks[best_idx]
            
            return box, lms
            
        except Exception as e:
            logger.error(f"mtcnn detection failed: {e}")
            return None
    
    def _detect_opencv(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """opencv cascade detection - no landmarks, just box"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # take largest face
        areas = [w * h for (x, y, w, h) in faces]
        best_idx = np.argmax(areas)
        x, y, w, h = faces[best_idx]
        
        box = np.array([x, y, x + w, y + h])
        
        # estimate landmarks from box since opencv doesnt give them
        # this is rough but works ok for frontal faces
        landmarks = self._estimate_landmarks_from_box(box)
        
        return box, landmarks
    
    def _estimate_landmarks_from_box(self, box: np.ndarray) -> np.ndarray:
        """
        rough landmark estimation when detector doesnt provide them
        assumes frontal face - not great but better than nothing
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # approximate positions based on average face proportions
        landmarks = np.array([
            [x1 + w * 0.3, y1 + h * 0.35],   # left eye
            [x1 + w * 0.7, y1 + h * 0.35],   # right eye
            [x1 + w * 0.5, y1 + h * 0.55],   # nose
            [x1 + w * 0.35, y1 + h * 0.75],  # left mouth
            [x1 + w * 0.65, y1 + h * 0.75]   # right mouth
        ], dtype=np.float32)
        
        return landmarks
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                   output_size: int = 112) -> np.ndarray:
        """
        align face using similarity transform
        
        this is important for the identity encoder - arcface expects
        aligned faces in a specific format
        """
        # scale reference landmarks to output size
        scale = output_size / 112.0
        ref_lms = REFERENCE_LANDMARKS * scale
        
        # compute similarity transform matrix
        # using umeyama algorithm - more stable than simple affine
        tform = self._get_similarity_transform(landmarks, ref_lms)
        
        # apply transform
        aligned = cv2.warpAffine(
            image,
            tform,
            (output_size, output_size),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned
    
    def _get_similarity_transform(self, src_pts: np.ndarray, 
                                   dst_pts: np.ndarray) -> np.ndarray:
        """
        compute similarity transform (rotation, scale, translation)
        
        based on umeyama's method - handles reflection correctly
        """
        num_pts = src_pts.shape[0]
        dim = src_pts.shape[1]
        
        # center the points
        src_mean = src_pts.mean(axis=0)
        dst_mean = dst_pts.mean(axis=0)
        
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        
        # compute scale
        src_std = np.sqrt(np.sum(src_centered ** 2) / num_pts)
        dst_std = np.sqrt(np.sum(dst_centered ** 2) / num_pts)
        
        src_norm = src_centered / src_std
        dst_norm = dst_centered / dst_std
        
        # compute rotation using svd
        u, s, vt = np.linalg.svd(dst_norm.T @ src_norm)
        
        # handle reflection
        det = np.linalg.det(u @ vt)
        if det < 0:
            vt[-1, :] *= -1
        
        rotation = u @ vt
        scale = dst_std / src_std * np.sum(s)
        
        # build transform matrix
        transform = np.zeros((2, 3), dtype=np.float32)
        transform[:2, :2] = scale * rotation
        transform[:, 2] = dst_mean - scale * rotation @ src_mean
        
        return transform
    
    def process(self, image_path: str) -> Optional[Image.Image]:
        """
        full preprocessing pipeline for a single image
        
        returns aligned face as PIL image or None if no face found
        """
        # load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"could not load image: {image_path}")
            return None
        
        # detect face
        result = self.detect_face(image)
        if result is None:
            return None
        
        box, landmarks = result
        
        # align to standard position
        aligned = self.align_face(image, landmarks, self.target_size)
        
        # convert bgr to rgb for pil
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(aligned_rgb)
    
    def process_batch(self, image_paths: List[str]) -> List[Optional[Image.Image]]:
        """process multiple images - could parallelize this later"""
        results = []
        for path in image_paths:
            results.append(self.process(path))
        return results
    
    def extract_face_region(self, image: np.ndarray, box: np.ndarray, 
                            margin: float = 0.2) -> np.ndarray:
        """
        crop face region with margin
        
        margin adds extra space around the face - useful for
        preserving context like hair
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # clamp to image bounds
        img_h, img_w = image.shape[:2]
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(img_w, x2 + margin_x)
        y2 = min(img_h, y2 + margin_y)
        
        return image[y1:y2, x1:x2]
