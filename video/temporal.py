# temporal.py - temporal smoothing for video face swap
# reduces flickering and improves frame-to-frame consistency

import numpy as np
from collections import deque
from typing import Optional, Tuple
import cv2
import logging

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    smooths face swap results across video frames
    
    without this, the output tends to flicker because each frame
    is processed independently. this class maintains a buffer of
    recent frames and blends them together
    
    uses exponential moving average with optional landmark tracking
    """
    
    def __init__(self, window_size: int = 5, alpha: float = 0.8):
        # how many frames to keep in buffer
        self.window_size = window_size
        
        # blending weight - higher means more weight on current frame
        # 0.8 seems to work well - responsive but smooth
        self.alpha = alpha
        
        # frame buffer
        self.buffer = deque(maxlen=window_size)
        
        # for landmark-based smoothing
        self.prev_landmarks = None
        self.landmark_smoother = LandmarkSmoother(alpha=0.7)
    
    def reset(self):
        """clear buffer - call this when starting a new video"""
        self.buffer.clear()
        self.prev_landmarks = None
        self.landmark_smoother.reset()
    
    def smooth(self, frame: np.ndarray) -> np.ndarray:
        """
        apply temporal smoothing to frame
        
        blends current frame with previous frames using ema
        """
        # add to buffer
        self.buffer.append(frame.astype(np.float32))
        
        if len(self.buffer) == 1:
            # first frame, nothing to smooth with
            return frame
        
        # exponential moving average
        smoothed = self.buffer[-1].copy()
        weight = 1.0
        total_weight = weight
        
        for i in range(len(self.buffer) - 2, -1, -1):
            weight *= (1 - self.alpha)
            smoothed += weight * self.buffer[i]
            total_weight += weight
        
        smoothed /= total_weight
        
        return smoothed.astype(np.uint8)
    
    def smooth_with_flow(self, frame: np.ndarray, 
                         prev_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        smoother using optical flow for better motion handling
        
        aligns previous frame to current using flow before blending
        this handles camera motion better than simple averaging
        """
        if prev_frame is None or len(self.buffer) == 0:
            self.buffer.append(frame.astype(np.float32))
            return frame
        
        # compute optical flow
        prev_gray = cv2.cvtColor(self.buffer[-1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # warp previous frame using flow
        h, w = frame.shape[:2]
        flow_map = np.column_stack((
            np.tile(np.arange(w), h),
            np.repeat(np.arange(h), w)
        )).reshape(h, w, 2).astype(np.float32)
        
        flow_map += flow
        
        warped_prev = cv2.remap(
            self.buffer[-1].astype(np.uint8),
            flow_map[:, :, 0],
            flow_map[:, :, 1],
            cv2.INTER_LINEAR
        )
        
        # blend warped previous with current
        blended = self.alpha * frame + (1 - self.alpha) * warped_prev
        
        self.buffer.append(blended)
        
        return blended.astype(np.uint8)


class LandmarkSmoother:
    """
    smooths facial landmarks across frames
    
    jittery landmarks cause the face region to jump around
    smoothing them first gives more stable results
    """
    
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.prev_landmarks = None
    
    def reset(self):
        self.prev_landmarks = None
    
    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """apply ema smoothing to landmarks"""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            return landmarks
        
        # exponential moving average
        smoothed = self.alpha * landmarks + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        
        return smoothed


class ColorMatcher:
    """
    matches color distribution between frames
    
    helps maintain consistent skin tone when lighting changes
    uses histogram matching in lab color space
    """
    
    def __init__(self):
        self.reference_hist = None
    
    def set_reference(self, frame: np.ndarray):
        """set reference frame for color matching"""
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        self.reference_hist = self._compute_histogram(lab)
    
    def match(self, frame: np.ndarray) -> np.ndarray:
        """match frame colors to reference"""
        if self.reference_hist is None:
            return frame
        
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        matched_lab = self._match_histogram(lab)
        matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
        
        return matched
    
    def _compute_histogram(self, lab_image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """compute histogram for each lab channel"""
        hists = []
        for i in range(3):
            hist = cv2.calcHist([lab_image], [i], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # normalize
            hists.append(hist)
        return tuple(hists)
    
    def _match_histogram(self, lab_image: np.ndarray) -> np.ndarray:
        """match histogram to reference"""
        result = lab_image.copy()
        
        for i in range(3):
            # compute source histogram
            src_hist = cv2.calcHist([lab_image], [i], None, [256], [0, 256])
            src_hist = src_hist.flatten() / src_hist.sum()
            
            # compute cdfs
            src_cdf = np.cumsum(src_hist)
            ref_cdf = np.cumsum(self.reference_hist[i])
            
            # create lookup table
            lut = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                # find closest value in reference cdf
                idx = np.argmin(np.abs(ref_cdf - src_cdf[j]))
                lut[j] = idx
            
            # apply lut
            result[:, :, i] = cv2.LUT(lab_image[:, :, i], lut)
        
        return result


class FlickerReducer:
    """
    specifically targets high-frequency flicker
    
    uses frequency domain filtering to remove rapid changes
    that appear as flicker to human eyes
    """
    
    def __init__(self, buffer_size: int = 7):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def reset(self):
        self.buffer.clear()
    
    def reduce(self, frame: np.ndarray) -> np.ndarray:
        """apply flicker reduction"""
        self.buffer.append(frame.astype(np.float32))
        
        if len(self.buffer) < 3:
            return frame
        
        # median filter across time dimension
        # removes outlier frames that cause flicker
        stack = np.stack(list(self.buffer), axis=0)
        median = np.median(stack, axis=0)
        
        # blend with current frame to preserve detail
        alpha = 0.6
        result = alpha * frame + (1 - alpha) * median
        
        return result.astype(np.uint8)


class AdaptiveSmoother:
    """
    adjusts smoothing strength based on motion
    
    less smoothing when theres fast motion (to avoid blur)
    more smoothing when scene is static (to reduce noise)
    """
    
    def __init__(self, min_alpha: float = 0.5, max_alpha: float = 0.95):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.prev_frame = None
        self.smoothed = None
    
    def reset(self):
        self.prev_frame = None
        self.smoothed = None
    
    def smooth(self, frame: np.ndarray) -> np.ndarray:
        """adaptively smooth based on motion"""
        if self.prev_frame is None:
            self.prev_frame = frame.astype(np.float32)
            self.smoothed = frame.astype(np.float32)
            return frame
        
        # estimate motion as frame difference
        diff = np.abs(frame.astype(np.float32) - self.prev_frame)
        motion = np.mean(diff) / 255.0  # normalize to 0-1
        
        # map motion to alpha - more motion = higher alpha (less smoothing)
        alpha = self.min_alpha + motion * (self.max_alpha - self.min_alpha)
        alpha = np.clip(alpha, self.min_alpha, self.max_alpha)
        
        # apply smoothing
        self.smoothed = alpha * frame + (1 - alpha) * self.smoothed
        self.prev_frame = frame.astype(np.float32)
        
        return self.smoothed.astype(np.uint8)
