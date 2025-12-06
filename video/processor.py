# processor.py - video face swap processing
# handles frame extraction, processing, and reconstruction

import os
import tempfile
from pathlib import Path
from typing import Optional, Callable, Generator, Tuple
import logging

import cv2
import numpy as np
from tqdm import tqdm

from api.inference import FaceSwapInference
from .temporal import TemporalSmoother

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    processes videos for face swapping
    
    pipeline:
    1. extract frames from video
    2. detect and swap face in each frame
    3. apply temporal smoothing to reduce flicker
    4. reconstruct video with audio
    """
    
    def __init__(self, inference_engine: FaceSwapInference, 
                 temporal_smooth: bool = True):
        self.inference = inference_engine
        self.temporal_smooth = temporal_smooth
        
        if temporal_smooth:
            self.smoother = TemporalSmoother()
        
        # temp directory for intermediate files
        self.temp_dir = Path(tempfile.gettempdir()) / "face_swap_video"
        self.temp_dir.mkdir(exist_ok=True)
    
    def process_video(self, source_image: np.ndarray, video_path: str,
                      output_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        swap face in entire video
        
        source_image: face to transfer (numpy array)
        video_path: path to input video
        output_path: where to save result
        progress_callback: optional function called with (current_frame, total_frames)
        """
        # open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"could not open video: {video_path}")
            return False
        
        # get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"processing video: {total_frames} frames at {fps} fps")
        
        # set up video writer
        temp_video = str(self.temp_dir / "temp_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        # reset smoother for new video
        if self.temporal_smooth:
            self.smoother.reset()
        
        # precompute source identity once
        source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        
        # process frames
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # swap face in this frame
            processed = self._process_frame(source_rgb, frame)
            
            # apply temporal smoothing
            if self.temporal_smooth and processed is not None:
                processed = self.smoother.smooth(processed)
            
            # if swap failed, use original frame
            if processed is None:
                processed = frame
            else:
                # convert back to bgr for video
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                # resize to match original if needed
                if processed.shape[:2] != (height, width):
                    processed = cv2.resize(processed, (width, height))
            
            writer.write(processed)
            
            frame_idx += 1
            pbar.update(1)
            
            if progress_callback:
                progress_callback(frame_idx, total_frames)
        
        pbar.close()
        cap.release()
        writer.release()
        
        # add audio from original video
        self._merge_audio(video_path, temp_video, output_path)
        
        # cleanup temp file
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        logger.info(f"video saved to {output_path}")
        return True
    
    def _process_frame(self, source: np.ndarray, frame: np.ndarray) -> Optional[np.ndarray]:
        """process single frame"""
        # convert frame to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # run face swap
        result, _ = self.inference.swap(source, frame_rgb)
        
        return result
    
    def _merge_audio(self, original_video: str, processed_video: str, output_path: str):
        """
        combine processed video with audio from original
        
        uses ffmpeg for this - more reliable than moviepy for audio sync
        """
        try:
            import subprocess
            
            # ffmpeg command to copy audio from original to processed
            cmd = [
                'ffmpeg', '-y',
                '-i', processed_video,
                '-i', original_video,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',  # ? makes audio optional
                '-shortest',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"ffmpeg warning: {result.stderr}")
                # fallback - just copy video without audio
                import shutil
                shutil.copy(processed_video, output_path)
                
        except FileNotFoundError:
            logger.warning("ffmpeg not found, output will have no audio")
            import shutil
            shutil.copy(processed_video, output_path)
    
    def process_video_streaming(self, source_image: np.ndarray, 
                                 video_path: str) -> Generator[np.ndarray, None, None]:
        """
        generator that yields processed frames one at a time
        
        useful for real-time display or streaming output
        """
        cap = cv2.VideoCapture(video_path)
        
        if self.temporal_smooth:
            self.smoother.reset()
        
        source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = self._process_frame(source_rgb, frame)
            
            if self.temporal_smooth and processed is not None:
                processed = self.smoother.smooth(processed)
            
            if processed is None:
                processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            yield processed
        
        cap.release()
    
    def get_video_info(self, video_path: str) -> dict:
        """get basic info about a video file"""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info


class RealtimeProcessor:
    """
    real-time face swap from webcam or video stream
    
    optimized for low latency - skips frames if processing is slow
    """
    
    def __init__(self, inference_engine: FaceSwapInference):
        self.inference = inference_engine
        self.smoother = TemporalSmoother(window_size=3)  # smaller window for real-time
        
        self.source_image = None
        self.running = False
    
    def set_source(self, image: np.ndarray):
        """set the source face to swap onto video"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # assume bgr, convert to rgb
            self.source_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            self.source_image = image
    
    def process_webcam(self, camera_id: int = 0, display: bool = True):
        """
        process webcam feed in real-time
        
        press 'q' to quit, 's' to save screenshot
        """
        if self.source_image is None:
            logger.error("set source image first with set_source()")
            return
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"could not open camera {camera_id}")
            return
        
        self.running = True
        self.smoother.reset()
        
        logger.info("starting real-time processing (press 'q' to quit)")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, latency = self.inference.swap(self.source_image, frame_rgb)
            
            if result is not None:
                result = self.smoother.smooth(result)
                display_frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            else:
                display_frame = frame
            
            # add latency overlay
            cv2.putText(
                display_frame,
                f"Latency: {latency:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            if display:
                cv2.imshow("Face Swap", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite("screenshot.jpg", display_frame)
                    logger.info("saved screenshot.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
        self.running = False
    
    def stop(self):
        """stop real-time processing"""
        self.running = False
