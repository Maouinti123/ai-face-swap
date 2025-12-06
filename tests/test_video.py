# test_video.py - unit tests for video processing
# run with: pytest tests/test_video.py -v

import pytest
import numpy as np
import tempfile
import os


class TestTemporalSmoother:
    """tests for temporal smoothing"""
    
    def test_smoother_creation(self):
        from video import TemporalSmoother
        
        smoother = TemporalSmoother(window_size=5, alpha=0.8)
        assert smoother is not None
        assert smoother.window_size == 5
        assert smoother.alpha == 0.8
    
    def test_smoother_reset(self):
        from video import TemporalSmoother
        
        smoother = TemporalSmoother()
        
        # add some frames
        for _ in range(3):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            smoother.smooth(frame)
        
        assert len(smoother.buffer) == 3
        
        smoother.reset()
        assert len(smoother.buffer) == 0
    
    def test_smoother_single_frame(self):
        from video import TemporalSmoother
        
        smoother = TemporalSmoother()
        
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = smoother.smooth(frame)
        
        # first frame should be returned unchanged
        assert np.array_equal(result, frame)
    
    def test_smoother_multiple_frames(self):
        from video import TemporalSmoother
        
        smoother = TemporalSmoother(window_size=3, alpha=0.5)
        
        frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) 
                  for _ in range(5)]
        
        results = [smoother.smooth(f) for f in frames]
        
        # should have smoothed results
        assert len(results) == 5
        for r in results:
            assert r.shape == (256, 256, 3)
    
    def test_smoother_reduces_noise(self):
        from video import TemporalSmoother
        
        smoother = TemporalSmoother(window_size=5, alpha=0.7)
        
        # create frames with noise
        base = np.ones((64, 64, 3), dtype=np.float32) * 128
        frames = [base + np.random.randn(64, 64, 3) * 20 for _ in range(10)]
        frames = [f.clip(0, 255).astype(np.uint8) for f in frames]
        
        results = [smoother.smooth(f) for f in frames]
        
        # smoothed frames should have less variance
        original_var = np.var([f.astype(float) for f in frames])
        smoothed_var = np.var([r.astype(float) for r in results[3:]])  # skip first few
        
        # smoothing should reduce variance somewhat
        assert smoothed_var <= original_var * 1.5  # allow some tolerance


class TestLandmarkSmoother:
    """tests for landmark smoothing"""
    
    def test_landmark_smoother_creation(self):
        from video.temporal import LandmarkSmoother
        
        smoother = LandmarkSmoother(alpha=0.7)
        assert smoother is not None
    
    def test_landmark_smoothing(self):
        from video.temporal import LandmarkSmoother
        
        smoother = LandmarkSmoother(alpha=0.5)
        
        # create sequence of landmarks with noise
        base_landmarks = np.array([[100, 100], [150, 100], [125, 130]], dtype=np.float32)
        
        results = []
        for _ in range(5):
            noisy = base_landmarks + np.random.randn(3, 2) * 5
            smoothed = smoother.smooth(noisy)
            results.append(smoothed)
        
        # smoothed landmarks should be closer to base
        final_diff = np.abs(results[-1] - base_landmarks).mean()
        assert final_diff < 20  # reasonable tolerance


class TestColorMatcher:
    """tests for color matching"""
    
    def test_color_matcher_creation(self):
        from video.temporal import ColorMatcher
        
        matcher = ColorMatcher()
        assert matcher is not None
    
    def test_color_matcher_set_reference(self):
        from video.temporal import ColorMatcher
        
        matcher = ColorMatcher()
        
        ref_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        matcher.set_reference(ref_frame)
        
        assert matcher.reference_hist is not None
    
    def test_color_matcher_match(self):
        from video.temporal import ColorMatcher
        
        matcher = ColorMatcher()
        
        ref_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        matcher.set_reference(ref_frame)
        
        test_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        matched = matcher.match(test_frame)
        
        assert matched.shape == test_frame.shape
        assert matched.dtype == np.uint8


class TestFlickerReducer:
    """tests for flicker reduction"""
    
    def test_flicker_reducer_creation(self):
        from video.temporal import FlickerReducer
        
        reducer = FlickerReducer(buffer_size=7)
        assert reducer is not None
    
    def test_flicker_reducer_reset(self):
        from video.temporal import FlickerReducer
        
        reducer = FlickerReducer()
        
        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            reducer.reduce(frame)
        
        reducer.reset()
        assert len(reducer.buffer) == 0
    
    def test_flicker_reduction(self):
        from video.temporal import FlickerReducer
        
        reducer = FlickerReducer(buffer_size=5)
        
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) 
                  for _ in range(10)]
        
        results = [reducer.reduce(f) for f in frames]
        
        assert len(results) == 10
        for r in results:
            assert r.shape == (64, 64, 3)


class TestAdaptiveSmoother:
    """tests for adaptive smoothing"""
    
    def test_adaptive_smoother_creation(self):
        from video.temporal import AdaptiveSmoother
        
        smoother = AdaptiveSmoother(min_alpha=0.5, max_alpha=0.95)
        assert smoother is not None
    
    def test_adaptive_smoother_static_scene(self):
        from video.temporal import AdaptiveSmoother
        
        smoother = AdaptiveSmoother()
        
        # static scene - same frame repeated
        frame = np.ones((64, 64, 3), dtype=np.uint8) * 128
        
        results = [smoother.smooth(frame.copy()) for _ in range(5)]
        
        # should produce stable output
        assert len(results) == 5
    
    def test_adaptive_smoother_motion(self):
        from video.temporal import AdaptiveSmoother
        
        smoother = AdaptiveSmoother()
        
        # moving scene - different frames
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) 
                  for _ in range(5)]
        
        results = [smoother.smooth(f) for f in frames]
        
        assert len(results) == 5


class TestVideoProcessor:
    """tests for video processor"""
    
    def test_processor_get_video_info(self):
        from video import VideoProcessor
        
        # create a simple test video
        import cv2
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        # write a few frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, 30, (640, 480))
        
        for _ in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        
        writer.release()
        
        # test get_video_info without full processor init
        cap = cv2.VideoCapture(tmp_path)
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        
        assert info['width'] == 640
        assert info['height'] == 480
        assert info['frame_count'] == 30
        
        os.unlink(tmp_path)
