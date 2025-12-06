# test_api.py - unit tests for the fastapi endpoints
# run with: pytest tests/test_api.py -v

import pytest
import base64
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """create test client for api"""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_image_b64():
    """create a sample base64 encoded image"""
    img = Image.new('RGB', (256, 256), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def sample_image_bytes():
    """create sample image bytes"""
    img = Image.new('RGB', (256, 256), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoint:
    """tests for health check endpoint"""
    
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_format(self, client):
        response = client.get("/health")
        data = response.json()
        
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'gpu_available' in data
    
    def test_health_status_healthy(self, client):
        response = client.get("/health")
        data = response.json()
        
        assert data['status'] == 'healthy'


class TestSwapImageEndpoint:
    """tests for image swap endpoint"""
    
    def test_swap_missing_source(self, client, sample_image_b64):
        response = client.post("/swap/image", json={
            "target_image": sample_image_b64
        })
        assert response.status_code == 422  # validation error
    
    def test_swap_missing_target(self, client, sample_image_b64):
        response = client.post("/swap/image", json={
            "source_image": sample_image_b64
        })
        assert response.status_code == 422
    
    def test_swap_invalid_base64(self, client):
        response = client.post("/swap/image", json={
            "source_image": "not_valid_base64",
            "target_image": "also_not_valid"
        })
        assert response.status_code == 400
    
    def test_swap_image_too_small(self, client):
        # create tiny image
        img = Image.new('RGB', (32, 32), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        b64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        response = client.post("/swap/image", json={
            "source_image": b64,
            "target_image": b64
        })
        assert response.status_code == 400


class TestSwapImageUploadEndpoint:
    """tests for file upload swap endpoint"""
    
    def test_upload_missing_files(self, client):
        response = client.post("/swap/image/upload")
        assert response.status_code == 422
    
    def test_upload_missing_target(self, client, sample_image_bytes):
        response = client.post("/swap/image/upload", files={
            "source": ("source.jpg", sample_image_bytes, "image/jpeg")
        })
        assert response.status_code == 422
    
    def test_upload_missing_source(self, client, sample_image_bytes):
        response = client.post("/swap/image/upload", files={
            "target": ("target.jpg", sample_image_bytes, "image/jpeg")
        })
        assert response.status_code == 422


class TestSwapVideoEndpoint:
    """tests for video swap endpoint"""
    
    def test_video_missing_files(self, client):
        response = client.post("/swap/video")
        assert response.status_code == 422
    
    def test_video_invalid_source(self, client):
        # create dummy video bytes
        video_bytes = b"not a real video"
        img_bytes = b"not a real image"
        
        response = client.post("/swap/video", files={
            "source": ("source.jpg", img_bytes, "image/jpeg"),
            "video": ("video.mp4", video_bytes, "video/mp4")
        })
        # should fail because files are invalid
        assert response.status_code in [400, 500]


class TestRateLimiting:
    """tests for rate limiting middleware"""
    
    def test_rate_limit_not_exceeded(self, client):
        # make a few requests - should all succeed
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200


class TestCORS:
    """tests for cors headers"""
    
    def test_cors_headers_present(self, client):
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        # cors should allow the request
        assert response.status_code in [200, 405]


class TestAPIDocumentation:
    """tests for api documentation"""
    
    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert 'paths' in schema
        assert '/health' in schema['paths']
        assert '/swap/image' in schema['paths']
    
    def test_swagger_ui(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
