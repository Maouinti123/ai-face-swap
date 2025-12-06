# test_dataset.py - unit tests for dataset handling
# run with: pytest tests/test_dataset.py -v

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil


class TestDatasetManager:
    """tests for dataset manager"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """create temporary data directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_manager_creation(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        assert manager is not None
        assert manager.version == "v1"
    
    def test_directory_setup(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        
        # check directories were created
        assert (Path(temp_data_dir) / "v1").exists()
        assert (Path(temp_data_dir) / "v1" / "aligned").exists()
        assert (Path(temp_data_dir) / "v1" / "raw").exists()
        assert (Path(temp_data_dir) / "new_images").exists()
    
    def test_metadata_creation(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        
        assert manager.metadata is not None
        assert 'version' in manager.metadata
        assert 'images' in manager.metadata
        assert 'identities' in manager.metadata
    
    def test_register_image(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        
        # create a test image
        test_img = Image.new('RGB', (256, 256), color='red')
        img_path = Path(temp_data_dir) / "v1" / "aligned" / "test_person_001.jpg"
        test_img.save(img_path)
        
        # register it
        result = manager.register_image(img_path, "test_person")
        
        assert result == True
        assert manager.metadata['total_images'] == 1
        assert 'test_person' in manager.metadata['identities']
    
    def test_duplicate_detection(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        
        # create and register same image twice
        test_img = Image.new('RGB', (256, 256), color='blue')
        img_path = Path(temp_data_dir) / "v1" / "aligned" / "test_001.jpg"
        test_img.save(img_path)
        
        result1 = manager.register_image(img_path, "person1")
        result2 = manager.register_image(img_path, "person1")
        
        assert result1 == True
        assert result2 == False  # duplicate should be rejected
        assert manager.metadata['total_images'] == 1
    
    def test_get_identities(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        
        # register images for different identities
        for i, name in enumerate(['alice', 'bob', 'charlie']):
            test_img = Image.new('RGB', (256, 256), color=(i*50, i*50, i*50))
            img_path = Path(temp_data_dir) / "v1" / "aligned" / f"{name}_001.jpg"
            test_img.save(img_path)
            manager.register_image(img_path, name)
        
        identities = manager.get_identities()
        
        assert len(identities) == 3
        assert 'alice' in identities
        assert 'bob' in identities
        assert 'charlie' in identities
    
    def test_create_new_version(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        new_version = manager.create_new_version()
        
        assert new_version == "v2"
        assert (Path(temp_data_dir) / "v2").exists()
    
    def test_get_stats(self, temp_data_dir):
        from dataset import DatasetManager
        
        manager = DatasetManager(temp_data_dir)
        stats = manager.get_stats()
        
        assert 'version' in stats
        assert 'total_images' in stats
        assert 'num_identities' in stats


class TestFacePreprocessor:
    """tests for face preprocessing"""
    
    def test_preprocessor_creation(self):
        from dataset import FacePreprocessor
        
        preprocessor = FacePreprocessor(target_size=256)
        assert preprocessor is not None
        assert preprocessor.target_size == 256
    
    def test_preprocessor_detector_init(self):
        from dataset import FacePreprocessor
        
        preprocessor = FacePreprocessor(target_size=256, detector_type="opencv")
        assert preprocessor.detector is not None
    
    def test_alignment_transform(self):
        from dataset import FacePreprocessor
        
        preprocessor = FacePreprocessor(target_size=112)
        
        # test with dummy landmarks
        src_landmarks = np.array([
            [40, 50], [70, 50], [55, 70], [42, 90], [68, 90]
        ], dtype=np.float32)
        
        dst_landmarks = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        
        transform = preprocessor._get_similarity_transform(src_landmarks, dst_landmarks)
        
        assert transform.shape == (2, 3)


class TestFaceSwapDataset:
    """tests for pytorch dataset"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """create temporary dataset with sample images"""
        temp_dir = tempfile.mkdtemp()
        
        # create sample images for two identities
        for identity in ['person_a', 'person_b']:
            identity_dir = Path(temp_dir) / "aligned" / identity
            identity_dir.mkdir(parents=True)
            
            for i in range(3):
                img = Image.new('RGB', (256, 256), 
                               color=(100 + i*20, 100, 100) if identity == 'person_a' 
                               else (100, 100 + i*20, 100))
                img.save(identity_dir / f"img_{i}.jpg")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dataset_creation(self, temp_dataset_dir):
        from dataset import FaceSwapDataset
        
        dataset = FaceSwapDataset(temp_dataset_dir, image_size=256)
        assert dataset is not None
        assert len(dataset) > 0
    
    def test_dataset_getitem(self, temp_dataset_dir):
        from dataset import FaceSwapDataset
        
        dataset = FaceSwapDataset(temp_dataset_dir, image_size=256, augment=False)
        
        sample = dataset[0]
        
        assert 'source' in sample
        assert 'target' in sample
        assert 'same_identity' in sample
        assert sample['source'].shape == (3, 256, 256)
        assert sample['target'].shape == (3, 256, 256)
    
    def test_dataset_normalization(self, temp_dataset_dir):
        from dataset import FaceSwapDataset
        
        dataset = FaceSwapDataset(temp_dataset_dir, image_size=256, augment=False)
        sample = dataset[0]
        
        # images should be normalized to [-1, 1]
        assert sample['source'].min() >= -1.0
        assert sample['source'].max() <= 1.0


class TestDataLoader:
    """tests for dataloader creation"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        temp_dir = tempfile.mkdtemp()
        
        for identity in ['person_a', 'person_b']:
            identity_dir = Path(temp_dir) / "aligned" / identity
            identity_dir.mkdir(parents=True)
            
            for i in range(5):
                img = Image.new('RGB', (256, 256), color=(100 + i*10, 100, 100))
                img.save(identity_dir / f"img_{i}.jpg")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dataloader_creation(self, temp_dataset_dir):
        from dataset import get_dataloader
        
        loader = get_dataloader(
            data_dir=temp_dataset_dir,
            batch_size=2,
            image_size=256,
            num_workers=0
        )
        
        assert loader is not None
    
    def test_dataloader_batch(self, temp_dataset_dir):
        from dataset import get_dataloader
        
        loader = get_dataloader(
            data_dir=temp_dataset_dir,
            batch_size=2,
            image_size=256,
            num_workers=0
        )
        
        batch = next(iter(loader))
        
        assert batch['source'].shape[0] == 2
        assert batch['target'].shape[0] == 2
