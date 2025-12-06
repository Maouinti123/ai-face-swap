# manager.py - handles dataset downloading and organization
# started with just lfw but made it extensible for other datasets

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import logging

# set up logging so i can debug issues later
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    manages the face dataset - downloading, versioning, and updates
    
    the idea is to keep track of all images with metadata so we can
    easily add new faces without reprocessing everything
    """
    
    def __init__(self, data_dir: str, version: str = "v1"):
        # store paths as Path objects - easier to work with
        self.data_dir = Path(data_dir)
        self.version = version
        self.version_dir = self.data_dir / version
        
        # metadata file keeps track of all processed images
        self.metadata_file = self.version_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # make sure directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        """create the folder structure if it doesnt exist"""
        dirs_to_create = [
            self.version_dir,
            self.version_dir / "aligned",
            self.version_dir / "raw",
            self.data_dir / "new_images"
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """load existing metadata or create empty one"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        # default structure for new metadata
        return {
            "version": self.version,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_images": 0,
            "identities": {},
            "images": []
        }
    
    def _save_metadata(self):
        """persist metadata to disk"""
        self.metadata["updated_at"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_hash(self, filepath: Path) -> str:
        """compute md5 hash to detect duplicates - faster than comparing pixels"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def download_lfw_dataset(self, kaggle_dataset: str = "atulanandjha/lfwpeople"):
        """
        download lfw from kaggle - need to have kaggle api key set up
        
        lfw is small but good for initial testing. for production
        would want something bigger like vggface2
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            logger.info(f"downloading {kaggle_dataset} from kaggle...")
            
            api = KaggleApi()
            api.authenticate()
            
            # download to raw folder
            download_path = self.version_dir / "raw"
            api.dataset_download_files(
                kaggle_dataset,
                path=str(download_path),
                unzip=True
            )
            
            logger.info("download complete, starting preprocessing...")
            return True
            
        except Exception as e:
            # kaggle can be flaky sometimes
            logger.error(f"failed to download dataset: {e}")
            logger.info("make sure you have ~/.kaggle/kaggle.json set up")
            return False
    
    def register_image(self, image_path: Path, identity: str, 
                       session: Optional[str] = None) -> bool:
        """
        add a new image to the dataset index
        
        identity is the person's name/id, session is optional
        for tracking different photo sessions
        """
        if not image_path.exists():
            logger.warning(f"image not found: {image_path}")
            return False
        
        # check for duplicates using hash
        img_hash = self._compute_hash(image_path)
        existing_hashes = [img["hash"] for img in self.metadata["images"]]
        
        if img_hash in existing_hashes:
            logger.info(f"duplicate image detected, skipping: {image_path.name}")
            return False
        
        # add to metadata
        image_entry = {
            "filename": image_path.name,
            "path": str(image_path),
            "identity": identity,
            "session": session or "default",
            "hash": img_hash,
            "added_at": datetime.now().isoformat()
        }
        
        self.metadata["images"].append(image_entry)
        
        # update identity count
        if identity not in self.metadata["identities"]:
            self.metadata["identities"][identity] = 0
        self.metadata["identities"][identity] += 1
        
        self.metadata["total_images"] += 1
        self._save_metadata()
        
        return True
    
    def process_new_images(self, preprocessor) -> int:
        """
        process images from the new_images folder
        
        this runs as a background job to handle dataset updates
        without interrupting training
        """
        new_images_dir = self.data_dir / "new_images"
        processed_count = 0
        
        # supported formats - could add more but these cover 99% of cases
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for img_file in new_images_dir.iterdir():
            if img_file.suffix.lower() not in image_extensions:
                continue
            
            try:
                # extract identity from filename - expecting format: identity_xxx.jpg
                identity = img_file.stem.split('_')[0]
                
                # run face detection and alignment
                aligned_face = preprocessor.process(str(img_file))
                
                if aligned_face is not None:
                    # save aligned face
                    output_path = self.version_dir / "aligned" / img_file.name
                    aligned_face.save(str(output_path))
                    
                    # register in metadata
                    self.register_image(output_path, identity)
                    processed_count += 1
                    
                    # move original to processed folder so we dont reprocess
                    processed_dir = new_images_dir / "processed"
                    processed_dir.mkdir(exist_ok=True)
                    shutil.move(str(img_file), str(processed_dir / img_file.name))
                else:
                    # no face detected - move to failed folder for manual review
                    failed_dir = new_images_dir / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    shutil.move(str(img_file), str(failed_dir / img_file.name))
                    logger.warning(f"no face detected in {img_file.name}")
                    
            except Exception as e:
                logger.error(f"error processing {img_file.name}: {e}")
        
        logger.info(f"processed {processed_count} new images")
        return processed_count
    
    def create_new_version(self) -> str:
        """
        create a new dataset version - useful when making significant changes
        
        copies metadata and creates new version folder
        """
        # figure out next version number
        existing_versions = [d.name for d in self.data_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('v')]
        version_nums = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
        next_version = f"v{max(version_nums, default=0) + 1}"
        
        # create new version directory
        new_version_dir = self.data_dir / next_version
        new_version_dir.mkdir(parents=True)
        
        # copy structure from current version
        (new_version_dir / "aligned").mkdir()
        (new_version_dir / "raw").mkdir()
        
        # update metadata for new version
        new_metadata = self.metadata.copy()
        new_metadata["version"] = next_version
        new_metadata["created_at"] = datetime.now().isoformat()
        new_metadata["parent_version"] = self.version
        
        with open(new_version_dir / "metadata.json", 'w') as f:
            json.dump(new_metadata, f, indent=2)
        
        logger.info(f"created new dataset version: {next_version}")
        return next_version
    
    def get_image_paths(self, identity: Optional[str] = None) -> List[Path]:
        """get all image paths, optionally filtered by identity"""
        paths = []
        for img in self.metadata["images"]:
            if identity is None or img["identity"] == identity:
                paths.append(Path(img["path"]))
        return paths
    
    def get_identities(self) -> List[str]:
        """return list of all unique identities in dataset"""
        return list(self.metadata["identities"].keys())
    
    def get_stats(self) -> Dict:
        """return dataset statistics for logging/debugging"""
        return {
            "version": self.version,
            "total_images": self.metadata["total_images"],
            "num_identities": len(self.metadata["identities"]),
            "images_per_identity": {
                k: v for k, v in sorted(
                    self.metadata["identities"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # top 10 only
            }
        }
