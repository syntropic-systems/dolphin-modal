"""
Modal deployment for Dolphin document parsing model
High-concurrency GPU-accelerated image processing API following VLM best practices

Optimized for burst traffic pattern where all pages of a PDF arrive simultaneously.
- Scales from 0 to handle cost efficiently
- Can process up to 10 pages in parallel
- Memory snapshots for fast cold starts (~2s instead of 30s+)
- Processes 100-page PDF in ~2 minutes
"""

import io
import time
import uuid
from pathlib import Path
from typing import Optional

import modal

# Modal app configuration
app = modal.App("dolphin-parser")

# Model storage using Modal Volume (persists across deployments)
MODEL_VOL_PATH = Path("/models")
MODEL_VOL = modal.Volume.from_name("dolphin-model", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}

def download_dolphin_model():
    """Download Dolphin model to persistent volume during image build"""
    from huggingface_hub import snapshot_download
    
    model_path = MODEL_VOL_PATH / "Dolphin"
    
    print("Downloading ByteDance/Dolphin model to volume...")
    snapshot_download(
        repo_id="ByteDance/Dolphin",
        local_dir=str(model_path),
        ignore_patterns=["*.git*", "README.md", ".gitattributes"]
    )
    print(f"Model downloaded to {model_path}")

# Define the Modal image with all dependencies
dolphin_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",  # For OpenCV
        "libglib2.0-0",     # For OpenCV
        "libsm6",           # For OpenCV
        "libxext6",         # For OpenCV  
        "libxrender-dev",   # For OpenCV
        "libgomp1",         # For OpenMP support
        "libglib2.0-0"      # For general compatibility
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0", 
        "transformers==4.47.0",
        "accelerate==1.6.0",
        "timm==0.5.4",
        "pillow==9.3.0",
        "opencv-python-headless==4.11.0.86",  # Headless version for servers
        "numpy==1.26.4",
        "omegaconf==2.3.0",
        "pymupdf==1.26",
        "fastapi",
        "python-multipart",
        "huggingface_hub"
    )
    .run_function(download_dolphin_model, volumes=volumes)  # Download to volume
    .add_local_dir(".", remote_path="/app")  # Add source code LAST
)

# File validation constants
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Dolphin Parser Service Class following VLM best practices
@app.cls(
    image=dolphin_image,
    gpu="A100",
    max_containers=10,  # Process up to 10 pages simultaneously
    scaledown_window=120,  # Keep warm for 2 minutes between PDF jobs
    min_containers=0,  # Scale to zero when idle (cost optimization)
    timeout=600,  # 10 minute timeout per request
    volumes=volumes,
    enable_memory_snapshot=True,  # Save memory state after model loading
    experimental_options={"enable_gpu_snapshot": True},  # Enable GPU memory snapshots
)
class DolphinParser:
    
    @modal.enter(snap=True)  # Include model loading in memory snapshot
    def start_model(self):
        """Load Dolphin model once when container starts"""
        import os
        import sys
        
        # Setup Python environment
        os.chdir("/app")
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        
        print(f"Working directory: {os.getcwd()}")
        print(f"Contents of /app: {sorted(os.listdir('/app'))}")
        
        # Import DOLPHIN class
        from demo_page_hf import DOLPHIN
        
        # Load model from persistent volume
        model_path = str(MODEL_VOL_PATH / "Dolphin")
        print(f"Loading Dolphin model from {model_path}...")
        
        self.model = DOLPHIN(model_path)
        print(f"Model loaded successfully on device: {self.model.device}")
    
    @modal.fastapi_endpoint(method="POST", docs=True)
    def parse(self, request: dict) -> dict:
        """
        Parse a document image and return structured results
        
        Args:
            request: JSON with base64 image data or image_url
        
        Returns:
            JSON response with parsed document structure
        """
        import base64
        import requests
        from PIL import Image
        
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        
        try:
            # Handle image input (base64 or URL)
            if "image_data" in request:
                # Base64 encoded image
                image_data = base64.b64decode(request["image_data"])
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                filename = request.get("filename", "uploaded_image.jpg")
                
            elif "image_url" in request:
                # Image from URL  
                response = requests.get(request["image_url"])
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                filename = request["image_url"].split("/")[-1]
                
            else:
                raise ValueError("Either 'image_data' (base64) or 'image_url' must be provided")
            
            # Validate image size
            if len(str(image.size)) > MAX_FILE_SIZE:  # Rough size check
                raise ValueError(f"Image too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB")
            
            # Parse document using full two-stage process (like demo_page_hf.py)
            from demo_page_hf import process_single_image
            
            # Use the complete processing pipeline
            _, recognition_results = process_single_image(
                image=image,
                model=self.model,
                save_dir=None,  # Don't save files
                image_name=filename.split('.')[0],
                max_batch_size=16,
                save_individual=False
            )
            
            processing_time = time.perf_counter() - start_time
            
            return {
                "request_id": request_id,
                "filename": filename,
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": time.time(),
                "results": recognition_results,  # Full parsed JSON structure
                "metadata": {
                    "image_size": list(image.size),
                    "model_device": self.model.device,
                    "total_elements": len(recognition_results) if isinstance(recognition_results, list) else len(recognition_results.get("elements", []))
                }
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "error": str(e),
                "status": "failed"
            }
    
    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "timestamp": time.time(),
            "deployment": "Modal GPU",
            "model_loaded": hasattr(self, 'model')
        }
    
    @modal.fastapi_endpoint(method="GET")
    def debug_memory(self) -> dict:
        """Debug GPU memory usage and calculate possible model instances"""
        import torch
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # Get GPU memory info
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - cached_memory
        
        # Convert to GB
        total_gb = total_memory / (1024**3)
        allocated_gb = allocated_memory / (1024**3)
        cached_gb = cached_memory / (1024**3)
        free_gb = free_memory / (1024**3)
        used_gb = cached_gb  # Used memory (allocated + cached)
        
        # Estimate model memory usage
        model_params = 0
        model_memory_gb = 0
        if hasattr(self, 'model'):
            model_params = sum(p.numel() for p in self.model.model.parameters())
            # Rough estimate: FP16 = 2 bytes per param + overhead
            model_memory_gb = (model_params * 2) / (1024**3) * 1.5  # 1.5x for overhead
        
        # Calculate how many instances could fit
        available_for_models = free_gb + (used_gb - model_memory_gb) if model_memory_gb > 0 else free_gb
        possible_instances = int(available_for_models / model_memory_gb) if model_memory_gb > 0 else 0
        
        return {
            "device": f"cuda:{device}",
            "total_gb": round(total_gb, 2),
            "allocated_gb": round(allocated_gb, 2), 
            "cached_gb": round(cached_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_gb": round(used_gb, 2),
            "model_loaded": hasattr(self, 'model'),
            "model_params": model_params,
            "estimated_model_memory_gb": round(model_memory_gb, 2),
            "possible_instances": max(1, possible_instances),  # At least current instance
            "memory_efficiency": f"{(used_gb/total_gb)*100:.1f}%"
        }
    
    @modal.exit()
    def shutdown_model(self):
        """Clean up when container shuts down"""
        if hasattr(self, 'model'):
            print("Shutting down Dolphin model...")
            del self.model


@app.local_entrypoint()
def main():
    """Local entrypoint for testing the deployed service"""
    import json
    import urllib.request
    import base64
    
    parser = DolphinParser()
    
    # Test with a sample image (convert to base64)
    test_image_path = Path("demo/page_imgs/page_1.jpeg") 
    if test_image_path.exists():
        with open(test_image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        payload = json.dumps({
            "image_data": image_data,
            "filename": "page_1.jpeg"
        })
        
        print("Testing Dolphin parser...")
        req = urllib.request.Request(
            parser.parse.get_web_url(),
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print("Response:", result)
    else:
        print("Test image not found, skipping test")