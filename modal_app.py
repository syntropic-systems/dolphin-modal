"""
Modal deployment for Dolphin document parsing model
High-concurrency GPU-accelerated image processing API with memory snapshots
"""

import io
import os
import time
import uuid
from datetime import datetime
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

pip_packages = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "transformers==4.47.0",
    "accelerate==1.6.0",
    "timm==0.5.4",
    "pillow==9.3.0",
    "opencv-python-headless==4.11.0.86",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "pymupdf==1.26",
    "fastapi",
    "python-multipart",
    "huggingface_hub",
]

# Define the Modal image with all dependencies
dolphin_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04")
    .apt_install(
        "python3-pip",
        "python-is-python3",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    # --- THIS IS THE FIX ---
    # Unpack the list of packages using the * operator
    # and pass the index URL as a keyword argument.
    .pip_install(
        *pip_packages,
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    # -----------------------
    .run_function(download_dolphin_model, volumes=volumes)
    .add_local_dir(".", remote_path="/app")
)

# Configuration constants
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per individual image

# T4 GPU Batch Processing Configuration
MAX_BATCH_SIZE = 32          # Maximum images per batch request (A10 GPU optimized)
ELEMENT_BATCH_SIZE = 32      # Maximum elements per GPU inference call
GPU_TYPE = "A10"              # GPU type (A10: 24GB VRAM, cost-optimized)
MAX_CONTAINERS = 4           # Scale up to 4 containers for burst traffic

# Dolphin Parser Service Class with memory snapshots
@app.cls(
    image=dolphin_image,
    gpu=GPU_TYPE,  # T4 GPU: 16GB VRAM, optimized for cost/performance
    max_containers=MAX_CONTAINERS,  # Scale up for burst traffic
    scaledown_window=120,
    min_containers=0,  # Scale to zero when idle (cost optimization)
    timeout=600,  # 10 minute timeout per request
    volumes=volumes,
    enable_memory_snapshot=True,  # Save memory state after model loading
    experimental_options={"enable_gpu_snapshot": True}  # Enable GPU memory snapshots
)
class DolphinParser:
    """Dolphin document parser with container tracking"""
    
    container_id: str = None
    container_start: float = None  
    requests_handled: int = 0
        
    @modal.enter(snap=True)  # Include model loading in memory snapshot
    def start_model(self):
        """Load Dolphin model once when container starts"""
        import sys
        
        # Initialize container tracking
        self.container_id = str(uuid.uuid4())[:8]
        self.container_start = time.perf_counter()
        self.requests_handled = 0
        
        start_time = time.perf_counter()
        
        # Check if model already loaded from snapshot
        if hasattr(self, 'model'):
            print(f"✅ Container {self.container_id} restored from snapshot (model ready)")
            return
        
        print(f"🆕 Container {self.container_id} loading model...")
        
        # Setup Python environment
        os.chdir("/app")
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        
        # Import and load model
        from demo_page_hf import DOLPHIN
        
        model_path = str(MODEL_VOL_PATH / "Dolphin")
        self.model = DOLPHIN(model_path)
        
        load_time = time.perf_counter() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s on {self.model.device}")
    
    @modal.fastapi_endpoint(method="POST", docs=True)  
    def parse_batch(self, request: dict) -> dict:
        """
        BATCH PROCESSING: Parse multiple document images in one GPU-optimized call
        
        Args:
            request: JSON with array of images: {"images": [{"image_data": "base64...", "filename": "page1.jpg"}, ...]}
        
        Returns:
            JSON response with results for all images: {"results": [{"filename": "page1.jpg", "elements": [...]}]}
        """
        import base64
        from PIL import Image
        
        start_time = time.perf_counter()
        batch_id = str(uuid.uuid4())[:8]
        self.requests_handled += 1
        
        try:
            images_data = request.get("images", [])
            if not images_data:
                raise ValueError("'images' array is required")
            
            # T4 GPU Memory Optimization: Validate batch size
            if len(images_data) > MAX_BATCH_SIZE:
                raise ValueError(f"Batch size limited to {MAX_BATCH_SIZE} images for {GPU_TYPE} GPU optimization, received {len(images_data)}")
            
            print(f"📦 BATCH {batch_id}: Processing {len(images_data)} images on container {self.container_id}")
            
            # Prepare images and paths for batch processing
            pil_images = []
            image_paths = []
            filenames = []
            
            for i, img_data in enumerate(images_data):
                if "image_data" not in img_data:
                    raise ValueError(f"Image {i}: 'image_data' is required")
                
                # Decode base64 image
                image_data = base64.b64decode(img_data["image_data"])
                pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                filename = img_data.get("filename", f"batch_image_{i}.jpg")
                
                pil_images.append(pil_image)
                image_paths.append(f"batch_{batch_id}_{i}")  # Temporary paths for processing
                filenames.append(filename)
            
            # Use the CUDA-optimized batch processing function
            from demo_page_hf import batch_detect_all_elements_pil
            
            batch_results = batch_detect_all_elements_pil(
                images=pil_images, 
                image_names=image_paths,
                model=self.model, 
                max_batch_size=ELEMENT_BATCH_SIZE  # T4 GPU element processing optimization
            )
            
            # Format results for response
            formatted_results = []
            for i, (image_path, elements) in enumerate(batch_results.items()):
                formatted_results.append({
                    "filename": filenames[i],
                    "elements": elements,
                    "element_count": len(elements)
                })
            
            processing_time = time.perf_counter() - start_time
            total_elements = sum(len(result["elements"]) for result in formatted_results)
            
            print(f"✅ BATCH {batch_id}: {total_elements} elements from {len(images_data)} images in {processing_time:.2f}s")
            
            return {
                "batch_id": batch_id,
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": time.time(),
                "images_processed": len(images_data),
                "total_elements": total_elements,
                "results": formatted_results,
                "metadata": {
                    "model_device": self.model.device,
                    "container_id": self.container_id,
                    "container_requests": self.requests_handled,
                    "batch_throughput": round(len(images_data) / processing_time, 2),
                    "elements_per_second": round(total_elements / processing_time, 2)
                }
            }
            
        except Exception as e:
            error_time = time.perf_counter() - start_time
            print(f"❌ BATCH {batch_id}: Error after {error_time:.2f}s - {str(e)}")
            return {
                "batch_id": batch_id,
                "error": str(e),
                "status": "failed",
                "processing_time_seconds": round(error_time, 2)
            }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "timestamp": time.time(),
            "deployment": "Modal GPU",
            "model_loaded": hasattr(self, 'model'),
            "container_id": self.container_id,
            "requests_handled": self.requests_handled,
            "container_age_seconds": round(time.perf_counter() - self.container_start, 2)
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
            "possible_instances": max(1, possible_instances),
            "memory_efficiency": f"{(used_gb/total_gb)*100:.1f}%",
            "container_id": self.container_id
        }
    
    @modal.exit()
    def shutdown_model(self):
        """Clean up when container shuts down"""
        print(f"🛑 Container {self.container_id} shutting down (handled {self.requests_handled} requests)")
        if hasattr(self, 'model'):
            del self.model