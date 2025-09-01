"""
Modal deployment for Dolphin document parsing model
High-concurrency GPU-accelerated image processing API
"""

import io
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import modal
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Modal app configuration
app = modal.App("dolphin-parser")

# Define the Modal image with all dependencies
dolphin_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.8.0",
        "torchvision==0.23.0", 
        "transformers==4.47.0",
        "accelerate==1.6.0",
        "timm==0.5.4",
        "pillow==9.3.0",
        "opencv-python==4.11.0.86",
        "numpy==1.26.4",
        "omegaconf==2.3.0",
        "pymupdf==1.26",
        "fastapi",
        "python-multipart"
    ])
    .copy_local_dir(".", "/app")
    .workdir("/app")
)

# FastAPI app
web_app = FastAPI(
    title="Dolphin Document Parser API",
    description="GPU-accelerated document parsing with Modal",
    version="2.0.0",
)

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File validation constants
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
            )


@app.function(
    image=dolphin_image,
    gpu=modal.gpu.A100(),  # Use A100 for optimal performance
    concurrency_limit=10,  # Max 10 parallel workers as requested
    container_idle_timeout=300,  # Keep containers warm for 5 minutes
    timeout=600,  # 10 minute timeout per request
    allow_concurrent_inputs=1,  # Process one image per container instance
)
@modal.asgi_app()
def fastapi_app():
    """Modal ASGI app wrapper"""
    
    # Import and initialize model inside the Modal function
    from demo_page_hf import DOLPHIN
    
    # Global model instance - loaded once per container
    model = None
    
    def get_model():
        nonlocal model
        if model is None:
            print("Loading Dolphin model...")
            model = DOLPHIN("./hf_model")
            print(f"Model loaded successfully on device: {model.device}")
        return model

    @web_app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "service": "Dolphin Document Parser API - Modal",
            "version": "2.0.0",
            "status": "running",
            "deployment": "Modal GPU",
            "endpoints": {
                "parse_image": "/parse",
                "health_check": "/health",
                "server_status": "/status"
            }
        }

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "deployment": "Modal GPU",
            "gpu_available": True
        }

    @web_app.get("/status")
    async def get_status():
        """Get server status"""
        return {
            "server": "running",
            "deployment": "Modal",
            "gpu_type": "A100",
            "concurrency_limit": 10,
            "supported_formats": list(SUPPORTED_IMAGE_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
        }

    @web_app.post("/parse")
    async def parse_image(
        file: UploadFile = File(...),
        max_batch_size: Optional[int] = 16
    ):
        """
        Parse a document image and return structured results
        
        Args:
            file: Image file to parse (jpg, jpeg, png)
            max_batch_size: Maximum batch size for element processing
        
        Returns:
            JSON response with parsed document structure
        """
        
        # Validate file
        validate_image_file(file)
        
        # Get model instance
        dolphin_model = get_model()
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        try:
            # Read file content
            file_content = await file.read()
            
            # Process image directly from memory
            start_time = time.perf_counter()
            
            # Open image from bytes
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            
            # Use the chat method for document parsing
            result = dolphin_model.chat("Parse the reading order of this document.", image)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            # Prepare response
            response_data = {
                "request_id": request_id,
                "filename": file.filename,
                "file_type": Path(file.filename).suffix.replace('.', '').upper() if file.filename else "JPG",
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": time.time(),
                "results": {
                    "content": result,
                    "format": "text"
                },
                "metadata": {
                    "image_size": image.size,
                    "model_device": dolphin_model.device,
                    "content_length": len(result)
                }
            }
            
            return JSONResponse(content=response_data)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Processing failed",
                    "message": str(e),
                    "request_id": request_id
                }
            )

    return web_app


@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("🚀 Deploying Dolphin Parser to Modal...")
    print("📝 Use `modal deploy modal_app.py` to deploy")
    print("🌐 Access via: https://{your-app-id}.modal.run/parse")