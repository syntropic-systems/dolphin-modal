# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download models (required before running)
# Option A: Original format for config-based parsing
# Download from Google Drive/Baidu Yun and place in ./checkpoints

# Option B: HuggingFace format (recommended)
git lfs install
git clone https://huggingface.co/ByteDance/Dolphin ./hf_model
```

## Development Commands

### Local Document Parsing

```bash
# Page-level parsing with HuggingFace model
python demo_page_hf.py --model_path ./hf_model --input_path <image/pdf/dir> --save_dir ./results --max_batch_size 16

# Batch processing multiple images (CUDA-optimized)
python benchmark_batch_processing.py --model_path ./hf_model --input_path ./demo/page_imgs --max_batch_size 16

# Element-level parsing
python demo_element_hf.py --model_path ./hf_model --input_path <image> --element_type <table/formula/text>

# Config-based parsing (original framework)
python demo_page.py --config ./config/Dolphin.yaml --input_path <image/pdf/dir> --save_dir ./results
```

### Modal Cloud Deployment

```bash
# Deploy to Modal
modal deploy modal_app.py

# Test batch endpoint
python test_batch_endpoint.py

# Benchmark Modal performance
python benchmark_modal_batch.py  # Compares individual vs batch processing
```

### Code Quality

```bash
# Format code (pre-commit configured)
black . --line-length 120
isort . --profile black
flake8 .
```

## Architecture

### Two-Stage Document Parsing Pipeline

1. **Stage 1: Layout Analysis** (`chat.py:78`)
   - Detects all elements and their reading order in one pass
   - Returns bounding boxes with labels (para, title, table, figure, etc.)

2. **Stage 2: Element Recognition** (`demo_page_hf.py:200-282`)
   - Batch processes detected elements by type
   - Parallel GPU inference for all text/table elements

### Core Components

- **`chat.py`**: DOLPHIN class with original framework support
- **`demo_page_hf.py`**: HuggingFace implementation with batch processing
  - `batch_detect_all_elements()`: File-based batch processing
  - `batch_detect_all_elements_pil()`: PIL image batch processing for Modal
- **`modal_app.py`**: Cloud deployment with FastAPI endpoints
  - `/parse`: Single image endpoint
  - `/parse_batch`: Batch endpoint (up to 32 images)
- **`utils/`**: Core model components
  - `model.py`: DonutModel architecture (Swin encoder + MBart decoder)
  - `processor.py`: Image/text preprocessing
  - `utils.py`: Coordinate mapping, bbox adjustment

### Modal Configuration (modal_app.py:75-83)

```python
MAX_BATCH_SIZE = 32       # T4 GPU optimized batch limit
ELEMENT_BATCH_SIZE = 32   # Elements per GPU inference
GPU_TYPE = "T4"           # 16GB VRAM, cost-optimized
MAX_CONTAINERS = 4        # Auto-scaling configuration
```

### Batch Processing Flow

1. Load N images → Prepare padded tensors
2. Single GPU call for layout detection on all N images
3. Collect all elements across images by type
4. Batch process elements (text/tables) in chunks of 32
5. Return results mapped to original images

## Key Implementation Details

### Coordinate System (utils.py:259-291)
- Images padded to square for model input
- Coordinates mapped back to original dimensions
- Bounding box edge adjustment for better OCR

### Memory Management
- T4 GPU (16GB): Max 32 images per batch
- Element batch size: 32 for optimal GPU utilization
- Memory snapshots enabled for fast cold starts

### Model Configuration (config/Dolphin.yaml)
- Input resolution: 896×896
- Max sequence length: 4096 tokens
- Vision backbone: Swin Transformer [2, 2, 14, 2]
- Text decoder: MBart-based causal LM

## API Endpoints

### Individual Processing
```
POST https://abhishekgautam011--dolphin-parser-dolphinparser-parse.modal.run
Body: {"image_data": "base64...", "filename": "doc.jpg"}
```

### Batch Processing (32 images max)
```
POST https://abhishekgautam011--dolphin-parser-dolphinparser-parse-batch.modal.run
Body: {"images": [{"image_data": "base64...", "filename": "page1.jpg"}, ...]}
```

## Performance Optimization

- Batch processing provides 10-50x speedup over sequential
- T4 GPU handles 32 images in ~30-60 seconds
- Use `batch_detect_all_elements_pil()` for in-memory batch processing
- Modal auto-scales 0-4 containers based on load