# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Core Demo Scripts:**
```bash
# Page-level parsing (config-based)
python demo_page.py --config ./config/Dolphin.yaml --input_path <image/pdf/directory> --save_dir ./results

# Page-level parsing (HuggingFace)
python demo_page_hf.py --model_path ./hf_model --input_path <image/pdf/directory> --save_dir ./results

# Element-level parsing (config-based)  
python demo_element.py --config ./config/Dolphin.yaml --input_path <image> --element_type <table/formula/text>

# Element-level parsing (HuggingFace)
python demo_element_hf.py --model_path ./hf_model --input_path <image> --element_type <table/formula/text>
```

**Code Quality (Pre-commit hooks configured):**
```bash
black . --line-length 120
isort . --profile black
flake8 .
```

## Architecture

**Tech Stack:** Pure Python with PyTorch-based vision-language model

**Core Components:**
- `chat.py` - Main DOLPHIN class implementation
- `utils/model.py` - DonutModel and SwinEncoder definitions  
- `utils/processor.py` - DolphinProcessor for image/text processing
- `config/Dolphin.yaml` - Model configuration (896×896 input, 4096 max length)

**Two-Stage Parsing Approach:**
1. Layout analysis and reading order determination
2. Parallel element parsing with heterogeneous anchors

**Dual Framework Support:**
- Original config-based framework (uses `config/Dolphin.yaml`)
- HuggingFace Transformers compatibility (uses `--model_path`)

## Key Dependencies

- **Deep Learning:** PyTorch 2.1.0, transformers 4.47.0, accelerate 1.6.0
- **Vision:** timm 0.5.4 (Swin Transformer), OpenCV 4.11.0  
- **Document Processing:** PyMuPDF 1.26 (PDF support)
- **Configuration:** OmegaConf 2.3.0 (YAML configs)

## Development Notes

- Models must be downloaded separately (not included in repository)
- Supports batch processing with configurable `max_batch_size`
- Input formats: images (JPEG/PNG/JPG) and PDFs
- Two parsing granularities: page-level (full documents) and element-level (tables/formulas/text)
- Production deployment options in `deployment/` (vLLM and TensorRT-LLM)

## Model Configuration

Default settings in `config/Dolphin.yaml`:
- Vision backbone: Swin Transformer with layers [2, 2, 14, 2]  
- Text decoder: MBart-based causal language model
- Image preprocessing: 896×896 resolution
- Max sequence length: 4096 tokens