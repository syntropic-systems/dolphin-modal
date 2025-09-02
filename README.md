<div align="center">
  <img src="./assets/dolphin.png" width="300">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.14059">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://huggingface.co/ByteDance/Dolphin">
    <img src="https://img.shields.io/badge/HuggingFace-Dolphin-yellow">
  </a>
  <a href="https://modelscope.cn/models/ByteDance/Dolphin">
    <img src="https://img.shields.io/badge/ModelScope-Dolphin-purple">
  </a>
  <a href="https://huggingface.co/spaces/ByteDance/Dolphin">
    <img src="https://img.shields.io/badge/Demo-Dolphin-blue">
  </a>
  <a href="https://github.com/bytedance/Dolphin">
    <img src="https://img.shields.io/badge/Code-Github-green">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-lightgray">
  </a>
  <br>
</div>

<br>

<div align="center">
  <img src="./assets/demo.gif" width="800">
</div>

# Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting

Dolphin (**Do**cument Image **P**arsing via **H**eterogeneous Anchor Prompt**in**g) is a novel multimodal document image parsing model following an analyze-then-parse paradigm. This repository contains the demo code and pre-trained models for Dolphin.

## Enhanced Mac Compatibility & Modal Deployment

This is an enhanced version of the original [ByteDance Dolphin repository](https://github.com/bytedance/Dolphin) with the following improvements:

- **Mac Apple Silicon (MPS) Support**: Added native Metal Performance Shaders (MPS) backend support for optimal performance on M1/M2/M3 Macs
- **Modal Cloud Deployment**: Configured for easy deployment and scaling on Modal's serverless platform
- **Cross-Platform Compatibility**: Enhanced CPU fallback and improved compatibility across different hardware configurations
- **Streamlined Dependencies**: Updated requirements for better Mac compatibility while maintaining original functionality

All original Dolphin functionality is preserved - this version simply makes it work seamlessly on Mac and provides cloud deployment options.

## 🚀 Modal Cloud Deployment

Deploy Dolphin on Modal.com for production-scale GPU-accelerated document parsing with auto-scaling and high concurrency.

### Quick Start

1. **Install Modal CLI:**
   ```bash
   pip install modal
   modal token new  # Authenticate
   ```

2. **Deploy to Modal:**
   ```bash
   modal deploy modal_app.py
   # Check deployment status
   modal app list
   ```
   
3. **Run Performance Benchmark:**
   ```bash
   python benchmark_modal.py  # Test with 20 simultaneous requests
   ```

4. **Test the API:**
   ```bash
   python test_modal_api.py https://your-app.modal.run ./demo/page_imgs/page_1.jpeg
   ```

### Modal API Usage

**FastAPI endpoints with base64 image data:**

```bash
# Health check
curl https://your-app--health.modal.run

# Parse document (POST with base64 data)
curl -X POST https://your-app--parse.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image_data",
    "filename": "document.jpg"
  }'

# Debug GPU memory usage
curl https://your-app--debug-memory.modal.run
```

**Response:**
```json
{
  "request_id": "abc123",
  "filename": "document.jpg", 
  "processing_time_seconds": 10.23,
  "timestamp": 1640995200.0,
  "results": {
    "content": "parsed document structure...",
    "elements": [...]
  },
  "metadata": {
    "image_size": [1024, 768],
    "model_device": "cuda",
    "total_elements": 9,
    "container_id": "abc12345",
    "container_requests": 3
  }
}
```

### 🚀 Production Performance & Benchmarks

**GPU Configuration**: NVIDIA L4 with Memory Snapshots
- **Cold Start Time**: 2-3s (with snapshots) vs 20-30s (without)
- **Processing Speed**: ~10.2s per page (including model inference)
- **Throughput**: 0.46 pages/second effective (burst traffic)
- **Cost**: $0.0106 per page @ $1.10/hour L4 pricing

**Real-World Burst Performance (20 pages simultaneously)**:
```
📊 BENCHMARK RESULTS (20-Page PDF Simulation)
┌─────────────────────────────┬─────────────────────────────┐
│ Metric                      │ Value                       │
├─────────────────────────────┼─────────────────────────────┤
│ Total Processing Time       │ 43.26 seconds               │
│ Containers Utilized         │ 10 containers               │
│ Parallelization Efficiency  │ 4.7x (excellent)           │
│ Cost per Page              │ $0.0106                     │
│ Success Rate               │ 100% (20/20 pages)         │
│ Cold Start Penalty         │ ~11s average                │
│ Memory Snapshots           │ ✅ Enabled (fast restarts)  │
└─────────────────────────────┴─────────────────────────────┘
```

**Scaling Projections**:
- **100-page document**: ~1.7 minutes, $1.06 cost
- **500-page document**: ~8.5 minutes, $5.29 cost  
- **1000-page document**: ~17.1 minutes, $10.58 cost

**Modal Architecture Benefits**:
- **GPU Acceleration**: NVIDIA L4 optimized for cost/performance
- **Memory Snapshots**: 10x faster cold starts (model pre-loaded)
- **Auto-scaling**: 0 to 20 containers based on demand
- **Burst Traffic Ready**: Handle entire PDF processing simultaneously
- **Cost Optimization**: Scale to zero between jobs

Perfect for document indexing: Upload PDF → All pages processed in parallel → Results in ~1-2 minutes.

## 📑 Overview

Document image parsing is challenging due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Dolphin addresses these challenges through a two-stage approach:

1. **🔍 Stage 1**: Comprehensive page-level layout analysis by generating element sequence in natural reading order
2. **🧩 Stage 2**: Efficient parallel parsing of document elements using heterogeneous anchors and task-specific prompts

<div align="center">
  <img src="./assets/framework.png" width="680">
</div>

Dolphin achieves promising performance across diverse page-level and element-level parsing tasks while ensuring superior efficiency through its lightweight architecture and parallel parsing mechanism.

## 🚀 Demo
Try our demo on [Demo-Dolphin](http://115.190.42.15:8888/dolphin/).

## 📅 Changelog
- 🔥 **2025.07.10** Released the *Fox-Page Benchmark*, a manually refined subset of the original [Fox dataset](https://github.com/ucaslcl/Fox). Download via: [Baidu Yun](https://pan.baidu.com/share/init?surl=t746ULp6iU5bUraVrPlMSw&pwd=fox1) | [Google Drive](https://drive.google.com/file/d/1yZQZqI34QCqvhB4Tmdl3X_XEvYvQyP0q/view?usp=sharing).
- 🔥 **2025.06.30** Added [TensorRT-LLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/tensorrt_llm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.27** Added [vLLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/vllm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.13** Added multi-page PDF document parsing capability.
- 🔥 **2025.05.21** Our demo is released at [link](http://115.190.42.15:8888/dolphin/). Check it out!
- 🔥 **2025.05.20** The pretrained model and inference code of Dolphin are released.
- 🔥 **2025.05.16** Our paper has been accepted by ACL 2025. Paper link: [arXiv](https://arxiv.org/abs/2505.14059).

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ByteDance/Dolphin.git
   cd Dolphin
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained models using one of the following options:

   **Option A: Original Model Format (config-based)**
   
   Download from [Baidu Yun](https://pan.baidu.com/s/15zcARoX0CTOHKbW8bFZovQ?pwd=9rpx) or [Google Drive](https://drive.google.com/drive/folders/1PQJ3UutepXvunizZEw-uGaQ0BCzf-mie?usp=sharing) and put them in the `./checkpoints` folder.

   **Option B: Hugging Face Model Format**
   
   Visit our Huggingface [model card](https://huggingface.co/ByteDance/Dolphin), or download model by:
   
   ```bash
   # Download the model from Hugging Face Hub
   git lfs install
   git clone https://huggingface.co/ByteDance/Dolphin ./hf_model
   # Or use the Hugging Face CLI
   pip install huggingface_hub
   huggingface-cli download ByteDance/Dolphin --local-dir ./hf_model
   ```

## ⚡ Inference

Dolphin provides two inference frameworks with support for two parsing granularities:
- **Page-level Parsing**: Parse the entire document page into a structured JSON and Markdown format
- **Element-level Parsing**: Parse individual document elements (text, table, formula)

### 📄 Page-level Parsing

#### Using Original Framework (config-based)

```bash
# Process a single document image
python demo_page.py --config ./config/Dolphin.yaml --input_path ./demo/page_imgs/page_1.jpeg --save_dir ./results

# Process a single document pdf
python demo_page.py --config ./config/Dolphin.yaml --input_path ./demo/page_imgs/page_6.pdf --save_dir ./results

# Process all documents in a directory
python demo_page.py --config ./config/Dolphin.yaml --input_path ./demo/page_imgs --save_dir ./results

# Process with custom batch size for parallel element decoding
python demo_page.py --config ./config/Dolphin.yaml --input_path ./demo/page_imgs --save_dir ./results --max_batch_size 8
```

#### Using Hugging Face Framework

```bash
# Process a single document image
python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs/page_1.jpeg --save_dir ./results

# Process a single document pdf
python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs/page_6.pdf --save_dir ./results

# Process all documents in a directory
python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs --save_dir ./results

# Process with custom batch size for parallel element decoding
python demo_page_hf.py --model_path ./hf_model --input_path ./demo/page_imgs --save_dir ./results --max_batch_size 16
```

### 🧩 Element-level Parsing

#### Using Original Framework (config-based)

```bash
# Process a single table image
python demo_element.py --config ./config/Dolphin.yaml --input_path ./demo/element_imgs/table_1.jpeg --element_type table

# Process a single formula image
python demo_element.py --config ./config/Dolphin.yaml --input_path ./demo/element_imgs/line_formula.jpeg --element_type formula

# Process a single text paragraph image
python demo_element.py --config ./config/Dolphin.yaml --input_path ./demo/element_imgs/para_1.jpg --element_type text
```

#### Using Hugging Face Framework

```bash
# Process a single table image
python demo_element_hf.py --model_path ./hf_model --input_path ./demo/element_imgs/table_1.jpeg --element_type table

# Process a single formula image
python demo_element_hf.py --model_path ./hf_model --input_path ./demo/element_imgs/line_formula.jpeg --element_type formula

# Process a single text paragraph image
python demo_element_hf.py --model_path ./hf_model --input_path ./demo/element_imgs/para_1.jpg --element_type text
```

## 🌟 Key Features

- 🔄 Two-stage analyze-then-parse approach based on a single VLM
- 📊 Promising performance on document parsing tasks
- 🔍 Natural reading order element sequence generation
- 🧩 Heterogeneous anchor prompting for different document elements
- ⏱️ Efficient parallel parsing mechanism
- 🤗 Support for Hugging Face Transformers for easier integration


## 📮 Notice
**Call for Bad Cases:** If you have encountered any cases where the model performs poorly, we would greatly appreciate it if you could share them in the issue. We are continuously working to optimize and improve the model.

## 💖 Acknowledgement

We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [Donut](https://github.com/clovaai/donut/)
- [Nougat](https://github.com/facebookresearch/nougat)
- [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [MinerU](https://github.com/opendatalab/MinerU/tree/master)
- [Swin](https://github.com/microsoft/Swin-Transformer)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 📝 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{feng2025dolphin,
  title={Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting},
  author={Feng, Hao and Wei, Shu and Fei, Xiang and Shi, Wei and Han, Yingdong and Liao, Lei and Lu, Jinghui and Wu, Binghong and Liu, Qi and Lin, Chunhui and others},
  journal={arXiv preprint arXiv:2505.14059},
  year={2025}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bytedance/Dolphin&type=Date)](https://www.star-history.com/#bytedance/Dolphin&Date)
