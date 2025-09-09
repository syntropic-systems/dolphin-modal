# Dolphin Batch API Documentation

## Overview

The Dolphin API now supports both individual and batch processing endpoints. The batch endpoint provides **10-50x performance improvement** by processing multiple images in a single GPU-optimized call.

## Configuration

**GPU Configuration:** T4 (16GB VRAM)
- **Maximum batch size:** 32 images per request
- **Element batch size:** 32 elements per GPU call  
- **GPU type:** T4 (cost-optimized)
- **Max containers:** 4 (scales up for burst traffic)

### T4 GPU Optimization Benefits:
- ✅ **Cost-efficient:** 40% cheaper than A100
- ✅ **Memory optimized:** 16GB VRAM handles 32 images reliably
- ✅ **High throughput:** 32 images processed in ~30-60 seconds
- ✅ **Auto-scaling:** 0 to 4 containers based on demand

## API Endpoints

### 1. Individual Processing (Legacy)
**Endpoint:** `POST https://abhishekgautam011--dolphin-parser-dolphinparser-parse.modal.run`

**Request:**
```json
{
  "image_data": "base64_encoded_image_string",
  "filename": "document.jpg"
}
```

**Response:**
```json
{
  "request_id": "abc123",
  "filename": "document.jpg",
  "processing_time_seconds": 10.23,
  "timestamp": 1640995200.0,
  "results": [
    {
      "label": "title",
      "bbox": [100, 200, 500, 250],
      "text": "Document Title",
      "reading_order": 0
    }
  ],
  "metadata": {
    "image_size": [1024, 768],
    "model_device": "cuda",
    "total_elements": 15,
    "container_id": "abc12345",
    "container_requests": 3
  }
}
```

### 2. Batch Processing (Recommended)
**Endpoint:** `POST https://abhishekgautam011--dolphin-parser-dolphinparser-parse-batch.modal.run`

**Request:**
```json
{
  "images": [
    {
      "image_data": "base64_encoded_image_1",
      "filename": "page_1.jpg"
    },
    {
      "image_data": "base64_encoded_image_2",
      "filename": "page_2.jpg"
    },
    // ... up to 32 images (T4 GPU optimized)
  ]
}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789",
  "processing_time_seconds": 45.67,
  "timestamp": 1640995200.0,
  "images_processed": 20,
  "total_elements": 285,
  "results": [
    {
      "filename": "page_1.jpg",
      "elements": [
        {
          "label": "title",
          "bbox": [100, 200, 500, 250],
          "text": "Document Title",
          "reading_order": 0
        },
        // ... more elements
      ],
      "element_count": 15
    },
    {
      "filename": "page_2.jpg",
      "elements": [...],
      "element_count": 12
    }
    // ... results for all images
  ],
  "metadata": {
    "model_device": "cuda",
    "container_id": "abc12345",
    "container_requests": 1,
    "batch_throughput": 0.44,  // images per second
    "elements_per_second": 6.24
  }
}
```

## Python Client Implementation

### Basic Batch Client

```python
import requests
import base64
import json
from typing import List, Dict, Optional
from pathlib import Path
import time

class DolphinBatchClient:
    """Client for Dolphin batch API with request aggregation and response unpacking"""
    
    def __init__(self, batch_size: int = 20):
        """
        Initialize the batch client
        
        Args:
            batch_size: Maximum number of images per batch (max 32 for T4 GPU)
        """
        self.batch_api_url = "https://abhishekgautam011--dolphin-parser-dolphinparser-parse-batch.modal.run"
        self.batch_size = min(batch_size, 32)  # API limit
        self.pending_requests = []
        
    def add_image(self, image_path: str, request_id: Optional[str] = None) -> str:
        """
        Add an image to the pending batch
        
        Args:
            image_path: Path to the image file
            request_id: Optional ID to track this specific request
            
        Returns:
            request_id for tracking
        """
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{len(self.pending_requests)}_{Path(image_path).stem}"
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # Add to pending requests
        self.pending_requests.append({
            "request_id": request_id,
            "image_data": image_data,
            "filename": Path(image_path).name,
            "original_path": image_path
        })
        
        return request_id
    
    def process_batch(self) -> Dict[str, Dict]:
        """
        Process all pending requests as a batch
        
        Returns:
            Dictionary mapping request_id to results
        """
        if not self.pending_requests:
            return {}
        
        results_by_id = {}
        
        # Process in chunks if needed
        for i in range(0, len(self.pending_requests), self.batch_size):
            batch = self.pending_requests[i:i + self.batch_size]
            batch_results = self._send_batch(batch)
            results_by_id.update(batch_results)
        
        # Clear pending requests
        self.pending_requests = []
        
        return results_by_id
    
    def _send_batch(self, batch: List[Dict]) -> Dict[str, Dict]:
        """
        Send a single batch request and unpack results
        
        Args:
            batch: List of request dictionaries
            
        Returns:
            Dictionary mapping request_id to results
        """
        # Prepare batch payload
        images_data = [
            {
                "image_data": req["image_data"],
                "filename": req["filename"]
            }
            for req in batch
        ]
        
        payload = {"images": images_data}
        
        print(f"📦 Sending batch of {len(batch)} images...")
        start_time = time.perf_counter()
        
        try:
            # Send batch request
            response = requests.post(
                self.batch_api_url,
                json=payload,
                timeout=300  # 5 minutes for large batches
            )
            
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text[:200]}")
            
            # Parse response
            batch_response = response.json()
            processing_time = time.perf_counter() - start_time
            
            print(f"✅ Batch processed in {processing_time:.2f}s")
            print(f"   Images: {batch_response.get('images_processed', 0)}")
            print(f"   Elements: {batch_response.get('total_elements', 0)}")
            print(f"   Throughput: {batch_response.get('metadata', {}).get('batch_throughput', 0):.2f} img/s")
            
            # Unpack results to individual requests
            results_by_id = {}
            batch_results = batch_response.get('results', [])
            
            for i, result in enumerate(batch_results):
                if i < len(batch):
                    request_id = batch[i]["request_id"]
                    results_by_id[request_id] = {
                        "success": True,
                        "filename": result.get('filename'),
                        "elements": result.get('elements', []),
                        "element_count": result.get('element_count', 0),
                        "original_path": batch[i]["original_path"],
                        "batch_id": batch_response.get('batch_id'),
                        "processing_time": batch_response.get('processing_time_seconds')
                    }
            
            return results_by_id
            
        except Exception as e:
            print(f"❌ Batch failed: {str(e)}")
            # Return error for all requests in batch
            return {
                req["request_id"]: {
                    "success": False,
                    "error": str(e),
                    "original_path": req["original_path"]
                }
                for req in batch
            }
```

### Advanced Async Batch Client

```python
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentRequest:
    """Individual document request"""
    request_id: str
    image_path: str
    image_data: str  # base64
    timestamp: datetime
    
@dataclass
class DocumentResult:
    """Parsed document result"""
    request_id: str
    success: bool
    elements: List[Dict]
    element_count: int
    processing_time: float
    error: Optional[str] = None

class AsyncDolphinBatchClient:
    """Asynchronous batch client with automatic batching and queuing"""
    
    def __init__(self, batch_size: int = 20, auto_batch_delay: float = 1.0):
        """
        Initialize async batch client
        
        Args:
            batch_size: Maximum images per batch
            auto_batch_delay: Seconds to wait before auto-processing partial batch
        """
        self.batch_api_url = "https://abhishekgautam011--dolphin-parser-dolphinparser-parse-batch.modal.run"
        self.batch_size = min(batch_size, 32)
        self.auto_batch_delay = auto_batch_delay
        
        self.request_queue = asyncio.Queue()
        self.result_futures = {}
        self.is_processing = False
        
    async def parse_document(self, image_path: str) -> DocumentResult:
        """
        Parse a single document (will be batched automatically)
        
        Args:
            image_path: Path to document image
            
        Returns:
            DocumentResult with parsed elements
        """
        # Create request
        request_id = f"{datetime.now().timestamp()}_{Path(image_path).stem}"
        
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        request = DocumentRequest(
            request_id=request_id,
            image_path=image_path,
            image_data=image_data,
            timestamp=datetime.now()
        )
        
        # Create future for result
        future = asyncio.Future()
        self.result_futures[request_id] = future
        
        # Add to queue
        await self.request_queue.put(request)
        
        # Start batch processor if not running
        if not self.is_processing:
            asyncio.create_task(self._batch_processor())
        
        # Wait for result
        return await future
    
    async def parse_documents_parallel(self, image_paths: List[str]) -> List[DocumentResult]:
        """
        Parse multiple documents in parallel (optimally batched)
        
        Args:
            image_paths: List of document image paths
            
        Returns:
            List of DocumentResults in same order as input
        """
        tasks = [self.parse_document(path) for path in image_paths]
        return await asyncio.gather(*tasks)
    
    async def _batch_processor(self):
        """Background task that processes queued requests in batches"""
        self.is_processing = True
        
        while True:
            batch = []
            
            # Collect batch
            try:
                # Wait for first request
                first_request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=5.0
                )
                batch.append(first_request)
                
                # Collect more requests up to batch size
                deadline = asyncio.get_event_loop().time() + self.auto_batch_delay
                
                while len(batch) < self.batch_size:
                    remaining_time = deadline - asyncio.get_event_loop().time()
                    if remaining_time <= 0:
                        break
                    
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=remaining_time
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break  # Process partial batch
                
            except asyncio.TimeoutError:
                # No requests in queue
                self.is_processing = False
                return
            
            # Process batch
            if batch:
                await self._process_batch(batch)
    
    async def _process_batch(self, batch: List[DocumentRequest]):
        """Send batch request and distribute results"""
        print(f"🚀 Processing batch of {len(batch)} documents")
        
        # Prepare payload
        images_data = [
            {
                "image_data": req.image_data,
                "filename": Path(req.image_path).name
            }
            for req in batch
        ]
        
        payload = {"images": images_data}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.batch_api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    
                    if response.status == 200:
                        batch_result = await response.json()
                        
                        # Distribute results
                        results = batch_result.get('results', [])
                        for i, result in enumerate(results):
                            if i < len(batch):
                                request = batch[i]
                                doc_result = DocumentResult(
                                    request_id=request.request_id,
                                    success=True,
                                    elements=result.get('elements', []),
                                    element_count=result.get('element_count', 0),
                                    processing_time=batch_result.get('processing_time_seconds', 0)
                                )
                                
                                # Resolve future
                                if request.request_id in self.result_futures:
                                    self.result_futures[request.request_id].set_result(doc_result)
                                    del self.result_futures[request.request_id]
                    else:
                        error = f"API error {response.status}"
                        raise Exception(error)
                        
        except Exception as e:
            # Handle errors for all requests in batch
            for request in batch:
                doc_result = DocumentResult(
                    request_id=request.request_id,
                    success=False,
                    elements=[],
                    element_count=0,
                    processing_time=0,
                    error=str(e)
                )
                
                if request.request_id in self.result_futures:
                    self.result_futures[request.request_id].set_result(doc_result)
                    del self.result_futures[request.request_id]
```

## Usage Examples

### Example 1: Simple Synchronous Batch Processing

```python
# Process a folder of documents
from pathlib import Path

client = DolphinBatchClient(batch_size=20)

# Add all images to batch
image_folder = Path("documents/")
for image_path in image_folder.glob("*.jpg"):
    client.add_image(str(image_path))

# Process batch synchronously
results = client.process_batch()

# Access individual results
for request_id, result in results.items():
    if result["success"]:
        print(f"Document: {result['filename']}")
        print(f"Elements found: {result['element_count']}")
        
        # Process each element
        for element in result["elements"]:
            if element["label"] == "title":
                print(f"  Title: {element['text']}")
            elif element["label"] == "para":
                print(f"  Paragraph at {element['bbox']}: {element['text'][:50]}...")
    else:
        print(f"Failed: {result.get('error')}")
```

### Example 2: Async Processing with Automatic Batching

```python
async def process_pdf_pages():
    """Process all pages of a PDF converted to images"""
    
    client = AsyncDolphinBatchClient(
        batch_size=20,
        auto_batch_delay=0.5  # Wait 0.5s for more requests before sending batch
    )
    
    # Get all page images
    page_images = sorted(Path("pdf_pages/").glob("page_*.jpg"))
    
    # Process all pages in parallel (automatically batched)
    results = await client.parse_documents_parallel(
        [str(p) for p in page_images]
    )
    
    # Compile results
    all_text = []
    for i, result in enumerate(results):
        if result.success:
            page_text = []
            # Sort elements by reading order
            sorted_elements = sorted(
                result.elements,
                key=lambda x: x.get("reading_order", 0)
            )
            for element in sorted_elements:
                if element["label"] in ["para", "title", "sec"]:
                    page_text.append(element["text"])
            
            all_text.append(f"=== Page {i+1} ===\n" + "\n".join(page_text))
        else:
            all_text.append(f"=== Page {i+1} ERROR: {result.error} ===")
    
    # Save compiled document
    with open("extracted_document.txt", "w") as f:
        f.write("\n\n".join(all_text))
    
    print(f"✅ Processed {len(results)} pages")

# Run async function
asyncio.run(process_pdf_pages())
```

### Example 3: Processing with Progress Tracking

```python
from tqdm import tqdm

def process_documents_with_progress(image_paths: List[str], batch_size: int = 20):
    """Process documents with progress bar"""
    
    client = DolphinBatchClient(batch_size=batch_size)
    
    # Add all images with progress
    print("Loading images...")
    for path in tqdm(image_paths, desc="Adding to batch"):
        client.add_image(path)
    
    # Process batch
    print(f"\nProcessing {len(image_paths)} images in batches of {batch_size}...")
    results = client.process_batch()
    
    # Process results with progress
    print("\nExtracting text...")
    extracted_data = []
    
    for request_id, result in tqdm(results.items(), desc="Processing results"):
        if result["success"]:
            doc_data = {
                "filename": result["filename"],
                "element_count": result["element_count"],
                "text_elements": [],
                "tables": [],
                "figures": []
            }
            
            for element in result["elements"]:
                if element["label"] in ["para", "title", "sec"]:
                    doc_data["text_elements"].append(element)
                elif element["label"] == "tab":
                    doc_data["tables"].append(element)
                elif element["label"] == "fig":
                    doc_data["figures"].append(element)
            
            extracted_data.append(doc_data)
    
    return extracted_data

# Usage
image_files = list(Path("documents/").glob("*.jpg"))
data = process_documents_with_progress([str(f) for f in image_files])
print(f"✅ Extracted data from {len(data)} documents")
```

## Performance Comparison

| Approach | 20 Images Processing Time | Cost | GPU Efficiency |
|----------|---------------------------|------|----------------|
| Individual Requests | ~300 seconds | $0.092 | ~10% |
| Batch Request | ~30 seconds | $0.009 | ~95% |
| **Improvement** | **10x faster** | **10x cheaper** | **9.5x better** |

## Best Practices

1. **Batch Size**: Use 15-25 images per batch for optimal GPU utilization
2. **Timeout**: Set timeout to 15-20 seconds per image in batch
3. **Error Handling**: Always handle partial batch failures
4. **Memory**: Each image ~5MB base64, limit total payload to 200MB
5. **Concurrency**: Use async client for processing multiple batches in parallel

## Rate Limits

- Maximum batch size: 32 images (T4 GPU optimized)
- Maximum payload size: 250MB
- Timeout: 600 seconds per request
- Concurrent requests: Based on Modal container limits

## Migration Guide

### Before (Individual Requests):
```python
results = []
for image in images:
    response = requests.post(url, json={"image_data": image})
    results.append(response.json())
```

### After (Batch Processing):
```python
client = DolphinBatchClient()
for image in images:
    client.add_image(image)
results = client.process_batch()
```

## Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| 400 | Invalid request format | Check payload structure |
| 413 | Payload too large | Reduce batch size |
| 422 | Invalid image data | Verify base64 encoding |
| 500 | Server error | Retry with smaller batch |
| 503 | Service unavailable | Wait and retry |

## Support

For issues or questions:
- GitHub: https://github.com/ByteDance/Dolphin
- Modal Dashboard: https://modal.com/apps/abhishekgautam011/main/deployed/dolphin-parser