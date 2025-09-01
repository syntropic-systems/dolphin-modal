#!/usr/bin/env python3
"""
CPU vs MPS Performance Benchmark for Dolphin Model
"""

import time
import torch
from PIL import Image
from demo_page_hf import DOLPHIN

def benchmark_device(model_path, image_path, device_name, num_runs=3):
    """Benchmark model performance on specified device"""
    print(f"\n{'='*50}")
    print(f"Benchmarking on {device_name.upper()}")
    print(f"{'='*50}")
    
    # Load model
    model = DOLPHIN(model_path)
    
    # Override device selection for benchmark
    if device_name == "cpu":
        model.device = "cpu"
        model.model = model.model.to("cpu").float()
    elif device_name == "mps":
        model.device = "mps" 
        model.model = model.model.to("mps").float()
    
    print(f"Model device: {model.device}")
    print(f"Model dtype: {next(model.model.parameters()).dtype}")
    
    # Load test image
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Warmup run
    print("Warming up...")
    _ = model.chat("Parse the reading order of this document.", image)
    
    # Benchmark runs
    times = []
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")
        start_time = time.perf_counter()
        
        result = model.chat("Parse the reading order of this document.", image)
        
        end_time = time.perf_counter()
        run_time = end_time - start_time
        times.append(run_time)
        print(f"  Time: {run_time:.2f}s")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults for {device_name.upper()}:")
    print(f"  Average time: {avg_time:.2f}s")
    print(f"  Min time: {min_time:.2f}s") 
    print(f"  Max time: {max_time:.2f}s")
    print(f"  Output length: {len(result)} characters")
    
    return avg_time, min_time, max_time

def main():
    model_path = "./hf_model"
    image_path = "./demo/page_imgs/page_1.jpeg"
    
    print("Dolphin Model Performance Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    # Benchmark CPU
    cpu_avg, cpu_min, cpu_max = benchmark_device(model_path, image_path, "cpu")
    
    # Benchmark MPS (if available)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        mps_avg, mps_min, mps_max = benchmark_device(model_path, image_path, "mps")
        
        # Comparison
        print(f"\n{'='*50}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*50}")
        print(f"CPU Average:  {cpu_avg:.2f}s")
        print(f"MPS Average:  {mps_avg:.2f}s")
        print(f"Speedup:      {cpu_avg/mps_avg:.2f}x")
        print(f"MPS is {((cpu_avg - mps_avg) / cpu_avg * 100):.1f}% faster")
    else:
        print("MPS not available - skipping MPS benchmark")

if __name__ == "__main__":
    main()