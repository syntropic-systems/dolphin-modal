#!/usr/bin/env python3
"""
Benchmark Modal Dolphin API and estimate costs for 1000 pages
"""

import requests
import base64
import json
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# API endpoints
API_URL = 'https://abhishekgautam011--dolphin-parser-dolphinparser-parse.modal.run'
HEALTH_URL = 'https://abhishekgautam011--dolphin-parser-dolphinparser-health.modal.run'
DEBUG_URL = 'https://abhishekgautam011--dolphin-parser-dolphinparser-debug-memory.modal.run'

def process_image(image_path, request_num):
    """Process a single image and return timing info"""
    try:
        # Convert to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        payload = {
            'image_data': image_data,
            'filename': f'{image_path.name}_req{request_num}'
        }
        
        # Make request
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=120)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'request_num': request_num,
                'total_time': total_time,
                'processing_time': result.get('processing_time_seconds', 0),
                'elements': result.get('metadata', {}).get('total_elements', 0)
            }
        else:
            return {
                'success': False,
                'request_num': request_num,
                'error': response.text
            }
    except Exception as e:
        return {
            'success': False,
            'request_num': request_num,
            'error': str(e)
        }

def get_memory_usage():
    """Get GPU memory usage from deployed model"""
    try:
        response = requests.get(DEBUG_URL, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Debug endpoint failed: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def run_benchmark(num_requests=10, concurrent=3):
    """Run benchmark with specified number of requests"""
    print(f"🚀 Starting benchmark: {num_requests} requests with {concurrent} concurrent connections")
    print("=" * 60)
    
    # First check memory usage
    print("🔍 Checking GPU memory usage...")
    memory_info = get_memory_usage()
    if "error" not in memory_info:
        print(f"📊 GPU Memory Info:")
        print(f"   Used: {memory_info.get('used_gb', 0):.2f} GB")
        print(f"   Total: {memory_info.get('total_gb', 0):.2f} GB") 
        print(f"   Free: {memory_info.get('free_gb', 0):.2f} GB")
        print(f"   Model instances possible: {memory_info.get('possible_instances', 'Unknown')}")
    else:
        print(f"⚠️  Could not get memory info: {memory_info.get('error', 'Unknown')}")
    print("=" * 60)
    
    # Get test images
    image_dir = Path('demo/page_imgs')
    images = list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg'))
    if not images:
        print("❌ No test images found")
        return
    
    print(f"📄 Found {len(images)} test images")
    
    results = []
    start_benchmark = time.time()
    
    # Run requests with concurrency - use the same image for consistency
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = []
        # Use the first image for all requests for consistent benchmarking
        test_image = images[0] if images else None
        print(f"Using image: {test_image.name if test_image else 'None'}")
        
        for i in range(num_requests):
            future = executor.submit(process_image, test_image, i+1)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result['success']:
                print(f"✅ Request {result['request_num']}: {result['total_time']:.2f}s (processing: {result['processing_time']}s)")
            else:
                print(f"❌ Request {result['request_num']}: Failed - {result.get('error', 'Unknown error')}")
    
    end_benchmark = time.time()
    total_benchmark_time = end_benchmark - start_benchmark
    
    # Calculate statistics
    successful = [r for r in results if r['success']]
    if successful:
        total_times = [r['total_time'] for r in successful]
        processing_times = [r['processing_time'] for r in successful]
        
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total requests: {num_requests}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {num_requests - len(successful)}")
        print(f"Total benchmark time: {total_benchmark_time:.2f} seconds")
        print(f"Throughput: {len(successful)/total_benchmark_time:.2f} pages/second")
        
        print(f"\n⏱️  TIMING STATISTICS (successful requests):")
        print(f"Average total time: {statistics.mean(total_times):.2f} seconds")
        print(f"Median total time: {statistics.median(total_times):.2f} seconds")
        print(f"Min total time: {min(total_times):.2f} seconds")
        print(f"Max total time: {max(total_times):.2f} seconds")
        
        print(f"\nAverage processing time: {statistics.mean(processing_times):.2f} seconds")
        print(f"Median processing time: {statistics.median(processing_times):.2f} seconds")
        
        # Create detailed table
        print("\n📋 DETAILED REQUEST TABLE:")
        print("┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓")
        print("┃ Request┃ Total Time (s)┃ Processing Time (s)┃ Network Overhead ┃ Elements  ┃")
        print("┣━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━┫")
        for r in successful:
            overhead = r['total_time'] - r['processing_time']
            print(f"┃ {r['request_num']:^6d} ┃ {r['total_time']:^13.2f} ┃ {r['processing_time']:^17.2f} ┃ {overhead:^16.2f} ┃ {r['elements']:^9d} ┃")
        print("┗━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━┛")
        
        # Cost estimation for 1000 pages
        print("\n" + "=" * 60)
        print("💰 COST ESTIMATION FOR 1000 PAGES")
        print("=" * 60)
        
        # Modal A100 pricing (as of 2024)
        # A100 40GB: ~$3.72/hour
        gpu_cost_per_hour = 3.72
        gpu_cost_per_second = gpu_cost_per_hour / 3600
        
        # Calculate based on actual processing time (GPU usage)
        avg_processing_time = statistics.mean(processing_times)
        total_gpu_seconds = avg_processing_time * 1000
        total_gpu_hours = total_gpu_seconds / 3600
        gpu_cost = total_gpu_seconds * gpu_cost_per_second
        
        print(f"Average processing time per page: {avg_processing_time:.2f} seconds")
        print(f"Total GPU time for 1000 pages: {total_gpu_hours:.2f} hours ({total_gpu_seconds:.0f} seconds)")
        print(f"GPU cost (A100 @ ${gpu_cost_per_hour}/hour): ${gpu_cost:.2f}")
        
        # With concurrency estimation
        if concurrent > 1:
            concurrent_time = total_gpu_seconds / concurrent
            concurrent_hours = concurrent_time / 3600
            print(f"\nWith {concurrent} concurrent workers:")
            print(f"Wall clock time: {concurrent_hours:.2f} hours ({concurrent_time:.0f} seconds)")
            print(f"Throughput: {1000/concurrent_time:.2f} pages/second")
        
        # Different batch sizes
        print(f"\n📈 THROUGHPUT & TIME ESTIMATES:")
        for workers in [1, 3, 5, 10]:
            wall_time = (avg_processing_time * 1000) / workers
            wall_hours = wall_time / 3600
            throughput = workers / avg_processing_time
            print(f"{workers:2d} workers: {wall_hours:6.2f} hours, {throughput:6.2f} pages/sec")
        
        # Optimization potential analysis
        print(f"\n" + "=" * 60)
        print("🚀 OPTIMIZATION POTENTIAL (A100-40GB)")
        print("=" * 60)
        
        # Get memory info again to show optimization potential
        memory_info = get_memory_usage()
        if "error" not in memory_info:
            model_memory = memory_info.get('estimated_model_memory_gb', 4.0)  # Default estimate
            possible_instances = memory_info.get('possible_instances', 1)
            
            print(f"Current model memory usage: {model_memory:.2f} GB")
            print(f"Possible model instances on A100-40GB: {possible_instances}")
            
            # Calculate optimization scenarios
            if possible_instances > 1:
                # Scenario 1: Multiple model instances with batching
                optimized_time = avg_processing_time / possible_instances
                optimized_cost = (optimized_time * 1000 * gpu_cost_per_second)
                cost_reduction = ((gpu_cost - optimized_cost) / gpu_cost) * 100
                
                print(f"\n📈 MULTI-INSTANCE OPTIMIZATION:")
                print(f"   With {possible_instances} model instances:")
                print(f"   Processing time per page: {optimized_time:.2f}s (vs {avg_processing_time:.2f}s)")
                print(f"   Cost for 1000 pages: ${optimized_cost:.2f} (vs ${gpu_cost:.2f})")
                print(f"   Cost reduction: {cost_reduction:.1f}%")
                print(f"   Throughput: {possible_instances/avg_processing_time:.2f} pages/sec")
                
                # Theoretical batch processing
                batch_sizes = [8, 16, 32]
                print(f"\n📦 BATCH PROCESSING POTENTIAL:")
                for batch_size in batch_sizes:
                    # Estimate batch speedup (not linear, diminishing returns)
                    batch_speedup = min(batch_size * 0.7, batch_size)  # 70% efficiency
                    batch_time = avg_processing_time / batch_speedup
                    batch_throughput = batch_size / avg_processing_time
                    batch_cost = (batch_time * 1000 * gpu_cost_per_second) / batch_size
                    
                    print(f"   Batch size {batch_size:2d}: {batch_time:.2f}s/page, {batch_throughput:.2f} pages/sec, ${batch_cost:.4f}/page")
        
        # Compare with Mistral's pricing
        mistral_cost_per_page = 0.001  # $1 per 1000 pages
        current_cost_per_page = gpu_cost / 1000
        cost_gap = current_cost_per_page / mistral_cost_per_page
        
        print(f"\n💰 COST COMPARISON:")
        print(f"Current cost per page: ${current_cost_per_page:.4f}")
        print(f"Mistral cost per page: ${mistral_cost_per_page:.4f}")
        print(f"Cost gap: {cost_gap:.1f}x more expensive")
        print(f"Target optimization needed: {cost_gap:.1f}x speedup or memory efficiency")

        print(f"\n💡 OPTIMIZATION NOTES:")
        print(f"• Current setup uses single model instance per container")
        print(f"• A100-40GB can potentially fit {memory_info.get('possible_instances', '?')} model instances")
        print(f"• Batch processing could provide 5-10x speedup for multiple pages")
        print(f"• INT8 quantization could double model instances (2x memory savings)")
        print(f"• Dynamic batching + multi-instance = path to Mistral-level pricing")
        
        return {
            'total_requests': num_requests,
            'successful': len(successful),
            'avg_processing_time': avg_processing_time,
            'estimated_cost_1000': gpu_cost,
            'throughput': len(successful)/total_benchmark_time
        }

if __name__ == "__main__":
    # Run benchmark with 10 requests, 3 concurrent workers
    results = run_benchmark(num_requests=10, concurrent=3)