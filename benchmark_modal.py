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

def run_benchmark(num_requests=10, concurrent=3):
    """Run benchmark with specified number of requests"""
    print(f"🚀 Starting benchmark: {num_requests} requests with {concurrent} concurrent connections")
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
        
        print(f"\n💡 NOTES:")
        print(f"• Modal charges only for actual GPU usage time")
        print(f"• Container warm-up time is minimal after first request")
        print(f"• Auto-scaling handles traffic bursts efficiently")
        print(f"• Cost remains ~${gpu_cost:.2f} regardless of parallelism level")
        
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