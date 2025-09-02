#!/usr/bin/env python3
"""
Benchmark Modal Dolphin API with burst traffic pattern
Simulates real-world usage: all PDF pages arrive simultaneously
"""

import requests
import base64
import json
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import random

# API endpoints (updated from modal deploy output)
API_URL = 'https://abhishekgautam011--dolphin-parser-dolphinparser-parse.modal.run'
HEALTH_URL = 'https://abhishekgautam011--dolphin-parser-dolphinparser-health.modal.run'
DEBUG_URL = 'https://abhishekgautam011--dolphin-parser-dolphinparser-debug-memory.modal.run'

def process_image(image_path: Path, request_num: int) -> Dict:
    """Process a single image and return timing info"""
    try:
        # Convert to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        payload = {
            'image_data': image_data,
            'filename': f'page_{request_num}.{image_path.suffix}'
        }
        
        # Make request with detailed timing
        start_time = time.perf_counter()
        response = requests.post(API_URL, json=payload, timeout=120)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            container_info = result.get('metadata', {})
            
            return {
                'success': True,
                'request_num': request_num,
                'total_time': total_time,
                'processing_time': result.get('processing_time_seconds', 0),
                'network_overhead': total_time - result.get('processing_time_seconds', 0),
                'elements': container_info.get('total_elements', 0),
                'container_id': container_info.get('container_id', 'unknown'),
                'container_requests': container_info.get('container_requests', 0),
                'timestamp': time.perf_counter()
            }
        else:
            return {
                'success': False,
                'request_num': request_num,
                'error': f"Status {response.status_code}: {response.text[:100]}",
                'timestamp': time.perf_counter()
            }
    except Exception as e:
        return {
            'success': False,
            'request_num': request_num,
            'error': str(e)[:100],
            'timestamp': time.perf_counter()
        }

def run_burst_benchmark(num_pages: int = 20):
    """
    Run benchmark simulating burst traffic pattern
    All requests sent simultaneously like a PDF being processed
    """
    print(f"🚀 BURST TRAFFIC BENCHMARK")
    print(f"   Simulating {num_pages}-page PDF processing")
    print(f"   All {num_pages} requests sent simultaneously")
    print("=" * 80)
    
    # Get test images
    image_dir = Path('demo/page_imgs')
    images = list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    if not images:
        print("❌ No test images found in demo/page_imgs/")
        return
    
    print(f"📄 Found {len(images)} test images")
    
    # Select images for the burst (cycle through if needed)
    test_images = []
    for i in range(num_pages):
        test_images.append(images[i % len(images)])
    
    print(f"📊 Using {len(set(test_images))} unique images for {num_pages} pages")
    
    # Warm up check (optional)
    print("\n🔥 Checking service health...")
    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        if health_response.status_code == 200:
            health = health_response.json()
            print(f"✅ Service healthy - Container ready: {health.get('container_id', 'unknown')}")
    except:
        print("⚠️  Could not reach health endpoint")
    
    print("\n" + "=" * 80)
    print("📤 SENDING BURST OF REQUESTS...")
    print("=" * 80)
    
    results = []
    burst_start = time.perf_counter()
    
    # Send all requests simultaneously
    with ThreadPoolExecutor(max_workers=num_pages) as executor:
        futures = {}
        
        # Submit all requests at once
        for i in range(num_pages):
            future = executor.submit(process_image, test_images[i], i+1)
            futures[future] = i+1
            print(f"→ Submitted request {i+1}/{num_pages}", end='\r')
        
        print(f"\n✅ All {num_pages} requests submitted\n")
        
        # Track completion order and timing
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            result = future.result()
            results.append(result)
            
            if result['success']:
                print(f"[{completed_count:2d}/{num_pages}] ✅ Page {result['request_num']:2d}: "
                      f"{result['total_time']:5.2f}s total "
                      f"({result['processing_time']:5.2f}s processing) "
                      f"on container {result.get('container_id', '?')}")
            else:
                print(f"[{completed_count:2d}/{num_pages}] ❌ Page {result['request_num']:2d}: "
                      f"{result.get('error', 'Unknown error')}")
    
    burst_end = time.perf_counter()
    total_burst_time = burst_end - burst_start
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if not successful:
        print("\n❌ All requests failed!")
        return
    
    print("\n" + "=" * 80)
    print("📊 BURST PROCESSING RESULTS")
    print("=" * 80)
    
    print(f"\n📈 OVERALL METRICS:")
    print(f"   Total pages processed: {num_pages}")
    print(f"   Successful: {len(successful)} ({len(successful)/num_pages*100:.1f}%)")
    print(f"   Failed: {len(failed)} ({len(failed)/num_pages*100:.1f}%)")
    print(f"   Total burst completion time: {total_burst_time:.2f} seconds")
    print(f"   Effective throughput: {len(successful)/total_burst_time:.2f} pages/second")
    
    # Container utilization analysis
    containers_used = len(set(r.get('container_id', 'unknown') for r in successful))
    print(f"\n🖥️  CONTAINER UTILIZATION:")
    print(f"   Unique containers used: {containers_used}")
    print(f"   Average pages per container: {len(successful)/containers_used:.1f}")
    
    # Timing analysis
    total_times = [r['total_time'] for r in successful]
    processing_times = [r['processing_time'] for r in successful]
    network_overheads = [r['network_overhead'] for r in successful]
    
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"   Average total time: {statistics.mean(total_times):.2f}s")
    print(f"   Average processing: {statistics.mean(processing_times):.2f}s")
    print(f"   Average network overhead: {statistics.mean(network_overheads):.2f}s")
    print(f"   Fastest page: {min(total_times):.2f}s")
    print(f"   Slowest page: {max(total_times):.2f}s")
    
    # Identify cold starts (first request on each container)
    container_first_requests = {}
    for r in successful:
        container_id = r.get('container_id', 'unknown')
        if container_id not in container_first_requests:
            container_first_requests[container_id] = r
    
    cold_start_times = [r['total_time'] for r in container_first_requests.values()]
    warm_times = [r['total_time'] for r in successful if r not in container_first_requests.values()]
    
    if cold_start_times:
        print(f"\n🥶 COLD START ANALYSIS:")
        print(f"   Cold starts detected: {len(cold_start_times)}")
        print(f"   Average cold start time: {statistics.mean(cold_start_times):.2f}s")
        if warm_times:
            print(f"   Average warm request time: {statistics.mean(warm_times):.2f}s")
            print(f"   Cold start penalty: {statistics.mean(cold_start_times) - statistics.mean(warm_times):.2f}s")
    
    # Detailed request table
    print(f"\n📋 DETAILED REQUEST TABLE:")
    print("┏━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓")
    print("┃ Page ┃ Total Time(s)┃ Processing(s) ┃ Network(s)   ┃ Container   ┃ Status     ┃")
    print("┣━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━╋━━━━━━━━━━━━┫")
    
    # Sort by request number for clarity
    sorted_results = sorted(successful, key=lambda x: x['request_num'])
    for r in sorted_results[:30]:  # Show first 30 for space
        container_id = r.get('container_id', 'unknown')[:8]
        is_cold = r in container_first_requests.values()
        status = "COLD" if is_cold else "WARM"
        print(f"┃ {r['request_num']:^4d} ┃ {r['total_time']:^12.2f} ┃ {r['processing_time']:^13.2f} ┃ "
              f"{r['network_overhead']:^12.2f} ┃ {container_id:^11s} ┃ {status:^10s} ┃")
    
    if len(sorted_results) > 30:
        print(f"┃  ... ┃     ...      ┃      ...      ┃     ...      ┃     ...     ┃    ...     ┃")
    print("┗━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━━┛")
    
    # Cost estimation
    print(f"\n" + "=" * 80)
    print("💰 COST ANALYSIS")
    print("=" * 80)
    
    # Modal GPU pricing (update based on current GPU type)
    # A100: $3.72/hour, L4: $1.10/hour, H100: $4.50/hour
    gpu_cost_per_hour = 1.10  # L4 pricing
    gpu_cost_per_second = gpu_cost_per_hour / 3600
    
    # Calculate actual GPU time (sum of all processing times)
    total_gpu_seconds = sum(processing_times)
    total_cost = total_gpu_seconds * gpu_cost_per_second
    cost_per_page = total_cost / len(successful)
    
    print(f"\n📊 This Burst:")
    print(f"   Total GPU compute time: {total_gpu_seconds:.2f} seconds")
    print(f"   Wall clock time: {total_burst_time:.2f} seconds")
    print(f"   Parallelization efficiency: {total_gpu_seconds/total_burst_time:.1f}x")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Cost per page: ${cost_per_page:.4f}")
    
    # Extrapolate to larger documents
    print(f"\n📈 EXTRAPOLATION:")
    for pages in [100, 500, 1000]:
        est_gpu_time = (total_gpu_seconds / len(successful)) * pages
        est_wall_time = est_gpu_time / containers_used  # Assume same parallelization
        est_cost = est_gpu_time * gpu_cost_per_second
        
        print(f"\n   {pages}-page document:")
        print(f"      Estimated completion: {est_wall_time:.1f} seconds ({est_wall_time/60:.1f} minutes)")
        print(f"      Estimated cost: ${est_cost:.2f}")
        print(f"      Throughput: {pages/est_wall_time:.1f} pages/second")
    
    # Optimization recommendations
    print(f"\n" + "=" * 80)
    print("💡 OPTIMIZATION INSIGHTS")
    print("=" * 80)
    
    if containers_used < num_pages:
        print(f"⚠️  Only {containers_used} containers used for {num_pages} pages")
        print(f"   → Consider increasing max_containers for better parallelization")
    
    if cold_start_times and warm_times:
        cold_penalty_pct = ((statistics.mean(cold_start_times) - statistics.mean(warm_times)) / statistics.mean(warm_times)) * 100
        if cold_penalty_pct > 50:
            print(f"⚠️  Cold starts add {cold_penalty_pct:.0f}% overhead")
            print(f"   → Memory snapshots can reduce this significantly")
    
    avg_processing = statistics.mean(processing_times)
    if avg_processing > 5:
        print(f"⚠️  Average processing time {avg_processing:.1f}s is high")
        print(f"   → Consider batch processing or model optimization")
    
    print(f"\n✅ Current configuration handles {len(successful)/total_burst_time:.1f} pages/second")
    print(f"   Perfect for burst traffic with {containers_used} parallel workers")

if __name__ == "__main__":
    # Run burst benchmark with 20 pages (simulating a typical PDF)
    run_burst_benchmark(num_pages=20)