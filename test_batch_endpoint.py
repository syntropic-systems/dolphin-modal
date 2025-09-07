#!/usr/bin/env python3
"""
Simple test script for the new batch endpoint
"""

import requests
import base64
import json
import time
from pathlib import Path

# Modal endpoints
BATCH_API_URL = 'https://abhishekgautam011--dolphin-parser-dolphinparser-parse-batch.modal.run'

def test_batch_endpoint(num_images=3):
    """Test the batch endpoint with a small number of images"""
    
    # Load test images
    image_dir = Path('demo/page_imgs')
    image_files = list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    if not image_files:
        print("❌ No test images found in demo/page_imgs/")
        return
    
    print(f"📄 Found {len(image_files)} test images")
    print(f"🧪 Testing with {num_images} images")
    
    # Prepare batch payload
    images_data = []
    for i in range(min(num_images, len(image_files))):
        image_file = image_files[i]
        
        with open(image_file, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        images_data.append({
            "image_data": image_data,
            "filename": f'test_{i+1}_{image_file.name}'
        })
    
    batch_payload = {
        "images": images_data
    }
    
    print(f"\n📦 Sending batch request...")
    print(f"   Payload size: ~{len(json.dumps(batch_payload)) / (1024*1024):.1f} MB")
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(BATCH_API_URL, json=batch_payload, timeout=120)
        total_time = time.perf_counter() - start_time
        
        print(f"\n⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS!")
            print(f"   Batch ID: {result.get('batch_id', 'N/A')}")
            print(f"   Processing time: {result.get('processing_time_seconds', 0):.2f}s")
            print(f"   Images processed: {result.get('images_processed', 0)}")
            print(f"   Total elements: {result.get('total_elements', 0)}")
            print(f"   Batch throughput: {result.get('metadata', {}).get('batch_throughput', 0):.2f} images/sec")
            print(f"   Elements/sec: {result.get('metadata', {}).get('elements_per_second', 0):.2f}")
            
            # Show results for each image
            print(f"\n📋 RESULTS BY IMAGE:")
            results = result.get('results', [])
            for i, img_result in enumerate(results):
                filename = img_result.get('filename', f'image_{i}')
                element_count = img_result.get('element_count', 0)
                print(f"   {i+1}. {filename}: {element_count} elements")
                
                # Show first few elements as sample
                elements = img_result.get('elements', [])
                if elements:
                    print(f"      Sample elements:")
                    for j, element in enumerate(elements[:3]):
                        label = element.get('label', 'unknown')
                        text = element.get('text', '')[:50]
                        bbox = element.get('bbox', [])
                        print(f"        - {label}: \"{text}...\" at {bbox}")
                    
                    if len(elements) > 3:
                        print(f"        - ... and {len(elements) - 3} more elements")
            
            # Save results for inspection
            output_file = f"batch_test_results_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Full results saved to: {output_file}")
            
            return True
        else:
            print(f"\n❌ REQUEST FAILED!")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n⏰ REQUEST TIMED OUT after {time.perf_counter() - start_time:.1f}s")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False


def main():
    print("🧪 BATCH ENDPOINT TEST")
    print("=" * 50)
    
    # Test with different batch sizes
    test_sizes = [1, 3, 6]  # Start small, work up
    
    for size in test_sizes:
        print(f"\n{'='*20} TESTING {size} IMAGES {'='*20}")
        
        success = test_batch_endpoint(size)
        
        if success:
            print(f"✅ Test with {size} images: PASSED")
        else:
            print(f"❌ Test with {size} images: FAILED")
            break  # Don't continue if smaller batch fails
        
        if size < max(test_sizes):
            print("\n⏸️  Waiting 5 seconds before next test...")
            time.sleep(5)
    
    print(f"\n🏁 Testing completed!")


if __name__ == "__main__":
    main()