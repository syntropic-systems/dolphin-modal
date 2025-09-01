#!/usr/bin/env python3
"""
Test script for Modal Dolphin API
"""

import requests
import time
import sys
from pathlib import Path

def test_modal_api(api_url, image_path):
    """Test the Modal API with an image"""
    
    print(f"🧪 Testing Modal API: {api_url}")
    print(f"📷 Image: {image_path}")
    print("=" * 50)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        health_response = requests.get(f"{api_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {health_response.json()}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
        
        print("\n2. Testing document parsing...")
        
        # Prepare file for upload
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            
            # Make request
            start_time = time.time()
            response = requests.post(
                f"{api_url}/parse",
                files=files,
                timeout=120  # 2 minute timeout
            )
            end_time = time.time()
            
            request_time = end_time - start_time
            print(f"⏱️  Request time: {request_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Parsing successful!")
            print(f"   Request ID: {result.get('request_id', 'N/A')}")
            print(f"   Filename: {result.get('filename', 'N/A')}")
            print(f"   Processing time: {result.get('processing_time_seconds', 'N/A')}s")
            print(f"   Content length: {result.get('metadata', {}).get('content_length', 'N/A')} chars")
            
            # Show first 200 characters of result
            content = result.get('results', {}).get('content', '')
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   Content preview: {preview}")
            
            return True
        else:
            print(f"❌ Parsing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout (>2 minutes)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print("Usage: python test_modal_api.py <modal_api_url> [image_path]")
        print("Example: python test_modal_api.py https://your-app.modal.run")
        sys.exit(1)
    
    api_url = sys.argv[1].rstrip('/')
    image_path = sys.argv[2] if len(sys.argv) > 2 else "./demo/page_imgs/page_1.jpeg"
    
    success = test_modal_api(api_url, image_path)
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n📝 Integration ready:")
        print("   • Same multipart/form-data request as current API")
        print("   • Synchronous responses (no polling needed)")  
        print("   • GPU-accelerated processing")
        print("   • Auto-scaling for burst traffic")
    else:
        print("\n❌ Tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()