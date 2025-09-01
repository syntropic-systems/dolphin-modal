#!/usr/bin/env python3
"""
Deployment script for Dolphin Modal app
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return None

def main():
    """Deploy Dolphin to Modal"""
    print("🚀 Dolphin Modal Deployment Script")
    print("=" * 50)
    
    # Check if Modal is installed
    try:
        subprocess.run(["modal", "--version"], check=True, capture_output=True)
        print("✅ Modal CLI is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Modal CLI not found. Install with:")
        print("   pip install modal")
        sys.exit(1)
    
    # Check if user is logged in
    result = run_command("modal token current", "Checking Modal authentication")
    if result is None:
        print("🔑 Please authenticate with Modal:")
        run_command("modal token new", "Setting up Modal authentication")
    
    # Deploy the app
    print("\n🚀 Deploying to Modal...")
    deploy_output = run_command("modal deploy modal_app.py", "Deploying Dolphin app")
    
    if deploy_output:
        print("\n🎉 Deployment successful!")
        print("=" * 50)
        
        # Extract URL from deployment output
        for line in deploy_output.split('\n'):
            if 'https://' in line and 'modal.run' in line:
                url = line.strip()
                print(f"🌐 API URL: {url}")
                print(f"📄 Docs: {url}/docs")
                print(f"🏥 Health: {url}/health")
                break
        
        print("\n📝 Usage Example:")
        print("curl -X POST \\")
        print("  -F 'file=@path/to/image.jpg' \\")
        print("  https://your-app.modal.run/parse")
        
        print("\n⚡ Features:")
        print("  • GPU-accelerated processing (A100)")
        print("  • Auto-scaling (0 to 10 workers)")  
        print("  • Same API as your current setup")
        print("  • ~0.5-1s processing per image")
        
    else:
        print("❌ Deployment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()