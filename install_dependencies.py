#!/usr/bin/env python3
"""
Dependency installer for Self-Learning Chatbot
Handles installation with proper error handling and fallbacks
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors gracefully"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {str(e)}")
        return False

def install_with_fallback():
    """Try multiple installation methods"""
    print("🤖 Self-Learning Chatbot - Dependency Installer")
    print("=" * 50)
    
    # Method 1: Try pip3 with user flag
    print("\n📦 Method 1: Installing with pip3 --user...")
    if run_command("pip3 install --user -r requirements.txt", "Installing dependencies"):
        return True
    
    # Method 2: Try with break-system-packages (if allowed)
    print("\n📦 Method 2: Installing with --break-system-packages...")
    if run_command("pip3 install --break-system-packages -r requirements.txt", "Installing with system packages"):
        return True
    
    # Method 3: Install core packages individually
    print("\n📦 Method 3: Installing core packages individually...")
    core_packages = [
        "fastapi",
        "uvicorn[standard]", 
        "websockets",
        "aiosqlite",
        "numpy",
        "torch --index-url https://download.pytorch.org/whl/cpu"
    ]
    
    success_count = 0
    for package in core_packages:
        if run_command(f"pip3 install --user {package}", f"Installing {package}"):
            success_count += 1
    
    if success_count >= 4:  # At least core packages installed
        print(f"✅ Installed {success_count}/{len(core_packages)} core packages")
        return True
    
    # Method 4: Try without heavy ML packages
    print("\n📦 Method 4: Installing minimal setup...")
    minimal_packages = [
        "fastapi",
        "uvicorn",
        "websockets", 
        "aiosqlite",
        "jinja2"
    ]
    
    success_count = 0
    for package in minimal_packages:
        if run_command(f"pip3 install --user {package}", f"Installing {package}"):
            success_count += 1
    
    if success_count == len(minimal_packages):
        print("✅ Minimal setup completed - chatbot will work with basic features")
        return True
    
    return False

def create_simple_requirements():
    """Create a simplified requirements file"""
    simple_reqs = """# Simplified requirements for basic functionality
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
aiosqlite==0.19.0
jinja2==3.1.2
pydantic==2.5.0
python-multipart==0.0.6
"""
    
    with open("requirements_simple.txt", "w") as f:
        f.write(simple_reqs)
    
    print("📝 Created requirements_simple.txt for basic functionality")

def main():
    """Main installation process"""
    try:
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("❌ Python 3.8+ required. Current version:", sys.version)
            return False
        
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
        
        # Create simplified requirements
        create_simple_requirements()
        
        # Try installation
        if install_with_fallback():
            print("\n🎉 Installation completed successfully!")
            print("\nNext steps:")
            print("1. Run: python3 run_chatbot.py")
            print("2. Open: http://localhost:8000")
            print("3. Start chatting!")
            return True
        else:
            print("\n⚠️ Installation failed. Try manual installation:")
            print("1. pip3 install --user fastapi uvicorn")
            print("2. pip3 install --user websockets aiosqlite")
            print("3. Then run: python3 run_chatbot.py")
            return False
            
    except Exception as e:
        print(f"💥 Installation script failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)