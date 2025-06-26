# setup.py

from setuptools import setup, find_packages
import subprocess
import sys
import os

def install_triton():
    """Install Triton with appropriate version for the system."""
    try:
        import triton
        print("Triton already installed.")
    except ImportError:
        print("Installing Triton...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "triton"])

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Some features may not work.")
        else:
            print(f"CUDA available: {torch.cuda.device_count()} devices")
    except ImportError:
        print("PyTorch not found. Please install PyTorch first.")

def get_requirements():
    """Get requirements from requirements.txt."""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # Add mono4dgs specific requirements
    mono4dgs_requirements = [
        "triton>=2.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "easydict>=1.9",
    ]
    
    requirements.extend(mono4dgs_requirements)
    return requirements

def get_long_description():
    """Get long description from README."""
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Pre-installation checks
check_cuda()
install_triton()

setup(
    name="mono4dgs",
    version="0.1.0",
    description="Monocular Depth- and Pose-Aware 4D Gaussian Splatting",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="4DGaussians + Mono4DGS Contributors",
    author_email="", 
    url="https://github.com/your-repo/mono4dgs",
    packages=find_packages(),
    
    # Include package data
    package_data={
        "": ["*.py", "*.md", "*.txt", "*.yaml", "*.yml"],
        "kernels": ["*.py"],
        "scene": ["*.py"],
        "tests": ["*.py"],
        "arguments": ["*.py"],
        "utils": ["*.py"],
        "gaussian_renderer": ["*.py"],
    },
    
    install_requires=get_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark>=3.4.0",
            "black>=22.0",
            "isort>=5.10",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "foundation": [
            # These would need to be installed separately from their respective repositories
            # "depth-pro @ git+https://github.com/apple/ml-depth-pro.git",
            # "mast3r @ git+https://github.com/naver/mast3r.git",
        ],
        "tensorboard": [
            "tensorboard>=2.8.0",
            "tensorboardX>=2.5",
        ],
    },
    
    python_requires=">=3.8",
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    
    keywords="gaussian splatting, 3d reconstruction, neural rendering, computer vision, depth estimation",
    
    entry_points={
        "console_scripts": [
            "mono4dgs-train=trainer_mono4dgs:main",
            "mono4dgs-test=tests.test_deform_kernel:run_performance_benchmark",
        ],
    },
    
    # Custom commands
    cmdclass={},
    
    # Zip safe
    zip_safe=False,
)

# Post-installation setup
def post_install():
    """Run post-installation setup."""
    print("\n" + "="*60)
    print("MONO4DGS INSTALLATION COMPLETE")
    print("="*60)
    
    # Check if CUDA is available before compiling kernels
    try:
        import torch
        if torch.cuda.is_available():
            try:
                from kernels.triton_deform import compile_kernels
                print("Compiling Triton kernels...")
                compile_kernels()
                print("✓ Kernel compilation successful")
            except Exception as e:
                print(f"✗ Kernel compilation failed: {e}")
                print("  You may need to compile kernels manually at runtime.")
        else:
            print("⚠ CUDA not available, skipping kernel compilation")
            print("  Kernels will be compiled at runtime if CUDA becomes available.")
    except ImportError:
        print("⚠ PyTorch not found, skipping kernel compilation")
    
    print("\nNext steps:")
    print("1. Install foundation models (optional):")
    print("   - Depth Pro: git clone https://github.com/apple/ml-depth-pro.git")
    print("   - MASt3R: git clone https://github.com/naver/mast3r.git")
    print("2. Run tests: python -m pytest tests/ -v")
    print("3. Try the example: python trainer_mono4dgs.py --config configs/example.yaml")
    print("\nFor help: python trainer_mono4dgs.py --help")
    print("="*60)

if __name__ == "__main__":
    # Run post-installation setup if this is run directly
    post_install()