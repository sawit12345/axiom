# Copyright 2025 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the "License");
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/axiom/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Setup script for axiomcuda - High-performance C++/CUDA backend for AXIOM

This package provides a CUDA-accelerated implementation of the AXIOM framework
for object-centric probabilistic modeling and inference.

Original AXIOM work by VERSES Research:
- Learning to Play Games in Minutes with Expanding Object-Centric Models
- https://github.com/VersesTech/axiom

Credit: This implementation builds upon the original AXIOM framework developed
by VERSES Research (© 2024-2025 VERSES AI, Inc.)
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# Package version
__version__ = "0.1.2"


class CMakeExtension(Extension):
    """CMake extension for building C++/CUDA code."""
    
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using CMake with CUDA support."""
    
    def run(self) -> None:
        """Run the build process."""
        self.check_cmake()
        self.check_cuda()
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def check_cmake(self) -> None:
        """Check if CMake is installed."""
        try:
            result = subprocess.run(
                ["cmake", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            version_line = result.stdout.decode().split('\n')[0]
            print(f"Found {version_line}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "CMake must be installed to build the axiomcuda package. "
                "Please install CMake 3.18 or higher."
            )
    
    def check_cuda(self) -> None:
        """Check if CUDA is available and set CUDA_HOME if found."""
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        
        if cuda_home and os.path.exists(cuda_home):
            print(f"Using CUDA from: {cuda_home}")
            return
        
        # Try common CUDA installation paths
        possible_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.5",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.3",
            "/usr/local/cuda-12.2",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-11.7",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["CUDA_HOME"] = path
                print(f"Found CUDA at: {path}")
                return
        
        print("WARNING: CUDA not found in standard locations.")
        print("Building with CPU-only support. Set CUDA_HOME environment variable for CUDA support.")
    
    def build_extension(self, ext: CMakeExtension) -> None:
        """Build a single extension using CMake."""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        # Debug or Release build
        cfg = "Debug" if self.debug else "Release"
        
        # CMake arguments
        use_cuda = os.environ.get("USE_CUDA", "ON")
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTS=OFF",
            f"-DUSE_CUDA={use_cuda}",
        ]
        
        # Add pybind11 cmake path if available
        try:
            import pybind11
            pybind11_cmake_dir = pybind11.get_cmake_dir()
            cmake_args.append(f"-Dpybind11_DIR={pybind11_cmake_dir}")
            print(f"Using pybind11 from: {pybind11_cmake_dir}")
        except ImportError:
            print("WARNING: pybind11 not available for import")
        
        # CUDA architecture flags from environment
        cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or os.environ.get("CUDA_ARCHITECTURES")
        if cuda_arch:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
        
        # Build arguments
        build_args = ["--config", cfg]
        
        # Parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Use available CPUs
            import multiprocessing
            ncpus = multiprocessing.cpu_count()
            build_args += ["--", f"-j{ncpus}"]
        
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        
        # Run CMake configuration
        print(f"Configuring CMake in {build_temp}...")
        print(f"CMake source directory: {ext.sourcedir}")
        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        
        # Build
        print(f"Building with CMake...")
        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True,
        )
        
        print(f"Build completed successfully.")


def get_cuda_version() -> str:
    """Get CUDA version if available."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line:
                    parts = line.split("release")
                    if len(parts) > 1:
                        version = parts[1].split(",")[0].strip()
                        return version
    except FileNotFoundError:
        pass
    return "not found"


def read_readme() -> str:
    """Read README file."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


# Setup configuration
setup(
    name="axiomcuda",
    version=__version__,
    author="VERSES Research",
    author_email="verses.research@verses.ai",
    description="AXIOMCUDA: High-performance C++/CUDA backend for AXIOM - Learning to Play Games with Object-Centric Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/VersesTech/axiom",
    license="VERSES Academic Research License",
    license_files=("LICENSE",),
    packages=find_packages(include=["axiomcuda", "axiomcuda.*", "gameworld", "gameworld.*"]),
    package_dir={
        "axiomcuda": "axiomcuda",
        "gameworld": "gameworld",
    },
    ext_modules=[CMakeExtension("axiomcuda.axiomcuda_backend", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.10,<3.13",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.10.0",
        "wandb>=0.18.0",
        "mediapy>=1.2.0",
        "moviepy>=1.0.3",
        "rich>=13.0.0",
        "multimethod>=1.12",
        "gymnasium>=0.28",
        "scipy>=1.10.0",
    ],
    extras_require={
        "cuda": [
            "nvidia-cublas-cu12>=12.0",
            "nvidia-cuda-runtime-cu12>=12.0",
            "nvidia-cudnn-cu12>=9.0",
            "nvidia-cufft-cu12>=11.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Operating System :: POSIX :: Linux",
    ],
    keywords=[
        "machine learning",
        "probabilistic inference",
        "variational inference",
        "mixture models",
        "object-centric learning",
        "CUDA",
        "GPU",
        "axiom",
    ],
    project_urls={
        "Homepage": "https://github.com/VersesTech/axiom",
        "Documentation": "https://docs.verses.ai/axiom",
        "Source": "https://github.com/VersesTech/axiom",
        "Bug Reports": "https://github.com/VersesTech/axiom/issues",
    },
)
