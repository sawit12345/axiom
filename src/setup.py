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
Setup script for axiomcuda_backend - CUDA-accelerated backend for Axiom
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for building C++/CUDA code."""
    
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using CMake."""
    
    def run(self) -> None:
        """Run the build process."""
        self.check_cmake()
        self.check_cuda()
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def check_cmake(self) -> None:
        """Check if CMake is installed."""
        try:
            subprocess.run(
                ["cmake", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
    
    def check_cuda(self) -> None:
        """Check if CUDA is available."""
        cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH"))
        
        if cuda_home and os.path.exists(cuda_home):
            print(f"Found CUDA at: {cuda_home}")
            return
        
        # Try to find CUDA
        possible_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-11.7",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                os.environ["CUDA_HOME"] = path
                print(f"Found CUDA at: {path}")
                return
        
        print("WARNING: CUDA not found. Building with CPU-only support.")
    
    def build_extension(self, ext: CMakeExtension) -> None:
        """Build a single extension."""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        # Debug or Release build
        cfg = "Debug" if self.debug else "Release"
        
        # CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTS=OFF",
            "-DUSE_CUDA=ON",
        ]
        
        # CUDA architecture flags
        cuda_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if cuda_arch:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
        
        # Build arguments
        build_args = ["--config", cfg]
        
        # Parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["--", "-j4"]
        
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        
        # Run CMake
        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        
        # Build
        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True,
        )


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
                    # Extract version like "12.0" from "release 12.0"
                    parts = line.split("release")
                    if len(parts) > 1:
                        version = parts[1].split(",")[0].strip()
                        return version
    except FileNotFoundError:
        pass
    return "unknown"


def read_readme() -> str:
    """Read README file."""
    readme_path = Path(__file__).parent.parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


# Setup configuration
setup(
    name="axiomcuda-backend",
    version="0.1.0",
    author="VERSES AI, Inc.",
    author_email="research@verses.ai",
    description="CUDA-accelerated backend for Axiom probabilistic inference",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/VersesTech/axiom",
    license="VERSES Academic Research License",
    license_files=("LICENSE",),
    packages=["axiomcuda"],
    package_dir={"axiomcuda": "axiomcuda"},
    ext_modules=[CMakeExtension("axiomcuda_backend", sourcedir="src")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords=[
        "machine learning",
        "probabilistic inference",
        "variational inference",
        "mixture models",
        "CUDA",
        "GPU",
    ],
    project_urls={
        "Bug Reports": "https://github.com/VersesTech/axiom/issues",
        "Source": "https://github.com/VersesTech/axiom",
        "Documentation": "https://docs.verses.ai/axiom",
    },
)
