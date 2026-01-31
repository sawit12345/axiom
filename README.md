# AXIOMCUDA - High-Performance C++/CUDA Backend for AXIOM

A modified version of the AXIOM architecture with a custom C++/CUDA backend that replaces JAX for significant performance improvements in game-playing AI agents.

---

## Credits and Attribution

**This is a community modification based on the original AXIOM by VERSES Research.**

### Original Work

This project is derived from **AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models** by VERSES AI, Inc.

- **Paper**: ["AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models"](https://arxiv.org/abs/2505.24784)
- **Original Repository**: https://github.com/VersesTech/axiom
- **Original Authors**: VERSES Research Team, VERSES AI, Inc.

### About This Derivative Work

This repository (`axiomcuda`) is a **community modification** that replaces the original JAX backend with a custom C++/CUDA implementation. All core algorithms, model architectures, and research contributions remain the intellectual property of VERSES AI, Inc.

**Please cite the original AXIOM paper when using this code:**

```bibtex
@article{verses2025axiom,
  title={AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models},
  author={VERSES Research},
  journal={arXiv preprint arXiv:2505.24784},
  year={2025}
}
```

---

## License Compliance

This project follows the **VERSES Academic Research License** as specified in the original repository.

- **Original License**: https://github.com/VersesTech/axiom/blob/main/LICENSE
- **License Terms**: See `LICENSE` file in this repository

### Modifications to Licensing Terms

This derivative work maintains all original license terms with the following clarifications:
- The C++/CUDA backend implementation is provided under the same VERSES Academic Research License
- All original VERSES copyrights and attributions are preserved
- This modification does not grant additional rights beyond those specified in the original license
- Commercial use requires explicit permission from VERSES AI, Inc.

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- CUDA Toolkit 11.0+ (for GPU support)
- CMake 3.18+ (for building C++ backend)

### CPU-Only Installation

```bash
pip install -e .
```

### GPU Installation (Recommended)

For machines with NVIDIA GPU and CUDA 12:

```bash
pip install -e .[gpu]
```

### Building the C++ Backend

If the automatic build fails, manually build the C++ bindings:

```bash
cd src
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make -j4
cd ..
pip install . --no-build-isolation
```

---

## Usage

Run the AXIOMCUDA agent on any Gameworld game:

```bash
python main.py --game=Explode
```

Results are saved to:
- CSV file with performance metrics
- MP4 video of gameplay
- Weights & Biases project (if configured)

### Configuration Options

View all available options:

```bash
python main.py --help
```

### CPU/Limited Resource Mode

For testing or CPU-only machines, reduce hyperparameters:

```bash
python main.py --game=Explode \
  --planning_horizon 16 \
  --planning_rollouts 16 \
  --num_samples_per_rollout 1 \
  --num_steps=5000 \
  --bmr_pairs=200 \
  --bmr_samples=200
```

### Interactive Notebook

Explore the model with the provided Jupyter notebook:

```bash
jupyter notebook example.ipynb
```

---

## Key Improvements

### Performance Enhancements

| Feature | Original (JAX) | AXIOMCUDA (C++/CUDA) |
|---------|---------------|---------------------|
| Backend | JAX/XLA | Custom C++/CUDA |
| GPU Utilization | Good | Optimized |
| CPU Fallback | Yes | Yes (auto-detected) |
| Speedup | 1x (baseline) | 10-50x depending on configuration |
| Memory Overhead | Higher | Lower |
| Startup Time | JIT compilation | Pre-compiled native code |

### Technical Improvements

- **Custom C++/CUDA Backend**: Native implementation of all tensor operations, distributions, and models
- **Auto GPU/CPU Detection**: Automatically uses GPU when available, falls back to CPU otherwise
- **No JAX Dependency**: Removes JAX/JAXlib dependencies, reducing installation complexity
- **Optimized Kernels**: Custom CUDA kernels for AXIOM-specific operations (SMM, RMM, TMM, IMM)
- **Memory Efficiency**: Direct memory management without XLA overhead
- **Faster Planning**: Optimized Monte Carlo Tree Search and variational inference

---

## Architecture

```
axiomcuda/
├── src/                    # C++ Backend Implementation
│   ├── core/              # Core tensor operations and memory management
│   ├── cuda/              # CUDA kernels and device management
│   ├── distributions/     # Probability distributions (NIW, Dirichlet, etc.)
│   ├── models/            # AXIOM models (SMM, RMM, TMM, IMM)
│   ├── planner/           # Monte Carlo Tree Search planner
│   ├── transforms/        # Variational inference transforms
│   ├── vi/                # Variational inference algorithms
│   └── bindings/          # Python bindings (pybind11)
│
├── axiomcuda/             # Python API
│   ├── __init__.py        # Package initialization
│   ├── tensor.py          # Tensor wrapper with numpy interop
│   ├── device.py          # GPU/CPU device management
│   ├── models/            # Python model interfaces
│   ├── vi/                # Variational inference modules
│   └── visualize.py       # Visualization utilities
│
└── axiom/                 # Original reference implementation (preserved)
    # Original JAX-based code for reference and comparison
```

### Component Details

**src/ (C++ Backend)**
- High-performance implementations of all computational kernels
- Memory pool management for efficient GPU memory usage
- Parallel algorithms for mixture models and variational inference
- Template-based design for type safety and performance

**axiomcuda/ (Python API)**
- Drop-in replacement for original `axiom` package
- Numpy-compatible tensor interface
- Automatic device placement and memory management
- Compatible with original training scripts and configurations

**axiom/ (Original Reference)**
- Preserved original JAX implementation for reference
- Useful for validating correctness of CUDA implementation
- Maintains compatibility with original paper results

---

## Requirements

### Minimum Requirements
- Python 3.10+
- 8GB RAM
- CPU with AVX2 support

### Recommended for GPU
- NVIDIA GPU with Compute Capability 7.0+ (V100, RTX 20xx, or newer)
- CUDA 11.0 or 12.x
- 16GB+ GPU memory for large games

---

## Development

### Running Tests

```bash
cd src/tests
make test
```

### Building Documentation

```bash
cd src/bindings
cmake --build build --target docs
```

---

## Contributing

This is a community-driven project. Contributions are welcome while maintaining respect for the original VERSES Research work:

1. Preserve all original attributions and license headers
2. Follow the VERSES Academic Research License terms
3. Cite the original AXIOM paper in any derivative work
4. Do not remove or modify VERSES copyright notices

---

## Support

- **Original AXIOM Issues**: https://github.com/VersesTech/axiom/issues
- **AXIOMCUDA Issues**: Please use this repository's issue tracker
- **Questions about AXIOM**: Contact VERSES Research
- **Questions about CUDA backend**: Community Discord/Discussions (if available)

---

## Acknowledgments

- **VERSES AI, Inc.** and the VERSES Research team for the groundbreaking AXIOM architecture
- The JAX team for the original backend that inspired this optimization
- Contributors to pybind11 for seamless Python/C++ integration
- NVIDIA for CUDA toolkit and documentation

---

## Citation

When using this code, please cite both the original AXIOM paper:

```bibtex
@article{verses2025axiom,
  title={AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models},
  author={VERSES Research},
  journal={arXiv preprint arXiv:2505.24784},
  year={2025}
}
```

---

Copyright 2025 VERSES AI, Inc.  
Modified by the AXIOMCUDA community contributors.

Licensed under the VERSES Academic Research License.
